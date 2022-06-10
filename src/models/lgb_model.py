from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn import metrics

from ..config import Config
from .base_model import BaseModel


def accuracy_score(
    y_true: np.ndarray, y_pred: np.ndarray, weight: np.ndarray, group: np.ndarray
) -> tuple[str, float, bool]:
    acc = metrics.accuracy_score(y_true, y_pred >= 0.5)
    return "accuracy", acc, True


def _lgb_datasets(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Make copies so that original datasets remain unchanged
    train_df = train_df.copy()
    test_df = test_df.copy()

    # Drop Transported and kfold
    drop = ["Transported", "kfold"]
    dropped = train_df[drop].values
    train_df = train_df.drop(drop, axis=1)

    # Drop PassengerId
    passenger_id = test_df["PassengerId"].values
    test_df = test_df.drop("PassengerId", axis=1)

    # Add suffix to index and store indices
    # So that the dataframes can be merged and split
    train_df = train_df.rename("train_{}".format)
    test_df = test_df.rename("test_{}".format)

    tr_idx = train_df.index
    te_idx = test_df.index

    # Merge
    df = pd.concat([train_df, test_df])

    oh_cols = ["CabinDeck", "HomePlanet", "Destination", "GroupSize"]
    df_cols: list[str] = df.columns.to_list()

    for oh_col in oh_cols:
        # Get all columns associated with the one-hot column
        columns = [column for column in df_cols if column.startswith(f"{oh_col}_")]

        # .idxmax() returns that column name which has the maximum value in the row
        values = df[columns].idxmax(axis=1)

        # Get all levels and make a mapping from level to index
        levels = values.value_counts().index
        mapping = {level: idx for idx, level in enumerate(levels)}

        # Add column with the mapping and specify type as category
        df[oh_col] = values.map(mapping).astype("category")

        # Drop one-hot columns
        df = df.drop(columns, axis=1)

    # Make sure other categorical features have the correct type
    missing = (col for col in df.columns if col.endswith("_missing"))
    others = ["CryoSleep", "VIP", "Alone", "CabinNum", "GroupId", *missing]
    df[others] = df[others].astype("category")

    # Split and add dropped columns
    train_df = df.loc[tr_idx, :]
    train_df[drop] = dropped

    test_df = df.loc[te_idx, :]
    test_df["PassengerId"] = passenger_id

    return train_df, test_df


class LGBMClassifierModel(BaseModel):
    name = "lgb"
    long_name = "LightGBM"

    def __init__(self, *, meta_mode: bool = False) -> None:
        super().__init__(preprocess=True, meta_mode=meta_mode)

    def preprocess_datasets(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        return _lgb_datasets(train_df, test_df)

    def optuna_parameters(self, trial: optuna.Trial) -> dict[str, Any]:
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 31, 100, log=True),
            "max_depth": trial.suggest_int("max_depth", 1, 100, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1.0),
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
            "min_child_samples": trial.suggest_int(
                "min_child_samples", 20, 50, log=True
            ),
            "subsample_for_bin": trial.suggest_int("subsample_for_bin", 2000, 8000),
        }

        if self.use_pruner is True:
            params["callbacks"] = [
                optuna.integration.LightGBMPruningCallback(trial, "accuracy", "valid_1")
            ]

        return params

    def init_classifier(self, params: dict[str, Any]) -> lgb.LGBMClassifier:
        init_params = {
            "num_leaves": params.get("num_leaves", 31),
            "max_depth": params.get("max_depth", -1),
            "learning_rate": params.get("learning_rate", 0.1),
            "n_estimators": params.get("n_estimators", 100),
            "reg_alpha": params.get("reg_alpha", 0.0),
            "reg_lambda": params.get("reg_lambda", 0.0),
            "min_child_samples": params.get("min_child_samples", 20),
            "subsample_for_bin": params.get("subsample_for_bin", 200000),
            "n_jobs": Config.N_JOBS,
            "verbose": -1,
            "objective": "binary",
        }

        return lgb.LGBMClassifier(**init_params)

    def extra_fit_parameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        callbacks = params.get("callbacks", []) + [lgb.log_evaluation(period=0)]
        return {
            "eval_set": [(X_train, y_train), (X_val, y_val)],
            "eval_metric": ["logloss", accuracy_score],
            "callbacks": callbacks,
        }
