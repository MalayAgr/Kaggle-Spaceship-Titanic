from __future__ import annotations

import functools
from typing import Any, Protocol, Type

import numpy as np
import optuna
import pandas as pd
import sklearn
from lightgbm import train
from sklearn import metrics


class Classifier(Protocol):
    def fit(self, X, y, *args, **kwargs) -> Classifier:
        ...

    def predict(self, X, *args, **kwargs) -> np.ndarray:
        ...

    def predict_proba(self, X, *args, **kwargs) -> np.ndarray:
        ...


class BaseModel:
    name: str = None
    REGISTRY: dict[str, Type[BaseModel]] = {}

    def __init__(self, *, use_pruner: bool = True) -> None:
        self.use_pruner = use_pruner

    def __init_subclass__(cls, **kwargs) -> None:
        if (name := cls.name) is not None:
            BaseModel.REGISTRY[name] = cls

    def preprocess_datasets(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        return train_df, test_df

    def hyperparameter_search(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame, n_trials: int
    ) -> dict[str, Any]:
        train_df, test_df = self.preprocess_datasets(train_df, test_df)

        objective = self.objective
        objective = functools.partial(objective, train_df=train_df, test_df=test_df)

        sampler = optuna.samplers.TPESampler(seed=42)

        pruner = optuna.pruners.HyperbandPruner() if self.use_pruner is True else None

        study = optuna.create_study(
            sampler=sampler,
            pruner=pruner,
            direction="maximize",
        )

        v = optuna.logging.get_verbosity()
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study.optimize(objective, n_trials=n_trials)

        optuna.logging.set_verbosity(v)

        return study.best_params

    def optuna_parameters(self, trial: optuna.Trial) -> dict[str, Any]:
        raise NotImplementedError("Base classes need to implement optuna_parameters().")

    def objective(
        self, trial: optuna.Trial, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> float:
        params = self.optuna_parameters(trial=trial)

        _, _, acc = self.train(
            train_df=train_df, test_df=test_df, params=params, verbose=False
        )

        return acc

    def init_classifier(self, params: dict[str, Any]) -> Classifier:
        raise NotImplementedError("Base classes need to implement init_classifier().")

    def extra_fit_parameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        return {}

    def _train_fold(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        fold: int,
        params: dict[str, Any],
        drop: list[str],
        *,
        verbose: bool = True,
    ) -> tuple[np.ndarray, float]:
        train = train_df[train_df["kfold"] != fold]

        y_train = train["Transported"]
        X_train = train.drop(drop, axis=1)

        val = train_df[train_df["kfold"] == fold]

        y_val = val["Transported"]
        X_val = val.drop(drop, axis=1)

        clf = self.init_classifier(params=params)

        fit_params = self.extra_fit_parameters(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            params=params,
        )

        clf.fit(
            X=X_train,
            y=y_train,
            **fit_params,
        )

        val_pred = clf.predict(X_val)
        acc = metrics.accuracy_score(y_val, val_pred)

        if verbose is True:
            print(f"\tFold {fold + 1} - Accuracy = {acc: .4f}")

        train_df.loc[val.index, "preds"] = clf.predict_proba(X_val)[:, 1]

        return clf.predict_proba(test_df)[:, 1], acc

    def train(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        params: dict[str, Any],
        *,
        preprocess: bool = False,
        verbose: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, float]:

        if preprocess is True:
            train_df, test_df = self.preprocess_datasets(
                train_df=train_df, test_df=test_df
            )

        train_df = train_df.copy()
        test_df = test_df.drop("PassengerId", axis=1)

        train_df["preds"] = np.nan

        drop = ["Transported", "preds", "kfold"]

        total_acc, test_preds = 0.0, []

        for fold in range(5):
            test_pred, acc = self._train_fold(
                train_df=train_df,
                test_df=test_df,
                fold=fold,
                params=params,
                drop=drop,
                verbose=verbose,
            )

            total_acc += acc
            test_preds.append(test_pred)

        acc = total_acc / 5

        if verbose is True:
            print(f"\tOverall accuracy = {acc: .4f}")

        test_preds = np.vstack(test_preds)
        test_preds = np.mean(test_preds, axis=0)

        return train_df["preds"].values, test_preds, acc


class BaseOptunaCVModel(BaseModel):
    pass
