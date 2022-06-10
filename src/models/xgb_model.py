from typing import Any

import numpy as np
import optuna
import xgboost as xgb

from ..config import Config
from .base_model import BaseModel


class XGBClassifierModel(BaseModel):
    name = "xgb"
    long_name = "XGBoost"

    def optuna_parameters(self, trial: optuna.Trial) -> dict[str, Any]:
        params = {
            "max_depth": trial.suggest_int("max_depth", 1, 11),
            "n_estimators": trial.suggest_int("n_estimators", 5, 500),
            "alpha": trial.suggest_uniform("alpha", 0.0, 5.0),
            "lambda": trial.suggest_float("lambda", 1.0, 5.0, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.8, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.2, 1.0),
            "min_child_weight": trial.suggest_uniform("min_child_weight", 1, 100),
            "sampling_method": trial.suggest_categorical(
                "sampling_method", ["uniform", "gradient_based"]
            ),
            "early_stopping_rounds": trial.suggest_int(
                "early_stopping_rounds", 5, 20, step=5
            ),
        }

        if self.use_pruner is True:
            params["callbacks"] = [
                optuna.integration.XGBoostPruningCallback(trial, "validation_1-logloss")
            ]

        return params

    def init_classifier(self, params: dict[str, Any]) -> xgb.XGBClassifier:
        init_params = {
            "max_depth": params.get("max_depth", 6),
            "n_estimators": params.get("n_estimators", 100),
            "alpha": params.get("alpha", 0.0),
            "lambda": params.get("lambda", 1.0),
            "learning_rate": params.get("learning_rate", 0.3),
            "colsample_bytree": params.get("colsample_bytree", 1.0),
            "colsample_bylevel": params.get("colsample_bylevel", 1.0),
            "min_child_weight": params.get("min_child_weight", 1.0),
            "sampling_method": params.get("sampling_method", "uniform"),
            "early_stopping_rounds": params.get("early_stopping_rounds", None),
            "use_label_encoder": False,
            "callbacks": params.get("callbacks", None),
            "n_jobs": Config.N_JOBS,
            "eval_metric": "logloss",
        }

        return xgb.XGBClassifier(**init_params)

    def extra_fit_parameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "eval_set": [(X_train, y_train), (X_val, y_val)],
            "verbose": False,
        }
