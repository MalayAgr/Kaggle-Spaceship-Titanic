from typing import Any

import optuna
from sklearn import ensemble

from ..config import Config
from .base_model import BaseModel


class RandomForestClassifierModel(BaseModel):
    name = "rf"
    long_name = "Random Forest"

    def __init__(self, *, meta_mode: bool = False) -> None:
        super().__init__(use_pruner=False, meta_mode=meta_mode)

    def optuna_parameters(self, trial: optuna.Trial) -> dict[str, Any]:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 1, 500),
            "max_depth": trial.suggest_int("max_depth", 1, 50),
            "min_samples_split": trial.suggest_int(
                "min_samples_split", 2, 10, log=True
            ),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5, log=True),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", None]
            ),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "ccp_alpha": trial.suggest_float("ccp_alpha", 0.01, 1.0, log=True),
        }

        if params["bootstrap"] is True:
            params["max_samples"] = trial.suggest_float(
                "max_samples", 0.01, 1.0, log=True
            )

        return params

    def init_classifier(
        self, params: dict[str, Any]
    ) -> ensemble.RandomForestClassifier:
        init_params = {
            "n_estimators": params.get("n_estimators", 100),
            "max_depth": params.get("max_depth"),
            "min_samples_split": params.get("min_samples_split", 2),
            "min_samples_leaf": params.get("min_samples_leaf", 1),
            "max_features": params.get("max_features", "sqrt"),
            "bootstrap": params.get("bootstrap", True),
            "ccp_alpha": params.get("ccp_alpha", 0.0),
            "max_samples": params.get("max_samples"),
            "n_jobs": Config.N_JOBS,
        }

        return ensemble.RandomForestClassifier(**init_params)
