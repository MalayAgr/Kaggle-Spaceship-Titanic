from typing import Any

import optuna
from sklearn.linear_model import LogisticRegression

from .base_model import BaseModel


class LogisticRegressionModel(BaseModel):
    name = "lr"

    def __init__(self) -> None:
        super().__init__(use_pruner=False)

    def optuna_parameters(self, trial: optuna.Trial) -> dict[str, Any]:
        return {
            "tol": trial.suggest_float("tol", 1e-6, 1e-4, log=True),
            "C": trial.suggest_float("C", 0.5, 2.0, log=True),
            "solver": trial.suggest_categorical(
                "solver", ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
            ),
        }

    def init_classifier(self, params: dict[str, Any]) -> LogisticRegression:
        init_params = {
            "tol": params.get("tol", 1e-4),
            "C": params.get("C", 1.0),
            "solver": params.get("solver", "lbfgs"),
            "max_iter": 1000,
            "verbose": False,
        }
        return LogisticRegression(**init_params)
