from typing import Any

import optuna
from sklearn import linear_model

from .base_model import BaseModel


class LogisticRegressionModel(BaseModel):
    name = "lr"
    long_name = "Logistic Regression"

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

    def init_classifier(
        self, params: dict[str, Any]
    ) -> linear_model.LogisticRegression:
        init_params = {
            "tol": params.get("tol", 1e-4),
            "C": params.get("C", 1.0),
            "solver": params.get("solver", "lbfgs"),
            "max_iter": 1000,
            "verbose": False,
        }
        return linear_model.LogisticRegression(**init_params)
