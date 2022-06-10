from typing import Any

import optuna
from sklearn import ensemble, tree

from .base_model import BaseModel


class AdaBoostModel(BaseModel):
    name = "ada"
    long_name = "AdaBoost"

    def __init__(self) -> None:
        super().__init__(use_pruner=False)

    def optuna_parameters(self, trial: optuna.Trial) -> dict[str, Any]:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.1, 5.0, log=True),
        }

        tune_estimator = trial.suggest_categorical("tune_estimator", [True, False])

        if tune_estimator:
            max_depth = trial.suggest_int("max_depth", 1, 50)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 10, log=True)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5, log=True)
            ccp_alpha = trial.suggest_float("ccp_alpha", 0.01, 1.0, log=True)

            params["base_estimator"] = tree.DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                ccp_alpha=ccp_alpha,
            )

        return params

    def init_classifier(self, params: dict[str, Any]) -> ensemble.AdaBoostClassifier:
        init_params = {
            "base_estimator": params.get("base_estimator"),
            "n_estimators": params.get("n_estimators", 50),
            "learning_rate": params.get("learning_rate", 1.0),
        }

        return ensemble.AdaBoostClassifier(**init_params)
