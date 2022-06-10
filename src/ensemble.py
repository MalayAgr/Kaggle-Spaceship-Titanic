import pandas as pd
from rich import print

from .config import Config
from .models import BaseModel


class Ensemble:
    def __init__(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        exclude: list[str] = None,
        l1_trials: int = Config.L1_N_TRIALS,
        l2_trials: int = Config.L2_N_TRIALS,
    ) -> None:
        self.train_df = train_df
        self.test_df = test_df
        self.l1_trials = l1_trials
        self.l2_trials = l2_trials

        self.exclude = exclude

        models = BaseModel.REGISTRY.keys()

        if exclude is not None:
            models = models - exclude

        self.models = [BaseModel.REGISTRY[model]() for model in models]

        columns = [f"{model.name}_preds" for model in self.models]
        extra_cols = ["Transported", "kfold"]

        meta_train_df = pd.DataFrame(columns=columns + extra_cols)
        meta_train_df[extra_cols] = train_df[extra_cols]

        self.meta_train_df = meta_train_df

        meta_test_df = pd.DataFrame(columns=["PassengerId"] + columns)
        meta_test_df["PassengerId"] = test_df["PassengerId"]

        self.meta_test_df = meta_test_df

    def train_level_one_models(self) -> None:
        for model in self.models:
            print(f"{model.long_name}:")

            print("\tFinding hyperparameters using Optuna...\n")
            params = model.hyperparameter_search(
                train_df=self.train_df,
                test_df=self.test_df,
                n_trials=self.l1_trials,
            )

            print(f"\tBest parameters: {params}")

            print("\tTraining model with best parameters...\n")

            preds, test_preds, _ = model.train(
                train_df=self.train_df,
                test_df=self.test_df,
                params=params,
            )

            self.meta_train_df[f"{model.name}_preds"] = preds
            self.meta_test_df[f"{model.name}_preds"] = test_preds

            print("\tDone!\n")

    def train_level_two_model(self, model_name="lr") -> pd.DataFrame:
        train_df = self.meta_train_df
        test_df = self.meta_test_df

        klass = BaseModel.REGISTRY[model_name]
        model = klass()

        print(f"Training {model.long_name} as level 2 model...")

        print("\tFinding optimal hyperparameters using Optuna...\n")
        params = model.hyperparameter_search(
            train_df=train_df,
            test_df=test_df,
            n_trials=self.l2_trials,
        )

        print(f"\tBest params: {params}\n")

        print("\tTraining model with optimal parameters...\n")

        _, test_preds, _ = model.train(
            train_df=train_df, test_df=test_df, params=params
        )

        print("\tDone!")

        self.meta_test_df["Transported"] = test_preds >= 0.5

        return self.meta_test_df
