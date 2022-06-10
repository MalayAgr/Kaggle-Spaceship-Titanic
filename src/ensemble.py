import pandas as pd
from rich import print

from .config import Config
from .models import BaseModel


class Ensemble:
    def __init__(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame, exclude: list[str] = None
    ) -> None:
        self.train_df = train_df
        self.test_df = test_df

        self.exclude = exclude

        models = BaseModel.REGISTRY.keys()

        if exclude is not None:
            models = models - exclude

        self.models = [BaseModel.REGISTRY[model]() for model in models]

        columns = [f"{m}_preds" for m in models]
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
                n_trials=Config.L1_N_TRIALS,
            )

            print(f"\tBest parameters: {params}")

            print("\tTraining model with best parameters...\n")

            preds, test_preds, acc = model.train(
                train_df=self.train_df,
                test_df=self.test_df,
                params=params,
            )

            self.meta_train_df[f"{model.name}_preds"] = preds
            self.meta_test_df[f"{model.name}_preds"] = test_preds

            print("\tDone!\n")

    def train_level_two_model(self) -> pd.DataFrame:
        print("Training a Logistic Regression model as level 2 model...")

        train_df = self.meta_train_df
        test_df = self.meta_test_df

        model = BaseModel.REGISTRY["lr"]()

        print("\tFinding optimal hyperparameters using Optuna...\n")
        params = model.hyperparameter_search(
            train_df=train_df,
            test_df=test_df,
            n_trials=Config.L2_N_TRIALS,
        )

        print(f"\tBest params: {params}\n")

        print("\tTraining model with optimal parameters...\n")

        _, test_preds, _ = model.train(
            train_df=train_df, test_df=test_df, params=params
        )

        print("\tDone!")

        self.meta_test_df["Transported"] = test_preds >= 0.5

        return self.meta_test_df
