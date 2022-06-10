import argparse

import pandas as pd
from src.config import Config
from src.ensemble import Ensemble


def main(args: argparse.Namespace):
    train_df = pd.read_csv(Config.filepath("train_prepared_both_le.csv"))
    test_df = pd.read_csv(Config.filepath("test_prepared_both_le.csv"))

    if (exclude := args.exclude) is not None:
        exclude = exclude.split(",")

    l1_trials = args.l1_trials
    l2_trials = args.l2_trials

    ensemble = Ensemble(
        train_df=train_df,
        test_df=test_df,
        exclude=exclude,
        l1_trials=l1_trials,
        l2_trials=l2_trials,
    )

    ensemble.train_level_one_models()

    model_name = args.l2_model
    result = ensemble.train_level_two_model(model_name=model_name)

    submission = result[["PassengerId", "Transported"]]
    submission.to_csv(Config.filepath("submission.csv"), index=False)


if __name__ == "__main__":
    desc = "Train an ensemble of models for Spaceship Titanic."
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "-e",
        "--exclude",
        type=str,
        required=False,
        default=None,
        help="Names of models to exclude from the ensemble, comma-separated.",
    )

    parser.add_argument(
        "-l1t",
        "-l1_trials",
        type=int,
        required=False,
        default=100,
        help="Number of Optuna trials for Level 1 models.",
    )

    parser.add_argument(
        "-l2t",
        "-l2_trials",
        type=int,
        required=False,
        default=20,
        help="Number of Optuna trials for Level 2 model.",
    )

    parser.add_argument(
        "-l2m",
        "--l2_model",
        type=str,
        required=False,
        default="lr",
        help="Name of the model to use as the Level 2 model.",
    )

    args = parser.parse_args()

    main(args)
