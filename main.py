import pandas as pd
from src.config import Config
from src.ensemble import Ensemble


def main():
    train_df = pd.read_csv(Config.filepath("train_prepared_both_le.csv"))
    test_df = pd.read_csv(Config.filepath("test_prepared_both_le.csv"))

    exclude = None

    ensemble = Ensemble(train_df=train_df, test_df=test_df, exclude=exclude)

    ensemble.train_level_one_models()

    model_name = "lr"
    result = ensemble.train_level_two_model(model_name=model_name)

    submission = result[["PassengerId", "Transported"]]
    submission.to_csv(Config.filepath("submission.csv"), index=False)


if __name__ == "__main__":
    main()
