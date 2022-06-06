import pandas as pd
from sklearn import model_selection


def make_folds(df: pd.DataFrame, n_folds: int = 5) -> pd.DataFrame:
    df = df.reset_index()

    df["kfold"] = -1
    kf = model_selection.KFold(n_splits=5, random_state=42, shuffle=True)

    for idx, (_, val_idx) in enumerate(kf.split(df)):
        df.loc[val_idx, "kfold"] = idx

    df = df.set_index("PassengerId")

    return df
