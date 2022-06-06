from typing import Callable

import numpy as np
import pandas as pd
from sklearn import impute, preprocessing


class CategoricalImputer:
    def __init__(self, feature_mode_cols: list[str] = None) -> None:
        if feature_mode_cols is None:
            feature_mode_cols = []

        if "CryoSleep" not in feature_mode_cols:
            feature_mode_cols.append("CryoSleep")

        self.feature_mode_cols = feature_mode_cols

    def _impute_using_feature_level_mode(self, df: pd.DataFrame) -> pd.DataFrame:
        columns = self.feature_mode_cols
        df[columns] = df[columns].fillna(df[columns].mode().iloc[0])
        return df

    def _impute_cabin_helper(
        self, df: pd.DataFrame, groupby: str | list[str]
    ) -> pd.DataFrame:
        # Find all passengers belonging to groups where
        # At least one member has a non-null column value
        temp = df.groupby(groupby).filter(lambda x: x["Cabin"].notna().any())

        # Replace by mode
        func = lambda x: x.fillna(x.mode().iloc[0]) if x.isna().any() else x
        temp["Cabin"] = temp.groupby(groupby)["Cabin"].transform(func)

        # Update the original dataframe
        df.loc[temp.index, "Cabin"] = temp["Cabin"]

        return df

    def _impute_cabin(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._impute_cabin_helper(df, groupby="GroupId")
        df = self._impute_cabin_helper(df, groupby=["HomePlanet", "Destination"])
        return df

    def _impute_vip_by_probs(self, df: pd.DataFrame) -> pd.DataFrame:
        probs = df["VIP"].value_counts(normalize=True)
        values = np.random.choice([False, True], size=df["VIP"].isna().sum(), p=probs)

        df.loc[df["VIP"].isna(), "VIP"] = values
        df["VIP"] = df["VIP"].astype(bool)

        return df

    def _impute_vip(self, df: pd.DataFrame) -> pd.DataFrame:
        df.loc[
            (df["VIP"].isna()) & (df["TotalExpense"] == 0.0) & (~df["CryoSleep"]), "VIP"
        ] = False

        df.loc[(df["VIP"].isna()) & (df["Age"] <= 12), "VIP"] = False

        df.loc[(df["VIP"].isna()) & (df["HomePlanet"] == "Earth"), "VIP"] = False

        df.loc[
            (df["VIP"].isna())
            & (df["Age"] >= 18)
            & (~df["CryoSleep"])
            & (df["Destination"] != "55 Cancri e"),
            "VIP",
        ] = True

        df = self._impute_vip_by_probs(df)

        return df

    def imputer_map(self) -> dict[str, Callable[[pd.DataFrame], pd.DataFrame]]:
        return {
            "feature_mode": self._impute_using_feature_level_mode,
            "cabin": self._impute_cabin,
            "vip": self._impute_vip,
        }

    def get_imputers(
        self, exclude: list[str] = None
    ) -> list[Callable[[pd.DataFrame], pd.DataFrame]]:
        imputer_map = self.imputer_map()

        if exclude is None:
            return list(imputer_map.values())

        if imputer_map.keys().isdisjoint(exclude):
            recognized = ", ".join(imputer_map)
            raise ValueError(
                f"exclude has unrecognized features. Recognized features: {recognized}."
            )

        return [func for feature, func in imputer_map.items() if feature not in exclude]

    def __call__(
        self, df: pd.DataFrame, *, copy: bool = False, exclude: list[str] | str = None
    ) -> pd.DataFrame:

        if isinstance(exclude, str):
            exclude = [exclude]

        imputers = self.get_imputers(exclude=exclude)

        df = df.copy() if copy is True else df

        for imputer in imputers:
            df = imputer(df)

        return df


def numeric_imputer(
    df: pd.DataFrame, numeric_cols: list[str], has_labels: bool = False
) -> pd.DataFrame:
    x = df

    if has_labels is True:
        transported = df["Transported"]
        x = df.drop("Transported", axis=1)

    # Scale values
    scaler = preprocessing.StandardScaler()
    x[numeric_cols] = scaler.fit_transform(x[numeric_cols])

    # Impute missing values
    imputer = impute.KNNImputer(n_neighbors=5, weights="distance")
    x = imputer.fit_transform(x)

    if has_labels is True:
        x = np.hstack((x, transported.values.reshape(-1, 1)))

    return pd.DataFrame(x, columns=df.columns, index=df.index)
