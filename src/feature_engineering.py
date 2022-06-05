from typing import Callable

import pandas as pd


class FeatureEngineer:
    def __init__(
        self,
        missing_value_cols: list[str] = None,
        alone: bool = True,
        total_expense_missing: bool = True,
    ) -> None:
        self.expenditure_cols = [
            "RoomService",
            "FoodCourt",
            "ShoppingMall",
            "Spa",
            "VRDeck",
        ]

        self.missing_value_cols = missing_value_cols
        self.alone = alone
        self.total_expense_missing = total_expense_missing

    def _from_passenger_id(self, df: pd.DataFrame) -> pd.DataFrame:
        split_id = df["PassengerId"].str.split("_", expand=True)

        df["GroupId"] = split_id[0]
        df["GroupSize"] = df.groupby("GroupId")["GroupId"].transform("count")

        if self.alone is True:
            df["Alone"] = df["GroupSize"] == 1

        return df

    def _from_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        for column in self.missing_value_cols:
            df[f"{column}_missing"] = df[column].isna()

        if self.total_expense_missing is True:
            df["TotalExpense_missing"] = (
                df[self.expenditure_cols].sum(axis=1, skipna=False).isna()
            )

        return df

    def _from_expenditure_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df["TotalExpense"] = df[self.expenditure_cols].sum(axis=1)
        return df

    def _from_cabin(self, df: pd.DataFrame) -> pd.DataFrame:
        df[["CabinDeck", "CabinNum", "CabinSide"]] = df["Cabin"].str.split(
            "/", expand=True
        )
        return df

    def engineer_map(self) -> dict[str, Callable[[pd.DataFrame], pd.DataFrame]]:
        return {
            "passenger_id": self._from_passenger_id,
            "missing_values": self._from_missing_values,
            "expenditure_columns": self._from_expenditure_columns,
            "cabin": self._from_cabin,
        }

    def get_engineers(
        self, exclude: list[str] = None
    ) -> list[Callable[[pd.DataFrame], pd.DataFrame]]:
        engineer_map = self.engineer_map()

        if exclude is None:
            return list(engineer_map.values())

        if engineer_map.keys().isdisjoint(exclude):
            recognized = ", ".join(engineer_map)
            raise ValueError(
                f"exclude has unrecognized features. Recognized features: {recognized}."
            )

        return [
            func for feature, func in engineer_map.items() if feature not in exclude
        ]

    def __call__(
        self, df: pd.DataFrame, *, copy: bool = False, exclude: list[str] | str = None
    ) -> pd.DataFrame:

        if isinstance(exclude, str):
            exclude = [exclude]

        engineers = self.get_engineers(exclude=exclude)

        df = df.copy() if copy is True else df

        for engineer in engineers:
            df = engineer(df)

        return df
