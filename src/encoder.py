import pandas as pd


def concat_train_test(
    train_df: pd.DataFrame, test_df: pd.DataFrame, *, has_labels: bool = True
) -> tuple[pd.DataFrame, pd.Index, pd.Index, pd.Series | None]:
    transported = None

    if has_labels is True:
        transported = train_df["Transported"].copy()
        train_df = train_df.drop("Transported", axis=1)

    # Store indices
    train_index = train_df.index
    test_index = test_df.index

    df = pd.concat([train_df, test_df])

    return df, train_index, test_index, transported


def split_train_test(
    df: pd.DataFrame,
    train_index: pd.Index,
    test_index: pd.Index,
    transported: pd.Series = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tr_df = df.loc[train_index, :]
    if transported is not None:
        tr_df["Transported"] = transported

    te_df = df.loc[test_index, :]

    return tr_df, te_df


def convert_bool2int(df: pd.DataFrame, exclude: list[str] = None) -> pd.DataFrame:
    columns = df.columns if exclude is None else df.columns.drop(exclude)

    if columns.empty:
        return df

    columns = [col for col in columns if df[col].dtype.name == "bool"]
    df[columns] = df[columns].astype(int)

    return df


def label_encoder(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        levels = df[column].value_counts().index
        mapping = {level: idx for idx, level in enumerate(levels)}
        df[column] = df[column].map(mapping)

    return df


def encode_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    bool_exclude: list[str] = None,
    one_hot_cols: list[str] = None,
    label_encoding_cols: list[str] = None,
    *,
    has_labels: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df, train_idx, test_idx, transported = concat_train_test(
        train_df=train_df, test_df=test_df, has_labels=has_labels
    )

    df = convert_bool2int(df, exclude=bool_exclude)

    if label_encoding_cols is not None:
        df = label_encoder(df, columns=label_encoding_cols)

    if one_hot_cols is not None:
        df = pd.get_dummies(df, columns=one_hot_cols)

    train_df, test_df = split_train_test(
        df=df, train_index=train_idx, test_index=test_idx, transported=transported
    )

    return train_df, test_df
