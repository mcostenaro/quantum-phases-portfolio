from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from .data import load_dataset


def stratified_split(
    df: pd.DataFrame,
    *,
    test_size=0.3,
    seed=42,
    label_col="labels",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        raise ValueError("Cannot split an empty dataframe.")
    if label_col not in df.columns:
        raise ValueError(f"Missing label column '{label_col}'.")

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=int(seed),
        stratify=df[label_col],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def sample_per_class(
    df: pd.DataFrame,
    n_per_class,
    *,
    seed=42,
    label_col="labels",
) -> pd.DataFrame:
    if df.empty:
        raise ValueError("Cannot sample from an empty dataframe.")
    if label_col not in df.columns:
        raise ValueError(f"Missing label column '{label_col}'.")

    n_per_class = int(n_per_class)
    if n_per_class < 1:
        raise ValueError(f"n_per_class must be >= 1, got {n_per_class}.")

    parts = []
    for class_idx, (label, group) in enumerate(df.groupby(label_col)):
        if len(group) < n_per_class:
            raise ValueError(
                f"Class {label} has only {len(group)} samples; "
                f"cannot sample {n_per_class}."
            )
        class_seed = int(seed + 1009 * class_idx)
        parts.append(group.sample(n=n_per_class, random_state=class_seed, replace=False))

    return pd.concat(parts, ignore_index=True)


def train_test_split_per_class(
    df: pd.DataFrame,
    n_train_per_class,
    *,
    seed=42,
    label_col="labels",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        raise ValueError("Cannot split an empty dataframe.")
    if label_col not in df.columns:
        raise ValueError(f"Missing label column '{label_col}'.")

    n_train_per_class = int(n_train_per_class)
    if n_train_per_class < 1:
        raise ValueError(
            f"n_train_per_class must be >= 1, got {n_train_per_class}."
        )

    train_parts = []
    test_parts = []
    for class_idx, (label, group) in enumerate(df.groupby(label_col)):
        if len(group) <= n_train_per_class:
            raise ValueError(
                f"Class {label} has {len(group)} samples; "
                f"cannot use {n_train_per_class} for train and still leave test samples."
            )

        class_seed = int(seed + 2039 * class_idx)
        train_part = group.sample(
            n=n_train_per_class,
            random_state=class_seed,
            replace=False,
        )
        test_part = group.drop(train_part.index)

        train_parts.append(train_part)
        test_parts.append(test_part)

    df_train = pd.concat(train_parts, ignore_index=True)
    df_test = pd.concat(test_parts, ignore_index=True)
    return df_train, df_test
