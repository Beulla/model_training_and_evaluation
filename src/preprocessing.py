import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

TARGET_COL = "final_result"
ID_COLS = ["code_module", "code_presentation", "id_student"]
CATEGORICAL_COLS = [
    "code_module",
    "code_presentation",
    "gender",
    "region",
    "highest_education",
    "imd_band",
    "age_band",
    "disability",
]
NUMERIC_COLS = [
    "num_of_prev_attempts",
    "studied_credits",
    "total_clicks",
    "active_days",
    "unique_activities",
    "mean_score",
    "max_score",
    "min_score",
    "num_assessments",
]


def prepare_features(df: pd.DataFrame):
    df = df.dropna(subset=[TARGET_COL]).copy()
    y_encoder = LabelEncoder()
    y = y_encoder.fit_transform(df[TARGET_COL])

    feature_df = df.drop(columns=ID_COLS + [TARGET_COL], errors="ignore")
    cat_cols = [c for c in CATEGORICAL_COLS if c in feature_df.columns]
    num_cols = [c for c in NUMERIC_COLS if c in feature_df.columns]

    for col in num_cols:
        feature_df[col] = feature_df[col].fillna(feature_df[col].median())

    return feature_df, y, y_encoder, cat_cols, num_cols


def build_preprocessor(cat_cols: list[str], num_cols: list[str]) -> ColumnTransformer:
    transformers = []
    if cat_cols:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_cols,
            )
        )
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def split_data(
    X: pd.DataFrame,
    y: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.15,
    random_state: int = 42,
) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    relative_val = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=relative_val,
        stratify=y_train,
        random_state=random_state,
    )
    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }
