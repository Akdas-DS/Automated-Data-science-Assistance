"""
feature_selection.py
────────────────────
Utilities for detecting problem type and preparing features
for model training.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


# ── Problem Type Detection ───────────────────────────────────────────────────

def detect_problem_type(df: pd.DataFrame, target_column: str) -> str:
    """
    Determine whether the task is classification or regression.

    Rules:
      • If the target column is non-numeric (object / category) → classification
      • If the target is numeric but has ≤ 20 unique values       → classification
      • Otherwise                                                   → regression
    """
    target = df[target_column]

    if target.dtype == "object" or target.dtype.name == "category":
        return "classification"

    # Numeric but low cardinality → likely classes
    if target.nunique() <= 20:
        return "classification"

    return "regression"


# ── Feature / Target Preparation ────────────────────────────────────────────

def prepare_features_and_target(
    df: pd.DataFrame,
    target_column: str,
) -> tuple:
    """
    Separate features (X) and target (y) from the DataFrame.

    • Drops non-numeric feature columns (simple approach — no one-hot encoding
      to keep the generated notebook beginner-friendly).
    • Label-encodes the target if it is categorical.

    Returns
    -------
    X : pd.DataFrame   — numeric feature matrix
    y : pd.Series       — target vector (encoded if needed)
    label_encoder : LabelEncoder | None
    """
    df = df.copy()

    # Separate
    y = df[target_column].copy()
    X = df.drop(columns=[target_column])

    # Encode categorical target (robust check for ANY non-numeric type)
    label_encoder = None
    if not np.issubdtype(y.dtype, np.number):
        label_encoder = LabelEncoder()
        y = pd.Series(label_encoder.fit_transform(y.astype(str)), name=target_column)
    else:
        # Even numeric classification targets need contiguous 0-indexed labels for XGBoost
        unique_vals = sorted(y.unique())
        if unique_vals != list(range(len(unique_vals))):
            label_encoder = LabelEncoder()
            y = pd.Series(label_encoder.fit_transform(y), name=target_column)

    # Encode categorical features automatically
    for col in X.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # Keep only numeric features
    X = X.select_dtypes(include=[np.number])

    # Replace any remaining infinities / NaNs (safety net)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    if X.shape[1] == 0:
        raise ValueError("No features remaining to train the model. Ensure the dataset contains valid structural data.")

    return X, y, label_encoder
