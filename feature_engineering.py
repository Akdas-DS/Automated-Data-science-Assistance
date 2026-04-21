"""
feature_engineering.py
──────────────────────
Automatic feature generation for numeric columns.

Currently generates:
  • Pairwise addition features   (A + B)
  • Pairwise multiplication features (A × B)

To keep the feature space manageable, we limit generation to the
first MAX_COLS numeric columns (excluding the target).
"""

import pandas as pd
import numpy as np
import itertools

# Maximum number of numeric columns to use for interaction features.
# Prevents combinatorial explosion on wide datasets.
MAX_COLS = 8


def generate_interaction_features(
    df: pd.DataFrame,
    target_column: str,
    max_cols: int = MAX_COLS,
) -> pd.DataFrame:
    """
    Create interaction features (sum & product) for numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        The (already cleaned) DataFrame.
    target_column : str
        Name of the target column — excluded from feature generation.
    max_cols : int
        Cap on how many numeric columns to combine.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with new interaction columns appended.
    new_features : list[str]
        Names of the newly created columns.
    """
    df = df.copy()

    # Select numeric columns, excluding the target
    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c != target_column
    ]

    # Limit to avoid blowup
    numeric_cols = numeric_cols[:max_cols]

    new_features = []

    # Generate pairwise combinations
    for col_a, col_b in itertools.combinations(numeric_cols, 2):
        # Addition feature
        sum_name = f"{col_a}_plus_{col_b}"
        df[sum_name] = df[col_a] + df[col_b]
        new_features.append(sum_name)

        # Multiplication feature
        prod_name = f"{col_a}_times_{col_b}"
        df[prod_name] = df[col_a] * df[col_b]
        new_features.append(prod_name)

    return df, new_features
