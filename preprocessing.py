"""
preprocessing.py
─────────────────
Functions for cleaning and preparing raw datasets.

Pipeline:
  1. Remove duplicate rows
  2. Fill missing values (mean for numeric, mode for categorical)
  3. Clip outliers using 1st / 99th percentile bounds
"""

import pandas as pd
import numpy as np


# ── 1. Remove Duplicates ────────────────────────────────────────────────────

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate rows and return the cleaned DataFrame."""
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    after = len(df)
    removed = before - after
    return df, removed


# ── 2. Handle Missing Values ────────────────────────────────────────────────

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values:
      • Numeric columns  → column mean
      • Categorical cols  → column mode (most frequent value)
    """
    missing_before = df.isnull().sum().sum()

    # Numeric columns — fill with mean
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().all():
            df = df.drop(columns=[col]) # Drop completely empty numeric columns
        elif df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mean())

    # Categorical columns — fill with mode
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns
    for col in categorical_cols:
        if df[col].isnull().all():
            df = df.drop(columns=[col]) # Drop completely empty categorical columns
        elif df[col].isnull().any():
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val[0])
            else:
                df[col] = df[col].fillna("Unknown")

    missing_after = df.isnull().sum().sum()
    filled = missing_before - missing_after
    return df, filled


# ── 3. Clip Outliers ────────────────────────────────────────────────────────

def clip_outliers(df: pd.DataFrame, lower_pct: float = 0.01, upper_pct: float = 0.99) -> pd.DataFrame:
    """
    Clip numeric columns to [1st percentile, 99th percentile] bounds.
    This reduces the influence of extreme values without removing rows.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    clipped_cols = []

    for col in numeric_cols:
        lower = df[col].quantile(lower_pct)
        upper = df[col].quantile(upper_pct)
        # Only clip if the bounds are different (avoids no-op on constant cols)
        if lower < upper:
            original = df[col].copy()
            df[col] = df[col].clip(lower=lower, upper=upper)
            if not original.equals(df[col]):
                clipped_cols.append(col)

    return df, clipped_cols


# ── Master Pipeline ─────────────────────────────────────────────────────────

def clean_dataset(df: pd.DataFrame) -> dict:
    """
    Run the full cleaning pipeline and return a report dict with:
      • cleaned_df
      • duplicates_removed
      • missing_filled
      • clipped_columns
    """
    df = df.copy()

    df, duplicates_removed = remove_duplicates(df)
    df, missing_filled = handle_missing_values(df)
    df, clipped_columns = clip_outliers(df)

    report = {
        "cleaned_df": df,
        "duplicates_removed": duplicates_removed,
        "missing_filled": missing_filled,
        "clipped_columns": clipped_columns,
    }
    return report
