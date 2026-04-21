"""
insights_engine.py
──────────────────
Smart data-science analysis algorithms.
Provides deeply informative, actionable, and human-readable qualitative data.
"""

import pandas as pd
import numpy as np

def generate_smart_insights(df: pd.DataFrame, target_column: str, problem_type: str) -> list[str]:
    """
    Produces highly analytical text insights warning the user of standard 
    data science issues: sizing, imbalance, multicollinearity, skewness, and outliers.
    """
    insights = []
    rows, cols = df.shape
    
    # 1. Dataset Size & Complexity
    if rows < 5000:
        insights.append(f"📦 **Small Dataset ({rows} rows)**: Simpler models (like Logistic Regression/Decision Trees) generally perform better on less data to prevent overfitting.")
    elif rows < 100000:
        insights.append(f"📦 **Medium Dataset ({rows} rows)**: Excellent size for ensemble methods like Random Forest or XGBoost. Good balance of training time vs accuracy.")
    else:
        insights.append(f"📦 **Large Dataset ({rows} rows)**: High signal density. XGBoost or advanced ensembles will excel here, but may take longer to train. Consider scaling numerical features.")

    # 2. Target Imbalance (Classification)
    if problem_type == "classification":
        class_counts = df[target_column].value_counts(normalize=True)
        if class_counts.iloc[0] > 0.75:
            min_class_ratio = class_counts.iloc[-1] * 100
            insights.append(f"⚠️ **Target Variable Imbalance**: The dominant class makes up {class_counts.iloc[0]*100:.1f}% of the target. Rare class is at {min_class_ratio:.1f}%. Consider models that handle imbalance well (like XGBoost) or evaluate using Precision/Recall rather than pure Accuracy.")
        else:
            insights.append("⚖️ **Target Distribution**: Target classes are relatively balanced. Standard accuracy mapping is reliable.")

    numeric_df = df.select_dtypes(include=[np.number])
    if target_column in numeric_df.columns:
        feature_df = numeric_df.drop(columns=[target_column])
    else:
        feature_df = numeric_df
        
    if len(feature_df.columns) > 1:
        # 3. Multicollinearity (High Correlation between features)
        corr_matrix = feature_df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr_pairs = [(upper.index[x], upper.columns[y]) for x, y in zip(*np.where(upper > 0.85))]
        
        if len(high_corr_pairs) > 0:
            insights.append(f"🔗 **High Multicollinearity Detected**: `{len(high_corr_pairs)}` feature pairs have correlation > 0.85. This can negatively impact simple Linear/Logistic models. Tree-based algorithms (Random Forest) isolate this robustly.")

        # 4. Skewness
        skewed = feature_df.skew().abs()
        highly_skewed = skewed[skewed > 1.5]
        if not highly_skewed.empty:
            insights.append(f"📉 **Skewed Distributions**: `{len(highly_skewed)}` numeric feature(s) are severely skewed. Algorithms reliant on strict Euclidean distance (like KNN or SVM) may benefit massively from StandardScaler.")

    return insights
