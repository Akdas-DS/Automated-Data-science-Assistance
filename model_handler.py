"""
model_handler.py
────────────────
Model suggestion, training, and evaluation helpers,
featuring advanced metrics and automated tuning capabilities.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    r2_score, mean_squared_error, mean_absolute_error
)

# ── Model Catalogues ─────────────────────────────────────────────────────────

CLASSIFICATION_MODELS = {
    "Logistic Regression": {
        "instance": LogisticRegression(max_iter=1000),
        "description": "Works well for simple linear data. Extremely fast and highly interpretable.",
        "params": {"C": [0.1, 1.0, 10.0]}
    },
    "Random Forest": {
        "instance": RandomForestClassifier(n_estimators=100, random_state=42),
        "description": "Excellent for capturing non-linear patterns and robust against outliers.",
        "params": {"n_estimators": [50, 100], "max_depth": [None, 10, 20]}
    },
    "Decision Tree": {
        "instance": DecisionTreeClassifier(random_state=42),
        "description": "Simple and interpretable, but prone to overfitting on complex datasets.",
        "params": {"max_depth": [None, 5, 10, 20]}
    },
    "K-Nearest Neighbors (KNN)": {
        "instance": KNeighborsClassifier(),
        "description": "Good for smaller datasets where distance between points defines the relationship.",
        "params": {"n_neighbors": [3, 5, 7]}
    },
    "Support Vector Machine (SVM)": {
        "instance": SVC(probability=True, random_state=42),
        "description": "Good for high-dimensional data. Requires scaled features for best results.",
        "params": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}
    },
    "XGBoost Classifier": {
        "instance": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        "description": "State-of-the-art gradient boosting. Dominates heavily structured tabular data.",
        "params": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1]}
    },
}

REGRESSION_MODELS = {
    "Linear Regression": {
        "instance": LinearRegression(),
        "description": "The baseline for regression. Best when relationships are strictly linear.",
        "params": {}
    },
    "Random Forest Regressor": {
        "instance": RandomForestRegressor(n_estimators=100, random_state=42),
        "description": "Robust ensemble method that averages multiple decision trees. Handles non-linearity well.",
        "params": {"n_estimators": [50, 100], "max_depth": [None, 10, 20]}
    },
    "Decision Tree Regressor": {
        "instance": DecisionTreeRegressor(random_state=42),
        "description": "Models relationships using direct logical splits. Fast but can overfit without depth tuning.",
        "params": {"max_depth": [None, 5, 10, 20]}
    },
    "KNN Regressor": {
        "instance": KNeighborsRegressor(),
        "description": "Predicts based on local feature proximity. Very sensitive to unscaled data.",
        "params": {"n_neighbors": [3, 5, 7]}
    },
    "Support Vector Regressor (SVR)": {
        "instance": SVR(),
        "description": "Finds a hyperplane to fit continuous data. Highly effective in high dimensional spaces.",
        "params": {"C": [0.1, 1, 10]}
    },
    "XGBoost Regressor": {
        "instance": XGBRegressor(random_state=42),
        "description": "Advanced boosting algorithm. Frequently wins Kaggle competitions for tabular data.",
        "params": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1]}
    },
}

# ── Functionality ────────────────────────────────────────────────────────────

def get_available_models(problem_type: str) -> dict:
    if problem_type == "classification":
        return CLASSIFICATION_MODELS
    return REGRESSION_MODELS


def estimate_training_time(rows: int, cols: int, model_name: str) -> str:
    """Provides a rough human-readable training time estimation."""
    complexity_score = rows * cols
    
    is_complex = "XGBoost" in model_name or "SVM" in model_name or "Random Forest" in model_name
    
    if complexity_score < 10000:
        return "< 2 seconds"
    elif complexity_score < 100000:
        return "2 - 5 seconds"
    elif complexity_score < 500000:
        return "5 - 15 seconds" if not is_complex else "10 - 30 seconds"
    else:
        return "30+ seconds" if not is_complex else "1+ minute(s)"


def extract_feature_importances(model, feature_names):
    """Attempt to safely extract feature importance or coefficients."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        return pd.Series(importances, index=feature_names).sort_values(ascending=False).head(10)
    elif hasattr(model, 'coef_'):
        # For multi-class logistic regression, coef_ is 2D
        importances = np.mean(np.abs(model.coef_), axis=0) if len(model.coef_.shape) > 1 else np.abs(model.coef_[0] if len(model.coef_.shape)==2 else model.coef_)
        return pd.Series(importances, index=feature_names).sort_values(ascending=False).head(10)
    return None


def calculate_metrics(y_true, y_pred, problem_type):
    if problem_type == "classification":
        # Handle multi-class gracefully via 'weighted' average
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
            "Recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
            "F1 Score": f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
    else:
        return {
            "R² Score": r2_score(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "MAE": mean_absolute_error(y_true, y_pred)
        }


def train_and_evaluate(
    model_name: str,
    model_dict: dict,
    X: pd.DataFrame,
    y: pd.Series,
    problem_type: str,
    improve: bool = False,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    
    # 1. Provide Training Estimation based on dataset size
    est_time = estimate_training_time(X.shape[0], X.shape[1], model_name)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
    )

    model = model_dict["instance"]
    scaler_used = False

    if improve:
        # Standard Scaling
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        scaler_used = True
        
        # Grid Search Hyperparameter tuning if params exist
        params = model_dict.get("params", {})
        if params:
            grid = GridSearchCV(model, params, cv=3, n_jobs=-1)
            grid.fit(X_train, y_train)
            model = grid.best_estimator_
        else:
            model.fit(X_train, y_train)
    else:
        # Ensure array shape matching for untouched sklearn pipelines
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred, problem_type)
    
    # Extract Important features using the column names of X
    feature_importances = extract_feature_importances(model, X.columns)

    return {
        "model": model,
        "model_name": model_name,
        "metrics": metrics,
        "feature_importances": feature_importances,
        "scaler_used": scaler_used,
        "estimated_time": est_time,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }
