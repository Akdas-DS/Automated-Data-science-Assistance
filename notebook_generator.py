"""
notebook_generator.py
─────────────────────
Creates a robust, deeply educational Jupyter Notebook (.ipynb) that mirrors
the advanced pipeline the user ran through the Streamlit app.
"""

import nbformat as nbf

MODEL_CODE_MAP = {
    "Logistic Regression": {
        "import": "from sklearn.linear_model import LogisticRegression",
        "init": "model = LogisticRegression(max_iter=1000)"
    },
    "Random Forest": {
        "import": "from sklearn.ensemble import RandomForestClassifier",
        "init": "model = RandomForestClassifier(n_estimators=100, random_state=42)"
    },
    "Decision Tree": {
        "import": "from sklearn.tree import DecisionTreeClassifier",
        "init": "model = DecisionTreeClassifier(random_state=42)"
    },
    "K-Nearest Neighbors (KNN)": {
        "import": "from sklearn.neighbors import KNeighborsClassifier",
        "init": "model = KNeighborsClassifier()"
    },
    "Support Vector Machine (SVM)": {
        "import": "from sklearn.svm import SVC",
        "init": "model = SVC(probability=True, random_state=42)"
    },
    "XGBoost Classifier": {
        "import": "from xgboost import XGBClassifier",
        "init": "model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)"
    },
    "Linear Regression": {
        "import": "from sklearn.linear_model import LinearRegression",
        "init": "model = LinearRegression()"
    },
    "Random Forest Regressor": {
        "import": "from sklearn.ensemble import RandomForestRegressor",
        "init": "model = RandomForestRegressor(n_estimators=100, random_state=42)"
    },
    "Decision Tree Regressor": {
        "import": "from sklearn.tree import DecisionTreeRegressor",
        "init": "model = DecisionTreeRegressor(random_state=42)"
    },
    "KNN Regressor": {
        "import": "from sklearn.neighbors import KNeighborsRegressor",
        "init": "model = KNeighborsRegressor()"
    },
    "Support Vector Regressor (SVR)": {
        "import": "from sklearn.svm import SVR",
        "init": "model = SVR()"
    },
    "XGBoost Regressor": {
        "import": "from xgboost import XGBRegressor",
        "init": "model = XGBRegressor(random_state=42)"
    }
}


def _md(text: str):
    return nbf.v4.new_markdown_cell(text)

def _code(source: str):
    return nbf.v4.new_code_cell(source)


def generate_notebook(
    target_column: str,
    problem_type: str,
    model_name: str,
    new_features: list[str],
    insights: list[str],
    scaler_used: bool,
) -> nbf.NotebookNode:
    """
    Build and return a NotebookNode containing clearly-separated
    cells. Includes Advanced EDA, Interpretability, Metrics, and standard models.
    """
    nb = nbf.v4.new_notebook()
    cells = []

    # ── 1. Title ─────────────────────────────────────────────────────────────
    cells.append(_md(
        "# 🔬 Advanced Data Science Analysis\n"
        "\n"
        "This notebook was **auto-generated** by AutoDS V2.0. \n"
        "It acts as a complete, highly-interpretable data science workflow.\n"
        "\n"
        f"- **Target column:** `{target_column}`\n"
        f"- **Problem type:** {problem_type}\n"
        f"- **Selected model:** {model_name}"
    ))

    # ── 2. Insights Engine Findings ──────────────────────────────────────────
    insights_text = "## 🧠 Intelligent Analysis Findings\n\nThe AutoDS Engine discovered the following:\n"
    for ins in insights:
        insights_text += f"- {ins}\n"
    cells.append(_md(insights_text))

    # ── 3. Import Libraries ──────────────────────────────────────────────────
    cells.append(_md("## 1 · Core Setup"))
    cells.append(_code(
        "import pandas as pd\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "from sklearn.model_selection import train_test_split\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score\n"
        "from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error\n"
        "\n"
        "# Set aesthetic style for advanced EDA\n"
        "sns.set_theme(style='whitegrid', palette='muted')\n"
        "%matplotlib inline"
    ))

    # ── 4. Load Dataset ──────────────────────────────────────────────────────
    cells.append(_md("## 2 · Load Dataset"))
    cells.append(_code(
        "# ⬇ Upload your CSV here (update the path if needed)\n"
        "df = pd.read_csv('dataset.csv')\n"
        "df.head()"
    ))

    # ── 5. Data Cleaning ─────────────────────────────────────────────────────
    cells.append(_md(
        "## 3 · Data Cleaning Pipeline\n"
        "1. Remove duplicates\n"
        "2. Safely handle empty columns & missing values (mean / mode)\n"
        "3. Clip numeric extremes to the [1st, 99th] percentiles to protect against wild outliers."
    ))
    cells.append(_code(
        "df = df.drop_duplicates().reset_index(drop=True)\n"
        "\n"
        "numeric_cols = df.select_dtypes(include=[np.number]).columns\n"
        "for col in numeric_cols:\n"
        "    if df[col].isnull().all():\n"
        "        df = df.drop(columns=[col])\n"
        "    elif df[col].isnull().any():\n"
        "        df[col] = df[col].fillna(df[col].mean())\n"
        "\n"
        "categorical_cols = df.select_dtypes(exclude=[np.number]).columns\n"
        "for col in categorical_cols:\n"
        "    if df[col].isnull().all():\n"
        "        df = df.drop(columns=[col])\n"
        "    elif df[col].isnull().any():\n"
        "        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')\n"
        "\n"
        "for col in df.select_dtypes(include=[np.number]).columns:\n"
        "    lower, upper = df[col].quantile(0.01), df[col].quantile(0.99)\n"
        "    if lower < upper:\n"
        "        df[col] = df[col].clip(lower=lower, upper=upper)\n"
        "\n"
        "print(f'Polished DataFrame shape: {df.shape}')"
    ))

    # ── 6. Advanced EDA ──────────────────────────────────────────────────────
    cells.append(_md("## 4 · Advanced EDA Explorations"))
    
    eda_code = (
        "numeric_cols = df.select_dtypes(include=[np.number]).columns\n"
        "if len(numeric_cols) > 0:\n"
        "    plt.figure(figsize=(12, 5))\n"
        "    sns.boxplot(data=df[numeric_cols[:10]], orient='h')\n"
        "    plt.title('Outlier Distribution across Top Numeric Features')\n"
        "    plt.show()\n"
        "\n"
        f"plt.figure(figsize=(8, 5))\n"
        f"sns.histplot(data=df, x='{target_column}', kde=True)\n"
        f"plt.title('Target Sequence Distribution: {target_column}')\n"
        f"plt.show()"
    )
    cells.append(_code(eda_code))

    # ── 7. Train-Test Split & Sub-ops ─────────────────────────────────────────
    cells.append(_md("## 5 · Target Preparation & Scaling"))

    split_code = (
        f"target_col = '{target_column}'\n"
        "y = df[target_col].copy()\n"
        "X = df.drop(columns=[target_col])\n"
        "\n"
        "# Encode categorical features\n"
        "from sklearn.preprocessing import LabelEncoder\n"
        "for col in X.select_dtypes(include=['object', 'category']).columns:\n"
        "    le = LabelEncoder()\n"
        "    X[col] = le.fit_transform(X[col].astype(str))\n"
        "\n"
        "X = X.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0)\n"
    )

    if problem_type == "classification":
        split_code += (
            "\n# Encode Categorical Targets\n"
            "from sklearn.preprocessing import LabelEncoder\n"
            "if y.dtype == 'object':\n"
            "    le = LabelEncoder()\n"
            "    y = pd.Series(le.fit_transform(y), name=target_col)\n"
        )

    split_code += (
        "\n"
        "X_train, X_test, y_train, y_test = train_test_split(\n"
        "    X, y, test_size=0.2, random_state=42\n"
        ")\n"
    )
    
    if scaler_used:
        split_code += (
            "\n"
            "# Feature Scaling was enabled via AutoDS Engine\n"
            "scaler = StandardScaler()\n"
            "X_train = scaler.fit_transform(X_train)\n"
            "X_test = scaler.transform(X_test)\n"
            "print('Features standardized.')\n"
        )

    cells.append(_code(split_code))

    # ── 8. Model Definition ──────────────────────────────────────────────────
    cells.append(_md(f"## 6 · Model Instance — {model_name}"))

    model_info = MODEL_CODE_MAP.get(model_name)
    if model_info:
        model_code = (
            f"{model_info['import']}\n\n"
            f"{model_info['init']}\n"
            "model.fit(X_train, y_train)\n"
            "print('Model trained successfully!')"
        )
    else:
        model_code = f"# Fallback\nmodel = None"

    cells.append(_code(model_code))

    # ── 9. Advanced Evaluation ───────────────────────────────────────────────
    cells.append(_md("## 7 · Evaluation Vectors"))

    if problem_type == "classification":
        eval_code = (
            "y_pred = model.predict(X_test)\n"
            "print(f'Accuracy : {accuracy_score(y_test, y_pred):.4f}')\n"
            "print(f'Precision: {precision_score(y_test, y_pred, average=\"weighted\", zero_division=0):.4f}')\n"
            "print(f'Recall   : {recall_score(y_test, y_pred, average=\"weighted\", zero_division=0):.4f}')\n"
            "print(f'F1 Score : {f1_score(y_test, y_pred, average=\"weighted\", zero_division=0):.4f}')"
        )
    else:
        eval_code = (
            "y_pred = model.predict(X_test)\n"
            "print(f'R² Score : {r2_score(y_test, y_pred):.4f}')\n"
            "print(f'RMSE     : {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}')\n"
            "print(f'MAE      : {mean_absolute_error(y_test, y_pred):.4f}')"
        )

    cells.append(_code(eval_code))
    
    # ── 10. Feature Importances ──────────────────────────────────────────────
    cells.append(_md("## 8 · Interpretability (Top Features)"))
    feature_imp_code = (
        "import numpy as np\n"
        "import pandas as pd\n\n"
        "if hasattr(model, 'feature_importances_'):\n"
        "    importances = model.feature_importances_\n"
        "    pd.Series(importances, index=X.columns).sort_values(ascending=False).head(10).plot(kind='barh', title='Top 10 Important Features')\n"
        "elif hasattr(model, 'coef_'):\n"
        "    coefs = np.mean(np.abs(model.coef_), axis=0) if len(model.coef_.shape) > 1 else np.abs(model.coef_[0] if len(model.coef_.shape)==2 else model.coef_)\n"
        "    pd.Series(coefs, index=X.columns).sort_values(ascending=False).head(10).plot(kind='barh', title='Top 10 Feature Coefficients')\n"
        "else:\n"
        "    print('Model does not expose standard interpretability coefficients.')"
    )
    cells.append(_code(feature_imp_code))

    nb.cells = cells
    return nb
