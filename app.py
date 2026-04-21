"""
app.py — AutoDS V2.0 : Intelligent Data Science Assistant
==========================================================
A fully automated, zero-code predictive analytics platform.
"""

import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nbformat

# ── Local modules ────────────────────────────────────────────────────────────
from preprocessing import clean_dataset
from feature_engineering import generate_interaction_features
from feature_selection import detect_problem_type, prepare_features_and_target
from insights_engine import generate_smart_insights
from model_handler import get_available_models, train_and_evaluate
from notebook_generator import generate_notebook
from visualizations import (
    render_univariate_distributions,
    render_feature_vs_target,
    render_correlation_heatmap,
    render_pairplot
)

# ════════════════════════════════════════════════════════════════════════════
# CACHING WRAPPERS
# ════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def run_cached_cleaning(df: pd.DataFrame) -> dict:
    return clean_dataset(df)

@st.cache_data(show_spinner=False)
def run_cached_feature_engineering(df: pd.DataFrame, target: str) -> tuple:
    return generate_interaction_features(df, target)


# ════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & PREMIUM AESTHETIC (NON-AI PALETTE)
# ════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="AutoDS V2.0",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Deep Slate, Emerald, and Amber color palette - avoiding common AI purple/cyan
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Header gradient bar - Deep Emerald to Slate */
    .main-header {
        background: linear-gradient(135deg, #064E3B 0%, #0F172A 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(15, 23, 42, .2);
        border-bottom: 4px solid #D97706; /* Amber accent */
    }
    .main-header h1 {
        color: #F8FAFC;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #94A3B8;
        font-size: 1.05rem;
        margin: .5rem 0 0 0;
        font-weight: 300;
    }

    /* Metric cards */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.4rem 1.6rem;
        text-align: center;
        transition: transform .2s ease, box-shadow .2s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 25px rgba(0,0,0,.05);
        border-color: #0F172A;
    }
    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        color: #0F172A; /* Slate */
        line-height: 1.1;
    }
    .metric-card .label {
        font-size: .85rem;
        color: #64748B;
        margin-top: .4rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Section headings */
    .section-heading {
        font-size: 1.4rem;
        font-weight: 600;
        color: #0F172A;
        border-left: 5px solid #059669;
        padding-left: .75rem;
        margin: 2.5rem 0 1rem 0;
    }

    /* Insight box */
    .insight-box {
        background: #F0FDF4;
        border-left: 4px solid #10B981;
        border-radius: 6px;
        padding: 1.2rem;
        margin-bottom: .8rem;
        font-size: .95rem;
        color: #064E3B;
    }

    /* Result badge */
    .result-badge {
        display: inline-block;
        background: #D97706; /* Amber */
        color: #fff;
        padding: .5rem 1.4rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 1.1rem;
        letter-spacing: .3px;
    }

    /* Button styling Overrides */
    .stButton > button {
        border-radius: 6px;
        font-weight: 600;
        transition: all .2s ease;
        border: 2px solid #0F172A;
        color: #0F172A;
    }
    .stButton > button:hover {
        background-color: #0F172A;
        color: #FFF;
    }
    
    .custom-divider {
        height: 1px;
        background: #E2E8F0;
        margin: 2rem 0;
        border: none;
    }
    
    /* Footer formatting */
    .footer-stamp {
        text-align: center;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #E2E8F0;
        color: #64748B;
        font-size: 0.85rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="main-header">
    <h1>Data Science Assistant V2.0</h1>
    <p>Upload a dataset to unlock Intelligent Insights, Advanced EDA, and Automated Predictive Modeling.</p>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 📂 Data Ingestion")
    uploaded_file = st.file_uploader(
        "Upload Source Dataset (CSV)",
        type=["csv"]
    )

    st.markdown("---")
    st.markdown("### ⚙️ Pipeline Configuration")

    if uploaded_file is not None:
        if "raw_df" not in st.session_state or st.session_state.get("file_name") != uploaded_file.name:
            st.session_state.raw_df = pd.read_csv(uploaded_file)
            st.session_state.file_name = uploaded_file.name
            for key in ["cleaning_report", "engineered_df", "new_features",
                         "problem_type", "train_results", "analysis_done", "smart_insights"]:
                st.session_state.pop(key, None)

        raw_df = st.session_state.raw_df

        if raw_df.empty:
            st.error("The uploaded dataset is empty. Please upload valid data.")
            st.stop()
        if len(raw_df.columns) < 2:
            st.error("Dataset must contain at least 2 columns (features + target).")
            st.stop()

        target_column = st.selectbox(
            "🎯 Predictive Target",
            options=raw_df.columns.tolist(),
        )

        run_analysis = st.button("🚀 Execute Analysis Pipeline", use_container_width=True)
    else:
        target_column = None
        run_analysis = False
        st.info("Awaiting Dataset Upload...")


# ════════════════════════════════════════════════════════════════════════════
# MAIN ARCHITECTURE
# ════════════════════════════════════════════════════════════════════════════

if uploaded_file is None:
    # Landing Dashboard
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card"><div class="value">🧠</div><div class="label">Intelligent Engine</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><div class="value">📊</div><div class="label">Advanced EDA</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><div class="value">⚙️</div><div class="label">Model Tuning</div></div>', unsafe_allow_html=True)
    st.stop()

raw_df = st.session_state.raw_df

if uploaded_file is not None and not st.session_state.get("analysis_done"):
    st.markdown('<div class="section-heading">📋 Dataset Preview</div>', unsafe_allow_html=True)
    st.dataframe(raw_df.head(20), use_container_width=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="value">{raw_df.shape[0]:,}</div><div class="label">Rows</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="value">{raw_df.shape[1]}</div><div class="label">Columns</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="value">{raw_df.isnull().sum().sum()}</div><div class="label">Missing Values</div></div>', unsafe_allow_html=True)


# ── 1. Pipeline Execution ────────────────────────────────────────────────────
if run_analysis:
    with st.spinner("Initiating AutoDS Pipeline..."):
        report = run_cached_cleaning(raw_df)
        cleaned_df = report["cleaned_df"]
        st.session_state.cleaning_report = report

        engineered_df, new_features = run_cached_feature_engineering(cleaned_df, target_column)
        st.session_state.engineered_df = engineered_df
        st.session_state.new_features = new_features

        problem_type = detect_problem_type(engineered_df, target_column)
        st.session_state.problem_type = problem_type
        
        # Intelligence Generation
        insights = generate_smart_insights(engineered_df, target_column, problem_type)
        st.session_state.smart_insights = insights

        st.session_state.target_column = target_column
        st.session_state.analysis_done = True

# ── 2. Render Analytics Dashboard ────────────────────────────────────────────
if st.session_state.get("analysis_done"):
    cleaning_report = st.session_state.cleaning_report
    engineered_df = st.session_state.engineered_df
    problem_type = st.session_state.problem_type
    target_column = st.session_state.target_column
    insights = st.session_state.smart_insights
    
    # ── Cleaning Summary ─────────────────────────────────────────────────────
    st.markdown('<div class="section-heading">🧹 Data Cleaning Summary</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="value">{cleaning_report["duplicates_removed"]}</div><div class="label">Duplicates Removed</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="value">{cleaning_report["missing_filled"]}</div><div class="label">Missing Values Filled</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="value">{len(cleaning_report["clipped_columns"])}</div><div class="label">Outliers Clipped</div></div>', unsafe_allow_html=True)

    # ── Section: Intelligent Insights ────────────────────────────────────────
    st.markdown('<div class="section-heading">🧠 Intelligent Data Insights</div>', unsafe_allow_html=True)
    
    if insights:
        for ins in insights:
            st.markdown(f'<div class="insight-box">{ins}</div>', unsafe_allow_html=True)
    else:
        st.info("No extreme anomalies (collinearity, severe imbalance, or skew) were detected by the engine.")

    # ── Section: Advanced EDA Dashboard ──────────────────────────────────────
    st.markdown('<div class="section-heading">📊 Advanced EDA Dashboard</div>', unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs([
        "Distributions & Outliers", 
        "Feature Interactions", 
        "Correlation Analysis", 
        "Dimensional Pairplot"
    ])
    
    with tab1:
        render_univariate_distributions(engineered_df)
    with tab2:
        render_feature_vs_target(engineered_df, target_column, problem_type)
    with tab3:
        render_correlation_heatmap(engineered_df, target_column)
    with tab4:
        render_pairplot(engineered_df, target_column)

    # ── Section: Model Training & Tuning ─────────────────────────────────────
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">⚙️ Model Configuration & Training</div>', unsafe_allow_html=True)
    
    emoji = "📂 Classification" if problem_type == "classification" else "📈 Regression"
    st.markdown(f"**Detected Supervised Task:** <span class='result-badge'>{emoji}</span>", unsafe_allow_html=True)
    st.write("")

    available_models = get_available_models(problem_type)
    
    c1, c2 = st.columns([1, 1.5])
    with c1:
        model_name = st.selectbox("Select Predictive Algorithm", options=list(available_models.keys()), key="model_select")
        improve_flag = st.checkbox("🔧 Improve Model Performance (Standardize + GridSearch Tuning)", value=False)
        train_clicked = st.button("🚂 Launch Training Sequence", use_container_width=True)
        
    with c2:
        st.markdown("##### Base Algorithm Description")
        st.info(available_models[model_name]["description"])
        
    # Execute Training
    if train_clicked:
        X, y, le = prepare_features_and_target(engineered_df, target_column)
        model_spec = available_models[model_name]
        
        with st.spinner(f"Training {model_name}... (Est: {model_spec.get('estimated_time', 'Computing')} )"):
            try:
                results = train_and_evaluate(
                    model_name=model_name,
                    model_dict=model_spec,
                    X=X,
                    y=y,
                    problem_type=problem_type,
                    improve=improve_flag
                )
                st.session_state.train_results = results
                st.session_state.selected_model_name = model_name
                st.session_state.train_error = None
            except Exception as e:
                st.session_state.train_error = f"Computation Error: {str(e)}"
                st.session_state.train_results = None

    # ── Section: Model Results ───────────────────────────────────────────────
    if st.session_state.get("train_error"):
        st.error(st.session_state.train_error)

    if st.session_state.get("train_results"):
        results = st.session_state.train_results
        metrics = results["metrics"]
        
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-heading">📈 Advanced Evaluation Metrics</div>', unsafe_allow_html=True)
        
        st.markdown(f"**Algorithm Deployed:** `{results['model_name']}`   |   **Scale & Tune Applied:** `{results['scaler_used']}`   |   **Time Class:** `{results['estimated_time']}`")
        
        # Dynamic Metric Rendering
        m_cols = st.columns(len(metrics))
        for idx, (m_name, m_val) in enumerate(metrics.items()):
            with m_cols[idx]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="value">{m_val:.4f}</div>
                    <div class="label">{m_name}</div>
                </div>
                """, unsafe_allow_html=True)
                
        # Interpretability Plot
        feat_importances = results.get("feature_importances")
        if feat_importances is not None:
            st.markdown("### Top Feature Determinants")
            fig_imp, ax_imp = plt.subplots(figsize=(10, 4))
            sns.barplot(x=feat_importances.values, y=feat_importances.index, palette="viridis", ax=ax_imp)
            ax_imp.set_title("Machine Learning Feature Impact / Coefficients")
            plt.tight_layout()
            st.pyplot(fig_imp)
            
        # ── Notebook Generation ──────────────────────────────────────────────
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-heading">📓 Export Educational Artifact</div>', unsafe_allow_html=True)
        
        nb = generate_notebook(
            target_column=target_column,
            problem_type=problem_type,
            model_name=results["model_name"],
            new_features=st.session_state.new_features,
            insights=insights,
            scaler_used=results["scaler_used"]
        )
        nb_bytes = nbformat.writes(nb).encode("utf-8")
        st.download_button(
            label="⬇️ Download Extensive Jupyter Notebook",
            data=nb_bytes,
            file_name="autods_v2_analysis.ipynb",
            mime="application/x-ipynb+json",
            use_container_width=True,
        )

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown('<div class="footer-stamp">Engineered by Mohammed Akdas Ansari | AutoDS v2.0</div>', unsafe_allow_html=True)
