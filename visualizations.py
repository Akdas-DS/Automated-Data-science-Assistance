"""
visualizations.py
─────────────────
Memory-optimized EDA visualization components for AutoDS platform.
Every figure is explicitly closed after rendering to prevent RAM leaks.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # Non-interactive backend — lower memory
import matplotlib.pyplot as plt
import seaborn as sns
import gc


def _safe_render(fig):
    """Render a figure to Streamlit and immediately free the memory."""
    st.pyplot(fig)
    plt.close(fig)
    gc.collect()


def render_univariate_distributions(df: pd.DataFrame):
    """Distributions: Boxplots/Violins for numerics, Countplots for categoricals."""
    st.markdown("### 📊 Distribution Explorer")
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    col_to_plot = st.selectbox("Select Feature to Examine:", options=df.columns.tolist())

    # Subsample for heavy datasets
    plot_df = df[[col_to_plot]].copy()
    if len(plot_df) > 2000:
        plot_df = plot_df.sample(n=2000, random_state=42)

    if col_to_plot in numeric_cols:
        plot_type = st.radio("Plot Type:", ["Histogram", "Boxplot", "Violin Plot"], horizontal=True)
        fig, ax = plt.subplots(figsize=(9, 4))

        if plot_type == "Histogram":
            sns.histplot(data=plot_df, x=col_to_plot, kde=True, color="#059669", ax=ax)
            ax.set_title(f"Distribution of {col_to_plot}")
        elif plot_type == "Boxplot":
            sns.boxplot(data=plot_df, x=col_to_plot, color="#0EA5E9", ax=ax, orient='h')
            ax.set_title(f"Outlier Spreads of {col_to_plot}")
        else:
            sns.violinplot(data=plot_df, x=col_to_plot, color="#D97706", ax=ax, orient='h')
            ax.set_title(f"Density Range of {col_to_plot}")

        plt.tight_layout()
        _safe_render(fig)

    else:
        st.info("📌 Categorical Feature Selected")
        val_counts = df[col_to_plot].value_counts()
        if len(val_counts) > 20:
            st.caption(f"Feature has {len(val_counts)} unique values — showing top 20.")
            plot_data = df[df[col_to_plot].isin(val_counts.head(20).index)]
        else:
            plot_data = df

        fig, ax = plt.subplots(figsize=(9, 4))
        sns.countplot(data=plot_data, y=col_to_plot,
                      order=plot_data[col_to_plot].value_counts().index,
                      palette="crest", ax=ax)
        ax.set_title(f"Top Categories in {col_to_plot}")
        plt.tight_layout()
        _safe_render(fig)


def render_feature_vs_target(df: pd.DataFrame, target_col: str, problem_type: str):
    """Feature-vs-Target relationship plots."""
    st.markdown("### ⚖️ Feature vs. Target Interaction")
    features = [c for c in df.columns if c != target_col]

    if not features:
        st.warning("No features aside from the target exist.")
        return

    analysis_feat = st.selectbox("Compare Target Against:", options=features)

    # Subsample
    plot_df = df[[analysis_feat, target_col]].dropna().copy()
    if len(plot_df) > 2000:
        plot_df = plot_df.sample(n=2000, random_state=42)

    fig, ax = plt.subplots(figsize=(9, 5))
    is_numeric_feat = pd.api.types.is_numeric_dtype(df[analysis_feat])

    if problem_type == "classification":
        if is_numeric_feat:
            sns.boxplot(data=plot_df, x=target_col, y=analysis_feat, palette="muted", ax=ax)
            ax.set_title(f"{analysis_feat} Grouped by {target_col}")
        else:
            if min(plot_df[target_col].nunique(), plot_df[analysis_feat].nunique()) > 15:
                st.warning("High cardinality restricts grouped count plots.")
                plt.close(fig)
                return
            sns.countplot(data=plot_df, x=analysis_feat, hue=target_col, palette="viridis", ax=ax)
            ax.set_title(f"{analysis_feat} Breakdown by {target_col}")
            plt.xticks(rotation=45)
    else:
        if is_numeric_feat:
            sns.regplot(data=plot_df, x=analysis_feat, y=target_col,
                        scatter_kws={'alpha': 0.4, 'color': '#0EA5E9', 's': 10},
                        line_kws={'color': '#D97706'}, ax=ax)
            ax.set_title(f"Trend: {analysis_feat} vs {target_col}")
        else:
            sns.boxplot(data=plot_df, x=analysis_feat, y=target_col, palette="crest", ax=ax)
            ax.set_title(f"Target across {analysis_feat} Classes")
            plt.xticks(rotation=45)

    plt.tight_layout()
    _safe_render(fig)


def render_correlation_heatmap(df: pd.DataFrame, target_col: str):
    """Compact correlation heatmap on top-variance numeric columns."""
    st.markdown("### 🕸️ Correlation Network (Pearson)")

    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        st.info("At least two numerical columns are required for a correlation heatmap.")
        return

    top_cols = numeric_df.var().nlargest(10).index.tolist()
    if target_col in numeric_df.columns and target_col not in top_cols:
        top_cols[-1] = target_col

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        numeric_df[top_cols].corr(),
        annot=len(top_cols) <= 10,
        fmt=".2f",
        cmap="vlag",
        center=0,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Top Volatility Correlation Grid")
    plt.tight_layout()
    _safe_render(fig)


def render_pairplot(df: pd.DataFrame, target_col: str):
    """On-demand pairplot — only generated when the user clicks the button."""
    st.markdown("### 🎲 Multi-Dimensional Pairplot")

    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_col]

    if not numeric_cols:
        st.info("No numerical features available for a pairplot.")
        return

    plot_cols = numeric_cols[:3] + [target_col]
    st.caption(f"Features: `{', '.join(plot_cols[:3])}` + Target")

    # Only render on explicit click to save memory
    if st.button("Generate Pairplot (memory-intensive)", key="pairplot_btn"):
        plot_df = df[plot_cols].dropna().copy()
        if len(plot_df) > 500:
            plot_df = plot_df.sample(n=500, random_state=42)

        hue = target_col if plot_df[target_col].nunique() <= 8 else None

        with st.spinner("Rendering..."):
            pair_fig = sns.pairplot(plot_df, hue=hue, palette="viridis",
                                   corner=True, plot_kws={"s": 12, "alpha": 0.6})
            st.pyplot(pair_fig.figure)
            plt.close(pair_fig.figure)
            gc.collect()
