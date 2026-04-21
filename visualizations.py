"""
visualizations.py
─────────────────
Modular Advanced EDA visualization components for AutoDS platform.
Ensures safe Fallbacks and handles categorical/numeric distinctions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def render_univariate_distributions(df: pd.DataFrame):
    """
    Renders general distributions of data: Boxplots and Violins for numerics,
    Countplots for categorical. Limits categories to top values to prevent UI crowding.
    """
    st.markdown("### 📊 Distribution Explorer")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    
    col_to_plot = st.selectbox("Select Feature to Examine:", options=df.columns.tolist())
    
    if col_to_plot in numeric_cols:
        plot_type = st.radio("Plot Type:", ["Histogram", "Boxplot", "Violin Plot"], horizontal=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        
        if plot_type == "Histogram":
            sns.histplot(data=df, x=col_to_plot, kde=True, color="#059669", ax=ax)
            ax.set_title(f"Distribution of {col_to_plot}")
        elif plot_type == "Boxplot":
            sns.boxplot(data=df, x=col_to_plot, color="#0EA5E9", ax=ax, orient='h')
            ax.set_title(f"Outlier Spreads of {col_to_plot}")
        else:
            sns.violinplot(data=df, x=col_to_plot, color="#D97706", ax=ax, orient='h')
            ax.set_title(f"Density Range of {col_to_plot}")
            
        plt.tight_layout()
        st.pyplot(fig)
        
    else:
        st.info("📌 Categorical Feature Selected")
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Prevent huge cardinality crashing the UI
        val_counts = df[col_to_plot].value_counts()
        if len(val_counts) > 25:
            st.warning(f"Feature `{col_to_plot}` has {len(val_counts)} unique values. Only top 25 are shown.")
            plot_data = df[df[col_to_plot].isin(val_counts.head(25).index)]
        else:
            plot_data = df
            
        sns.countplot(data=plot_data, y=col_to_plot, order=plot_data[col_to_plot].value_counts().index, palette="crest", ax=ax)
        ax.set_title(f"Top Categories in {col_to_plot}")
        plt.tight_layout()
        st.pyplot(fig)


def render_feature_vs_target(df: pd.DataFrame, target_col: str, problem_type: str):
    """
    Determines and renders the best relationship plot bridging any feature to the central target.
    """
    st.markdown("### ⚖️ Feature vs. Target Interaction")
    features = [c for c in df.columns if c != target_col]
    
    if not features:
        st.warning("No features aside from the target exist.")
        return
        
    analysis_feat = st.selectbox("Compare Target Against:", options=features)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    is_numeric_feat = np.issubdtype(df[analysis_feat].dtype, np.number)
    
    if problem_type == "classification":
        if is_numeric_feat:
            # Boxplot overlapping class
            sns.boxplot(data=df, x=target_col, y=analysis_feat, palette="muted", ax=ax)
            ax.set_title(f"{analysis_feat} Variance Grouped by {target_col}")
        else:
            # Countplot grouped by target class
            if min(df[target_col].nunique(), df[analysis_feat].nunique()) > 15:
                st.warning("High cardinality restricts robust grouped count plots.")
                return
            sns.countplot(data=df, x=analysis_feat, hue=target_col, palette="viridis", ax=ax)
            ax.set_title(f"{analysis_feat} Class Breakdown relative to {target_col}")
            plt.xticks(rotation=45)
            
    else: # Regression
        if is_numeric_feat:
            # Scatter sequence with regression correlation tracking
            st.write(f"Evaluating linear slope dependency between `{analysis_feat}` and `{target_col}`.")
            # Subsample for speed if enormous dataset
            plot_df = df.sample(n=min(5000, len(df)), random_state=42)
            sns.regplot(data=plot_df, x=analysis_feat, y=target_col, scatter_kws={'alpha':0.5, 'color':'#0EA5E9'}, line_kws={'color': '#D97706'}, ax=ax)
            ax.set_title(f"Correlation Trend: {analysis_feat} vs {target_col}")
        else:
            # Swarm/Strip/Box
            sns.boxplot(data=df, x=analysis_feat, y=target_col, palette="crest", ax=ax)
            ax.set_title(f"Target Distribution Spans across {analysis_feat} Classes")
            plt.xticks(rotation=45)
            
    plt.tight_layout()
    st.pyplot(fig)


def render_correlation_heatmap(df: pd.DataFrame, target_col: str):
    """
    Renders correlation matrices on strictly numeric data. 
    Filters strictly top variance features to prevent UI crowding on extremely wide tables.
    """
    st.markdown("### 🕸️ Correlation Network (Pearson)")
    
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        st.info("At least two numerical columns are required to draw a correlation heatmap.")
        return
        
    top_cols = numeric_df.var().nlargest(15).index.tolist()
    if target_col in numeric_df.columns and target_col not in top_cols:
        top_cols[-1] = target_col
        
    fig, ax = plt.subplots(figsize=(10, 7))
    corr_matrix = numeric_df[top_cols].corr()
    
    sns.heatmap(
        corr_matrix,
        annot=numeric_df.shape[1] <= 12,
        fmt=".2f",
        cmap="vlag",
        center=0,
        linewidths=0.5,
        ax=ax
    )
    ax.set_title("Top Volatility Correlation Grid")
    plt.tight_layout()
    st.pyplot(fig)


def render_pairplot(df: pd.DataFrame, target_col: str):
    """
    Renders a complex Pairplot spanning numeric dependencies relative to the target grouping.
    Employs sub-sampling to heavily optimize execution on >800 record arrays.
    """
    st.markdown("### 🎲 Multi-Dimensional Array (Pairplot)")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Find up to top 4 numeric columns + target
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    if not numeric_cols:
         st.info("No numerical traits available for a dimensionality pair layout.")
         return
         
    plot_cols = numeric_cols[:4] + [target_col]
    
    st.info(f"Mapping Array Combinations across top features: `{', '.join(plot_cols[:4])}`")
    
    # Execution optimization
    plot_df = df[plot_cols].copy()
    if len(plot_df) > 800:
        st.caption("*(Random Sub-Sampling active (N=800) to optimize processing payload)*")
        plot_df = plot_df.sample(n=800, random_state=42)
        
    hue_param = target_col if plot_df[target_col].nunique() <= 10 else None
    
    with st.spinner("Generating High-Resolution Grid..."):
        fig = sns.pairplot(plot_df, hue=hue_param, palette="viridis", corner=True)
        st.pyplot(fig)
