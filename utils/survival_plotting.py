"""
Survival Curves Plotting Utilities for PoF3 Pipeline

This module provides clean functions for visualizing survival analysis results.
"""

import logging
from typing import Optional
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend


def plot_survival_curves_by_class(
    df: pd.DataFrame,
    equipment_col: str = "Ekipman_Tipi",
    duration_col: str = "duration_days",
    event_col: str = "event",
    output_path: str = None,
    logger: Optional[logging.Logger] = None
) -> str:
    """
    Plot Kaplan-Meier survival curves grouped by equipment type.

    Args:
        df: DataFrame with survival data
        equipment_col: Equipment type column name
        duration_col: Duration column name
        event_col: Event (failure) column name
        output_path: Path to save the plot
        logger: Logger instance

    Returns:
        Path to saved plot
    """
    if logger:
        logger.info("[PLOT] Plotting survival curves by equipment class...")

    try:
        from lifelines import KaplanMeierFitter
    except ImportError:
        if logger:
            logger.warning("[WARN] lifelines not installed. Skipping survival curves.")
        return None

    # Get equipment types
    equipment_types = df[equipment_col].unique()

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    kmf = KaplanMeierFitter()
    colors = plt.cm.tab10(np.linspace(0, 1, len(equipment_types)))

    for idx, eq_type in enumerate(sorted(equipment_types)):
        mask = df[equipment_col] == eq_type
        data = df[mask]

        if len(data) < 5:  # Skip if too few samples
            continue

        kmf.fit(
            durations=data[duration_col],
            event_observed=data[event_col],
            label=eq_type
        )

        kmf.plot_survival_function(ax=ax, color=colors[idx], linewidth=2)

    ax.set_xlabel("Time (days)", fontsize=12)
    ax.set_ylabel("Survival Probability", fontsize=12)
    ax.set_title("Kaplan-Meier Survival Curves by Equipment Type", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        if logger:
            logger.info(f"[PLOT] Survival curves saved → {output_path}")

    plt.close()
    return output_path


def plot_cox_coefficients(
    cox_model,
    output_path: str = None,
    logger: Optional[logging.Logger] = None,
    top_n: int = 15
) -> str:
    """
    Plot Cox model coefficients (hazard ratios).

    Args:
        cox_model: Fitted CoxPHFitter model
        output_path: Path to save the plot
        logger: Logger instance
        top_n: Number of top features to show

    Returns:
        Path to saved plot
    """
    if logger:
        logger.info("[PLOT] Plotting Cox model coefficients...")

    try:
        # Get coefficients
        coef_df = cox_model.summary[['coef', 'exp(coef)', 'p']].copy()
        coef_df = coef_df.sort_values('coef', key=abs, ascending=False).head(top_n)

        fig, ax = plt.subplots(figsize=(10, 8))

        y_pos = np.arange(len(coef_df))
        colors = ['red' if c > 0 else 'blue' for c in coef_df['coef']]

        ax.barh(y_pos, coef_df['coef'], color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(coef_df.index, fontsize=10)
        ax.set_xlabel('Coefficient (log hazard ratio)', fontsize=12)
        ax.set_title(f'Top {top_n} Cox Model Coefficients', fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            if logger:
                logger.info(f"[PLOT] Cox coefficients saved → {output_path}")

        plt.close()
        return output_path

    except Exception as e:
        if logger:
            logger.error(f"[ERROR] Cox coefficients plotting failed: {e}")
        return None


def plot_feature_importance_comparison(
    rsf_importance: pd.DataFrame,
    shap_importance: pd.DataFrame,
    output_path: str = None,
    logger: Optional[logging.Logger] = None,
    top_n: int = 15
) -> str:
    """
    Plot side-by-side comparison of RSF and SHAP feature importance.

    Args:
        rsf_importance: RSF importance DataFrame
        shap_importance: SHAP importance DataFrame
        output_path: Path to save the plot
        logger: Logger instance
        top_n: Number of top features to show

    Returns:
        Path to saved plot
    """
    if logger:
        logger.info("[PLOT] Plotting feature importance comparison...")

    try:
        # Get top features from both
        rsf_top = rsf_importance.head(top_n).set_index('feature')['importance']
        shap_top = shap_importance.head(top_n).set_index('feature')['abs_importance']

        # Combine and normalize
        combined = pd.DataFrame({
            'RSF': rsf_top,
            'SHAP': shap_top
        }).fillna(0)

        # Normalize to 0-1
        combined['RSF'] = combined['RSF'] / combined['RSF'].max() if combined['RSF'].max() > 0 else 0
        combined['SHAP'] = combined['SHAP'] / combined['SHAP'].max() if combined['SHAP'].max() > 0 else 0

        # Sort by average importance
        combined['avg'] = (combined['RSF'] + combined['SHAP']) / 2
        combined = combined.sort_values('avg', ascending=True)
        combined = combined.drop('avg', axis=1)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))

        y_pos = np.arange(len(combined))
        width = 0.35

        ax.barh(y_pos - width/2, combined['RSF'], width, label='RSF', color='steelblue', alpha=0.8)
        ax.barh(y_pos + width/2, combined['SHAP'], width, label='SHAP', color='coral', alpha=0.8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(combined.index, fontsize=9)
        ax.set_xlabel('Normalized Importance', fontsize=12)
        ax.set_title('Feature Importance Comparison (RSF vs SHAP)', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            if logger:
                logger.info(f"[PLOT] Feature importance comparison saved → {output_path}")

        plt.close()
        return output_path

    except Exception as e:
        if logger:
            logger.error(f"[ERROR] Feature importance comparison plotting failed: {e}")
        return None
