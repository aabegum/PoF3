"""
Advanced ML Utilities for PoF3 Pipeline

This module provides clean, reusable functions for:
- Temporal Cross-Validation
- SHAP-based feature importance
- Second ML reference window validation
- Feature selection

Designed to be imported and used without code duplication.
"""

import logging
from typing import Dict, List, Tuple, Optional
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, average_precision_score


def temporal_cross_validation(
    X: pd.DataFrame,
    y: pd.Series,
    model_fn,
    n_splits: int = 3,
    logger: Optional[logging.Logger] = None
) -> Dict[str, List[float]]:
    """
    Perform temporal cross-validation for time-series data.

    Args:
        X: Feature matrix
        y: Target vector
        model_fn: Function that returns a new model instance
        n_splits: Number of CV splits
        logger: Logger instance

    Returns:
        Dictionary with 'auc' and 'ap' lists
    """
    if logger:
        logger.info(f"[TEMPORAL CV] Starting {n_splits}-fold temporal cross-validation...")

    tscv = TimeSeriesSplit(n_splits=n_splits)
    auc_scores = []
    ap_scores = []

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = model_fn()
        model.fit(X_train, y_train)

        y_pred = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_pred)
        ap = average_precision_score(y_test, y_pred)

        auc_scores.append(auc)
        ap_scores.append(ap)

        if logger:
            logger.info(f"  Fold {fold_idx}/{n_splits}: AUC={auc:.3f}, AP={ap:.3f}")

    mean_auc = np.mean(auc_scores)
    std_auc = np.std(auc_scores)
    mean_ap = np.mean(ap_scores)
    std_ap = np.std(ap_scores)

    if logger:
        logger.info(f"[TEMPORAL CV] Mean AUC: {mean_auc:.3f} ± {std_auc:.3f}")
        logger.info(f"[TEMPORAL CV] Mean AP:  {mean_ap:.3f} ± {std_ap:.3f}")

    return {
        "auc_scores": auc_scores,
        "ap_scores": ap_scores,
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "mean_ap": mean_ap,
        "std_ap": std_ap,
    }


def compute_shap_importance(
    model,
    X: pd.DataFrame,
    max_samples: int = 1000,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Compute SHAP-based feature importance.

    Args:
        model: Trained model (XGBoost or CatBoost)
        X: Feature matrix
        max_samples: Maximum samples for SHAP explainer
        logger: Logger instance

    Returns:
        DataFrame with columns: feature, importance, abs_importance
    """
    try:
        import shap
    except ImportError:
        if logger:
            logger.warning("[WARN] SHAP not installed. Skipping SHAP importance.")
        return pd.DataFrame(columns=["feature", "importance", "abs_importance"])

    if logger:
        logger.info("[SHAP] Computing SHAP feature importance...")

    # Sample data if too large
    if len(X) > max_samples:
        X_sample = X.sample(n=max_samples, random_state=42)
    else:
        X_sample = X

    try:
        # Create explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        # Handle binary classification (shap_values might be a list)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Positive class

        # Calculate mean absolute SHAP values
        mean_shap = np.abs(shap_values).mean(axis=0)

        # Create DataFrame
        importance_df = pd.DataFrame({
            "feature": X.columns,
            "abs_importance": mean_shap,
        }).sort_values("abs_importance", ascending=False)

        if logger:
            logger.info("[SHAP] Top 10 features:")
            for idx, row in importance_df.head(10).iterrows():
                logger.info(f"  {row['feature']:30s}: {row['abs_importance']:.4f}")

        return importance_df

    except Exception as e:
        if logger:
            logger.error(f"[ERROR] SHAP computation failed: {e}")
        return pd.DataFrame(columns=["feature", "abs_importance"])


def train_second_reference_window(
    events: pd.DataFrame,
    eq: pd.DataFrame,
    ref_date_1: pd.Timestamp,
    window_days: int,
    model_fn,
    numeric_cols: List[str],
    cat_cols: List[str],
    logger: Optional[logging.Logger] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Train model with a second reference window for robustness validation.

    Args:
        events: Fault events DataFrame
        eq: Equipment master DataFrame
        ref_date_1: First reference date (from main training)
        window_days: Prediction window in days
        model_fn: Function that returns a new model instance
        numeric_cols: List of numeric feature columns
        cat_cols: List of categorical feature columns
        logger: Logger instance

    Returns:
        (predictions_df, metrics_dict)
    """
    # Calculate second reference date (6 months before first)
    ref_date_2 = ref_date_1 - timedelta(days=180)
    window_end_2 = ref_date_2 + timedelta(days=window_days)

    if logger:
        logger.info("")
        logger.info("[SECOND WINDOW] Training with alternative reference date...")
        logger.info(f"[SECOND WINDOW] T_ref_2 = {ref_date_2.date()}")
        logger.info(f"[SECOND WINDOW] Window = {ref_date_2.date()} -> {window_end_2.date()}")

    # Split past & future
    past = events[events["Ariza_Baslangic_Zamani"] <= ref_date_2].copy()
    future = events[
        (events["Ariza_Baslangic_Zamani"] > ref_date_2)
        & (events["Ariza_Baslangic_Zamani"] <= window_end_2)
    ].copy()

    if logger:
        logger.info(f"[SECOND WINDOW] Past faults: {len(past):,}")
        logger.info(f"[SECOND WINDOW] Future faults: {len(future):,}")

    # Build features (similar to main training)
    rows = []
    for cid, grp in past.groupby("cbs_id"):
        times = grp["Ariza_Baslangic_Zamani"].dropna().sort_values().values
        cnt = len(times)

        if cnt < 2:
            mtbf = np.nan
        else:
            diffs = np.diff(times).astype("timedelta64[D]").astype(int)
            mtbf = float(np.mean(diffs))

        last = times[-1] if cnt > 0 else None
        days_since_last = (ref_date_2 - last).days if last is not None else np.nan

        rows.append((cid, cnt, mtbf, days_since_last))

    past_feat = pd.DataFrame(
        rows,
        columns=["cbs_id", "Ariza_Sayisi_Gecmis", "MTBF_Gun_Gecmis", "Son_Ariza_Gun_Sayisi_Gecmis"]
    )

    # Build labels
    if future.empty:
        target = pd.DataFrame({"cbs_id": eq["cbs_id"], "Label_Ariza_Pencere": 0})
    else:
        tmp = future.groupby("cbs_id").size().rename("Label_Ariza_Pencere").reset_index()
        tmp["Label_Ariza_Pencere"] = 1
        target = tmp

    # Base equipment snapshot
    base = eq.copy()
    base = base[base["Kurulum_Tarihi"] <= ref_date_2].copy()
    base["Ekipman_Yasi_Gun_ML"] = (ref_date_2 - base["Kurulum_Tarihi"]).dt.days.clip(lower=0)

    # Merge
    df = (
        base[["cbs_id"] + cat_cols + ["Ekipman_Yasi_Gun_ML"]]
        .merge(past_feat, on="cbs_id", how="left")
        .merge(target, on="cbs_id", how="left")
    )

    df["Label_Ariza_Pencere"] = df["Label_Ariza_Pencere"].fillna(0).astype(int)
    df["Ariza_Sayisi_Gecmis"] = df["Ariza_Sayisi_Gecmis"].fillna(0).astype(int)
    df["MTBF_Gun_Gecmis"] = df["MTBF_Gun_Gecmis"].fillna(df["MTBF_Gun_Gecmis"].median())
    df["Son_Ariza_Gun_Sayisi_Gecmis"] = df["Son_Ariza_Gun_Sayisi_Gecmis"].fillna(
        df["Ekipman_Yasi_Gun_ML"]
    )

    # Train model
    y = df["Label_Ariza_Pencere"]
    X_num = df[numeric_cols].copy()
    X_cat = df[cat_cols].astype(str)
    X = pd.concat([X_num, pd.get_dummies(X_cat, drop_first=True)], axis=1)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = model_fn()
    model.fit(X_train, y_train)

    y_pred = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    ap = average_precision_score(y_test, y_pred)

    if logger:
        logger.info(f"[SECOND WINDOW] Model AUC: {auc:.3f}, AP: {ap:.3f}")

    # Return predictions
    predictions = pd.DataFrame({
        "cbs_id": df["cbs_id"],
        "PoF_Second_Window": model.predict_proba(X)[:, 1]
    })

    metrics = {
        "ref_date": ref_date_2,
        "auc": auc,
        "ap": ap,
        "n_positives": y.sum(),
        "n_total": len(y),
    }

    return predictions, metrics


def select_features_by_importance(
    importance_df: pd.DataFrame,
    min_importance: float = 0.01,
    top_k: Optional[int] = None,
    logger: Optional[logging.Logger] = None
) -> List[str]:
    """
    Select features based on importance threshold or top-k.

    Args:
        importance_df: DataFrame with 'feature' and 'abs_importance' columns
        min_importance: Minimum importance threshold (0-1)
        top_k: If specified, select top-k features regardless of threshold
        logger: Logger instance

    Returns:
        List of selected feature names
    """
    if importance_df.empty:
        return []

    if top_k is not None:
        selected = importance_df.head(top_k)["feature"].tolist()
        if logger:
            logger.info(f"[FEATURE SELECTION] Selected top {top_k} features")
    else:
        selected = importance_df[importance_df["abs_importance"] >= min_importance]["feature"].tolist()
        if logger:
            logger.info(f"[FEATURE SELECTION] Selected {len(selected)} features with importance >= {min_importance}")

    return selected
