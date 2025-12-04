"""
03_survival_models.py (PoF3)

Purpose:
- Fit Cox Proportional Hazards model on survival data
- Predict Probability of Failure (PoF) at multiple time horizons
- Optional: Random Survival Forest for comparison

Inputs:
    data/intermediate/survival_base.csv
    data/intermediate/features_pof3.csv

Outputs:
    data/outputs/pof_cox_3m.csv
    data/outputs/pof_cox_6m.csv
    data/outputs/pof_cox_12m.csv
"""

import os
import sys
import logging
from datetime import datetime

# Add project root to Python path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
from lifelines import CoxPHFitter

import warnings
warnings.filterwarnings("ignore")

try:
    from config.config import (
        INTERMEDIATE_PATHS,
        OUTPUT_DIR,
        SURVIVAL_HORIZONS_MONTHS,
        MIN_EQUIPMENT_PER_CLASS,
    )
except ImportError:
    INTERMEDIATE_PATHS = {
        "survival_base": "data/intermediate/survival_base.csv",
        "features_pof3": "data/intermediate/features_pof3.csv",
    }
    OUTPUT_DIR = "data/outputs"
    SURVIVAL_HORIZONS_MONTHS = [3, 6, 12]
    MIN_EQUIPMENT_PER_CLASS = 30


LOG_DIR = "logs"
STEP_NAME = "03_survival_models"


# ------------------------------------------------------------------------------------
# LOGGING
# ------------------------------------------------------------------------------------

def setup_logger(step_name: str) -> logging.Logger:
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"{step_name}_{ts}.log")

    logger = logging.getLogger(step_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    logger.info("=" * 80)
    logger.info(f"{step_name} - PoF3 Survival Modeling")
    logger.info("=" * 80)
    logger.info("")
    return logger


# ------------------------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------------------------

def prepare_cox_data(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Prepare survival data for Cox model:
    - Filter valid durations
    - Handle rare equipment classes
    - Select features
    - One-hot encode categorical variables
    """
    logger.info("[STEP] Preparing data for Cox model.")

    # Check required columns
    required = ["duration_days", "event", "Ekipman_Tipi"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df = df.copy()

    # Filter valid durations
    before = len(df)
    df = df[df["duration_days"] > 0].copy()
    dropped = before - len(df)
    if dropped > 0:
        logger.warning(f"[WARN] Dropped {dropped:,} records with non-positive duration.")

    # Handle rare equipment classes
    counts = df["Ekipman_Tipi"].value_counts()
    rare = counts[counts < MIN_EQUIPMENT_PER_CLASS].index.tolist()
    if rare:
        logger.info(f"[INFO] Grouping rare classes into 'Other': {rare}")
        df.loc[df["Ekipman_Tipi"].isin(rare), "Ekipman_Tipi"] = "Other"

    logger.info(f"[INFO] Equipment classes: {df['Ekipman_Tipi'].nunique()} classes - {sorted(df['Ekipman_Tipi'].unique())}")

    # Select features for Cox model
    feature_cols = ["Ekipman_Yasi_Gun", "MTBF_Gun", "Ekipman_Tipi"]
    available_features = [col for col in feature_cols if col in df.columns]

    required_cols = available_features + ["duration_days", "event", "cbs_id"]
    df = df[required_cols].copy()

    # Handle missing values
    if "MTBF_Gun" in df.columns:
        df["MTBF_Gun"] = df["MTBF_Gun"].fillna(df["MTBF_Gun"].median())

    # Save cbs_id before one-hot encoding
    cbs_ids = df[["cbs_id"]].copy()

    # One-hot encode equipment type
    df = pd.get_dummies(df, columns=["Ekipman_Tipi"], drop_first=True)

    # Add cbs_id back if it was removed
    if "cbs_id" not in df.columns:
        df["cbs_id"] = cbs_ids["cbs_id"].values

    logger.info(f"[OK] Cox data prepared: {len(df):,} rows, {len(df.columns)} features")
    return df


def fit_cox_model(df: pd.DataFrame, logger: logging.Logger) -> CoxPHFitter:
    """
    Fit Cox Proportional Hazards model.
    """
    logger.info("[STEP] Fitting Cox Proportional Hazards model.")

    # Exclude cbs_id from model training
    train_df = df.drop(columns=["cbs_id"]).copy()

    cph = CoxPHFitter(penalizer=0.01)

    try:
        cph.fit(train_df, duration_col="duration_days", event_col="event")
        logger.info("[OK] Cox model fitted successfully.")
        logger.info(f"[INFO] Concordance index: {cph.concordance_index_:.3f}")
    except Exception as e:
        logger.exception(f"[FATAL] Cox model fitting failed: {e}")
        raise

    return cph


def compute_pof(cph: CoxPHFitter, df: pd.DataFrame,
                horizons_months: list, logger: logging.Logger) -> dict:
    """
    Compute Probability of Failure (PoF) at specified time horizons.
    """
    logger.info("[STEP] Computing PoF for multiple horizons.")

    # Prepare data for prediction (exclude duration, event, cbs_id)
    cbs_ids = df["cbs_id"].copy()
    cols_to_drop = ["duration_days", "event"]
    if "cbs_id" in df.columns:
        cols_to_drop.append("cbs_id")

    prediction_features = df.drop(columns=cols_to_drop).copy()

    pof_results = {}

    for horizon_months in horizons_months:
        horizon_days = horizon_months * 30
        logger.info(f"  Computing PoF for {horizon_months}M ({horizon_days} days)...")

        try:
            survival_prob = cph.predict_survival_function(prediction_features, times=[horizon_days]).T
            pof = 1 - survival_prob[horizon_days]
            pof.index = cbs_ids.values
            pof_results[horizon_months] = pof
            logger.info(f"    Mean PoF: {pof.mean():.3f}, Max: {pof.max():.3f}")
        except Exception as e:
            logger.error(f"    [ERROR] PoF prediction failed for {horizon_months}M: {e}")
            continue

    logger.info(f"[OK] PoF computed for {len(pof_results)} horizons.")
    return pof_results


# ------------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------------

def main():
    logger = setup_logger(STEP_NAME)

    try:
        # -------------------------------------------------------------------------
        # Load data
        # -------------------------------------------------------------------------
        logger.info("[STEP] Loading survival base and features.")

        surv_path = INTERMEDIATE_PATHS["survival_base"]
        feat_path = INTERMEDIATE_PATHS["features_pof3"]

        if not os.path.exists(surv_path):
            raise FileNotFoundError(f"Survival base not found: {surv_path}")
        if not os.path.exists(feat_path):
            raise FileNotFoundError(f"Features not found: {feat_path}")

        survival_base = pd.read_csv(surv_path, encoding="utf-8-sig")
        features = pd.read_csv(feat_path, encoding="utf-8-sig")

        # Normalize column names from Turkish to internal format
        survival_base.rename(columns={"CBS_ID": "cbs_id", "Sure_Gun": "duration_days", "Olay": "event"}, inplace=True)
        features.rename(columns={"CBS_ID": "cbs_id"}, inplace=True)

        logger.info(f"[OK] Loaded survival base: {len(survival_base):,} rows")
        logger.info(f"[OK] Loaded features: {len(features):,} rows")

        # -------------------------------------------------------------------------
        # Merge datasets (get MTBF and chronic flag from features)
        # -------------------------------------------------------------------------
        feature_cols = ["cbs_id", "MTBF_Gun", "Tekrarlayan_Ariza_90g_Flag", "Son_Ariza_Gun_Sayisi"]
        df = survival_base.merge(features[feature_cols], on="cbs_id", how="left")
        logger.info(f"[OK] Merged dataset: {len(df):,} rows")
        logger.info("")

        # -------------------------------------------------------------------------
        # Prepare for Cox model
        # -------------------------------------------------------------------------
        df_cox = prepare_cox_data(df, logger)
        logger.info("")

        # -------------------------------------------------------------------------
        # Fit Cox model
        # -------------------------------------------------------------------------
        cph = fit_cox_model(df_cox, logger)
        logger.info("")

        # -------------------------------------------------------------------------
        # Compute PoF at horizons
        # -------------------------------------------------------------------------
        pof_results = compute_pof(cph, df_cox, SURVIVAL_HORIZONS_MONTHS, logger)
        logger.info("")

        # -------------------------------------------------------------------------
        # Save outputs
        # -------------------------------------------------------------------------
        logger.info("[STEP] Saving PoF predictions.")
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        for horizon_months, pof_series in pof_results.items():
            output_df = pd.DataFrame({
                "cbs_id": pof_series.index,
                f"PoF_{horizon_months}M": pof_series.values
            })
            output_path = os.path.join(OUTPUT_DIR, f"pof_cox_{horizon_months}m.csv")
            output_df.to_csv(output_path, index=False, encoding="utf-8-sig")
            logger.info(f"[OK] Saved {output_path}")

        logger.info("")
        logger.info("[SUCCESS] 03_survival_models completed successfully.")
        logger.info("=" * 80)

    except Exception as e:
        logger.exception(f"[FATAL] 03_survival_models failed: {e}")
        raise


if __name__ == "__main__":
    main()
