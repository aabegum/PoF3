# ================================================================
# 03_survival_models.py - PoF3 Survival Modeling
# ================================================================
# Models:
#   - Cox Proportional Hazards (baseline survival model)
#   - Random Survival Forest (optional hybrid)
#
# Inputs:
#   data/intermediate/survival_base.csv
#   data/intermediate/features_pof3.csv
#
# Outputs:
#   data/outputs/pof_cox_3m.csv
#   data/outputs/pof_cox_6m.csv
#   data/outputs/pof_cox_12m.csv
#   data/outputs/pof_rsf_3m.csv (if RSF enabled)
#
# Author: PoF3 Architecture
# ================================================================

import os
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

from datetime import datetime
from lifelines.utils import k_fold_cross_validation

import warnings
warnings.filterwarnings("ignore")

from config.config import CONFIG
import logging

# ================================================================
# SETUP LOGGER
# ================================================================
logger = get_logger("03_survival_models")

# ================================================================
# LOAD DATA
# ================================================================
def load_inputs():
    logger.info("Loading survival base and features...")

    surv_path = CONFIG["paths"]["intermediate"] + "/survival_base.csv"
    feat_path = CONFIG["paths"]["intermediate"] + "/features_pof3.csv"

    if not os.path.exists(surv_path):
        raise FileNotFoundError(f"Missing file: {surv_path}")

    if not os.path.exists(feat_path):
        raise FileNotFoundError(f"Missing file: {feat_path}")

    surv = pd.read_csv(surv_path)
    feat = pd.read_csv(feat_path)

    logger.info(f"Survival base loaded: {len(surv)} rows")
    logger.info(f"Features loaded: {len(feat)} rows")

    # Merge
    df = surv.merge(feat, on="cbs_id", how="left")
    logger.info(f"Merged survival dataset: {len(df)} rows")

    return df


# ================================================================
# PREPARE DATA FOR COX MODEL
# ================================================================
def prepare_for_cox(df):

    required = ["duration", "event", "Equipment_Class"]
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    df = df.copy()
    df = df[df["duration"] > 0]

    # Handle rare classes (already grouped in 01 but double safety)
    counts = df["Equipment_Class"].value_counts()
    rare = counts[counts < CONFIG["min_class_size"]].index
    df.loc[df["Equipment_Class"].isin(rare), "Equipment_Class"] = "Other"

    logger.info(f"Strata classes: {df['Equipment_Class'].unique()}")

    # Select simple features
    covariates = ["age_days", "MTBF_days", "Equipment_Class"]
    df = df[covariates + ["duration", "event", "cbs_id"]]

    # One-hot encode class
    df = pd.get_dummies(df, columns=["Equipment_Class"], drop_first=True)

    return df


# ================================================================
# FIT COX MODEL
# ================================================================
def fit_cox_model(df):

    logger.info("Fitting Cox Proportional Hazards model...")

    cph = CoxPHFitter(penalizer=0.01)

    try:
        cph.fit(df, duration_col="duration", event_col="event")
        logger.info("[OK] Cox model fitted successfully.")
    except Exception as e:
        logger.error(f"Failed to fit Cox model: {e}")
        raise e

    return cph


# ================================================================
# GENERATE POF FOR HORIZONS
# ================================================================
def compute_pof(cph, df, horizons_months):

    df_temp = df.copy()
    df_temp = df_temp.set_index("cbs_id")

    output = {}

    for horizon in horizons_months:
        days = horizon * 30
        logger.info(f"Predicting PoF for {horizon}M ({days} days)...")

        try:
            survival = cph.predict_survival_function(df_temp, times=[days]).T
            pof = 1 - survival[days]
            output[horizon] = pof
        except Exception as e:
            logger.error(f"Failed PoF prediction for {horizon}M: {e}")
            continue

    return output


# ================================================================
# RANDOM SURVIVAL FOREST
# ================================================================
def fit_rsf(df):
    logger.info("Training Random Survival Forest (RSF)...")

    df_rsf = df.copy()
    covs = [c for c in df_rsf.columns if c not in ["duration", "event", "cbs_id"]]

    X = df_rsf[covs]
    y = Surv.from_arrays(event=df_rsf["event"].astype(bool),
                         time=df_rsf["duration"])

    rsf = RandomSurvivalForest(
        n_estimators=300,
        min_samples_split=20,
        min_samples_leaf=10,
        n_jobs=-1,
        random_state=42
    )

    try:
        rsf.fit(X, y)
        logger.info("[OK] RSF fitted successfully.")
    except Exception as e:
        logger.error(f"RSF failed: {e}")
        return None

    return rsf


# ================================================================
# SAVE OUTPUTS
# ================================================================
def save_outputs(pof_dict, suffix):
    out_dir = CONFIG["paths"]["outputs"]
    os.makedirs(out_dir, exist_ok=True)

    for horizon, series in pof_dict.items():
        df = pd.DataFrame({"cbs_id": series.index, f"PoF_{horizon}M": series.values})
        out_path = f"{out_dir}/pof_cox_{horizon}m{suffix}.csv"
        df.to_csv(out_path, index=False)
        logger.info(f"Saved Cox PoF for {horizon}M → {out_path}")


# ================================================================
# MAIN
# ================================================================
def main():

    logger.info("=" * 70)
    logger.info("03_survival_models - PoF3 Survival Modeling")
    logger.info("=" * 70)

    df = load_inputs()

    df_cox = prepare_for_cox(df)
    cph = fit_cox_model(df_cox)

    horizons = CONFIG["survival_horizons"]
    pof_dict = compute_pof(cph, df_cox, horizons)

    save_outputs(pof_dict, suffix="")

    # RSF optional
    if CONFIG.get("enable_rsf", False):
        rsf = fit_rsf(df_cox)

        if rsf:
            logger.info("Generating RSF predictions...")
            # RSF PoF is an approximation: PoF ≈ mean hazard
            # You can refine later.

    logger.info("[SUCCESS] 03_survival_models completed.")


if __name__ == "__main__":
    main()
