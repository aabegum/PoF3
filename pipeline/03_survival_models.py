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
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

# ML-based PoF models
try:
    import xgboost as xgb
    from catboost import CatBoostClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, average_precision_score
    HAS_ML = True
except ImportError:
    HAS_ML = False


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
# RANDOM SURVIVAL FOREST (RSF) MODEL
# ------------------------------------------------------------------------------------
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

def fit_rsf_model(df: pd.DataFrame, logger: logging.Logger) -> RandomSurvivalForest:
    logger.info("[STEP] Fitting Random Survival Forest model.")

    # Prepare data
    train_df = df.drop(columns=["cbs_id"])
    y = Surv.from_arrays(
        event=train_df["event"].astype(bool),
        time=train_df["duration_days"].astype(float),
    )
    X = train_df.drop(columns=["event", "duration_days"])

    try:
        rsf = RandomSurvivalForest(
            n_estimators=400,
            min_samples_split=20,
            min_samples_leaf=5,
            max_features="sqrt",
            n_jobs=-1,
            random_state=42,
        )
        rsf.fit(X, y)
        logger.info("[OK] RSF model fitted successfully.")
        return rsf

    except Exception as e:
        logger.exception(f"[FATAL] RSF model fitting failed: {e}")
        raise


def compute_rsf_pof(rsf_model, df: pd.DataFrame, horizons_months, logger):
    logger.info("[STEP] Predicting PoF using RSF.")

    cbs_ids = df["cbs_id"].copy()
    X = df.drop(columns=["duration_days", "event", "cbs_id"])

    pof_results = {}

    for horizon in horizons_months:
        horizon_days = horizon * 30
        logger.info(f"  RSF PoF for {horizon}M ({horizon_days} days)...")

        surv_fn = rsf_model.predict_survival_function(X)
        pof_vals = np.array([1 - fn(horizon_days) for fn in surv_fn])

        pof_results[horizon] = pd.Series(pof_vals, index=cbs_ids.values)

        logger.info(
            f"    Mean RSF PoF: {pof_results[horizon].mean():.3f}, "
            f"Max: {pof_results[horizon].max():.3f}"
        )

    return pof_results
def train_ml_pof_models(df: pd.DataFrame, logger: logging.Logger) -> None:
    """
    ML tabanlı PoF modeli (XGBoost + CatBoost).

    Hedef: event (0/1) → ekipman bugüne kadar en az bir kez arıza yapmış mı?
    Bu model zaman ufkuna bağlı değil, 'statik PoF / arıza eğilimi' skoru üretir.
    """

    if not HAS_ML:
        logger.warning("[WARN] XGBoost/CatBoost kütüphaneleri bulunamadı. ML PoF eğitimi atlanıyor.")
        return

    logger.info("")
    logger.info("[STEP] Training ML-based PoF models (XGBoost + CatBoost).")

    # Güvenli kopya
    data = df.copy()

    # Hedef: event (0/1)
    if "event" not in data.columns:
        raise ValueError("ML PoF modeli için 'event' kolonu bulunamadı.")

    y = data["event"].astype(int)

    # Kimlik ve saf süre kolonlarını ayır
    id_col = "cbs_id"
    drop_cols = {id_col, "event", "duration_days", "First_Fault_Date"}

    # Sayısal / kategorik ayrımı
    numeric_cols = [
        c
        for c in data.columns
        if c not in drop_cols
        and pd.api.types.is_numeric_dtype(data[c])
    ]
    cat_cols = []
    if "Ekipman_Tipi" in data.columns:
        cat_cols.append("Ekipman_Tipi")

    logger.info(f"[INFO] Numeric features: {numeric_cols}")
    logger.info(f"[INFO] Categorical features: {cat_cols}")

    # XGBoost için: sadece sayısal + one-hot kategori
    X_xgb = data[numeric_cols].copy()
    if cat_cols:
        X_dummies = pd.get_dummies(data[cat_cols], drop_first=True)
        X_xgb = pd.concat([X_xgb, X_dummies], axis=1)

    # CatBoost için: sayısal + kategorik birlikte
    feature_cols_cat = numeric_cols + cat_cols
    X_cat = data[feature_cols_cat].copy()
    cat_indices = [feature_cols_cat.index(c) for c in cat_cols]

    # Train / test split
    X_xgb_train, X_xgb_test, y_train, y_test = train_test_split(
        X_xgb, y, test_size=0.3, random_state=CONFIG["RANDOM_STATE"], stratify=y
    )

    X_cat_train, X_cat_test, _, _ = train_test_split(
        X_cat, y, test_size=0.3, random_state=CONFIG["RANDOM_STATE"], stratify=y
    )

    # ------------------------------------------------------------------
    # XGBoost
    # ------------------------------------------------------------------
    logger.info("[STEP] Fitting XGBoost PoF model.")
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=CONFIG["RANDOM_STATE"],
        n_jobs=CONFIG.get("N_JOBS", -1),
    )

    xgb_model.fit(X_xgb_train, y_train)
    y_proba_test = xgb_model.predict_proba(X_xgb_test)[:, 1]

    auc_xgb = roc_auc_score(y_test, y_proba_test)
    ap_xgb = average_precision_score(y_test, y_proba_test)
    logger.info(f"[OK] XGBoost test AUC: {auc_xgb:.3f}, AP: {ap_xgb:.3f}")

    # Tüm veri için tahmin
    xgb_proba_all = xgb_model.predict_proba(X_xgb)[:, 1]

    # ------------------------------------------------------------------
    # CatBoost
    # ------------------------------------------------------------------
    logger.info("[STEP] Fitting CatBoost PoF model.")
    cat_model = CatBoostClassifier(
        iterations=200,
        depth=4,
        learning_rate=0.05,
        l2_leaf_reg=3.0,
        loss_function="Logloss",
        eval_metric="AUC",
        verbose=False,
        random_state=CONFIG["RANDOM_STATE"],
    )

    cat_model.fit(
        X_cat_train,
        y_train,
        cat_features=cat_indices,
        eval_set=(X_cat_test, y_test),
        verbose=False,
    )
    y_proba_cat_test = cat_model.predict_proba(X_cat_test)[:, 1]

    auc_cat = roc_auc_score(y_test, y_proba_cat_test)
    ap_cat = average_precision_score(y_test, y_proba_cat_test)
    logger.info(f"[OK] CatBoost test AUC: {auc_cat:.3f}, AP: {ap_cat:.3f}")

    cat_proba_all = cat_model.predict_proba(X_cat)[:, 1]

    # ------------------------------------------------------------------
    # Sonuçları kaydet
    # ------------------------------------------------------------------
    out_dir = CONFIG["PATHS"]["OUTPUT_POF"]
    os.makedirs(out_dir, exist_ok=True)

    out_df = pd.DataFrame(
        {
            id_col: data[id_col],
            "PoF_ML_XGB": xgb_proba_all,
            "PoF_ML_CatBoost": cat_proba_all,
        }
    )

    out_path = os.path.join(out_dir, "pof_ml_static.csv")
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info(f"[OK] ML PoF skorları kaydedildi: {out_path}")

    # Modelleri de saklamak istersen:
    models_dir = CONFIG["PATHS"].get("MODELS", "models")
    os.makedirs(models_dir, exist_ok=True)
    xgb_model.save_model(os.path.join(models_dir, "pof_ml_xgb.json"))
    cat_model.save_model(os.path.join(models_dir, "pof_ml_catboost.cbm"))
    logger.info("[OK] ML modelleri kaydedildi (XGBoost + CatBoost).")

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
        # -------------------------- RSF Modeling --------------------------
        logger.info("")
        logger.info("[STEP] Training Random Survival Forest")
        rsf = fit_rsf_model(df_cox, logger)

        logger.info("[STEP] Predicting RSF PoF horizons")
        rsf_pof = compute_rsf_pof(rsf, df_cox, SURVIVAL_HORIZONS_MONTHS, logger)

        # Save RSF PoF outputs
        for horizon, pof_series in rsf_pof.items():
            output_df = pd.DataFrame({
                "cbs_id": pof_series.index,
                f"PoF_RSF_{horizon}M": pof_series.values
            })
            output_path = os.path.join(OUTPUT_DIR, f"pof_rsf_{horizon}m.csv")
            output_df.to_csv(output_path, index=False, encoding="utf-8-sig")
            logger.info(f"[OK] Saved {output_path}")
        # 3) ML-based PoF (XGBoost + CatBoost) — statik PoF
        train_ml_pof_models(merged, logger)

    except Exception as e:
        logger.exception(f"[FATAL] 03_survival_models failed: {e}")
        raise


if __name__ == "__main__":
    main()
