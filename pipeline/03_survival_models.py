"""
03_survival_models.py (PoF3)

Purpose:
- Fit Cox Proportional Hazards model on survival data
- Fit optional Random Survival Forest (RSF) model
- (Optional) Train ML-based static PoF models (XGBoost + CatBoost)
- Predict Probability of Failure (PoF) at multiple time horizons

Inputs (intermediate):
    data/intermediate/survival_base.csv
    data/intermediate/features_pof3.csv

Outputs:
    data/outputs/pof_cox_3m.csv
    data/outputs/pof_cox_6m.csv
    data/outputs/pof_cox_12m.csv
    data/outputs/pof_rsf_3m.csv  (if sksurv available)
    data/outputs/pof_rsf_6m.csv
    data/outputs/pof_rsf_12m.csv
    data/outputs/ml/pof_ml_static.csv  (if XGBoost/CatBoost available)
    models/pof_ml_xgb.json              (if ML available)
    models/pof_ml_catboost.cbm
"""

import os
import sys
import logging
from datetime import datetime

import numpy as np
import pandas as pd

# UTF-8 stdout for Windows
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# -------------------------------------------------------------------------
# Optional ML imports (XGBoost + CatBoost)
# -------------------------------------------------------------------------
try:
    import xgboost as xgb
    from catboost import CatBoostClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, average_precision_score
    HAS_ML = True
except ImportError:
    HAS_ML = False

# -------------------------------------------------------------------------
# Optional RSF imports (sksurv)
# -------------------------------------------------------------------------
try:
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.util import Surv
    HAS_RSF = True
except ImportError:
    HAS_RSF = False

# -------------------------------------------------------------------------
# Add project root to Python path
# -------------------------------------------------------------------------
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from lifelines import CoxPHFitter

import warnings
warnings.filterwarnings("ignore")

# -------------------------------------------------------------------------
# CONFIG IMPORT WITH SAFE FALLBACKS
# -------------------------------------------------------------------------
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

# ML-related defaults (do NOT depend on CONFIG to avoid breakage)
RANDOM_STATE = 42
N_JOBS = -1
ML_OUTPUT_SUBDIR = "ml"
MODELS_DIR = "models"

LOG_DIR = "logs"
STEP_NAME = "03_survival_models"


# =============================================================================
# LOGGING
# =============================================================================

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


# =============================================================================
# HELPERS - COX / RSF
# =============================================================================

def prepare_cox_data(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Prepare survival data for Cox/RSF models:
    - Ensure required columns
    - Filter valid durations
    - Handle rare equipment classes
    - Select features
    - One-hot encode categorical variables
    """
    logger.info("[STEP] Preparing data for Cox model.")

    required = ["duration_days", "event", "Ekipman_Tipi", "cbs_id"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in survival data: {missing}")

    df = df.copy()

    # Filter invalid durations
    before = len(df)
    df = df[df["duration_days"] > 0].copy()
    dropped = before - len(df)
    if dropped > 0:
        logger.warning(f"[WARN] Dropped {dropped:,} records with non-positive duration_days.")

    # Handle rare equipment classes
    counts = df["Ekipman_Tipi"].value_counts()
    rare = counts[counts < MIN_EQUIPMENT_PER_CLASS].index.tolist()
    if rare:
        logger.info(f"[INFO] Grouping rare classes into 'Other': {list(rare)}")
        df.loc[df["Ekipman_Tipi"].isin(rare), "Ekipman_Tipi"] = "Other"

    logger.info(
        f"[INFO] Equipment classes: {df['Ekipman_Tipi'].nunique()} classes - "
        f"{sorted(df['Ekipman_Tipi'].unique())}"
    )

    # Feature set for Cox / RSF
    candidate_features = [
        "Ekipman_Yasi_Gun",
        "MTBF_Gun",
        "Son_Ariza_Gun_Sayisi",
        "Tekrarlayan_Ariza_90g_Flag",
        "Ekipman_Tipi",
    ]
    feature_cols = [c for c in candidate_features if c in df.columns]

    required_cols = feature_cols + ["duration_days", "event", "cbs_id"]
    df = df[required_cols].copy()

    # Handle missing values in numeric features
    numeric_feats = [c for c in feature_cols if c != "Ekipman_Tipi"]
    for col in numeric_feats:
        if col in df.columns:
            if df[col].isna().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                logger.info(f"[INFO] Filled NaNs in {col} with median={median_val:.2f}")

    # Save cbs_id before encoding
    cbs_ids = df[["cbs_id"]].copy()

    # One-hot encode Ekipman_Tipi
    if "Ekipman_Tipi" in df.columns:
        df = pd.get_dummies(df, columns=["Ekipman_Tipi"], drop_first=True)

    # Add cbs_id back if removed
    if "cbs_id" not in df.columns:
        df["cbs_id"] = cbs_ids["cbs_id"].values

    logger.info(f"[OK] Cox/RSF data prepared: {len(df):,} rows, {len(df.columns)} columns")
    return df


def fit_cox_model(df: pd.DataFrame, logger: logging.Logger) -> CoxPHFitter:
    """
    Fit Cox Proportional Hazards model.
    """
    logger.info("[STEP] Fitting Cox Proportional Hazards model.")

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


def compute_pof(
    cph: CoxPHFitter,
    df: pd.DataFrame,
    horizons_months: list,
    logger: logging.Logger,
) -> dict:
    """
    Compute Probability of Failure (PoF) at specified time horizons.
    Returns dict[months] -> pd.Series(index=cbs_id, values=PoF).
    """
    logger.info("[STEP] Computing PoF for multiple horizons (Cox).")

    cbs_ids = df["cbs_id"].copy()
    cols_to_drop = ["duration_days", "event", "cbs_id"]
    prediction_features = df.drop(columns=cols_to_drop).copy()

    pof_results = {}

    for horizon_months in horizons_months:
        horizon_days = horizon_months * 30
        logger.info(f"  Computing PoF for {horizon_months}M ({horizon_days} days)...")

        try:
            surv = cph.predict_survival_function(prediction_features, times=[horizon_days]).T
            pof = 1.0 - surv[horizon_days]
            pof.index = cbs_ids.values
            pof_results[horizon_months] = pof
            logger.info(f"    Mean PoF: {pof.mean():.3f}, Max: {pof.max():.3f}")
        except Exception as e:
            logger.error(f"    [ERROR] Cox PoF prediction failed for {horizon_months}M: {e}")
            continue

    logger.info(f"[OK] Cox PoF computed for {len(pof_results)} horizons.")
    return pof_results


# =============================================================================
# RANDOM SURVIVAL FOREST (RSF)
# =============================================================================

def fit_rsf_model(df: pd.DataFrame, logger: logging.Logger):
    """
    Fit Random Survival Forest (if sksurv is available).
    """
    if not HAS_RSF:
        logger.warning("[WARN] sksurv is not installed. RSF model will be skipped.")
        return None

    logger.info("[STEP] Fitting Random Survival Forest model.")

    train_df = df.drop(columns=["cbs_id"]).copy()
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
            random_state=RANDOM_STATE,
        )
        rsf.fit(X, y)
        logger.info("[OK] RSF model fitted successfully.")
        return rsf
    except Exception as e:
        logger.exception(f"[FATAL] RSF model fitting failed: {e}")
        return None


def compute_rsf_pof(rsf_model, df: pd.DataFrame, horizons_months, logger):
    """
    Compute RSF-based PoF at given horizons.
    Returns dict[months] -> pd.Series(index=cbs_id, values=PoF).
    """
    if rsf_model is None:
        logger.warning("[WARN] RSF model is None; skipping RSF PoF computation.")
        return {}

    logger.info("[STEP] Predicting PoF using RSF.")

    cbs_ids = df["cbs_id"].copy()
    X = df.drop(columns=["duration_days", "event", "cbs_id"])

    pof_results = {}

    for horizon in horizons_months:
        horizon_days = horizon * 30
        logger.info(f"  RSF PoF for {horizon}M ({horizon_days} days)...")

        try:
            surv_fns = rsf_model.predict_survival_function(X)
            pof_vals = np.array([1.0 - fn(horizon_days) for fn in surv_fns])
            pof_series = pd.Series(pof_vals, index=cbs_ids.values)

            logger.info(
                f"    Mean RSF PoF: {pof_series.mean():.3f}, "
                f"Max: {pof_series.max():.3f}"
            )

            pof_results[horizon] = pof_series
        except Exception as e:
            logger.error(f"    [ERROR] RSF PoF prediction failed for {horizon}M: {e}")

    return pof_results


# =============================================================================
# ML-BASED STATIC PoF (XGBoost + CatBoost)
# =============================================================================

def train_ml_pof_models(data: pd.DataFrame, logger: logging.Logger) -> None:
    """
    ML tabanlı PoF modeli (XGBoost + CatBoost).

    Hedef: event (0/1) → ekipman bugüne kadar en az bir kez arıza yapmış mı?
    Bu model zaman ufkuna bağlı değil, 'statik PoF / arıza eğilimi' skoru üretir.
    """
    if not HAS_ML:
        logger.warning("[WARN] XGBoost/CatBoost kütüphaneleri bulunamadı. ML PoF eğitimi atlanıyor.")
        return

    if "event" not in data.columns or "cbs_id" not in data.columns:
        logger.warning("[WARN] ML PoF için gerekli kolonlar (event, cbs_id) bulunamadı. ML adımı atlanıyor.")
        return

    logger.info("")
    logger.info("[STEP] Training ML-based PoF models (XGBoost + CatBoost).")

    df = data.copy()

    # Target
    y = df["event"].astype(int)
    if y.nunique() < 2:
        logger.warning("[WARN] ML için hedef değişken tek sınıflı. ML PoF modeli eğitilmeyecek.")
        return

    id_col = "cbs_id"
    drop_cols = {id_col, "event"}

    # Feature selection: all numeric + one categorical (Ekipman_Tipi)
    numeric_cols = [
        c
        for c in df.columns
        if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])
    ]
    cat_cols = []
    if "Ekipman_Tipi" in df.columns:
        cat_cols.append("Ekipman_Tipi")

    logger.info(f"[INFO] Numeric features: {numeric_cols}")
    logger.info(f"[INFO] Categorical features: {cat_cols}")

    # XGBoost: numeric + one-hot encoded categories
    X_xgb = df[numeric_cols].copy()
    if cat_cols:
        X_dummies = pd.get_dummies(df[cat_cols], drop_first=True)
        X_xgb = pd.concat([X_xgb, X_dummies], axis=1)

    # CatBoost: numeric + categorical directly
    feature_cols_cat = numeric_cols + cat_cols
    X_cat = df[feature_cols_cat].copy()
    cat_indices = [feature_cols_cat.index(c) for c in cat_cols]

    # Train/test split
    X_xgb_train, X_xgb_test, y_train, y_test = train_test_split(
        X_xgb, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    X_cat_train, X_cat_test, _, _ = train_test_split(
        X_cat, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
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
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )

    xgb_model.fit(X_xgb_train, y_train)
    y_proba_test = xgb_model.predict_proba(X_xgb_test)[:, 1]

    auc_xgb = roc_auc_score(y_test, y_proba_test)
    ap_xgb = average_precision_score(y_test, y_proba_test)
    logger.info(f"[OK] XGBoost test AUC: {auc_xgb:.3f}, AP: {ap_xgb:.3f}")

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
        random_state=RANDOM_STATE,
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
    # Save ML outputs
    # ------------------------------------------------------------------
    out_dir = os.path.join(OUTPUT_DIR, ML_OUTPUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)

    out_df = pd.DataFrame(
        {
            id_col: df[id_col].values,
            "PoF_ML_XGB": xgb_proba_all,
            "PoF_ML_CatBoost": cat_proba_all,
        }
    )
    out_path = os.path.join(out_dir, "pof_ml_static.csv")
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info(f"[OK] ML PoF skorları kaydedildi: {out_path}")

    # Save models
    os.makedirs(MODELS_DIR, exist_ok=True)
    xgb_model.save_model(os.path.join(MODELS_DIR, "pof_ml_xgb.json"))
    cat_model.save_model(os.path.join(MODELS_DIR, "pof_ml_catboost.cbm"))
    logger.info("[OK] ML modelleri kaydedildi (XGBoost + CatBoost).")


# =============================================================================
# MAIN
# =============================================================================

def main():
    logger = setup_logger(STEP_NAME)

    try:
        # ---------------------------------------------------------------------
        # Load data
        # ---------------------------------------------------------------------
        logger.info("[STEP] Loading survival base and features.")

        surv_path = INTERMEDIATE_PATHS["survival_base"]
        feat_path = INTERMEDIATE_PATHS["features_pof3"]

        if not os.path.exists(surv_path):
            raise FileNotFoundError(f"Survival base not found: {surv_path}")
        if not os.path.exists(feat_path):
            raise FileNotFoundError(f"Features not found: {feat_path}")

        survival_base = pd.read_csv(surv_path, encoding="utf-8-sig")
        features = pd.read_csv(feat_path, encoding="utf-8-sig")

        # Normalise ID/duration/event columns (support both Turkish + English)
        if "cbs_id" not in survival_base.columns and "CBS_ID" in survival_base.columns:
            survival_base.rename(columns={"CBS_ID": "cbs_id"}, inplace=True)
        if "duration_days" not in survival_base.columns and "Sure_Gun" in survival_base.columns:
            survival_base.rename(columns={"Sure_Gun": "duration_days"}, inplace=True)
        if "event" not in survival_base.columns and "Olay" in survival_base.columns:
            survival_base.rename(columns={"Olay": "event"}, inplace=True)

        if "cbs_id" not in features.columns and "CBS_ID" in features.columns:
            features.rename(columns={"CBS_ID": "cbs_id"}, inplace=True)

        if "Ekipman_Tipi" not in survival_base.columns:
            logger.error("Ekipman_Tipi column is missing from survival_base. Check 01_data_processing outputs.")
            raise KeyError("Ekipman_Tipi not found in survival_base.")

        logger.info(f"[OK] Loaded survival base: {len(survival_base):,} rows")
        logger.info(f"[OK] Loaded features: {len(features):,} rows")

        # ---------------------------------------------------------------------
        # Merge datasets (MTBF, chronic, last-fault info)
        # ---------------------------------------------------------------------
        merge_cols = ["cbs_id", "MTBF_Gun", "Tekrarlayan_Ariza_90g_Flag", "Son_Ariza_Gun_Sayisi"]
        merge_cols = [c for c in merge_cols if c in features.columns]

        df = survival_base.merge(features[merge_cols], on="cbs_id", how="left")
        logger.info(f"[OK] Merged dataset: {len(df):,} rows")
        logger.info("")

        # Keep a copy for ML models before encoding
        df_for_ml = df.copy()

        # ---------------------------------------------------------------------
        # Prepare for Cox / RSF
        # ---------------------------------------------------------------------
        df_cox = prepare_cox_data(df, logger)
        logger.info("")

        # ---------------------------------------------------------------------
        # Fit Cox model
        # ---------------------------------------------------------------------
        cph = fit_cox_model(df_cox, logger)
        logger.info("")

        # ---------------------------------------------------------------------
        # Compute PoF at horizons (Cox)
        # ---------------------------------------------------------------------
        pof_results = compute_pof(cph, df_cox, SURVIVAL_HORIZONS_MONTHS, logger)
        logger.info("")

        # ---------------------------------------------------------------------
        # Save Cox PoF outputs
        # ---------------------------------------------------------------------
        logger.info("[STEP] Saving Cox PoF predictions.")
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        for horizon_months, pof_series in pof_results.items():
            output_df = pd.DataFrame({
                "cbs_id": pof_series.index,
                f"PoF_{horizon_months}M": pof_series.values,
            })
            output_path = os.path.join(OUTPUT_DIR, f"pof_cox_{horizon_months}m.csv")
            output_df.to_csv(output_path, index=False, encoding="utf-8-sig")
            logger.info(f"[OK] Saved {output_path}")

        logger.info("")
        logger.info("[SUCCESS] Cox-based survival PoF completed.")
        logger.info("=" * 80)

        # ---------------------------------------------------------------------
        # RSF Modeling
        # ---------------------------------------------------------------------
        logger.info("")
        logger.info("[STEP] Training Random Survival Forest")
        rsf_model = fit_rsf_model(df_cox, logger)

        if rsf_model is not None:
            logger.info("[STEP] Predicting RSF PoF horizons")
            rsf_pof = compute_rsf_pof(rsf_model, df_cox, SURVIVAL_HORIZONS_MONTHS, logger)

            for horizon, pof_series in rsf_pof.items():
                output_df = pd.DataFrame({
                    "cbs_id": pof_series.index,
                    f"PoF_RSF_{horizon}M": pof_series.values,
                })
                output_path = os.path.join(OUTPUT_DIR, f"pof_rsf_{horizon}m.csv")
                output_df.to_csv(output_path, index=False, encoding="utf-8-sig")
                logger.info(f"[OK] Saved {output_path}")

        # ---------------------------------------------------------------------
        # ML-based static PoF (XGBoost + CatBoost)
        # ---------------------------------------------------------------------
        train_ml_pof_models(df_for_ml, logger)

        logger.info("")
        logger.info("[SUCCESS] 03_survival_models completed successfully.")
        logger.info("=" * 80)

    except Exception as e:
        logger.exception(f"[FATAL] 03_survival_models failed: {e}")
        raise


if __name__ == "__main__":
    main()
