"""
03_sagkalim_modelleri.py (PoF3 - Enhanced v3)

MAJOR IMPROVEMENTS v3:
========================
1. Cox PoF Calibration - Fixes underestimation (median 0.5% → 3%)
2. Weibull AFT Models - Parametric survival with better extrapolation
3. Stratified Models - Separate models for Sigorta/Ayırıcı/Other (solves class imbalance)
4. Equipment Generation Features - Captures temporal population shifts
5. Ensemble Model - Weighted combination (Cox + Weibull + RSF + ML)
6. Improved RSF Feature Importance - Multiple fallback methods
7. Model Disagreement Diagnostics - Understand why models differ

OUTPUTS:
========
- Cox PoF (calibrated) - 4 horizons
- Weibull AFT PoF (stratified) - 4 horizons
- RSF PoF - 4 horizons
- ML PoF - 1 year window
- Ensemble PoF - 4 horizons (RECOMMENDED FOR PRODUCTION)
- Stratified model performance reports
- Weibull shape parameters (aging analysis)
"""

import os
import sys
import logging
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

warnings.filterwarnings("ignore")

from utils.date_parser import parse_date_safely
from lifelines import CoxPHFitter, WeibullAFTFitter, KaplanMeierFitter

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

# RSF
try:
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.util import Surv
    HAS_RSF = True
except Exception:
    HAS_RSF = False

# ML
try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except Exception:
    HAS_CATBOOST = False

HAS_ML = HAS_XGB or HAS_CATBOOST

# SHAP
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

# Config
try:
    from config.config import (
        ANALYSIS_DATE,
        INTERMEDIATE_PATHS,
        FEATURE_OUTPUT_PATH,
        OUTPUT_DIR,
        SURVIVAL_HORIZONS_MONTHS,
        ML_PREDICTION_WINDOW_DAYS,
        MIN_EQUIPMENT_PER_CLASS,
        RANDOM_STATE,
        LOG_DIR,
    )
except ImportError:
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    ANALYSIS_DATE = pd.Timestamp.today().normalize()
    INTERMEDIATE_PATHS = {
        "survival_base": os.path.join(DATA_DIR, "ara_ciktilar", "survival_base.csv"),
        "fault_events_clean": os.path.join(DATA_DIR, "ara_ciktilar", "fault_events_clean.csv"),
    }
    FEATURE_OUTPUT_PATH = os.path.join(DATA_DIR, "ara_ciktilar", "ozellikler_pof3.csv")
    OUTPUT_DIR = os.path.join(DATA_DIR, "sonuclar")
    SURVIVAL_HORIZONS_MONTHS = [3, 6, 12, 24]
    ML_PREDICTION_WINDOW_DAYS = 365
    MIN_EQUIPMENT_PER_CLASS = 30
    RANDOM_STATE = 42
    LOG_DIR = os.path.join(PROJECT_ROOT, "loglar")

STEP_NAME = "03_sagkalim_modelleri_v3"
MODELS_DIR = os.path.join(PROJECT_ROOT, "modeller")

# Ensemble weights (tuned based on model strengths)
ENSEMBLE_WEIGHTS = {
    'cox': 0.15,      # Lower due to underestimation (even with calibration)
    'weibull': 0.25,  # Higher - better calibration + extrapolation
    'rsf': 0.30,      # Higher - good discrimination
    'ml': 0.30        # Higher - best AUC
}


# ==============================================================================
# LOGGING
# ==============================================================================
def setup_logger(step_name: str) -> logging.Logger:
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"{step_name}_{ts}.log")

    logger = logging.getLogger(step_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    logger.info("=" * 80)
    logger.info(f"{step_name} - PoF3 Survival Models (Enhanced v3)")
    logger.info("=" * 80)
    logger.info("")
    logger.info("NEW IN v3:")
    logger.info("  - Cox PoF Calibration")
    logger.info("  - Weibull AFT Models (Stratified)")
    logger.info("  - Stratified Models (Sigorta/Ayırıcı/Other)")
    logger.info("  - Equipment Generation Features")
    logger.info("  - Ensemble Model")
    logger.info("=" * 80)
    logger.info("")
    return logger


# ==============================================================================
# TEMPORAL STABILITY ANALYSIS
# ==============================================================================
def analyze_temporal_stability(logger: logging.Logger) -> None:
    """Year-by-year temporal analysis."""
    logger.info("=" * 80)
    logger.info("[TEMPORAL STABILITY] Analyzing data trends...")
    logger.info("=" * 80)
    
    events = pd.read_csv(INTERMEDIATE_PATHS["fault_events_clean"], encoding="utf-8-sig")
    features = pd.read_csv(FEATURE_OUTPUT_PATH, encoding="utf-8-sig")
    
    events["Ariza_Baslangic_Zamani"] = pd.to_datetime(events["Ariza_Baslangic_Zamani"])
    events["Year"] = events["Ariza_Baslangic_Zamani"].dt.year
    
    faults_by_year = events.groupby("Year").size()
    
    logger.info("[TEMPORAL] Fault counts by year:")
    for year, count in faults_by_year.items():
        logger.info(f"  {year}: {count:,} faults")
    
    logger.info("=" * 80)
    logger.info("")


# ==============================================================================
# COLUMN NORMALIZATION
# ==============================================================================
def normalize_columns(df_surv: pd.DataFrame, df_feat: pd.DataFrame, logger: logging.Logger):
    """Normalize column names."""
    surv = df_surv.rename(columns={
        "CBS_ID": "cbs_id",
        "Sure_Gun": "duration_days",
        "Olay": "event",
    })

    feat = df_feat.rename(columns={
        "CBS_ID": "cbs_id",
        "Fault_Count": "Ariza_Sayisi",
        "Has_Ariza_Gecmisi": "Ariza_Gecmisi",
    })

    logger.info("[OK] Column names normalized")
    return surv, feat


# ==============================================================================
# ENHANCED DATA PREPARATION WITH EQUIPMENT GENERATION
# ==============================================================================
def prepare_cox_data(df_full: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Enhanced feature preparation with:
    - Equipment generation cohorts (temporal shift handling)
    - Maintenance flags
    - Proper imputation
    """
    required = ["duration_days", "event", "Ekipman_Tipi", "Ekipman_Yasi_Gun", "cbs_id"]
    missing = [c for c in required if c not in df_full.columns]
    if missing:
        raise KeyError(f"Required columns missing: {missing}")

    df = df_full.copy()

    # Filter duration
    before = len(df)
    df = df[df["duration_days"] > 0].copy()
    dropped = before - len(df)
    if dropped > 0:
        logger.warning(f"[WARN] Dropped {dropped:,} records with duration_days <= 0")

    # Group rare equipment types
    counts = df["Ekipman_Tipi"].value_counts()
    rare = counts[counts < MIN_EQUIPMENT_PER_CLASS].index.tolist()
    if rare:
        logger.info(f"[INFO] Grouping rare equipment into 'Other': {rare}")
        df.loc[df["Ekipman_Tipi"].isin(rare), "Ekipman_Tipi"] = "Other"

    logger.info(f"[INFO] Equipment types: {sorted(df['Ekipman_Tipi'].unique())}")

    # ==============================================================================
    # NEW: EQUIPMENT GENERATION FEATURES (v3)
    # ==============================================================================
    if "Kurulum_Tarihi" in df.columns:
        df["Kurulum_Tarihi"] = pd.to_datetime(df["Kurulum_Tarihi"], errors="coerce")
        df["Install_Year"] = df["Kurulum_Tarihi"].dt.year
        
        # Define cohorts based on temporal analysis
        # 2017 equipment: 10.6% chronic rate (HIGH RISK)
        # 2018+ equipment: 0.4% chronic rate (LOW RISK)
        df["Equipment_Generation"] = pd.cut(
            df["Install_Year"],
            bins=[2000, 2017, 2020, 2026],
            labels=["Pre2017_HighRisk", "2017-2020_Medium", "Post2020_Low"],
            right=False
        )
        
        logger.info("")
        logger.info("[GENERATION] Equipment cohort distribution:")
        for gen in ["Pre2017_HighRisk", "2017-2020_Medium", "Post2020_Low"]:
            count = (df["Equipment_Generation"] == gen).sum()
            pct = 100 * count / len(df)
            logger.info(f"  {gen}: {count:,} ({pct:.1f}%)")
        logger.info("")

    # Feature set
    base_features = [
        "Ekipman_Yasi_Gun", "MTBF_Gun", "TFF_Gun",
        "Ariza_Sayisi", "Son_Ariza_Gun_Sayisi", "Faults_Last_365d",
    ]

    chronic_features = ["Kronik_Kritik", "Kronik_Yuksek", "Kronik_Orta"]
    
    maintenance_raw = ["Bakim_Sayisi", "Bakim_Var_Mi", "Son_Bakimdan_Gecen_Gun"]

    equipment_features = []
    
    # Voltage
    if "Gerilim_Seviyesi" in df.columns:
        try:
            df["Gerilim_Seviyesi_kV"] = (
                df["Gerilim_Seviyesi"].astype(str)
                .str.extract(r"([\d\.,]+)", expand=False)
                .str.replace(",", ".", regex=False).astype(float)
            )
            equipment_features.append("Gerilim_Seviyesi_kV")
        except Exception:
            pass

    # Categorical features (sparse handling)
    categorical_features = []
    if "Marka" in df.columns and df["Marka"].notna().sum() / len(df) >= 0.05:
        categorical_features.append("Marka")
    if "Son_Bakim_Tipi" in df.columns and df["Son_Bakim_Tipi"].notna().sum() / len(df) >= 0.05:
        categorical_features.append("Son_Bakim_Tipi")

    all_features = base_features + chronic_features + maintenance_raw + equipment_features + categorical_features
    
    feature_cols = ["Ekipman_Tipi"] + [c for c in all_features if c in df.columns]
    
    # Add Equipment_Generation if created
    if "Equipment_Generation" in df.columns:
        feature_cols.append("Equipment_Generation")

    # Maintenance transformation
    if "Son_Bakimdan_Gecen_Gun" in df.columns:
        df["Never_Maintained"] = df["Son_Bakimdan_Gecen_Gun"].isna().astype(int)
        df["Son_Bakimdan_Gecen_Gun"] = df["Son_Bakimdan_Gecen_Gun"].fillna(0)
        feature_cols.append("Never_Maintained")

    required_cols = feature_cols + ["duration_days", "event", "cbs_id"]
    df = df[required_cols].copy()

    # Imputation
    logger.info("[IMPUTATION] Filling missing values...")
    numeric_cols = [c for c in df.columns if c not in ["Ekipman_Tipi", "Equipment_Generation", "cbs_id"] + categorical_features and pd.api.types.is_numeric_dtype(df[c])]

    for col in numeric_cols:
        if df[col].isna().any():
            if col in ["Bakim_Sayisi", "Bakim_Var_Mi"]:
                df[col] = df[col].fillna(0)
            else:
                med = df[col].median()
                df[col] = df[col].fillna(med if not pd.isna(med) else 0)

    for col in categorical_features:
        if col in df.columns:
            df[col] = df[col].fillna("UNKNOWN").astype(str)

    # One-hot encoding
    cbs_ids = df[["cbs_id"]].copy()
    all_categoricals = ["Ekipman_Tipi"] + categorical_features
    if "Equipment_Generation" in df.columns:
        all_categoricals.append("Equipment_Generation")
    
    df = pd.get_dummies(df, columns=all_categoricals, drop_first=True)
    
    if "cbs_id" not in df.columns:
        df["cbs_id"] = cbs_ids["cbs_id"].values

    # Remove constant columns
    numeric_features = [c for c in df.columns if c not in ["cbs_id", "duration_days", "event"]]
    constant_cols = [col for col in numeric_features if df[col].std() == 0]
    
    if constant_cols:
        logger.warning(f"[WARN] Removing constant columns: {constant_cols}")
        df = df.drop(columns=constant_cols)

    logger.info(f"[OK] Data ready: {len(df):,} rows, {len(df.columns)} columns")
    logger.info("")
    return df


# ==============================================================================
# POF DISTRIBUTION ANALYSIS
# ==============================================================================
def analyze_pof_distribution(pof_series: pd.Series, horizon_label: str, logger: logging.Logger) -> dict:
    """Comprehensive PoF distribution analysis."""
    stats = {
        "Horizon": horizon_label,
        "Mean": pof_series.mean(),
        "Median": pof_series.median(),
        "Std": pof_series.std(),
        "P25": pof_series.quantile(0.25),
        "P75": pof_series.quantile(0.75),
        "P95": pof_series.quantile(0.95),
        "P99": pof_series.quantile(0.99),
        "HighRisk_50": (pof_series > 0.5).sum(),
        "HighRisk_30": (pof_series > 0.3).sum(),
        "HighRisk_10": (pof_series > 0.1).sum(),
    }
    
    logger.info(f"  {horizon_label}:")
    logger.info(f"    Mean={stats['Mean']:.3f}, Median={stats['Median']:.3f}, Std={stats['Std']:.3f}")
    logger.info(f"    High-risk: >0.5={stats['HighRisk_50']:,}, >0.3={stats['HighRisk_30']:,}, >0.1={stats['HighRisk_10']:,}")
    
    return stats


# ==============================================================================
# STRATIFIED COX MODELS (v3 - CLASS IMBALANCE FIX)
# ==============================================================================
def fit_stratified_cox_models(df_cox: pd.DataFrame, logger: logging.Logger) -> dict:
    """
    Train separate Cox models for major equipment groups.
    Solves class imbalance (Sigorta 87% dominance).
    """
    logger.info("=" * 80)
    logger.info("[STRATIFIED COX] Training equipment-specific models...")
    logger.info("=" * 80)
    
    # Define groups
    equipment_groups = {
        'Sigorta': ['Sigorta'],
        'Ayırıcı': ['Ayırıcı'],
        'Other': ['Trafo', 'Direk', 'Hat', 'Pano', 'Other']
    }
    
    stratified_models = {}
    
    for group_name, equipment_types in equipment_groups.items():
        # Check if equipment type columns exist
        equip_cols = [c for c in df_cox.columns if any(f'Ekipman_Tipi_{et}' in c for et in equipment_types)]
        
        if not equip_cols:
            logger.warning(f"[STRATIFIED] Skipping {group_name}: no matching columns")
            continue
        
        # Filter data
        group_mask = df_cox[[c for c in equip_cols if c in df_cox.columns]].sum(axis=1) > 0
        group_data = df_cox[group_mask].copy()
        
        if len(group_data) < 50:
            logger.warning(f"[STRATIFIED] Skipping {group_name}: only {len(group_data)} samples")
            continue
        
        logger.info(f"[STRATIFIED] {group_name}: {len(group_data):,} equipment")
        
        # Drop equipment type columns (not needed within group)
        drop_cols = ['cbs_id', 'duration_days', 'event'] + [c for c in group_data.columns if 'Ekipman_Tipi_' in c]
        X_group = group_data.drop(columns=[c for c in drop_cols if c in group_data.columns])
        
        # Train Cox
        train_df = group_data[['duration_days', 'event'] + list(X_group.columns)].copy()
        
        cph = CoxPHFitter(penalizer=0.01)
        try:
            cph.fit(train_df, duration_col="duration_days", event_col="event")
            c_index = cph.concordance_index_
            logger.info(f"  C-index: {c_index:.3f}")
            
            stratified_models[group_name] = {
                'model': cph,
                'equipment_types': equipment_types,
                'n_samples': len(group_data),
                'c_index': c_index
            }
        except Exception as e:
            logger.error(f"  Training failed: {e}")
    
    logger.info("=" * 80)
    logger.info("")
    return stratified_models


# ==============================================================================
# CALIBRATED COX POF (v3 - FIXES UNDERESTIMATION)
# ==============================================================================
def compute_pof_from_cox_calibrated(
    stratified_models: dict,
    df_cox: pd.DataFrame,
    horizons_months,
    logger: logging.Logger,
) -> dict:
    """
    Calibrated Cox PoF using stratified models.
    Applies empirical calibration to match observed failure rates.
    """
    logger.info("[COX CALIBRATED] Computing PoF with stratification + calibration...")
    
    results = {}
    calibration_info = []
    
    for m in horizons_months:
        days = m * 30
        all_pofs = []
        
        for group_name, model_info in stratified_models.items():
            cph = model_info['model']
            equipment_types = model_info['equipment_types']
            
            # Get equipment in this group
            equip_cols = [c for c in df_cox.columns if any(f'Ekipman_Tipi_{et}' in c for et in equipment_types)]
            group_mask = df_cox[[c for c in equip_cols if c in df_cox.columns]].sum(axis=1) > 0
            group_data = df_cox[group_mask].copy()
            
            if group_data.empty:
                continue
            
            # Prepare features
            drop_cols = ['cbs_id', 'duration_days', 'event'] + [c for c in group_data.columns if 'Ekipman_Tipi_' in c]
            X = group_data.drop(columns=[c for c in drop_cols if c in group_data.columns])
            
            # Raw prediction
            try:
                surv = cph.predict_survival_function(X, times=[days]).T
                pof_raw = 1.0 - surv[days]
                
                # Calibration
                obs_failures = group_data[(group_data['event'] == 1) & (group_data['duration_days'] <= days)].shape[0]
                obs_rate = obs_failures / len(group_data)
                pred_mean = pof_raw.mean()
                
                if pred_mean > 0.001:
                    calibration_factor = obs_rate / pred_mean
                else:
                    calibration_factor = 1.0
                
                pof_calibrated = (pof_raw * calibration_factor).clip(0, 1)
                pof_calibrated.index = group_data['cbs_id'].values
                
                all_pofs.append(pof_calibrated)
                
                calibration_info.append({
                    "Horizon_Months": m,
                    "Group": group_name,
                    "Observed_Rate": obs_rate,
                    "Raw_Mean": pred_mean,
                    "Calibration_Factor": calibration_factor,
                    "Calibrated_Mean": pof_calibrated.mean()
                })
                
            except Exception as e:
                logger.error(f"  {group_name} {m}mo failed: {e}")
        
        if all_pofs:
            combined = pd.concat(all_pofs)
            results[m] = combined
            
            # Log
            stats = analyze_pof_distribution(combined, f"Cox-Cal {m}mo", logger)
    
    # Save calibration report
    if calibration_info:
        calib_df = pd.DataFrame(calibration_info)
        calib_path = os.path.join(OUTPUT_DIR, "cox_calibration_report.csv")
        calib_df.to_csv(calib_path, index=False, encoding="utf-8-sig")
        logger.info(f"[OK] Calibration report → {calib_path}")
    
    logger.info("")
    return results


# ==============================================================================
# WEIBULL AFT MODELS (v3 - NEW)
# ==============================================================================
def fit_weibull_aft_stratified(df_cox: pd.DataFrame, logger: logging.Logger) -> dict:
    """
    Fit Weibull AFT models (stratified by equipment group).
    Provides shape parameters for aging analysis.
    """
    logger.info("=" * 80)
    logger.info("[WEIBULL AFT] Training parametric models (stratified)...")
    logger.info("=" * 80)
    
    equipment_groups = {
        'Sigorta': ['Sigorta'],
        'Ayırıcı': ['Ayırıcı'],
        'Other': ['Trafo', 'Direk', 'Hat', 'Pano', 'Other']
    }
    
    weibull_models = {}
    
    for group_name, equipment_types in equipment_groups.items():
        equip_cols = [c for c in df_cox.columns if any(f'Ekipman_Tipi_{et}' in c for et in equipment_types)]
        
        if not equip_cols:
            continue
        
        group_mask = df_cox[[c for c in equip_cols if c in df_cox.columns]].sum(axis=1) > 0
        group_data = df_cox[group_mask].copy()
        
        if len(group_data) < 50:
            continue
        
        logger.info(f"[WEIBULL] {group_name}: {len(group_data):,} equipment")
        
        drop_cols = ['cbs_id', 'duration_days', 'event'] + [c for c in group_data.columns if 'Ekipman_Tipi_' in c]
        X_group = group_data.drop(columns=[c for c in drop_cols if c in group_data.columns])
        
        train_df = group_data[['duration_days', 'event'] + list(X_group.columns)].copy()
        
        wf = WeibullAFTFitter(penalizer=0.01)
        try:
            wf.fit(train_df, duration_col='duration_days', event_col='event')
            c_index = wf.concordance_index_
            
            # Shape parameter
            try:
                rho = float(wf.rho_)
            except:
                try:
                    rho = float(wf.lambda_.values[0])
                except:
                    rho = np.nan

            
            logger.info(f"  C-index: {c_index:.3f}")
            logger.info(f"  Shape (ρ): {rho:.2f}", end="")
            
            if rho < 1:
                logger.info(" → Early failures (infant mortality)")
            elif abs(rho - 1) < 0.1:
                logger.info(" → Constant failure rate")
            else:
                logger.info(" → Wear-out failures (aging)")
            
            weibull_models[group_name] = {
                'model': wf,
                'equipment_types': equipment_types,
                'n_samples': len(group_data),
                'c_index': c_index,
                'shape': rho
            }
        except Exception as e:
            logger.error(f"  Training failed: {e}")
    
    logger.info("=" * 80)
    logger.info("")
    return weibull_models


def compute_pof_from_weibull(
    weibull_models: dict,
    df_cox: pd.DataFrame,
    horizons_months,
    logger: logging.Logger
) -> dict:
    """Compute PoF from Weibull AFT models."""
    logger.info("[WEIBULL] Computing PoF from AFT models...")
    
    results = {}
    
    for m in horizons_months:
        days = m * 30
        all_pofs = []
        
        for group_name, model_info in weibull_models.items():
            wf = model_info['model']
            equipment_types = model_info['equipment_types']
            
            equip_cols = [c for c in df_cox.columns if any(f'Ekipman_Tipi_{et}' in c for et in equipment_types)]
            group_mask = df_cox[[c for c in equip_cols if c in df_cox.columns]].sum(axis=1) > 0
            group_data = df_cox[group_mask].copy()
            
            if group_data.empty:
                continue
            
            drop_cols = ['cbs_id', 'duration_days', 'event'] + [c for c in group_data.columns if 'Ekipman_Tipi_' in c]
            X = group_data.drop(columns=[c for c in drop_cols if c in group_data.columns])
            
            try:
                surv = wf.predict_survival_function(X, times=[days]).T
                pof = 1.0 - surv[days]
                pof.index = group_data['cbs_id'].values
                all_pofs.append(pof)
            except Exception as e:
                logger.error(f"  {group_name} {m}mo failed: {e}")
        
        if all_pofs:
            combined = pd.concat(all_pofs)
            results[m] = combined
            
            stats = analyze_pof_distribution(combined, f"Weibull {m}mo", logger)
    
    logger.info("")
    return results

from sklearn.linear_model import LassoCV
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

def feature_selection_block(df, logger):
    logger.info("[FEATURE SELECTION] Running VIF + LASSO + RFE + Consensus...")

    df = df.copy()
    target = df["event"]
    features = df.drop(columns=["cbs_id", "duration_days", "event"])
    # --------------------------------------------
    # FULL HARD CLEAN FOR VIF
    # --------------------------------------------

    # 1) Ensure all columns are numeric
    for col in features.columns:
        features[col] = pd.to_numeric(features[col], errors='coerce')

    # 2) Replace inf / -inf
    features = features.replace([np.inf, -np.inf], np.nan)

    # 3) Drop all-NaN columns
    features = features.dropna(axis=1, how='all')

    # 4) Fill NaN with median
    features = features.fillna(features.median())

    # 5) Force float64 dtype for ALL columns
    features = features.astype('float64')

    # --------------------------------------------
    # 1) VIF FILTER
    # --------------------------------------------
    
    # --------------------------------------------
    # 1) VIF FILTER
    # --------------------------------------------

    # HARD CLEANING: ensure VIF-safe matrix
    for col in features.columns:
        features[col] = pd.to_numeric(features[col], errors='coerce')

    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.dropna(axis=1, how="all")
    features = features.fillna(features.median())
    features = features.astype("float64")

    logger.info(f"[DEBUG] VIF input dtypes:\n{features.dtypes}")

    vif_df = pd.DataFrame()
    vif_df["feature"] = features.columns
    vif_df["VIF"] = [
        variance_inflation_factor(features.values, i)
        for i in range(features.shape[1])
    ]

    high_vif = vif_df[vif_df["VIF"] > 15]["feature"].tolist()
    logger.info(f"[VIF] Removing high-VIF features: {high_vif}")

    features_v1 = features.drop(columns=high_vif)


    # --------------------------------------------
    # 2) LASSO (sparse selection)
    # --------------------------------------------
    scaler = StandardScaler()
    Xs = scaler.fit_transform(features_v1)

    try:
        lasso = LassoCV(cv=5, random_state=RANDOM_STATE)
        lasso.fit(Xs, target)

        lasso_keep = [
            f for f, coef in zip(features_v1.columns, lasso.coef_) if abs(coef) > 1e-6
        ]
        logger.info(f"[LASSO] Selected features: {lasso_keep}")
    except Exception:
        logger.warning("[LASSO] Failed — keeping all VIF-clean features")
        lasso_keep = list(features_v1.columns)

    features_v2 = features_v1[lasso_keep]

    # --------------------------------------------
    # 3) RFE (Recursive Feature Elimination)
    # --------------------------------------------
    from sklearn.linear_model import LogisticRegression
    est = LogisticRegression(max_iter=200, n_jobs=-1)

    try:
        selector = RFE(est, n_features_to_select=max(10, len(features_v2)//2))
        selector.fit(features_v2, target)
        rfe_keep = features_v2.columns[selector.support_].tolist()
        logger.info(f"[RFE] Selected: {rfe_keep}")
    except Exception:
        logger.warning("[RFE] Failed — skipping")
        rfe_keep = lasso_keep

    # --------------------------------------------
    # 4) CONSENSUS
    # --------------------------------------------
    consensus = list(set(lasso_keep) & set(rfe_keep))
    logger.info(f"[CONSENSUS] Final selected features: {consensus}")

    return df[["cbs_id","duration_days","event"] + consensus]

# ==============================================================================
# RSF MODEL (IMPROVED)
# ==============================================================================
def fit_rsf_model(df_cox: pd.DataFrame, logger: logging.Logger):
    if not HAS_RSF:
        logger.warning("[RSF] scikit-survival not installed, skipping")
        return None

    import time
    from datetime import timedelta
    from sksurv.util import Surv
    from sksurv.ensemble import RandomSurvivalForest

    logger.info("[RSF] Training Random Survival Forest...")

    work = df_cox.copy()
    work = work[work["duration_days"] > 0].copy()

    try:
        y = Surv.from_arrays(
            event=work["event"].astype(bool),
            time=work["duration_days"].astype(float),
        )

        # All RSF features (after your feature_selection_block)
        X = work.drop(columns=["event", "duration_days", "cbs_id"])

        rsf = RandomSurvivalForest(
            n_estimators=400,
            min_samples_split=20,
            min_samples_leaf=5,
            max_features="sqrt",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        rsf.fit(X, y)
        logger.info("[RSF] Model trained successfully")

        # ----------------------------------------------------------
        # 1) Try built-in feature_importances_
        # ----------------------------------------------------------
        logger.info("[RSF] Computing feature importance...")

        try:
            if hasattr(rsf, "feature_importances_") and rsf.feature_importances_ is not None:
                importance = rsf.feature_importances_

                if importance is None or len(importance) != X.shape[1]:
                    raise ValueError("Invalid feature_importances_ shape")

                df_imp = pd.DataFrame({"feature": X.columns, "importance": importance})
                df_imp = df_imp.sort_values("importance", ascending=False)

                logger.info("[RSF] Top 10 features (built-in):")
                for _, row in df_imp.head(10).iterrows():
                    logger.info(f"  {row['feature']:40s}: {row['importance']:.4f}")

                importance_path = os.path.join(OUTPUT_DIR, "rsf_feature_importance.csv")
                df_imp.to_csv(importance_path, index=False, encoding="utf-8-sig")
                logger.info(f"[RSF] Importance saved → {importance_path}")

                return rsf
            else:
                raise AttributeError("No usable feature_importances_ on RSF")

        except Exception as e:
            logger.warning(f"[RSF] Built-in importance failed; switching to permutation importance: {e}")

        # ----------------------------------------------------------
        # 2) Manual permutation importance WITH LIVE PROGRESS
        #    - Uses rsf.score(X, y) (concordance index)
        # ----------------------------------------------------------
        n_features = X.shape[1]
        n_repeats = 10

        logger.info(f"[RSF] Permutation importance started: {n_features} features × {n_repeats} repeats")

        start_total = time.time()

        try:
            baseline_score = rsf.score(X, y)
            logger.info(f"[RSF] Baseline concordance score: {baseline_score:.4f}")
        except Exception as e:
            logger.error(f"[RSF] Baseline score failed, skipping permutation importance: {e}")
            return rsf

        importances = np.zeros(n_features)

        for feat_idx, col in enumerate(X.columns):
            feat_start = time.time()
            scores = []

            logger.info(f"[RSF] ---- Feature {feat_idx+1}/{n_features}: '{col}' ----")

            for r in range(n_repeats):
                loop_start = time.time()

                X_perm = X.copy()
                X_perm.iloc[:, feat_idx] = np.random.permutation(X_perm.iloc[:, feat_idx].values)

                try:
                    score_perm = rsf.score(X_perm, y)
                except Exception as e:
                    logger.error(
                        f"[RSF] Error in permutation (feature='{col}', repeat={r+1}/{n_repeats}): {e}"
                    )
                    score_perm = np.nan

                scores.append(score_perm)

                loop_elapsed = time.time() - loop_start
                logger.info(
                    f"[RSF]   repeat {r+1}/{n_repeats} for '{col}' finished in {loop_elapsed:.2f}s"
                )

            valid_scores = [s for s in scores if not np.isnan(s)]
            if valid_scores:
                mean_drop = baseline_score - np.mean(valid_scores)
            else:
                mean_drop = 0.0

            importances[feat_idx] = max(mean_drop, 0.0)

            feat_elapsed = time.time() - feat_start
            remaining = (n_features - feat_idx - 1) * feat_elapsed

            logger.info(
                f"[RSF] Completed feature '{col}' in {timedelta(seconds=int(feat_elapsed))}. "
                f"Estimated remaining ≈ {timedelta(seconds=int(remaining))}"
            )

        total_elapsed = time.time() - start_total
        logger.info(f"[RSF] Permutation importance completed in {timedelta(seconds=int(total_elapsed))}")

        df_imp = pd.DataFrame({"feature": X.columns, "importance": importances})
        df_imp = df_imp.sort_values("importance", ascending=False)

        logger.info("[RSF] Top 10 features (permutation):")
        for _, row in df_imp.head(10).iterrows():
            logger.info(f"  {row['feature']:40s}: {row['importance']:.4f}")

        importance_path = os.path.join(OUTPUT_DIR, "rsf_feature_importance.csv")
        df_imp.to_csv(importance_path, index=False, encoding="utf-8-sig")
        logger.info(f"[RSF] Importance saved → {importance_path}")

        return rsf

    except Exception as e:
        logger.exception(f"[RSF] Training failed: {e}")
        return None



def compute_pof_from_rsf(rsf_model, df_cox, horizons_months, logger):
    if rsf_model is None:
        return {}

    logger.info("[RSF] Computing PoF...")

    work = df_cox.copy()
    cbs_ids = work["cbs_id"].copy()
    X = work.drop(columns=["duration_days", "event", "cbs_id"])

    results = {}

    try:
        surv_fns = rsf_model.predict_survival_function(X)

        for m in horizons_months:
            days = m * 30
            try:
                pof_vals = np.array([1.0 - fn(days) for fn in surv_fns])
                series = pd.Series(pof_vals, index=cbs_ids.values)
                results[m] = series
                
                analyze_pof_distribution(series, f"RSF {m}mo", logger)
                
            except Exception as e:
                logger.error(f"  RSF {m}mo failed: {e}")

    except Exception as e:
        logger.error(f"[RSF] Prediction failed: {e}")

    logger.info("")
    return results


# ==============================================================================
# ML MODELS (LEAKAGE-FREE)
# ==============================================================================
def build_leakage_free_ml_dataset(logger):
    logger.info("[ML] Building leakage-free dataset...")

    metadata_path = os.path.join(
        os.path.dirname(INTERMEDIATE_PATHS["fault_events_clean"]),
        "data_range_metadata.csv"
    )

    if os.path.exists(metadata_path):
        metadata = pd.read_csv(metadata_path, encoding="utf-8-sig")
        DATA_END_DATE = pd.to_datetime(metadata.loc[metadata["Parameter"] == "DATA_END_DATE", "Value"].iloc[0])
        DATA_START_DATE = pd.to_datetime(metadata.loc[metadata["Parameter"] == "DATA_START_DATE", "Value"].iloc[0])
    else:
        DATA_END_DATE = pd.to_datetime(ANALYSIS_DATE)
        DATA_START_DATE = DATA_END_DATE - timedelta(days=1636)

    ref_date = DATA_END_DATE - timedelta(days=ML_PREDICTION_WINDOW_DAYS)
    window_end = ref_date + timedelta(days=ML_PREDICTION_WINDOW_DAYS)

    logger.info(f"[ML] Reference date: {ref_date.date()}")
    logger.info(f"[ML] Prediction window: {ref_date.date()} → {window_end.date()}")

    events = pd.read_csv(INTERMEDIATE_PATHS["fault_events_clean"], encoding="utf-8-sig")
    eq = pd.read_csv(INTERMEDIATE_PATHS["equipment_master"], encoding="utf-8-sig")

    events["Ariza_Baslangic_Zamani"] = events["Ariza_Baslangic_Zamani"].apply(parse_date_safely)
    eq["Kurulum_Tarihi"] = eq["Kurulum_Tarihi"].apply(parse_date_safely)

    events["cbs_id"] = events["cbs_id"].astype(str).str.lower().str.strip()
    eq["cbs_id"] = eq["cbs_id"].astype(str).str.lower().str.strip()

    past = events[events["Ariza_Baslangic_Zamani"] <= ref_date].copy()
    future = events[(events["Ariza_Baslangic_Zamani"] > ref_date) & (events["Ariza_Baslangic_Zamani"] <= window_end)].copy()

    # Build past features
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
        days_since = (ref_date - last).days if last is not None else np.nan

        rows.append((cid, cnt, mtbf, days_since))

    past_feat = pd.DataFrame(rows, columns=["cbs_id", "Ariza_Sayisi_Gecmis", "MTBF_Gun_Gecmis", "Son_Ariza_Gun_Sayisi_Gecmis"])

    # Labels
    if future.empty:
        target = pd.DataFrame({"cbs_id": eq["cbs_id"], "Label_Ariza_Pencere": 0})
    else:
        tmp = future.groupby("cbs_id").size().rename("Label_Ariza_Pencere").reset_index()
        tmp["Label_Ariza_Pencere"] = 1
        target = tmp

    base = eq.copy()
    base = base[base["Kurulum_Tarihi"] <= ref_date].copy()
    base["Ekipman_Yasi_Gun_ML"] = (ref_date - base["Kurulum_Tarihi"]).dt.days.clip(lower=0)

    df = (
        base[["cbs_id", "Ekipman_Tipi", "Gerilim_Seviyesi", "Marka", "Ekipman_Yasi_Gun_ML"]]
        .merge(past_feat, on="cbs_id", how="left")
        .merge(target, on="cbs_id", how="left")
    )

    df["Label_Ariza_Pencere"] = df["Label_Ariza_Pencere"].fillna(0).astype(int)
    df["Ariza_Sayisi_Gecmis"] = df["Ariza_Sayisi_Gecmis"].fillna(0).astype(int)
    df["MTBF_Gun_Gecmis"] = df["MTBF_Gun_Gecmis"].fillna(df["MTBF_Gun_Gecmis"].median())
    df["Son_Ariza_Gun_Sayisi_Gecmis"] = df["Son_Ariza_Gun_Sayisi_Gecmis"].fillna(df["Ekipman_Yasi_Gun_ML"])

    logger.info(f"[ML] Dataset: {len(df):,} equipment, {df['Label_Ariza_Pencere'].sum():,} positives")
    logger.info("")

    return df


def train_leakage_free_ml_models(df, logger):
    logger.info("[ML] Training XGBoost + CatBoost...")

    y = df["Label_Ariza_Pencere"]

    numeric_cols = ["Ekipman_Yasi_Gun_ML", "Ariza_Sayisi_Gecmis", "MTBF_Gun_Gecmis", "Son_Ariza_Gun_Sayisi_Gecmis"]
    cat_cols = ["Ekipman_Tipi", "Gerilim_Seviyesi", "Marka"]

    X_num = df[numeric_cols].copy()
    X_cat = df[cat_cols].astype(str)

    X_xgb = pd.concat([X_num, pd.get_dummies(X_cat, drop_first=True)], axis=1)
    X_catb = pd.concat([X_num, X_cat], axis=1)
    cat_idx = [X_catb.columns.get_loc(c) for c in cat_cols]

    X_train_xgb, X_test_xgb, y_train, y_test = train_test_split(
        X_xgb, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )

    X_train_cat, X_test_cat, _, _ = train_test_split(
        X_catb, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )

    # XGBoost
    model_xgb = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_STATE, eval_metric="logloss"
    )
    model_xgb.fit(X_train_xgb, y_train)

    proba_xgb = model_xgb.predict_proba(X_test_xgb)[:, 1]
    auc_xgb = roc_auc_score(y_test, proba_xgb)
    logger.info(f"[ML] XGBoost AUC={auc_xgb:.3f}")

    # CatBoost
    if HAS_CATBOOST:
        model_cat = CatBoostClassifier(
            iterations=400, depth=4, learning_rate=0.05,
            eval_metric="AUC", random_state=RANDOM_STATE, verbose=False
        )
        model_cat.fit(X_train_cat, y_train, cat_features=cat_idx, eval_set=(X_test_cat, y_test), verbose=False)

        proba_cat = model_cat.predict_proba(X_test_cat)[:, 1]
        auc_cat = roc_auc_score(y_test, proba_cat)
        logger.info(f"[ML] CatBoost AUC={auc_cat:.3f}")
    else:
        model_cat = None

    # SHAP
    if HAS_SHAP:
        logger.info("[SHAP] Computing feature importance...")
        try:
            sample_size = min(1000, len(X_xgb))
            X_sample = X_xgb.sample(n=sample_size, random_state=RANDOM_STATE)
            
            explainer = shap.TreeExplainer(model_xgb)
            shap_values = explainer.shap_values(X_sample)
            
            shap_importance = pd.DataFrame({
                "feature": X_xgb.columns,
                "shap_importance": np.abs(shap_values).mean(axis=0)
            }).sort_values("shap_importance", ascending=False)
            
            logger.info("[SHAP] Top 10 features:")
            for _, row in shap_importance.head(10).iterrows():
                logger.info(f"  {row['feature']:40s}: {row['shap_importance']:.4f}")
            
            shap_path = os.path.join(OUTPUT_DIR, "shap_feature_importance.csv")
            shap_importance.to_csv(shap_path, index=False, encoding="utf-8-sig")
        except Exception as e:
            logger.warning(f"[SHAP] Failed: {e}")

    # Save predictions
    out_data = {"cbs_id": df["cbs_id"], "PoF_ML_XGB": model_xgb.predict_proba(X_xgb)[:, 1]}
    
    if model_cat:
        out_data["PoF_ML_CatBoost"] = model_cat.predict_proba(X_catb)[:, 1]
    
    out_df = pd.DataFrame(out_data)
    
    analyze_pof_distribution(out_df["PoF_ML_XGB"], "ML XGBoost", logger)
    
    out_path = os.path.join(OUTPUT_DIR, "leakage_free_ml_pof.csv")
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info(f"[ML] Predictions saved → {out_path}")

    # Save models
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_xgb.save_model(os.path.join(MODELS_DIR, "ml_xgb.json"))
    if model_cat:
        model_cat.save_model(os.path.join(MODELS_DIR, "ml_catboost.cbm"))

    logger.info("")


# ==============================================================================
# ENSEMBLE MODEL (v3 - NEW)
# ==============================================================================
def create_ensemble_pof(
    cox_pof: dict,
    weibull_pof: dict,
    rsf_pof: dict,
    ml_pof_path: str,
    logger: logging.Logger
) -> dict:
    """
    Weighted ensemble from all models.
    Weights: Cox=0.15, Weibull=0.25, RSF=0.30, ML=0.30
    """
    logger.info("=" * 80)
    logger.info("[ENSEMBLE] Creating weighted ensemble model...")
    logger.info("=" * 80)
    logger.info(f"[ENSEMBLE] Weights: {ENSEMBLE_WEIGHTS}")
    
    ml_df = pd.read_csv(ml_pof_path, encoding="utf-8-sig")
    
    ensemble = {}
    
    for horizon in sorted(cox_pof.keys()):
        if horizon not in weibull_pof or horizon not in rsf_pof:
            continue
            
        cox_series = cox_pof[horizon]
        weibull_series = weibull_pof[horizon]
        rsf_series = rsf_pof[horizon]
        ml_series = ml_df.set_index("cbs_id")["PoF_ML_XGB"]
        
        common_ids = (
            set(cox_series.index) & 
            set(weibull_series.index) & 
            set(rsf_series.index) & 
            set(ml_series.index)
        )
        common_ids = list(common_ids)
        
        if not common_ids:
            continue
        
        cox_aligned = cox_series.loc[common_ids]
        weibull_aligned = weibull_series.loc[common_ids]
        rsf_aligned = rsf_series.loc[common_ids]
        ml_aligned = ml_series.loc[common_ids]
        
        ensemble_pof = (
            ENSEMBLE_WEIGHTS['cox'] * cox_aligned +
            ENSEMBLE_WEIGHTS['weibull'] * weibull_aligned +
            ENSEMBLE_WEIGHTS['rsf'] * rsf_aligned +
            ENSEMBLE_WEIGHTS['ml'] * ml_aligned
        )
        
        ensemble[horizon] = ensemble_pof
        
        analyze_pof_distribution(ensemble_pof, f"ENSEMBLE {horizon}mo", logger)
    
    logger.info("=" * 80)
    logger.info("")
    return ensemble


# ==============================================================================
# MODEL DIAGNOSTICS (v3 - NEW)
# ==============================================================================
def diagnose_model_disagreement(
    cox_pof: dict,
    weibull_pof: dict,
    rsf_pof: dict,
    ml_pof_path: str,
    stratified_cox: dict,
    logger: logging.Logger
) -> None:
    """Diagnose why models disagree."""
    logger.info("=" * 80)
    logger.info("[DIAGNOSIS] Model disagreement analysis...")
    logger.info("=" * 80)
    
    # Cox coefficients
    logger.info("[COX] Coefficient analysis:")
    for group_name, model_info in stratified_cox.items():
        cph = model_info['model']
        cox_coef = cph.params_.abs().sort_values(ascending=False)
        logger.info(f"  {group_name} top 3:")
        for feat, _ in cox_coef.head(3).items():
            logger.info(f"    {feat}: {cph.params_[feat]:.3f}")
    
    # Prediction variance
    horizon = 24
    if horizon in cox_pof and horizon in weibull_pof and horizon in rsf_pof:
        logger.info("")
        logger.info(f"[VARIANCE] Prediction std at {horizon}mo:")
        logger.info(f"  Cox: {cox_pof[horizon].std():.4f}")
        logger.info(f"  Weibull: {weibull_pof[horizon].std():.4f}")
        logger.info(f"  RSF: {rsf_pof[horizon].std():.4f}")
    
    # High-risk overlap
    ml_df = pd.read_csv(ml_pof_path, encoding="utf-8-sig")
    
    for horizon in [12, 24]:
        if horizon not in cox_pof or horizon not in weibull_pof or horizon not in rsf_pof:
            continue
        
        common = set(cox_pof[horizon].index) & set(weibull_pof[horizon].index) & set(rsf_pof[horizon].index) & set(ml_df["cbs_id"])
        
        cox_high = set(cox_pof[horizon].loc[common][cox_pof[horizon].loc[common] > 0.3].index)
        weibull_high = set(weibull_pof[horizon].loc[common][weibull_pof[horizon].loc[common] > 0.3].index)
        rsf_high = set(rsf_pof[horizon].loc[common][rsf_pof[horizon].loc[common] > 0.3].index)
        ml_high = set(ml_df.set_index("cbs_id").loc[common]["PoF_ML_XGB"][ml_df.set_index("cbs_id").loc[common]["PoF_ML_XGB"] > 0.3].index)
        
        logger.info("")
        logger.info(f"[OVERLAP] High-risk (PoF>0.3) at {horizon}mo:")
        logger.info(f"  Cox: {len(cox_high):,}")
        logger.info(f"  Weibull: {len(weibull_high):,}")
        logger.info(f"  RSF: {len(rsf_high):,}")
        logger.info(f"  ML: {len(ml_high):,}")
        logger.info(f"  All 4 agree: {len(cox_high & weibull_high & rsf_high & ml_high):,}")
    
    logger.info("=" * 80)
    logger.info("")


# ==============================================================================
# MAIN WORKFLOW
# ==============================================================================
def main():
    logger = setup_logger(STEP_NAME)

    try:
        # Temporal analysis
        analyze_temporal_stability(logger)

        # Load data
        logger.info("[STEP] Loading data...")
        survival_base = pd.read_csv(INTERMEDIATE_PATHS["survival_base"], encoding="utf-8-sig")
        features = pd.read_csv(FEATURE_OUTPUT_PATH, encoding="utf-8-sig")

        survival_base, features = normalize_columns(survival_base, features, logger)

        logger.info(f"[OK] Loaded: {len(survival_base):,} survival records, {len(features):,} features")

        # Merge
        surv_cols = ["cbs_id", "event", "duration_days"]
        df_surv = survival_base[surv_cols].copy()

        merge_cols = [
            "cbs_id", "Ekipman_Tipi", "Ekipman_Yasi_Gun", "MTBF_Gun", "TFF_Gun",
            "Ariza_Sayisi", "Ariza_Gecmisi", "Faults_Last_365d",
            "Kronik_Kritik", "Kronik_Yuksek", "Kronik_Orta",
            "Son_Ariza_Gun_Sayisi", "Bakim_Sayisi", "Bakim_Var_Mi",
            "Kurulum_Tarihi", "Son_Bakimdan_Gecen_Gun",
            "Son_Bakim_Tipi", "Gerilim_Seviyesi", "Marka",
        ]
        cols_in_features = [c for c in merge_cols if c in features.columns]
        feat_sub = features[cols_in_features].copy()

        df_full = df_surv.merge(feat_sub, on="cbs_id", how="left")
        logger.info(f"[OK] Merged: {len(df_full):,} rows")
        logger.info("")

        # Prepare data
        df_cox = prepare_cox_data(df_full, logger)

        # ====================================================================
        # STRATIFIED COX (v3)
        # ====================================================================
        stratified_cox = fit_stratified_cox_models(df_cox, logger)
        cox_pof = compute_pof_from_cox_calibrated(stratified_cox, df_cox, SURVIVAL_HORIZONS_MONTHS, logger)

        # Save Cox outputs
        logger.info("[STEP] Saving Cox PoF (calibrated)...")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        for m, series in cox_pof.items():
            out_df = pd.DataFrame({"cbs_id": series.index, f"PoF_Cox_Cal_{m}Ay": series.values})
            out_path = os.path.join(OUTPUT_DIR, f"cox_sagkalim_{m}ay_ariza_olasiligi.csv")
            out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
            logger.info(f"  → {out_path}")
        logger.info("")

        # ====================================================================
        # WEIBULL AFT (v3 - NEW)
        # ====================================================================
        weibull_models = fit_weibull_aft_stratified(df_cox, logger)
        weibull_pof = compute_pof_from_weibull(weibull_models, df_cox, SURVIVAL_HORIZONS_MONTHS, logger)

        # Save Weibull outputs
        logger.info("[STEP] Saving Weibull AFT PoF...")
        for m, series in weibull_pof.items():
            out_df = pd.DataFrame({"cbs_id": series.index, f"PoF_Weibull_{m}Ay": series.values})
            out_path = os.path.join(OUTPUT_DIR, f"weibull_sagkalim_{m}ay_ariza_olasiligi.csv")
            out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
            logger.info(f"  → {out_path}")
        logger.info("")

        # ====================================================================
        # RSF (IMPROVED)
        # ====================================================================
        logger.info("[STEP] Running feature selection before RSF...")
        df_rsf = feature_selection_block(df_cox, logger)
        rsf = fit_rsf_model(df_rsf, logger)
        rsf_pof = compute_pof_from_rsf(rsf, df_rsf, SURVIVAL_HORIZONS_MONTHS, logger)

        rsf = fit_rsf_model(df_cox, logger)
        rsf_pof = compute_pof_from_rsf(rsf, df_cox, SURVIVAL_HORIZONS_MONTHS, logger)

        if rsf_pof:
            logger.info("[STEP] Saving RSF PoF...")
            for m, series in rsf_pof.items():
                out_df = pd.DataFrame({"cbs_id": series.index, f"PoF_RSF_{m}Ay": series.values})
                out_path = os.path.join(OUTPUT_DIR, f"rsf_sagkalim_{m}ay_ariza_olasiligi.csv")
                out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
                logger.info(f"  → {out_path}")
            logger.info("")

        # ====================================================================
        # ML MODELS
        # ====================================================================
        df_ml = build_leakage_free_ml_dataset(logger)
        train_leakage_free_ml_models(df_ml, logger)

        # ====================================================================
        # ENSEMBLE (v3 - NEW)
        # ====================================================================
        ml_pof_path = os.path.join(OUTPUT_DIR, "leakage_free_ml_pof.csv")
        if os.path.exists(ml_pof_path) and rsf_pof and weibull_pof:
            ensemble_pof = create_ensemble_pof(cox_pof, weibull_pof, rsf_pof, ml_pof_path, logger)
            
            # Save ensemble outputs
            logger.info("[STEP] Saving ENSEMBLE PoF (RECOMMENDED)...")
            for m, series in ensemble_pof.items():
                out_df = pd.DataFrame({"cbs_id": series.index, f"PoF_Ensemble_{m}Ay": series.values})
                out_path = os.path.join(OUTPUT_DIR, f"ensemble_sagkalim_{m}ay_ariza_olasiligi.csv")
                out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
                logger.info(f"  → {out_path}")
            logger.info("")

        # ====================================================================
        # DIAGNOSTICS (v3 - NEW)
        # ====================================================================
        if os.path.exists(ml_pof_path) and rsf_pof and weibull_pof:
            diagnose_model_disagreement(
                cox_pof, weibull_pof, rsf_pof, ml_pof_path,
                stratified_cox, logger
            )

        logger.info("=" * 80)
        logger.info("[SUCCESS] 03_sagkalim_modelleri_v3 completed!")
        logger.info("=" * 80)
        logger.info("")
        logger.info("OUTPUTS:")
        logger.info("  - Cox PoF (calibrated, stratified)")
        logger.info("  - Weibull AFT PoF (stratified)")
        logger.info("  - RSF PoF")
        logger.info("  - ML PoF (XGBoost + CatBoost)")
        logger.info("  - ENSEMBLE PoF ← RECOMMENDED FOR PRODUCTION")
        logger.info("=" * 80)

    except Exception as e:
        logger.exception(f"[FATAL] Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()