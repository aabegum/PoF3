"""
03_sagkalim_modelleri.py (PoF3 - Enhanced v2)

Improvements:
- Temporal stability analysis (year-by-year)
- Fixed maintenance feature handling (binary flag + continuous)
- Updated to IEEE chronic flags from Step 02
- Better RSF error handling
- PoF distribution analysis (P50, P95, P99, high-risk counts)
- Model comparison report
- Graceful handling of sparse categorical features (Marka, Son_Bakim_Tipi)
- SHAP feature importance (when installed)

Amaç:
- Robust survival and ML models with comprehensive diagnostics
- Industry-standard PoF predictions with temporal validation
- Detailed model performance comparison
"""

import os
import sys
import logging
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Explicitly import sklearn components before try-except
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

warnings.filterwarnings("ignore")

from utils.date_parser import parse_date_safely
from lifelines import CoxPHFitter

# RSF (optional)
try:
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.util import Surv
    HAS_RSF = True
except Exception:
    HAS_RSF = False

# ML libraries
try:
    import xgboost as xgb
    HAS_XGB = True
except Exception as e:
    HAS_XGB = False
    print(f"Warning: XGBoost not available: {e}")

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError as e:
    HAS_CATBOOST = False
    print(f"Warning: CatBoost not available: {e}")
except Exception as e:
    # Handle NumPy binary incompatibility
    HAS_CATBOOST = False
    if "numpy.dtype size changed" in str(e):
        print(f"Warning: CatBoost has NumPy compatibility issue. Try: pip uninstall catboost && pip install catboost --no-cache-dir")
    else:
        print(f"Warning: CatBoost import failed: {e}")

HAS_ML = HAS_XGB or HAS_CATBOOST

# SHAP (optional but recommended)
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

try:
    from config.config import (
        ANALYSIS_DATE,
        INTERMEDIATE_PATHS,
        FEATURE_OUTPUT_PATH,
        OUTPUT_DIR,
        RESULT_PATHS,
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
        "equipment_master": os.path.join(DATA_DIR, "ara_ciktilar", "equipment_master.csv"),
    }
    FEATURE_OUTPUT_PATH = os.path.join(DATA_DIR, "ara_ciktilar", "ozellikler_pof3.csv")
    OUTPUT_DIR = os.path.join(DATA_DIR, "sonuclar")
    RESULT_PATHS = {"POF": os.path.join(DATA_DIR, "sonuclar")}
    SURVIVAL_HORIZONS_MONTHS = [3, 6, 12, 24]
    ML_PREDICTION_WINDOW_DAYS = 365
    MIN_EQUIPMENT_PER_CLASS = 30
    RANDOM_STATE = 42
    LOG_DIR = os.path.join(PROJECT_ROOT, "loglar")

STEP_NAME = "03_sagkalim_modelleri"
MODELS_DIR = os.path.join(PROJECT_ROOT, "modeller")


# ----------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------
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
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    logger.info("=" * 80)
    logger.info(f"{step_name} - PoF3 Survival Models (Enhanced v2)")
    logger.info("=" * 80)
    logger.info("")
    return logger


# ----------------------------------------------------------------------
# TEMPORAL STABILITY ANALYSIS
# ----------------------------------------------------------------------
def analyze_temporal_stability(logger: logging.Logger) -> None:
    """
    Year-by-year analysis to understand temporal CV degradation.
    Analyzes fault rates, chronic rates, and equipment population changes over time.
    """
    logger.info("=" * 80)
    logger.info("[TEMPORAL STABILITY] Analyzing data trends by year...")
    logger.info("=" * 80)
    
    # Load data
    events = pd.read_csv(INTERMEDIATE_PATHS["fault_events_clean"], encoding="utf-8-sig")
    features = pd.read_csv(FEATURE_OUTPUT_PATH, encoding="utf-8-sig")
    
    events["Ariza_Baslangic_Zamani"] = pd.to_datetime(events["Ariza_Baslangic_Zamani"])
    
    # 1. Fault rate by year
    events["Year"] = events["Ariza_Baslangic_Zamani"].dt.year
    faults_by_year = events.groupby("Year").size()
    
    logger.info("")
    logger.info("[TEMPORAL] Fault counts by year:")
    for year, count in faults_by_year.items():
        logger.info(f"  {year}: {count:,} faults")
    
    # 2. Chronic equipment rate by installation year (if available)
    if "Kurulum_Tarihi" in features.columns and "Kronik_Kritik" in features.columns:
        features["Kurulum_Tarihi"] = pd.to_datetime(features["Kurulum_Tarihi"], errors="coerce")
        features["Install_Year"] = features["Kurulum_Tarihi"].dt.year
        
        chronic_by_install_year = features.groupby("Install_Year").agg({
            "Kronik_Kritik": "mean",
            "Kronik_Yuksek": "mean",
            "cbs_id": "size"
        }).rename(columns={"cbs_id": "Count"})
        
        logger.info("")
        logger.info("[TEMPORAL] Chronic rates by equipment installation year:")
        logger.info(f"{'Year':<8} {'Count':<10} {'Kritik %':<12} {'Yuksek %':<12}")
        for year, row in chronic_by_install_year.iterrows():
            if pd.notna(year) and year >= 2000:  # Filter realistic years
                logger.info(
                    f"{int(year):<8} {int(row['Count']):<10} "
                    f"{row['Kronik_Kritik']*100:>10.1f}% {row['Kronik_Yuksek']*100:>10.1f}%"
                )
    
    # 3. Equipment type mix by fault year
    if "Ekipman_Tipi" in events.columns:
        equip_mix = events.groupby(["Year", "Ekipman_Tipi"]).size().unstack(fill_value=0)
        
        logger.info("")
        logger.info("[TEMPORAL] Equipment type distribution by fault year:")
        logger.info(equip_mix.to_string())
    
    # 4. Save detailed temporal report
    temporal_report_path = os.path.join(OUTPUT_DIR, "temporal_stability_report.csv")
    
    summary_data = []
    for year in faults_by_year.index:
        year_events = events[events["Year"] == year]
        summary_data.append({
            "Year": year,
            "Total_Faults": len(year_events),
            "Unique_Equipment": year_events["cbs_id"].nunique(),
            "Avg_Faults_Per_Equipment": len(year_events) / year_events["cbs_id"].nunique() if year_events["cbs_id"].nunique() > 0 else 0
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(temporal_report_path, index=False, encoding="utf-8-sig")
    logger.info(f"[TEMPORAL] Detailed report saved → {temporal_report_path}")
    logger.info("=" * 80)
    logger.info("")


# ----------------------------------------------------------------------
# COLUMN NORMALIZATION
# ----------------------------------------------------------------------
def normalize_columns(df_surv: pd.DataFrame, df_feat: pd.DataFrame, logger: logging.Logger):
    """
    Normalize column names to PoF3 Turkish standard.
    UPDATED: Maps old Kronik_90g_Flag to new IEEE chronic flags.
    """
    surv = df_surv.rename(
        columns={
            "CBS_ID": "cbs_id",
            "Sure_Gun": "duration_days",
            "Olay": "event",
        }
    )

    feat = df_feat.rename(
        columns={
            "CBS_ID": "cbs_id",
            "Fault_Count": "Ariza_Sayisi",
            "Has_Ariza_Gecmisi": "Ariza_Gecmisi",
            "Kronik_90g_Flag": "Kronik_Flag_Legacy",  # Old name for backward compatibility
            "Gerilim": "Gerilim_Seviyesi",
            "MARKA": "Marka",
            "MARKA ": "Marka",
            "component voltage": "Gerilim_Seviyesi",
            "voltage_level": "Gerilim_Seviyesi",
            "voltage_level ": "Gerilim_Seviyesi",
        }
    )

    # Validation
    if "cbs_id" not in surv.columns:
        raise KeyError("survival_base içinde 'cbs_id' kolonu bulunamadı.")
    if "duration_days" not in surv.columns:
        raise KeyError("survival_base içinde 'duration_days' kolonu bulunamadı.")
    if "event" not in surv.columns:
        raise KeyError("survival_base içinde 'event' kolonu bulunamadı.")

    # Fallbacks for missing columns
    if "Ariza_Sayisi" not in feat.columns and "Fault_Count" in feat.columns:
        feat["Ariza_Sayisi"] = feat["Fault_Count"]
    if "Ariza_Gecmisi" not in feat.columns and "Has_Ariza_Gecmisi" in feat.columns:
        feat["Ariza_Gecmisi"] = feat["Has_Ariza_Gecmisi"]

    logger.info("[OK] Column names normalized to PoF3 standard.")
    return surv, feat


# ----------------------------------------------------------------------
# IMPROVED DATA PREPARATION (WITH MAINTENANCE FLAGS)
# ----------------------------------------------------------------------
def prepare_cox_data(df_full: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    ENHANCED: Proper maintenance feature handling with binary flags.
    - Creates "Never_Maintained" binary indicator
    - Separates maintenance timing from maintenance status
    - Handles sparse categorical features gracefully
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
        logger.info(f"[INFO] Grouping rare equipment types into 'Other': {list(rare)}")
        df.loc[df["Ekipman_Tipi"].isin(rare), "Ekipman_Tipi"] = "Other"

    logger.info(
        f"[INFO] Equipment types: {df['Ekipman_Tipi'].nunique()} classes - "
        f"{sorted(df['Ekipman_Tipi'].unique())}"
    )

    # ============================================
    # FEATURE SET DEFINITION
    # ============================================
    
    # Core reliability features
    base_features = [
        "Ekipman_Yasi_Gun",
        "MTBF_Gun",
        "TFF_Gun",  # NEW: Time to First Failure
        "Ariza_Sayisi",
        "Son_Ariza_Gun_Sayisi",
        "Faults_Last_365d",  # NEW: From Step 02
    ]

    # IEEE chronic flags (NEW from Step 02)
    chronic_features = [
        "Kronik_Kritik",
        "Kronik_Yuksek",
        "Kronik_Orta",
    ]

    # Maintenance features (will be transformed)
    maintenance_raw = [
        "Bakim_Sayisi",
        "Bakim_Var_Mi",
        "Son_Bakimdan_Gecen_Gun",
    ]

    # Equipment attributes
    equipment_features = []
    
    # Voltage (numeric extraction)
    if "Gerilim_Seviyesi" in df.columns:
        try:
            df["Gerilim_Seviyesi_kV"] = (
                df["Gerilim_Seviyesi"]
                .astype(str)
                .str.extract(r"([\d\.,]+)", expand=False)
                .str.replace(",", ".", regex=False)
                .astype(float)
            )
            equipment_features.append("Gerilim_Seviyesi_kV")
            logger.info("[INFO] Gerilim_Seviyesi_kV created")
        except Exception:
            logger.warning("[WARN] Gerilim_Seviyesi numeric conversion failed")

    # Categorical features (graceful handling for sparse data)
    categorical_features = []
    
    # Marka (Brand) - only if sufficient non-null values
    if "Marka" in df.columns:
        non_null_marka = df["Marka"].notna().sum()
        coverage = non_null_marka / len(df) * 100
        if coverage >= 5.0:  # At least 5% coverage
            categorical_features.append("Marka")
            logger.info(f"[INFO] Including Marka ({coverage:.1f}% coverage)")
        else:
            logger.info(f"[INFO] Excluding Marka (only {coverage:.1f}% coverage)")
    
    # Son_Bakim_Tipi - only if sufficient data
    if "Son_Bakim_Tipi" in df.columns:
        non_null_bakim_tipi = df["Son_Bakim_Tipi"].notna().sum()
        coverage = non_null_bakim_tipi / len(df) * 100
        if coverage >= 5.0:
            categorical_features.append("Son_Bakim_Tipi")
            logger.info(f"[INFO] Including Son_Bakim_Tipi ({coverage:.1f}% coverage)")
        else:
            logger.info(f"[INFO] Excluding Son_Bakim_Tipi (only {coverage:.1f}% coverage)")

    # Combine all candidate features
    all_features = (
        base_features + chronic_features + maintenance_raw + 
        equipment_features + categorical_features
    )
    
    feature_cols = ["Ekipman_Tipi"] + [c for c in all_features if c in df.columns]
    
    # ============================================
    # MAINTENANCE FEATURE TRANSFORMATION
    # ============================================
    logger.info("")
    logger.info("[MAINTENANCE] Transforming maintenance features...")
    
    if "Son_Bakimdan_Gecen_Gun" in df.columns:
        # Create binary "never maintained" flag
        df["Never_Maintained"] = df["Son_Bakimdan_Gecen_Gun"].isna().astype(int)
        
        # For equipment WITH maintenance: keep actual days
        # For equipment WITHOUT maintenance: set to 0 (will be offset by Never_Maintained flag)
        df["Son_Bakimdan_Gecen_Gun"] = df["Son_Bakimdan_Gecen_Gun"].fillna(0)
        
        feature_cols.append("Never_Maintained")
        
        never_maintained_count = df["Never_Maintained"].sum()
        logger.info(f"[MAINTENANCE] Never maintained: {never_maintained_count:,} equipment ({100*never_maintained_count/len(df):.1f}%)")
    
    # Required columns for survival analysis
    required_cols = feature_cols + ["duration_days", "event", "cbs_id"]
    df = df[required_cols].copy()

    # ============================================
    # IMPUTATION STRATEGY
    # ============================================
    logger.info("")
    logger.info("[IMPUTATION] Filling missing values...")
    
    numeric_cols = [
        c for c in df.columns
        if c not in ["Ekipman_Tipi", "cbs_id"] + categorical_features
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    for col in numeric_cols:
        if df[col].isna().any():
            if col == "Never_Maintained":
                # Already handled above
                continue
            elif col in ["Bakim_Sayisi", "Bakim_Var_Mi"]:
                # Maintenance count: 0 for no maintenance
                df[col] = df[col].fillna(0)
                logger.info(f"[IMPUTATION] {col}: filled NaN with 0 (no maintenance)")
            else:
                # Other numeric: use median
                med = df[col].median()
                if pd.isna(med):
                    df[col] = df[col].fillna(0)
                    logger.info(f"[IMPUTATION] {col}: filled NaN with 0 (median unavailable)")
                else:
                    df[col] = df[col].fillna(med)
                    logger.info(f"[IMPUTATION] {col}: filled NaN with median={med:.2f}")

    # Categorical imputation
    for col in categorical_features:
        if col in df.columns:
            df[col] = df[col].fillna("UNKNOWN").astype(str)
            logger.info(f"[IMPUTATION] {col}: filled NaN with 'UNKNOWN'")

    # ============================================
    # ONE-HOT ENCODING
    # ============================================
    cbs_ids = df[["cbs_id"]].copy()
    
    # One-hot encode all categorical features
    all_categoricals = ["Ekipman_Tipi"] + categorical_features
    df = pd.get_dummies(df, columns=all_categoricals, drop_first=True)
    
    # Restore cbs_id
    if "cbs_id" not in df.columns:
        df["cbs_id"] = cbs_ids["cbs_id"].values

    # Final NaN check
    nan_counts = df.isna().sum()
    if nan_counts.any():
        logger.warning("[WARN] Remaining NaN values detected:")
        for col in nan_counts[nan_counts > 0].index:
            logger.warning(f"  {col}: {nan_counts[col]} NaN")
        df = df.fillna(0)
        logger.info("[IMPUTATION] Filled remaining NaN with 0")

    # Remove constant columns
    numeric_features = [c for c in df.columns if c not in ["cbs_id", "duration_days", "event"]]
    constant_cols = [col for col in numeric_features if df[col].std() == 0]
    
    if constant_cols:
        logger.warning(f"[WARN] Removing constant columns: {constant_cols}")
        df = df.drop(columns=constant_cols)

    logger.info(f"[OK] Cox/RSF data ready: {len(df):,} rows, {len(df.columns)} columns")
    logger.info("")
    return df


# ----------------------------------------------------------------------
# POF DISTRIBUTION ANALYSIS
# ----------------------------------------------------------------------
def analyze_pof_distribution(pof_series: pd.Series, horizon_label: str, logger: logging.Logger) -> dict:
    """
    Comprehensive PoF distribution analysis with quantiles and risk counts.
    """
    stats = {
        "Horizon": horizon_label,
        "Mean": pof_series.mean(),
        "Median": pof_series.median(),
        "Std": pof_series.std(),
        "Min": pof_series.min(),
        "Max": pof_series.max(),
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
    logger.info(f"    P25={stats['P25']:.3f}, P75={stats['P75']:.3f}, P95={stats['P95']:.3f}, P99={stats['P99']:.3f}")
    logger.info(f"    High-risk counts: PoF>0.5={stats['HighRisk_50']:,}, PoF>0.3={stats['HighRisk_30']:,}, PoF>0.1={stats['HighRisk_10']:,}")
    
    return stats


# ----------------------------------------------------------------------
# COX MODEL
# ----------------------------------------------------------------------
def fit_cox_model(df_cox: pd.DataFrame, logger: logging.Logger) -> CoxPHFitter:
    logger.info("[STEP] Training Cox Proportional Hazards model")

    train_df = df_cox.drop(columns=["cbs_id"]).copy()

    cph = CoxPHFitter(penalizer=0.01)
    try:
        cph.fit(train_df, duration_col="duration_days", event_col="event")
        logger.info("[OK] Cox model trained successfully")
        logger.info(f"[INFO] Concordance index: {cph.concordance_index_:.3f}")
    except Exception as e:
        logger.exception(f"[FATAL] Cox model training failed: {e}")
        raise

    return cph


def compute_pof_from_cox(
    cph: CoxPHFitter,
    df_cox: pd.DataFrame,
    horizons_months,
    logger: logging.Logger,
) -> dict:
    """
    ENHANCED: Includes distribution analysis for each horizon.
    """
    logger.info("[STEP] Computing PoF from Cox model with distribution analysis")

    cbs_ids = df_cox["cbs_id"].copy()
    drop_cols = ["duration_days", "event", "cbs_id"]
    X = df_cox.drop(columns=drop_cols).copy()

    results = {}
    dist_stats = []

    for m in horizons_months:
        days = m * 30
        try:
            surv = cph.predict_survival_function(X, times=[days]).T
            pof = 1.0 - surv[days]
            pof.index = cbs_ids.values
            results[m] = pof
            
            # Distribution analysis
            stats = analyze_pof_distribution(pof, f"Cox {m}mo", logger)
            dist_stats.append(stats)
            
        except Exception as e:
            logger.error(f"    [ERROR] Cox PoF computation failed for {m} months: {e}")

    # Save distribution stats
    if dist_stats:
        dist_df = pd.DataFrame(dist_stats)
        dist_path = os.path.join(OUTPUT_DIR, "cox_pof_distribution_stats.csv")
        dist_df.to_csv(dist_path, index=False, encoding="utf-8-sig")
        logger.info(f"[OK] Cox PoF distribution stats saved → {dist_path}")

    logger.info(f"[OK] Cox PoF computed for {len(results)} horizons")
    logger.info("")
    return results


# ----------------------------------------------------------------------
# RSF MODEL (ENHANCED ERROR HANDLING)
# ----------------------------------------------------------------------
def fit_rsf_model(df_cox: pd.DataFrame, logger: logging.Logger):
    if not HAS_RSF:
        logger.warning("[WARN] scikit-survival not installed. Skipping RSF. Install: pip install scikit-survival")
        return None

    logger.info("[STEP] Training Random Survival Forest (RSF)")

    work = df_cox.copy()
    work = work[work["duration_days"] > 0].copy()

    try:
        y = Surv.from_arrays(
            event=work["event"].astype(bool),
            time=work["duration_days"].astype(float),
        )
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
        logger.info("[OK] RSF model trained successfully")

        # ENHANCED: Better feature importance handling
        logger.info("[RSF] Computing feature importance...")
        try:
            if hasattr(rsf, 'feature_importances_'):
                importance = rsf.feature_importances_
                importance_df = pd.DataFrame({
                    "feature": X.columns,
                    "importance": importance
                }).sort_values("importance", ascending=False)

                logger.info("[RSF] Top 10 features:")
                for _, row in importance_df.head(10).iterrows():
                    logger.info(f"  {row['feature']:40s}: {row['importance']:.4f}")

                # Save importance
                importance_path = os.path.join(OUTPUT_DIR, "rsf_feature_importance.csv")
                importance_df.to_csv(importance_path, index=False, encoding="utf-8-sig")
                logger.info(f"[RSF] Feature importance saved → {importance_path}")
            else:
                logger.warning("[RSF] Model doesn't support feature_importances_ attribute")
        except AttributeError as e:
            logger.warning(f"[RSF] Feature importance not available: {str(e)}")
        except Exception as e:
            logger.warning(f"[RSF] Feature importance computation failed: {str(e)}")

        return rsf
        
    except Exception as e:
        logger.exception(f"[FATAL] RSF training failed: {e}")
        return None


def compute_pof_from_rsf(
    rsf_model,
    df_cox: pd.DataFrame,
    horizons_months,
    logger: logging.Logger,
) -> dict:
    """
    ENHANCED: Includes distribution analysis.
    """
    if rsf_model is None:
        logger.warning("[WARN] RSF model is None, skipping PoF computation")
        return {}

    logger.info("[STEP] Computing PoF from RSF with distribution analysis")

    work = df_cox.copy()
    cbs_ids = work["cbs_id"].copy()
    X = work.drop(columns=["duration_days", "event", "cbs_id"])

    results = {}
    dist_stats = []

    try:
        surv_fns = rsf_model.predict_survival_function(X)

        for m in horizons_months:
            days = m * 30
            try:
                pof_vals = np.array([1.0 - fn(days) for fn in surv_fns])
                series = pd.Series(pof_vals, index=cbs_ids.values)
                results[m] = series

                # Distribution analysis
                stats = analyze_pof_distribution(series, f"RSF {m}mo", logger)
                dist_stats.append(stats)
                
            except Exception as e:
                logger.error(f"    [ERROR] RSF PoF computation failed for {m} months: {e}")

        # Save distribution stats
        if dist_stats:
            dist_df = pd.DataFrame(dist_stats)
            dist_path = os.path.join(OUTPUT_DIR, "rsf_pof_distribution_stats.csv")
            dist_df.to_csv(dist_path, index=False, encoding="utf-8-sig")
            logger.info(f"[OK] RSF PoF distribution stats saved → {dist_path}")

    except Exception as e:
        logger.error(f"[ERROR] RSF prediction failed: {e}")
        return {}

    logger.info(f"[OK] RSF PoF computed for {len(results)} horizons")
    logger.info("")
    return results


# ----------------------------------------------------------------------
# ML LEAKAGE-FREE DATASET
# ----------------------------------------------------------------------
def build_leakage_free_ml_dataset(logger):
    logger.info("")
    logger.info("[ML] Building leakage-free dataset...")

    # Load DATA_END_DATE
    metadata_path = os.path.join(
        os.path.dirname(INTERMEDIATE_PATHS["fault_events_clean"]),
        "data_range_metadata.csv"
    )

    if not os.path.exists(metadata_path):
        logger.warning(f"[WARN] Metadata not found: {metadata_path}")
        logger.warning("[WARN] Falling back to ANALYSIS_DATE")
        analysis_dt = pd.to_datetime(ANALYSIS_DATE)
    else:
        metadata = pd.read_csv(metadata_path, encoding="utf-8-sig")
        DATA_END_DATE_str = metadata.loc[metadata["Parameter"] == "DATA_END_DATE", "Value"].iloc[0]
        DATA_END_DATE = pd.to_datetime(DATA_END_DATE_str)
        DATA_START_DATE_str = metadata.loc[metadata["Parameter"] == "DATA_START_DATE", "Value"].iloc[0]
        DATA_START_DATE = pd.to_datetime(DATA_START_DATE_str)

        logger.info(f"[ML] DATA_START_DATE = {DATA_START_DATE.date()}")
        logger.info(f"[ML] DATA_END_DATE   = {DATA_END_DATE.date()}")
        analysis_dt = DATA_END_DATE

    # Calculate T_ref
    ref_date = analysis_dt - timedelta(days=ML_PREDICTION_WINDOW_DAYS)
    window_end = ref_date + timedelta(days=ML_PREDICTION_WINDOW_DAYS)

    logger.info(f"[ML] Reference date (T_ref) = {ref_date.date()}")
    logger.info(f"[ML] Prediction window = {ref_date.date()} → {window_end.date()}")

    # Validation
    if os.path.exists(metadata_path):
        training_span_days = (ref_date - DATA_START_DATE).days
        training_span_years = training_span_days / 365.25
        logger.info(f"[ML] Training data span = {training_span_years:.2f} years ({training_span_days:,} days)")

        MIN_TRAIN_YEARS = 2.0
        if training_span_years < MIN_TRAIN_YEARS:
            logger.error(f"[FATAL] Insufficient training data: {training_span_years:.2f} years < {MIN_TRAIN_YEARS}")
            raise ValueError(f"Insufficient training data: {training_span_years:.2f} years")
        logger.info(f"[OK] Training data validation passed: {training_span_years:.2f} years >= {MIN_TRAIN_YEARS}")

    # Load data
    events = pd.read_csv(INTERMEDIATE_PATHS["fault_events_clean"], encoding="utf-8-sig")
    eq = pd.read_csv(INTERMEDIATE_PATHS["equipment_master"], encoding="utf-8-sig")

    events.rename(columns={"CBS_ID": "cbs_id"}, inplace=True)
    eq.rename(columns={"CBS_ID": "cbs_id"}, inplace=True)

    events["Ariza_Baslangic_Zamani"] = events["Ariza_Baslangic_Zamani"].apply(parse_date_safely)
    eq["Kurulum_Tarihi"] = eq["Kurulum_Tarihi"].apply(parse_date_safely)

    events["cbs_id"] = events["cbs_id"].astype(str).str.lower().str.strip()
    eq["cbs_id"] = eq["cbs_id"].astype(str).str.lower().str.strip()

    # Split past & future
    past = events[events["Ariza_Baslangic_Zamani"] <= ref_date].copy()
    future = events[
        (events["Ariza_Baslangic_Zamani"] > ref_date) &
        (events["Ariza_Baslangic_Zamani"] <= window_end)
    ].copy()

    logger.info(f"[ML] Past faults: {len(past):,}")
    logger.info(f"[ML] Future faults (window): {len(future):,}")

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
        days_since_last = (ref_date - last).days if last is not None else np.nan

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

    # Base equipment
    base = eq.copy()
    base = base[base["Kurulum_Tarihi"] <= ref_date].copy()
    base["Ekipman_Yasi_Gun_ML"] = (ref_date - base["Kurulum_Tarihi"]).dt.days.clip(lower=0)

    if "Gerilim_Seviyesi" not in eq.columns:
        base["Gerilim_Seviyesi"] = "UNKNOWN"
    if "Marka" not in eq.columns:
        base["Marka"] = "UNKNOWN"

    # Merge
    df = (
        base[["cbs_id", "Ekipman_Tipi", "Gerilim_Seviyesi", "Marka", "Ekipman_Yasi_Gun_ML"]]
        .merge(past_feat, on="cbs_id", how="left")
        .merge(target, on="cbs_id", how="left")
    )

    df["Label_Ariza_Pencere"] = df["Label_Ariza_Pencere"].fillna(0).astype(int)
    df["Ariza_Sayisi_Gecmis"] = df["Ariza_Sayisi_Gecmis"].fillna(0).astype(int)
    df["MTBF_Gun_Gecmis"] = df["MTBF_Gun_Gecmis"].fillna(df["MTBF_Gun_Gecmis"].median())
    df["Son_Ariza_Gun_Sayisi_Gecmis"] = df["Son_Ariza_Gun_Sayisi_Gecmis"].fillna(df["Ekipman_Yasi_Gun_ML"])

    logger.info(f"[ML] ML dataset: {len(df):,} equipment")
    logger.info(f"[ML] Positive labels: {df['Label_Ariza_Pencere'].sum():,}")
    logger.info("")

    return df


# ----------------------------------------------------------------------
# ML MODEL TRAINING (ENHANCED WITH SHAP)
# ----------------------------------------------------------------------
def train_leakage_free_ml_models(df, logger):
    logger.info("[STEP] Training leakage-free ML models (XGBoost + CatBoost)")

    y = df["Label_Ariza_Pencere"]

    numeric_cols = [
        "Ekipman_Yasi_Gun_ML",
        "Ariza_Sayisi_Gecmis",
        "MTBF_Gun_Gecmis",
        "Son_Ariza_Gun_Sayisi_Gecmis",
    ]
    cat_cols = ["Ekipman_Tipi", "Gerilim_Seviyesi", "Marka"]

    X_num = df[numeric_cols].copy()
    X_cat = df[cat_cols].astype(str)

    # XGBoost: one-hot
    X_xgb = pd.concat([X_num, pd.get_dummies(X_cat, drop_first=True)], axis=1)

    # CatBoost: categorical
    X_catb = pd.concat([X_num, X_cat], axis=1)
    cat_idx = [X_catb.columns.get_loc(c) for c in cat_cols]

    X_train_xgb, X_test_xgb, y_train, y_test = train_test_split(
        X_xgb, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )

    X_train_cat, X_test_cat, _, _ = train_test_split(
        X_catb, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )

    # XGBoost
    logger.info("[ML] Training XGBoost...")
    model_xgb = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
    )
    model_xgb.fit(X_train_xgb, y_train)

    proba_xgb = model_xgb.predict_proba(X_test_xgb)[:, 1]
    auc_xgb = roc_auc_score(y_test, proba_xgb)
    ap_xgb = average_precision_score(y_test, proba_xgb)

    logger.info(f"[ML] XGBoost AUC={auc_xgb:.3f}, AP={ap_xgb:.3f}")

    # CatBoost (if available)
    if HAS_CATBOOST:
        logger.info("[ML] Training CatBoost...")
        model_cat = CatBoostClassifier(
            iterations=400,
            depth=4,
            learning_rate=0.05,
            eval_metric="AUC",
            loss_function="Logloss",
            random_state=RANDOM_STATE,
            verbose=False,
        )
        model_cat.fit(
            X_train_cat,
            y_train,
            cat_features=cat_idx,
            eval_set=(X_test_cat, y_test),
            verbose=False,
        )

        proba_cat = model_cat.predict_proba(X_test_cat)[:, 1]
        auc_cat = roc_auc_score(y_test, proba_cat)
        ap_cat = average_precision_score(y_test, proba_cat)

        logger.info(f"[ML] CatBoost AUC={auc_cat:.3f}, AP={ap_cat:.3f}")
    else:
        logger.warning("[ML] CatBoost not available, skipping. Install: pip install catboost")
        model_cat = None

    # SHAP importance (if available)
    logger.info("")
    logger.info("[ADVANCED] Computing SHAP feature importance...")
    if HAS_SHAP:
        try:
            # Sample for SHAP (1000 rows max for speed)
            sample_size = min(1000, len(X_xgb))
            X_sample = X_xgb.sample(n=sample_size, random_state=RANDOM_STATE)
            
            explainer = shap.TreeExplainer(model_xgb)
            shap_values = explainer.shap_values(X_sample)
            
            # Mean absolute SHAP values
            shap_importance = pd.DataFrame({
                "feature": X_xgb.columns,
                "shap_importance": np.abs(shap_values).mean(axis=0)
            }).sort_values("shap_importance", ascending=False)
            
            logger.info("[SHAP] Top 10 features:")
            for _, row in shap_importance.head(10).iterrows():
                logger.info(f"  {row['feature']:40s}: {row['shap_importance']:.4f}")
            
            shap_path = os.path.join(OUTPUT_DIR, "shap_feature_importance.csv")
            shap_importance.to_csv(shap_path, index=False, encoding="utf-8-sig")
            logger.info(f"[SHAP] Importance saved → {shap_path}")
            
        except Exception as e:
            logger.warning(f"[WARN] SHAP computation failed: {e}")
    else:
        logger.warning("[WARN] SHAP not installed. Skipping. Install: pip install shap")

    # Temporal CV
    logger.info("")
    logger.info("[ADVANCED] Running temporal cross-validation...")
    try:
        from utils.ml_advanced import temporal_cross_validation

        cv_results = temporal_cross_validation(
            X=X_xgb,
            y=y,
            model_fn=lambda: xgb.XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE, eval_metric="logloss"
            ),
            n_splits=3,
            logger=logger
        )

        cv_df = pd.DataFrame({
            "metric": ["AUC"] * 3 + ["AP"] * 3,
            "fold": [1, 2, 3, 1, 2, 3],
            "score": cv_results["auc_scores"] + cv_results["ap_scores"]
        })
        cv_path = os.path.join(OUTPUT_DIR, "temporal_cv_scores.csv")
        cv_df.to_csv(cv_path, index=False, encoding="utf-8-sig")
        logger.info(f"[ADVANCED] Temporal CV saved → {cv_path}")
    except Exception as e:
        logger.warning(f"[WARN] Temporal CV failed: {e}")

    # Save full-model outputs
    out_data = {
        "cbs_id": df["cbs_id"],
        "PoF_ML_XGB": model_xgb.predict_proba(X_xgb)[:, 1],
    }
    
    if model_cat is not None:
        out_data["PoF_ML_CatBoost"] = model_cat.predict_proba(X_catb)[:, 1]
    
    out_df = pd.DataFrame(out_data)
    
    # PoF distribution analysis
    logger.info("")
    logger.info("[ML] PoF distribution analysis:")
    analyze_pof_distribution(out_df["PoF_ML_XGB"], "ML XGBoost", logger)
    if "PoF_ML_CatBoost" in out_df.columns:
        analyze_pof_distribution(out_df["PoF_ML_CatBoost"], "ML CatBoost", logger)
    
    out_path = os.path.join(OUTPUT_DIR, "leakage_free_ml_pof.csv")
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info(f"[ML] ML PoF scores saved → {out_path}")

    # Save models
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_xgb.save_model(os.path.join(MODELS_DIR, "leakage_free_pof_xgb.json"))
    if model_cat is not None:
        model_cat.save_model(os.path.join(MODELS_DIR, "leakage_free_pof_catboost.cbm"))
        logger.info(f"[ML] Both models saved → {MODELS_DIR}")
    else:
        logger.info(f"[ML] XGBoost model saved → {MODELS_DIR}")
    logger.info("")


# ----------------------------------------------------------------------
# MODEL COMPARISON REPORT
# ----------------------------------------------------------------------
def generate_model_comparison_report(
    cox_pof: dict,
    rsf_pof: dict,
    ml_pof_path: str,
    logger: logging.Logger
) -> None:
    """
    Generate comprehensive model comparison across all approaches.
    Handles missing CatBoost gracefully.
    """
    logger.info("=" * 80)
    logger.info("[MODEL COMPARISON] Generating cross-model comparison report...")
    logger.info("=" * 80)
    
    # Load ML predictions
    ml_df = pd.read_csv(ml_pof_path, encoding="utf-8-sig")
    
    # Check which ML models are available
    has_catboost_col = "PoF_ML_CatBoost" in ml_df.columns
    
    # Build comparison dataframe
    comparison_data = []
    
    for horizon in sorted(cox_pof.keys()):
        if horizon not in rsf_pof:
            continue
            
        cox_series = cox_pof[horizon]
        rsf_series = rsf_pof[horizon]
        
        # Align by cbs_id
        common_ids = set(cox_series.index) & set(rsf_series.index) & set(ml_df["cbs_id"])
        common_ids = list(common_ids)
        
        if not common_ids:
            logger.warning(f"[WARN] No common equipment IDs for {horizon} month horizon")
            continue
        
        cox_aligned = cox_series.loc[common_ids]
        rsf_aligned = rsf_series.loc[common_ids]
        ml_xgb_aligned = ml_df.set_index("cbs_id").loc[common_ids, "PoF_ML_XGB"]
        
        # Calculate correlations
        cox_rsf_corr = cox_aligned.corr(rsf_aligned)
        cox_ml_corr = cox_aligned.corr(ml_xgb_aligned)
        rsf_ml_corr = rsf_aligned.corr(ml_xgb_aligned)
        
        # Agreement metrics (high-risk equipment)
        cox_high_risk = set(cox_aligned[cox_aligned > 0.3].index)
        rsf_high_risk = set(rsf_aligned[rsf_aligned > 0.3].index)
        ml_high_risk = set(ml_xgb_aligned[ml_xgb_aligned > 0.3].index)
        
        cox_rsf_agreement = len(cox_high_risk & rsf_high_risk) / len(cox_high_risk | rsf_high_risk) if len(cox_high_risk | rsf_high_risk) > 0 else 0
        cox_ml_agreement = len(cox_high_risk & ml_high_risk) / len(cox_high_risk | ml_high_risk) if len(cox_high_risk | ml_high_risk) > 0 else 0
        rsf_ml_agreement = len(rsf_high_risk & ml_high_risk) / len(rsf_high_risk | ml_high_risk) if len(rsf_high_risk | ml_high_risk) > 0 else 0
        
        row_data = {
            "Horizon_Months": horizon,
            "N_Equipment": len(common_ids),
            "Cox_Mean_PoF": cox_aligned.mean(),
            "RSF_Mean_PoF": rsf_aligned.mean(),
            "ML_XGB_Mean_PoF": ml_xgb_aligned.mean(),
            "Cox_RSF_Correlation": cox_rsf_corr,
            "Cox_ML_Correlation": cox_ml_corr,
            "RSF_ML_Correlation": rsf_ml_corr,
            "Cox_RSF_HighRisk_Agreement": cox_rsf_agreement,
            "Cox_ML_HighRisk_Agreement": cox_ml_agreement,
            "RSF_ML_HighRisk_Agreement": rsf_ml_agreement,
            "Cox_HighRisk_Count": len(cox_high_risk),
            "RSF_HighRisk_Count": len(rsf_high_risk),
            "ML_HighRisk_Count": len(ml_high_risk),
        }
        
        # Add CatBoost metrics if available
        if has_catboost_col:
            ml_cat_aligned = ml_df.set_index("cbs_id").loc[common_ids, "PoF_ML_CatBoost"]
            row_data["ML_Cat_Mean_PoF"] = ml_cat_aligned.mean()
        
        comparison_data.append(row_data)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Log summary
    logger.info("")
    logger.info("[COMPARISON] Model correlation summary:")
    for _, row in comparison_df.iterrows():
        logger.info(f"  {int(row['Horizon_Months'])} months:")
        logger.info(f"    Cox-RSF correlation: {row['Cox_RSF_Correlation']:.3f}")
        logger.info(f"    Cox-ML correlation:  {row['Cox_ML_Correlation']:.3f}")
        logger.info(f"    High-risk agreement (Cox-RSF): {row['Cox_RSF_HighRisk_Agreement']:.2%}")
        logger.info(f"    High-risk agreement (Cox-ML):  {row['Cox_ML_HighRisk_Agreement']:.2%}")
    
    # Save report
    comparison_path = os.path.join(OUTPUT_DIR, "model_comparison_report.csv")
    comparison_df.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    logger.info(f"[COMPARISON] Report saved → {comparison_path}")
    logger.info("=" * 80)
    logger.info("")


# ----------------------------------------------------------------------
# MAIN WORKFLOW
# ----------------------------------------------------------------------
def main():
    logger = setup_logger(STEP_NAME)

    try:
        # Temporal stability analysis
        analyze_temporal_stability(logger)

        # Load data
        logger.info("[STEP] Loading survival_base and ozellikler_pof3")

        surv_path = INTERMEDIATE_PATHS.get("survival_base")
        if surv_path is None:
            raise KeyError("INTERMEDIATE_PATHS['survival_base'] not defined")

        if not os.path.exists(surv_path):
            raise FileNotFoundError(f"Survival base not found: {surv_path}")

        if FEATURE_OUTPUT_PATH is None or not os.path.exists(FEATURE_OUTPUT_PATH):
            raise FileNotFoundError(f"Feature file not found: {FEATURE_OUTPUT_PATH}")

        survival_base = pd.read_csv(surv_path, encoding="utf-8-sig")
        features = pd.read_csv(FEATURE_OUTPUT_PATH, encoding="utf-8-sig")

        survival_base, features = normalize_columns(survival_base, features, logger)

        logger.info(f"[OK] survival_base loaded: {len(survival_base):,} rows")
        logger.info(f"[OK] ozellikler_pof3 loaded: {len(features):,} rows")

        # Merge survival + features
        surv_cols = ["cbs_id", "event", "duration_days"]
        df_surv = survival_base[surv_cols].copy()

        merge_cols = [
            "cbs_id", "Ekipman_Tipi", "Ekipman_Yasi_Gun", "MTBF_Gun", "TFF_Gun",
            "Ariza_Sayisi", "Ariza_Gecmisi", "Faults_Last_365d",
            "Kronik_Kritik", "Kronik_Yuksek", "Kronik_Orta",
            "Son_Ariza_Gun_Sayisi", "Bakim_Sayisi", "Bakim_Var_Mi",
            "Ilk_Bakim_Tarihi", "Son_Bakim_Tarihi", "Son_Bakimdan_Gecen_Gun",
            "Son_Bakim_Tipi", "Bakim_Is_Emri_Tipleri",
            "Gerilim_Seviyesi", "kVA_Rating", "Marka",
        ]
        cols_in_features = [c for c in merge_cols if c in features.columns]
        feat_sub = features[cols_in_features].copy()

        df_full = df_surv.merge(feat_sub, on="cbs_id", how="left")
        logger.info(f"[OK] Merged dataset: {len(df_full):,} rows")
        logger.info("")

        # Prepare Cox data
        df_cox = prepare_cox_data(df_full, logger)

        # Cox model
        cph = fit_cox_model(df_cox, logger)
        pof_cox = compute_pof_from_cox(cph, df_cox, SURVIVAL_HORIZONS_MONTHS, logger)

        # Save Cox outputs
        logger.info("[STEP] Saving Cox PoF outputs")
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        for m, series in pof_cox.items():
            out_df = pd.DataFrame({
                "cbs_id": series.index,
                f"PoF_Cox_{m}Ay": series.values,
            })
            out_path = os.path.join(OUTPUT_DIR, f"cox_sagkalim_{m}ay_ariza_olasiligi.csv")
            out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
            logger.info(f"[OK] {out_path}")

        # RSF model
        logger.info("")
        logger.info("[STEP] Random Survival Forest (RSF)")
        rsf = fit_rsf_model(df_cox, logger)
        rsf_pof = compute_pof_from_rsf(rsf, df_cox, SURVIVAL_HORIZONS_MONTHS, logger)

        if rsf_pof:
            logger.info("[STEP] Saving RSF PoF outputs")
            for m, series in rsf_pof.items():
                out_df = pd.DataFrame({
                    "cbs_id": series.index,
                    f"PoF_RSF_{m}Ay": series.values,
                })
                out_path = os.path.join(OUTPUT_DIR, f"rsf_sagkalim_{m}ay_ariza_olasiligi.csv")
                out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
                logger.info(f"[OK] {out_path}")

        # ML models
        logger.info("")
        logger.info("[STEP] Leakage-free ML PoF")

        df_ml = build_leakage_free_ml_dataset(logger)
        train_leakage_free_ml_models(df_ml, logger)

        # Model comparison report
        ml_pof_path = os.path.join(OUTPUT_DIR, "leakage_free_ml_pof.csv")
        if os.path.exists(ml_pof_path) and rsf_pof:
            generate_model_comparison_report(pof_cox, rsf_pof, ml_pof_path, logger)

        # Visualizations
        logger.info("")
        logger.info("[STEP] Generating visualizations...")

        try:
            from utils.survival_plotting import (
                plot_survival_curves_by_class,
                plot_cox_coefficients,
                plot_feature_importance_comparison
            )
            from config.config import VISUAL_DIR

            surv_plot_path = os.path.join(VISUAL_DIR, "survival_curves_by_class.png")
            plot_survival_curves_by_class(df=df_full, output_path=surv_plot_path, logger=logger)

            if cph is not None:
                cox_plot_path = os.path.join(VISUAL_DIR, "cox_coefficients.png")
                plot_cox_coefficients(cox_model=cph, output_path=cox_plot_path, logger=logger)

            rsf_imp_path = os.path.join(OUTPUT_DIR, "rsf_feature_importance.csv")
            shap_imp_path = os.path.join(OUTPUT_DIR, "shap_feature_importance.csv")
            if os.path.exists(rsf_imp_path) and os.path.exists(shap_imp_path):
                rsf_imp = pd.read_csv(rsf_imp_path, encoding="utf-8-sig")
                shap_imp = pd.read_csv(shap_imp_path, encoding="utf-8-sig")
                comp_plot_path = os.path.join(VISUAL_DIR, "feature_importance_comparison.png")
                plot_feature_importance_comparison(
                    rsf_importance=rsf_imp,
                    shap_importance=shap_imp,
                    output_path=comp_plot_path,
                    logger=logger
                )

        except Exception as e:
            logger.warning(f"[WARN] Visualization generation failed: {e}")

        logger.info("")
        logger.info("[SUCCESS] 03_sagkalim_modelleri completed successfully!")
        logger.info("=" * 80)

    except Exception as e:
        logger.exception(f"[FATAL] 03_sagkalim_modelleri failed: {e}")
        raise


if __name__ == "__main__":
    main()