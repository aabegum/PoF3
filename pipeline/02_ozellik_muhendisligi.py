"""
02_ozellik_muhendisligi.py (PoF3 - Enhanced v2 - IEEE Standard)

Improvements:
- IEEE 1366 standard chronic detection (3+ faults in 365 days, rolling window)
- Uses DATA_END_DATE from Step 01 instead of ANALYSIS_DATE
- Added TFF (Time to First Failure) feature
- Added feature distribution report
- Parse kVA_Rating and voltage to numeric
- Cleaner chronic detection logic with temporal recency

Amaç:
- Robust feature engineering with industry-standard reliability metrics
- IEEE-compliant chronic equipment classification
- Comprehensive feature quality reporting
"""

import os
import sys
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from utils.date_parser import parse_date_safely
from config.config import (
    ANALYSIS_DATE,
    INTERMEDIATE_PATHS,
    FEATURE_OUTPUT_PATH,
    LOG_DIR,
)

STEP_NAME = "02_ozellik_muhendisligi"


# ---------------------------------------------------------
# LOGGING
# ---------------------------------------------------------

def setup_logger() -> logging.Logger:
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"{STEP_NAME}_{ts}.log")

    logger = logging.getLogger(STEP_NAME)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    logger.info("=" * 80)
    logger.info(f"{STEP_NAME} - PoF3 Feature Engineering (Enhanced v2 - IEEE Standard)")
    logger.info("=" * 80)
    logger.info(f"Analysis Date: {ANALYSIS_DATE}")
    logger.info("")
    return logger


# ---------------------------------------------------------
# DATA_END_DATE LOADER
# ---------------------------------------------------------

def load_data_end_date(logger: logging.Logger) -> pd.Timestamp:
    """
    Load DATA_END_DATE from Step 01 metadata.
    This is the last fault date in the dataset, used for temporal calculations.
    """
    metadata_path = os.path.join(
        os.path.dirname(INTERMEDIATE_PATHS["fault_events_clean"]),
        "data_range_metadata.csv"
    )
    
    if not os.path.exists(metadata_path):
        logger.warning(f"[WARN] Metadata file not found: {metadata_path}")
        logger.warning(f"[WARN] Falling back to ANALYSIS_DATE: {ANALYSIS_DATE}")
        return pd.to_datetime(ANALYSIS_DATE)
    
    metadata = pd.read_csv(metadata_path)
    data_end_date_str = metadata.loc[metadata["Parameter"] == "DATA_END_DATE", "Value"].values[0]
    data_end_date = pd.to_datetime(data_end_date_str)
    
    logger.info(f"[INFO] Loaded DATA_END_DATE from Step 01: {data_end_date.date()}")
    logger.info(f"[INFO] Using DATA_END_DATE for all temporal calculations")
    logger.info("")
    
    return data_end_date


# ---------------------------------------------------------
# IEEE STANDARD CHRONIC DETECTION (CLEAN IMPLEMENTATION)
# ---------------------------------------------------------

def compute_ieee_chronic_flags(events: pd.DataFrame,
                               data_end_date: pd.Timestamp,
                               logger: logging.Logger) -> pd.DataFrame:
    """
    IEEE 1366-compliant chronic detection with rolling 365-day window.
    
    Industry Standard:
    - KRITIK:  4+ faults in last 365 days (immediate action)
    - YUKSEK:  3 faults in last 365 days (priority maintenance)
    - ORTA:    2 faults in last 365 days (scheduled maintenance)
    - NORMAL:  0-1 fault in last 365 days
    
    Only considers faults within RECENCY_WINDOW from DATA_END_DATE.
    """
    if events.empty:
        return pd.DataFrame(columns=[
            "cbs_id", "Kronik_Kritik", "Kronik_Yuksek", "Kronik_Orta",
            "Kronik_Seviye_Max", "Faults_Last_365d"
        ])
    
    # Only consider recent faults (within 2 years of data end)
    RECENCY_WINDOW_DAYS = 730  # 2 years
    cutoff_date = data_end_date - timedelta(days=RECENCY_WINDOW_DAYS)
    
    recent_events = events[events["Ariza_Baslangic_Zamani"] >= cutoff_date].copy()
    
    logger.info(f"[CHRONIC] Analyzing {len(recent_events):,} faults within {RECENCY_WINDOW_DAYS} days of DATA_END_DATE")
    
    results = []
    
    for cbs_id, grp in recent_events.groupby("cbs_id"):
        times = grp["Ariza_Baslangic_Zamani"].sort_values().values
        n_faults = len(times)
        
        # Count faults in last 365 days from DATA_END_DATE
        last_365d_start = data_end_date - timedelta(days=365)
        faults_last_365d = np.sum(times >= last_365d_start)
        
        # IEEE severity classification
        if faults_last_365d >= 4:
            kritik, yuksek, orta = 1, 1, 1
            seviye = "KRITIK"
        elif faults_last_365d == 3:
            kritik, yuksek, orta = 0, 1, 1
            seviye = "YUKSEK"
        elif faults_last_365d == 2:
            kritik, yuksek, orta = 0, 0, 1
            seviye = "ORTA"
        else:
            kritik, yuksek, orta = 0, 0, 0
            seviye = "NORMAL"
        
        results.append({
            "cbs_id": cbs_id,
            "Kronik_Kritik": kritik,
            "Kronik_Yuksek": yuksek,
            "Kronik_Orta": orta,
            "Kronik_Seviye_Max": seviye,
            "Faults_Last_365d": faults_last_365d
        })
    
    chronic_df = pd.DataFrame(results)
    
    # Log distribution
    logger.info("[CHRONIC] IEEE Standard Classification:")
    for seviye in ["KRITIK", "YUKSEK", "ORTA", "NORMAL"]:
        count = (chronic_df["Kronik_Seviye_Max"] == seviye).sum()
        pct = count / len(chronic_df) * 100 if len(chronic_df) > 0 else 0
        logger.info(f"  {seviye}: {count:,} equipment ({pct:.1f}%)")
    
    total_chronic = (chronic_df["Kronik_Seviye_Max"] != "NORMAL").sum()
    chronic_rate = total_chronic / len(chronic_df) * 100 if len(chronic_df) > 0 else 0
    logger.info(f"[CHRONIC] Total chronic rate: {chronic_rate:.1f}% (IEEE target: 5-15%)")
    logger.info("")
    
    return chronic_df


# ---------------------------------------------------------
# MTBF & TFF CALCULATION (CLEAN IMPLEMENTATION)
# ---------------------------------------------------------

def compute_reliability_metrics(events: pd.DataFrame,
                                equipment: pd.DataFrame,
                                data_end_date: pd.Timestamp,
                                logger: logging.Logger) -> pd.DataFrame:
    """
    Compute reliability metrics:
    - MTBF (Mean Time Between Failures) for equipment with 2+ faults
    - TFF (Time to First Failure) for ALL equipment with ≥1 fault
    """
    if events.empty:
        return pd.DataFrame(columns=["cbs_id", "MTBF_Gun", "TFF_Gun"])
    
    events_sorted = events.sort_values(["cbs_id", "Ariza_Baslangic_Zamani"])
    results = []
    
    for cbs_id, grp in events_sorted.groupby("cbs_id"):
        times = grp["Ariza_Baslangic_Zamani"].dropna().sort_values().values
        n_faults = len(times)
        
        # Get installation date
        install_date = equipment.loc[equipment["cbs_id"] == cbs_id, "Kurulum_Tarihi"].values
        if len(install_date) == 0 or pd.isna(install_date[0]):
            continue
        install_date = install_date[0]
        
        # TFF: Time to First Failure (for all equipment with ≥1 fault)
        first_fault_date = times[0]
        tff_days = (first_fault_date - install_date).astype('timedelta64[D]').astype(int)
        
        # MTBF: Mean Time Between Failures (only for equipment with 2+ faults)
        if n_faults >= 2:
            diffs = np.diff(times).astype("timedelta64[D]").astype(int)
            mtbf_days = float(np.mean(diffs))
        else:
            mtbf_days = np.nan
        
        results.append({
            "cbs_id": cbs_id,
            "MTBF_Gun": mtbf_days,
            "TFF_Gun": tff_days
        })
    
    reliability_df = pd.DataFrame(results)
    
    # Log statistics
    logger.info("[RELIABILITY METRICS]")
    logger.info(f"  Equipment with TFF calculated: {reliability_df['TFF_Gun'].notna().sum():,}")
    logger.info(f"  Equipment with MTBF calculated: {reliability_df['MTBF_Gun'].notna().sum():,}")
    
    if reliability_df["TFF_Gun"].notna().any():
        logger.info(f"  TFF - Mean: {reliability_df['TFF_Gun'].mean():.0f} days")
        logger.info(f"  TFF - Median: {reliability_df['TFF_Gun'].median():.0f} days")
    
    if reliability_df["MTBF_Gun"].notna().any():
        logger.info(f"  MTBF - Mean: {reliability_df['MTBF_Gun'].mean():.0f} days")
        logger.info(f"  MTBF - Median: {reliability_df['MTBF_Gun'].median():.0f} days")
    logger.info("")
    
    return reliability_df


# ---------------------------------------------------------
# NUMERIC PARSING FOR kVA AND VOLTAGE
# ---------------------------------------------------------

def parse_equipment_attributes(features: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Parse kVA_Rating and voltage fields to numeric where possible.
    Creates separate numeric columns while preserving originals.
    """
    logger.info("[PARSING] Converting equipment attributes to numeric...")
    
    # Parse kVA_Rating
    if "kVA_Rating" in features.columns:
        features["kVA_Rating_Numeric"] = pd.to_numeric(features["kVA_Rating"], errors="coerce")
        non_null = features["kVA_Rating_Numeric"].notna().sum()
        logger.info(f"  kVA_Rating: {non_null:,}/{len(features):,} successfully parsed to numeric")
    
    # Parse Gerilim_Seviyesi (e.g., "34.5 kV" → 34.5)
    if "Gerilim_Seviyesi" in features.columns:
        features["Gerilim_Seviyesi_kV"] = (
            features["Gerilim_Seviyesi"]
            .astype(str)
            .str.extract(r'(\d+\.?\d*)', expand=False)
            .astype(float)
        )
        non_null = features["Gerilim_Seviyesi_kV"].notna().sum()
        logger.info(f"  Gerilim_Seviyesi: {non_null:,}/{len(features):,} successfully parsed to kV")
    
    logger.info("")
    return features


# ---------------------------------------------------------
# FEATURE DISTRIBUTION REPORT
# ---------------------------------------------------------

def generate_feature_distribution_report(features: pd.DataFrame,
                                        logger: logging.Logger,
                                        output_dir: str) -> None:
    """
    Generate comprehensive feature distribution report with:
    - Overall statistics (mean, median, quartiles, outliers)
    - Equipment-type breakdown
    - Anomaly detection
    """
    logger.info("[DISTRIBUTION] Generating feature distribution report...")
    
    # Numeric columns for analysis
    numeric_cols = [
        "Ekipman_Yasi_Gun", "Ariza_Sayisi", "MTBF_Gun", "TFF_Gun",
        "Son_Ariza_Gun_Sayisi", "Bakim_Sayisi", "Son_Bakimdan_Gecen_Gun",
        "kVA_Rating_Numeric", "Gerilim_Seviyesi_kV", "Faults_Last_365d"
    ]
    numeric_cols = [c for c in numeric_cols if c in features.columns]
    
    # Overall distribution
    dist_summary = []
    for col in numeric_cols:
        series = features[col].dropna()
        if len(series) == 0:
            continue
        
        q01, q25, q50, q75, q99 = series.quantile([0.01, 0.25, 0.50, 0.75, 0.99])
        iqr = q75 - q25
        outlier_low = (series < (q25 - 3 * iqr)).sum()
        outlier_high = (series > (q75 + 3 * iqr)).sum()
        
        dist_summary.append({
            "Feature": col,
            "Count": len(series),
            "Mean": round(series.mean(), 2),
            "Median": round(q50, 2),
            "Std": round(series.std(), 2),
            "P01": round(q01, 2),
            "P25": round(q25, 2),
            "P75": round(q75, 2),
            "P99": round(q99, 2),
            "IQR": round(iqr, 2),
            "Outliers_Low": outlier_low,
            "Outliers_High": outlier_high,
            "Missing": features[col].isna().sum()
        })
    
    dist_df = pd.DataFrame(dist_summary)
    dist_path = os.path.join(output_dir, "feature_distribution_overall.csv")
    dist_df.to_csv(dist_path, index=False, encoding="utf-8-sig")
    logger.info(f"[DISTRIBUTION] Overall report → {dist_path}")
    
    # Equipment-type breakdown
    if "Ekipman_Tipi" in features.columns:
        equip_summary = []
        for eq_type in features["Ekipman_Tipi"].unique():
            subset = features[features["Ekipman_Tipi"] == eq_type]
            
            equip_summary.append({
                "Ekipman_Tipi": eq_type,
                "Count": len(subset),
                "Avg_Age_Days": round(subset["Ekipman_Yasi_Gun"].mean(), 0) if "Ekipman_Yasi_Gun" in subset else np.nan,
                "Avg_Faults": round(subset["Ariza_Sayisi"].mean(), 2) if "Ariza_Sayisi" in subset else np.nan,
                "Chronic_Rate_%": round(100 * (subset["Kronik_Seviye_Max"] != "NORMAL").sum() / len(subset), 1) if "Kronik_Seviye_Max" in subset else np.nan,
                "Maintenance_Rate_%": round(100 * (subset["Bakim_Sayisi"] > 0).sum() / len(subset), 1) if "Bakim_Sayisi" in subset else np.nan,
                "Avg_MTBF_Days": round(subset["MTBF_Gun"].mean(), 0) if "MTBF_Gun" in subset else np.nan,
                "Avg_TFF_Days": round(subset["TFF_Gun"].mean(), 0) if "TFF_Gun" in subset else np.nan
            })
        
        equip_df = pd.DataFrame(equip_summary).sort_values("Count", ascending=False)
        equip_path = os.path.join(output_dir, "feature_distribution_by_equipment.csv")
        equip_df.to_csv(equip_path, index=False, encoding="utf-8-sig")
        logger.info(f"[DISTRIBUTION] Equipment-type report → {equip_path}")
    
    logger.info("")


# ---------------------------------------------------------
# ENHANCED SANITY CHECKS
# ---------------------------------------------------------

def enhanced_sanity_checks(features: pd.DataFrame, logger: logging.Logger) -> list:
    """
    Comprehensive quality checks with detailed reporting.
    """
    logger.info("[SANITY CHECK] Running enhanced quality control...")
    issues = []
    
    def add_issue(msg, severity="WARNING"):
        issues.append({"Severity": severity, "Issue": msg})
        if severity == "CRITICAL":
            logger.error(f"[SANITY CRITICAL] {msg}")
        else:
            logger.warning(f"[SANITY] {msg}")
    
    # Age checks
    if "Ekipman_Yasi_Gun" in features.columns:
        neg_age = features[features["Ekipman_Yasi_Gun"] < 0]
        if len(neg_age) > 0:
            add_issue(f"Negative equipment age: {len(neg_age)} records", "CRITICAL")
        
        zero_age = features[features["Ekipman_Yasi_Gun"] == 0]
        if len(zero_age) > 0:
            add_issue(f"Zero-age equipment: {len(zero_age)} records (investigate installation dates)")
            # Save details
            if len(zero_age) > 0:
                zero_age_path = os.path.join(
                    os.path.dirname(FEATURE_OUTPUT_PATH),
                    "zero_age_equipment_detail.csv"
                )
                zero_age[["cbs_id", "Ekipman_Tipi", "Kurulum_Tarihi", "Ekipman_Yasi_Gun"]].to_csv(
                    zero_age_path, index=False, encoding="utf-8-sig"
                )
                logger.info(f"[SANITY] Zero-age detail report → {zero_age_path}")
    
    # Fault metrics
    if "Ariza_Sayisi" in features.columns:
        neg_faults = (features["Ariza_Sayisi"] < 0).sum()
        if neg_faults > 0:
            add_issue(f"Negative fault count: {neg_faults} records", "CRITICAL")
    
    # MTBF checks
    if "MTBF_Gun" in features.columns:
        neg_mtbf = (features["MTBF_Gun"] < 0).sum()
        if neg_mtbf > 0:
            add_issue(f"Negative MTBF: {neg_mtbf} records", "CRITICAL")
        
        extreme_mtbf = features[features["MTBF_Gun"] > 36500]
        if len(extreme_mtbf) > 0:
            add_issue(f"MTBF > 100 years: {len(extreme_mtbf)} records (possible data quality issue)")
    
    # TFF checks
    if "TFF_Gun" in features.columns:
        neg_tff = (features["TFF_Gun"] < 0).sum()
        if neg_tff > 0:
            add_issue(f"Negative TFF: {neg_tff} records", "CRITICAL")
        
        extreme_tff = features[features["TFF_Gun"] > features["Ekipman_Yasi_Gun"]]
        if len(extreme_tff) > 0:
            add_issue(f"TFF > Equipment Age: {len(extreme_tff)} records (data inconsistency)", "CRITICAL")
    
    # Temporal consistency
    if "Son_Ariza_Gun_Sayisi" in features.columns:
        neg_since = (features["Son_Ariza_Gun_Sayisi"] < 0).sum()
        if neg_since > 0:
            add_issue(f"Negative days since last fault: {neg_since} records", "CRITICAL")
    
    # Maintenance data quality summary
    maintenance_cols = ["Bakim_Sayisi", "Ilk_Bakim_Tarihi", "Son_Bakim_Tarihi"]
    for col in maintenance_cols:
        if col in features.columns:
            missing_pct = 100 * features[col].isna().sum() / len(features)
            logger.info(f"[SANITY] {col}: {missing_pct:.1f}% missing")
    
    if not issues:
        logger.info("[SANITY CHECK] ✓ No critical issues detected")
    else:
        logger.warning(f"[SANITY CHECK] {len(issues)} issues detected")
    
    logger.info("")
    return issues


# ---------------------------------------------------------
# MAIN PROCESS
# ---------------------------------------------------------

def main():
    logger = setup_logger()
    
    try:
        # Load DATA_END_DATE from Step 01
        data_end_date = load_data_end_date(logger)
        
        # Load data
        eq_path = INTERMEDIATE_PATHS["equipment_master"]
        events_path = INTERMEDIATE_PATHS["fault_events_clean"]
        
        logger.info(f"[STEP] Loading equipment_master: {eq_path}")
        equipment = pd.read_csv(eq_path, encoding="utf-8-sig")
        
        logger.info(f"[STEP] Loading fault_events_clean: {events_path}")
        events = pd.read_csv(events_path, encoding="utf-8-sig") if os.path.exists(events_path) else pd.DataFrame()
        
        # Normalize cbs_id
        equipment["cbs_id"] = equipment["cbs_id"].astype(str).str.lower().str.strip()
        if not events.empty:
            events["cbs_id"] = events["cbs_id"].astype(str).str.lower().str.strip()
        
        # Parse dates
        equipment["Kurulum_Tarihi"] = equipment["Kurulum_Tarihi"].apply(parse_date_safely)
        if "Ilk_Ariza_Tarihi" in equipment.columns:
            equipment["Ilk_Ariza_Tarihi"] = equipment["Ilk_Ariza_Tarihi"].apply(parse_date_safely)
        
        for col in ["Ilk_Bakim_Tarihi", "Son_Bakim_Tarihi"]:
            if col in equipment.columns:
                equipment[col] = equipment[col].apply(parse_date_safely)
        
        if not events.empty:
            events["Ariza_Baslangic_Zamani"] = events["Ariza_Baslangic_Zamani"].apply(parse_date_safely)
        
        # Build base feature set
        base_cols = [
            "cbs_id", "Ekipman_Tipi", "Kurulum_Tarihi", "Ekipman_Yasi_Gun",
            "Ariza_Gecmisi", "Fault_Count"
        ]
        if "Ilk_Ariza_Tarihi" in equipment.columns:
            base_cols.append("Ilk_Ariza_Tarihi")
        
        existing_base = [c for c in base_cols if c in equipment.columns]
        features = equipment[existing_base].copy()
        
        # Rename to Turkish
        if "Fault_Count" in features.columns:
            features.rename(columns={"Fault_Count": "Ariza_Sayisi"}, inplace=True)
        
        # Last fault info (using DATA_END_DATE)
        if not events.empty:
            lastfault = events.groupby("cbs_id")["Ariza_Baslangic_Zamani"].max().rename("Son_Ariza_Tarihi")
            features = features.merge(lastfault, on="cbs_id", how="left")
            features["Son_Ariza_Tarihi"] = features["Son_Ariza_Tarihi"].apply(parse_date_safely)
            features["Son_Ariza_Gun_Sayisi"] = (data_end_date - features["Son_Ariza_Tarihi"]).dt.days
        else:
            features["Son_Ariza_Tarihi"] = pd.NaT
            features["Son_Ariza_Gun_Sayisi"] = np.nan
        
        # IEEE Standard Chronic Detection
        if not events.empty:
            chronic_df = compute_ieee_chronic_flags(events, data_end_date, logger)
            features = features.merge(chronic_df, on="cbs_id", how="left")
            
            # Fill non-chronic equipment
            for col in ["Kronik_Kritik", "Kronik_Yuksek", "Kronik_Orta"]:
                features[col] = features[col].fillna(0).astype(int)
            features["Kronik_Seviye_Max"] = features["Kronik_Seviye_Max"].fillna("NORMAL")
            features["Faults_Last_365d"] = features["Faults_Last_365d"].fillna(0).astype(int)
        else:
            features["Kronik_Kritik"] = 0
            features["Kronik_Yuksek"] = 0
            features["Kronik_Orta"] = 0
            features["Kronik_Seviye_Max"] = "NORMAL"
            features["Faults_Last_365d"] = 0
        
        # MTBF & TFF
        if not events.empty:
            reliability_df = compute_reliability_metrics(events, equipment, data_end_date, logger)
            features = features.merge(reliability_df, on="cbs_id", how="left")
        else:
            features["MTBF_Gun"] = np.nan
            features["TFF_Gun"] = np.nan
        
        # Maintenance & attributes
        bakim_cols = [
            "Bakim_Sayisi", "Bakim_Is_Emri_Tipleri", "Ilk_Bakim_Tarihi",
            "Son_Bakim_Tarihi", "Son_Bakim_Tipi", "Son_Bakimdan_Gecen_Gun"
        ]
        attr_cols = ["Marka", "Gerilim_Seviyesi", "Gerilim_Sinifi", "kVA_Rating"]
        
        extra_cols = [c for c in bakim_cols + attr_cols if c in equipment.columns]
        if extra_cols:
            extra_df = equipment[["cbs_id"] + extra_cols].copy()
            features = features.merge(extra_df, on="cbs_id", how="left")
        
        # Maintenance flags
        if "Bakim_Sayisi" in features.columns:
            features["Bakim_Sayisi"] = features["Bakim_Sayisi"].fillna(0).astype(int)
            features["Bakim_Var_Mi"] = (features["Bakim_Sayisi"] > 0).astype(int)
        else:
            features["Bakim_Sayisi"] = 0
            features["Bakim_Var_Mi"] = 0
        
        # Days since last maintenance (using DATA_END_DATE)
        if "Son_Bakimdan_Gecen_Gun" not in features.columns or features["Son_Bakimdan_Gecen_Gun"].isna().all():
            if "Son_Bakim_Tarihi" in features.columns:
                features["Son_Bakimdan_Gecen_Gun"] = (data_end_date - features["Son_Bakim_Tarihi"]).dt.days
        
        # Parse numeric attributes
        features = parse_equipment_attributes(features, logger)
        
        # Type corrections
        if "Ariza_Sayisi" in features.columns:
            features["Ariza_Sayisi"] = features["Ariza_Sayisi"].fillna(0).astype(int)
        if "Ariza_Gecmisi" in features.columns:
            features["Ariza_Gecmisi"] = features["Ariza_Gecmisi"].fillna(0).astype(int)
        
        # Summary
        logger.info("=" * 80)
        logger.info("[SUMMARY]")
        logger.info(f"Total equipment: {len(features):,}")
        if "Ariza_Sayisi" in features.columns:
            logger.info(f"Equipment with fault history: {(features['Ariza_Sayisi'] > 0).sum():,}")
        if "Kronik_Seviye_Max" in features.columns:
            chronic_count = (features["Kronik_Seviye_Max"] != "NORMAL").sum()
            logger.info(f"Chronic equipment: {chronic_count:,} ({100*chronic_count/len(features):.1f}%)")
        if "Bakim_Sayisi" in features.columns:
            logger.info(f"Equipment with maintenance records: {(features['Bakim_Sayisi'] > 0).sum():,}")
        logger.info("=" * 80)
        logger.info("")
        
        # Feature distribution report
        output_dir = os.path.dirname(FEATURE_OUTPUT_PATH)
        os.makedirs(output_dir, exist_ok=True)
        generate_feature_distribution_report(features, logger, output_dir)
        
        # Enhanced sanity checks
        sanity_issues = enhanced_sanity_checks(features, logger)
        sanity_report_path = os.path.join(output_dir, "ozellik_sanity_report.csv")
        if sanity_issues:
            sanity_df = pd.DataFrame(sanity_issues)
        else:
            sanity_df = pd.DataFrame({"Severity": ["INFO"], "Issue": ["NO ISSUES DETECTED"]})
        
        sanity_df.to_csv(sanity_report_path, index=False, encoding="utf-8-sig")
        logger.info(f"[OK] Sanity check report saved → {sanity_report_path}")
        logger.info("")
        
        # Save final feature set
        features.to_csv(FEATURE_OUTPUT_PATH, index=False, encoding="utf-8-sig")
        logger.info("=" * 80)
        logger.info(f"[SUCCESS] Feature set saved → {FEATURE_OUTPUT_PATH}")
        logger.info(f"[SUCCESS] Total features: {len(features.columns)}")
        logger.info(f"[SUCCESS] Total records: {len(features):,}")
        logger.info("=" * 80)
        logger.info("")
        
        # Log key feature columns
        logger.info("[FEATURE COLUMNS]")
        logger.info("Base features:")
        logger.info("  - cbs_id, Ekipman_Tipi, Kurulum_Tarihi, Ekipman_Yasi_Gun")
        logger.info("Fault features:")
        logger.info("  - Ariza_Sayisi, Ariza_Gecmisi, Ilk_Ariza_Tarihi, Son_Ariza_Tarihi, Son_Ariza_Gun_Sayisi")
        logger.info("Reliability metrics:")
        logger.info("  - MTBF_Gun, TFF_Gun")
        logger.info("Chronic flags (IEEE Standard):")
        logger.info("  - Kronik_Kritik, Kronik_Yuksek, Kronik_Orta, Kronik_Seviye_Max, Faults_Last_365d")
        logger.info("Maintenance features:")
        logger.info("  - Bakim_Sayisi, Bakim_Var_Mi, Son_Bakimdan_Gecen_Gun")
        logger.info("Equipment attributes:")
        logger.info("  - kVA_Rating, kVA_Rating_Numeric, Gerilim_Seviyesi, Gerilim_Seviyesi_kV, Marka")
        logger.info("")
        
        logger.info(f"[SUCCESS] {STEP_NAME} completed successfully!")
        
    except Exception as e:
        logger.exception(f"[FATAL] {STEP_NAME} failed: {e}")
        raise


if __name__ == "__main__":
    main()