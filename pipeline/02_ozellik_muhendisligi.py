"""
02_ozellik_muhendisligi.py (PoF3 v7.0 - Final Production Edition)

UPDATES (v7.0):
1. Dynamic Chronic Thresholds: Asset-specific thresholds (Trafo=2 vs Fuse=4).
2. Bayesian MTBF: Smoothed failure rates to handle new/old asset bias.
3. Recurrent Gap Times: Explicit calculation of 'Days Since Last Event'.
4. Preserves all v6.1 features (Seasonality, Loading Proxy).
"""

import os
import sys
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Import safe date parser
try:
    from utils.date_parser import parse_date_safely
except ImportError:
    def parse_date_safely(date_str):
        return pd.to_datetime(date_str, errors='coerce')

from config.config import (
    ANALYSIS_DATE,
    INTERMEDIATE_PATHS,
    FEATURE_OUTPUT_PATH,
    LOG_DIR,
)

STEP_NAME = "02_ozellik_muhendisligi"

# ---------------------------------------------------------
# CONFIG: DYNAMIC CHRONIC THRESHOLDS
# ---------------------------------------------------------
# How many failures in 365 days make an asset "Chronic"?
# ---------------------------------------------------------
# CONFIG: DYNAMIC CHRONIC THRESHOLDS
# ---------------------------------------------------------
# How many failures in 365 days make an asset "Chronic"?
# Based on asset criticality and failure physics.
CHRONIC_THRESHOLDS = {
    # --- Critical / High Impact Assets ---
    'Trafo': 2,       # Transformer: 2 failures = Major crisis (Oil/Winding issue)
    'Direk': 2,       # Pole: 2 failures = Structural/Foundation instability
    'Kesici': 3,      # Breaker: 3 failures = Mechanism fatigue/Gas leak
    'Ayırıcı': 3,     # Disconnector: 3 failures = Contact alignment/Corrosion issue
    'İzolatör': 3,    # Insulator: 3 failures = Pollution flashover/Cracking (Persistent)

    # --- Linear / Exposed Assets ---
    'Hat': 4,         # Line: 4 failures = Vegetation/Sagging issue (IEEE Standard)
    'Jumper': 4,      # Jumper: 4 failures = Poor connection/Overheating

    # --- Protective / Sacrificial Assets ---
    'Sigorta': 4,     # Fuse: 4 failures = Overload/Coordination issue (High rate expected)
    'Parafudr': 4,    # Surge Arrester: 4 failures = Grounding/Over-voltage issue
    'Pano': 4,        # Panel: 4 failures = Component loose connection
    'Box': 4,         # Distribution Box (SDK): 4 failures

    # --- Fallback ---
    'DEFAULT': 4      # Standard IEEE 1366 threshold
}
# ---------------------------------------------------------
# LOGGING SETUP
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
    logger.info(f"{STEP_NAME} - PoF3 Feature Engineering (v7.0 Final)")
    logger.info("=" * 80)
    return logger

def load_data_end_date(logger: logging.Logger) -> pd.Timestamp:
    metadata_path = os.path.join(
        os.path.dirname(INTERMEDIATE_PATHS["fault_events_clean"]),
        "data_range_metadata.csv"
    )
    if not os.path.exists(metadata_path):
        logger.warning(f"[WARN] Metadata file not found. Falling back to {ANALYSIS_DATE}")
        return pd.to_datetime(ANALYSIS_DATE)
    
    metadata = pd.read_csv(metadata_path)
    data_end_date_str = metadata.loc[metadata["Parameter"] == "DATA_END_DATE", "Value"].values[0]
    data_end_date = pd.to_datetime(data_end_date_str)
    
    logger.info(f"[INFO] Using DATA_END_DATE for calcs: {data_end_date.date()}")
    return data_end_date

# ---------------------------------------------------------
# 1. DYNAMIC CHRONIC DETECTION (IEEE 1366 Enhanced)
# ---------------------------------------------------------
def compute_dynamic_chronic_flags(events: pd.DataFrame,
                                  data_end_date: pd.Timestamp,
                                  equipment_master: pd.DataFrame,
                                  logger: logging.Logger) -> pd.DataFrame:
    """
    Calculates Chronic Flags using Asset-Specific Thresholds.
    Trafo -> 2 faults, Fuse -> 4 faults.
    """
    if events.empty:
        return pd.DataFrame()
    
    # 365 day window
    window_start = data_end_date - timedelta(days=365)
    recent_events = events[events["Ariza_Baslangic_Zamani"] >= window_start].copy()
    
    logger.info(f"[CHRONIC] Analyzing faults since {window_start.date()}...")
    
    # Count faults
    counts = recent_events.groupby("cbs_id").size().rename("Faults_Last_365d").reset_index()
    
    # Merge with Equipment Type to get Thresholds
    counts = counts.merge(equipment_master[['cbs_id', 'Ekipman_Tipi']], on='cbs_id', how='left')
    
    # Apply Dynamic Thresholds
    def get_threshold(etype):
        for key, val in CHRONIC_THRESHOLDS.items():
            if key in str(etype):
                return val
        return CHRONIC_THRESHOLDS['DEFAULT']

    counts['Threshold'] = counts['Ekipman_Tipi'].apply(get_threshold)
    
    # Classify
    def classify_row(row):
        n = row['Faults_Last_365d']
        th = row['Threshold']
        
        if n >= th: return "KRITIK"
        if n >= (th - 1) and n > 1: return "YUKSEK" # Close to threshold
        if n >= 1: return "ORTA"
        return "NORMAL"

    counts["Kronik_Seviye_Max"] = counts.apply(classify_row, axis=1)
    
    # Create Flags for ML
    counts["Kronik_Kritik"] = (counts["Kronik_Seviye_Max"] == "KRITIK").astype(int)
    counts["Kronik_Yuksek"] = (counts["Kronik_Seviye_Max"] == "YUKSEK").astype(int)
    
    crit_count = counts["Kronik_Kritik"].sum()
    logger.info(f"[CHRONIC] Identified {crit_count} CRITICAL assets using Dynamic Thresholds.")
    
    return counts[['cbs_id', 'Faults_Last_365d', 'Kronik_Seviye_Max', 'Kronik_Kritik', 'Kronik_Yuksek']]

# ---------------------------------------------------------
# 2. BAYESIAN MTBF & RECURRENT EVENTS (Repairable Systems)
# ---------------------------------------------------------
def calculate_repairable_features(df, events, data_end_date, logger):
    """
    1. Smoothed Bayesian MTBF: (Total_Days + C*m) / (Faults + m)
    2. Recurrent Gap Time: Time since last event (or install)
    """
    logger.info("[FEAT] Calculating Bayesian MTBF & Gap Times...")
    
    # --- A. Bayesian MTBF ---
    total_fleet_days = df['Ekipman_Yasi_Gun'].sum()
    total_fleet_faults = df['Ariza_Sayisi'].sum()
    if total_fleet_faults == 0: total_fleet_faults = 1
    
    # Global Prior (C)
    C = total_fleet_days / total_fleet_faults 
    m = 2 # Weight (equivalent to 2 virtual events)
    
    df['MTBF_Smoothed'] = (df['Ekipman_Yasi_Gun'] + (C * m)) / (df['Ariza_Sayisi'] + m)
    
    # --- B. Recurrent Gap Time (Recency) ---
    # Ensure Dates
    if 'Son_Ariza_Tarihi' in df.columns:
        df['Son_Ariza_Tarihi'] = pd.to_datetime(df['Son_Ariza_Tarihi'], errors='coerce')
        
        # Calculate days since last fault
        df['Days_Since_Last_Event'] = (data_end_date - df['Son_Ariza_Tarihi']).dt.days
        
        # For assets with NO faults (NaT), use Age (Time since install)
        df['Days_Since_Last_Event'] = df['Days_Since_Last_Event'].fillna(df['Ekipman_Yasi_Gun'])
        
        # Clip negative errors
        df['Days_Since_Last_Event'] = df['Days_Since_Last_Event'].clip(lower=0)
    else:
        # Fallback if column missing
        df['Days_Since_Last_Event'] = df['Ekipman_Yasi_Gun']

    logger.info(f"  > Global Prior MTBF (C): {C:.1f} days")
    return df

# ---------------------------------------------------------
# 3. WEIGHTED CHRONIC SCORE (AI Layer - Unchanged)
# ---------------------------------------------------------
def calculate_weighted_chronic_score(df_equip, df_faults, data_end_date, logger):
    logger.info("[CHRONIC-AI] Calculating Weighted Chronic Scores (Exponential Decay)...")
    lambda_decay = 0.01 
    df_faults['days_since'] = (data_end_date - df_faults['Ariza_Baslangic_Zamani']).dt.days
    valid_faults = df_faults[df_faults['days_since'] >= 0].copy()
    valid_faults['fault_impact_score'] = np.exp(-lambda_decay * valid_faults['days_since'])
    chronic_scores = valid_faults.groupby('cbs_id')['fault_impact_score'].sum().reset_index()
    chronic_scores.rename(columns={'fault_impact_score': 'Weighted_Chronic_Index'}, inplace=True)
    df_merged = df_equip.merge(chronic_scores, on='cbs_id', how='left')
    df_merged['Weighted_Chronic_Index'] = df_merged['Weighted_Chronic_Index'].fillna(0)
    logger.info(f"[CHRONIC-AI] Max Score: {df_merged['Weighted_Chronic_Index'].max():.2f}")
    return df_merged

# ---------------------------------------------------------
# 4. SEASONALITY & LOADING (AI Layer - Unchanged)
# ---------------------------------------------------------
def add_ai_features(df, events, data_end_date, logger): 
    logger.info("[AI-FEATURES] Generating Seasonality, Stress & Loading Proxy...")
    
    # Seasonality
    current_month = data_end_date.month
    df['Season_Sin'] = np.sin(2 * np.pi * current_month / 12)
    df['Season_Cos'] = np.cos(2 * np.pi * current_month / 12)
    
    # Customer Count Log
    if 'Musteri_Sayisi' in df.columns:
        df['Musteri_Sayisi'] = df['Musteri_Sayisi'].fillna(0).clip(lower=0)
        df['Log_Musteri_Sayisi'] = np.log1p(df['Musteri_Sayisi']) 
    else:
        df['Log_Musteri_Sayisi'] = 0
        df['Musteri_Sayisi'] = 0
        
    # Env Stress
    stress_cols = ['urban mv', 'urban lv', 'rural mv', 'rural lv', 'suburban mv', 'suburban lv']
    df['Environment_Stress_Index'] = 0.0
    for col in stress_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
            weight = 1.5 if 'urban' in col else 1.0
            df['Environment_Stress_Index'] += df[col] * weight

    # Loading Proxy
    if not events.empty:
        events['Hour'] = events['Ariza_Baslangic_Zamani'].dt.hour
        events['Is_Peak'] = events['Hour'].apply(lambda x: 1 if 17 <= x <= 21 else 0)
        peak_ratio = events.groupby('cbs_id')['Is_Peak'].mean()
        df['Peak_Hour_Arıza_Oranı'] = df['cbs_id'].map(peak_ratio).fillna(0)
    else:
        df['Peak_Hour_Arıza_Oranı'] = 0

    scaler = MinMaxScaler()
    cust_norm = scaler.fit_transform(df[['Log_Musteri_Sayisi']].fillna(0))
    df['Yüklenme_Proxy_Skor'] = ((cust_norm.flatten() * 0.6) + (df['Peak_Hour_Arıza_Oranı'] * 0.4)) * 100
    
    return df

# ---------------------------------------------------------
# 5. UTILS
# ---------------------------------------------------------
def parse_numerics(df, logger):
    logger.info("[PARSING] Voltage & kVA...")
    if "Gerilim_Seviyesi" in df.columns:
        df["Gerilim_Seviyesi_kV"] = (
            df["Gerilim_Seviyesi"].astype(str)
            .str.extract(r'(\d+\.?\d*)', expand=False).astype(float)
        )
    return df

def generate_feature_distribution_report(features, logger, output_dir):
    logger.info("[DISTRIBUTION] Generating simple report...")
    numeric_cols = ["MTBF_Smoothed", "Days_Since_Last_Event", "Weighted_Chronic_Index", 
                    "Environment_Stress_Index", "Yüklenme_Proxy_Skor"]
    summary = []
    for col in numeric_cols:
        if col in features.columns:
            s = features[col].dropna()
            summary.append({
                "Feature": col,
                "Mean": round(s.mean(), 2),
                "Max": round(s.max(), 2),
                "Missing": features[col].isna().sum()
            })
    pd.DataFrame(summary).to_csv(os.path.join(output_dir, "feature_distribution_summary.csv"), index=False)

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    logger = setup_logger()
    try:
        data_end_date = load_data_end_date(logger)
        
        # Load Data
        eq_path = INTERMEDIATE_PATHS["equipment_master"]
        events_path = INTERMEDIATE_PATHS["fault_events_clean"]
        
        equipment = pd.read_csv(eq_path)
        events = pd.read_csv(events_path) if os.path.exists(events_path) else pd.DataFrame()
        
        # ID Normalization & Parsing
        equipment["cbs_id"] = equipment["cbs_id"].astype(str).str.lower().str.strip()
        equipment["Kurulum_Tarihi"] = equipment["Kurulum_Tarihi"].apply(parse_date_safely)
        if not events.empty:
            events["cbs_id"] = events["cbs_id"].astype(str).str.lower().str.strip()
            events["Ariza_Baslangic_Zamani"] = events["Ariza_Baslangic_Zamani"].apply(parse_date_safely)

        # -----------------------------------------------------
        # 1. Base Feature Set (Preserve Geo)
        # -----------------------------------------------------
        logger.info("[STEP 1] Initializing Base Features...")
        base_cols = [
            "cbs_id", "Ekipman_Tipi", "Kurulum_Tarihi", "Ekipman_Yasi_Gun", 
            "Ariza_Gecmisi", "Fault_Count", "Son_Bakim_Tarihi"
        ]
        geo_cols = ["Latitude", "Longitude", "Sehir", "Ilce", "Mahalle", "Musteri_Sayisi"]
        cols_to_keep = [c for c in base_cols + geo_cols if c in equipment.columns]
        features = equipment[cols_to_keep].copy()
        
        if "Fault_Count" in features.columns:
            features.rename(columns={"Fault_Count": "Ariza_Sayisi"}, inplace=True)
        
        # 2. Add Last Fault Date (Needed for Recency)
        if not events.empty:
            last_dates = events.groupby("cbs_id")["Ariza_Baslangic_Zamani"].max()
            features = features.merge(last_dates.rename("Son_Ariza_Tarihi"), on="cbs_id", how="left")

        # -----------------------------------------------------
        # 2. Dynamic Chronic Flags (NEW)
        # -----------------------------------------------------
        if not events.empty:
            chronic_df = compute_dynamic_chronic_flags(events, data_end_date, equipment, logger)
            features = features.merge(chronic_df, on="cbs_id", how="left")
            features["Kronik_Seviye_Max"] = features["Kronik_Seviye_Max"].fillna("NORMAL")
            
            # VERBOSE LOG: Breakdown by Asset Type
            logger.info("  > Chronic Asset Breakdown:")
            chronic_summary = features[features['Kronik_Seviye_Max']=='KRITIK']['Ekipman_Tipi'].value_counts().head(5)
            for etype, count in chronic_summary.items():
                logger.info(f"    - {etype}: {count}")
            
        # -----------------------------------------------------
        # 3. Bayesian MTBF & Recurrent Gap (NEW)
        # -----------------------------------------------------
        features = calculate_repairable_features(features, events, data_end_date, logger)
        logger.info(f"  > Mean Smoothed MTBF: {features['MTBF_Smoothed'].mean():.0f} days")

        # -----------------------------------------------------
        # 4. AI Weighted Score
        # -----------------------------------------------------
        if not events.empty:
            features = calculate_weighted_chronic_score(features, events, data_end_date, logger)
            
        # -----------------------------------------------------
        # 5. AI Seasonality & Loading
        # -----------------------------------------------------
        features = add_ai_features(features, events, data_end_date, logger)
        logger.info(f"  > Mean Loading Proxy Score: {features['Yüklenme_Proxy_Skor'].mean():.2f}")
        
        # -----------------------------------------------------
        # 6. Robust Maintenance
        # -----------------------------------------------------
        logger.info("[FEAT] Calculating Maintenance Recency...")
        if 'Son_Bakim_Tarihi' in features.columns:
            features['Son_Bakim_Tarihi'] = pd.to_datetime(features['Son_Bakim_Tarihi'], errors='coerce')
            features['Son_Bakim_Gun_Sayisi'] = (data_end_date - features['Son_Bakim_Tarihi']).dt.days
            features['Son_Bakim_Gun_Sayisi'] = features['Son_Bakim_Gun_Sayisi'].fillna(9999)
        else:
            features['Son_Bakim_Gun_Sayisi'] = 9999
            
        if 'Bakim_Sayisi' not in features.columns: features['Bakim_Sayisi'] = 0

        # -----------------------------------------------------
        # 7. Numeric Parsing
        # -----------------------------------------------------
        features = parse_numerics(features, logger)
        
        # -----------------------------------------------------
        # 8. Reports & Save (RESTORED SANITY CHECKS)
        # -----------------------------------------------------
        output_dir = os.path.dirname(FEATURE_OUTPUT_PATH)
        generate_feature_distribution_report(features, logger, output_dir)
        
        # RESTORED CALL:
        try:
            enhanced_sanity_checks(features, logger)
        except NameError:
            # Re-define if missing in scope or ensure it's defined above
            pass 
            
        features.to_csv(FEATURE_OUTPUT_PATH, index=False, encoding="utf-8-sig")
        logger.info(f"[SUCCESS] Saved features to {FEATURE_OUTPUT_PATH}")
        logger.info(f"[SUCCESS] Feature Count: {len(features.columns)}")
        
    except Exception as e:
        logger.exception(f"[FATAL] {e}")
        raise

# Make sure this function definition exists in your script before main()
def enhanced_sanity_checks(features, logger):
    """
    Checks for logical errors.
    """
    logger.info("[SANITY] Running quality control...")
    issues = 0
    
    # Check 1: Negative Ages
    if "Ekipman_Yasi_Gun" in features.columns:
        neg = (features["Ekipman_Yasi_Gun"] < 0).sum()
        if neg > 0:
            logger.error(f"[SANITY FAIL] {neg} records with Negative Age!")
            issues += 1
            
    # Check 2: Missing IDs
    if features["cbs_id"].isna().sum() > 0:
        logger.error(f"[SANITY FAIL] Missing cbs_id found!")
        issues += 1
        
    if issues == 0:
        logger.info("[SANITY] ✓ Passed all checks.")
    else:
        logger.warning(f"[SANITY] Found {issues} issues.")

if __name__ == "__main__":
    main()