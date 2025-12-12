"""
02_ozellik_muhendisligi.py (PoF3 v6.1 - Extended Production Edition)

COMBINES:
1. IEEE 1366 Reporting (Kritik/Yuksek/Orta) -> Preserved from v3
2. AI Predictive Features (Weighted Score, Seasonality) -> Preserved from v3
3. Robust Sanity Checks & Reporting -> Preserved from v3
4. Geo-Location Pass-Through -> Preserved from v3
5. NEW: Loading Proxy Score (Customer Count + Peak Hour Stress) -> Added
6. NEW: Robust Maintenance Calculation -> Added
"""

import os
import sys
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler  # <--- Added for Loading Proxy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Assuming utils exists as per your previous file
# If utils/date_parser.py is missing, this line can be replaced with a simple function
try:
    from utils.date_parser import parse_date_safely
except ImportError:
    # Fallback if util is missing
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
    logger.info(f"{STEP_NAME} - PoF3 Feature Engineering (Extended Production v6.1)")
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
# 1. IEEE STANDARD CHRONIC DETECTION (Reporting Layer)
# ---------------------------------------------------------
def compute_ieee_chronic_flags(events: pd.DataFrame,
                               data_end_date: pd.Timestamp,
                               logger: logging.Logger) -> pd.DataFrame:
    """
    IEEE 1366: Rolling 365-day window counts.
    """
    if events.empty:
        return pd.DataFrame()
    
    # Filter 2 years for efficiency
    cutoff_date = data_end_date - timedelta(days=730)
    recent_events = events[events["Ariza_Baslangic_Zamani"] >= cutoff_date].copy()
    
    logger.info(f"[CHRONIC-IEEE] Analyzing {len(recent_events):,} recent faults...")
    
    # 365 day window start
    window_start = data_end_date - timedelta(days=365)
    
    # Fast grouping
    counts = recent_events[recent_events["Ariza_Baslangic_Zamani"] >= window_start].groupby("cbs_id").size()
    
    # Map to DataFrame
    chronic_df = pd.DataFrame(counts, columns=["Faults_Last_365d"]).reset_index()
    
    # Classify
    def classify(n):
        if n >= 4: return "KRITIK", 1, 1, 1
        if n == 3: return "YUKSEK", 0, 1, 1
        if n == 2: return "ORTA", 0, 0, 1
        return "NORMAL", 0, 0, 0

    chronic_df[["Kronik_Seviye_Max", "Kronik_Kritik", "Kronik_Yuksek", "Kronik_Orta"]] = \
        chronic_df["Faults_Last_365d"].apply(lambda x: pd.Series(classify(x)))
        
    # Stats
    crit_count = (chronic_df["Kronik_Seviye_Max"] == "KRITIK").sum()
    logger.info(f"[CHRONIC-IEEE] Identified {crit_count} CRITICAL assets (4+ faults/year).")
    return chronic_df

# ---------------------------------------------------------
# 2. WEIGHTED CHRONIC SCORE (AI Layer)
# ---------------------------------------------------------
def calculate_weighted_chronic_score(df_equip, df_faults, data_end_date, logger):
    """
    Exponential Decay Score:
    Recent faults matter MUCH more than old faults.
    Formula: Sum( e^(-lambda * days_ago) )
    """
    logger.info("[CHRONIC-AI] Calculating Weighted Chronic Scores (Exponential Decay)...")
    
    lambda_decay = 0.01 
    
    # Calculate Days Ago
    df_faults['days_since'] = (data_end_date - df_faults['Ariza_Baslangic_Zamani']).dt.days
    
    # Filter only past faults (sanity)
    valid_faults = df_faults[df_faults['days_since'] >= 0].copy()
    
    # Calculate Score
    valid_faults['fault_impact_score'] = np.exp(-lambda_decay * valid_faults['days_since'])
    
    # Aggregate
    chronic_scores = valid_faults.groupby('cbs_id')['fault_impact_score'].sum().reset_index()
    chronic_scores.rename(columns={'fault_impact_score': 'Weighted_Chronic_Index'}, inplace=True)
    
    # Merge
    df_merged = df_equip.merge(chronic_scores, on='cbs_id', how='left')
    df_merged['Weighted_Chronic_Index'] = df_merged['Weighted_Chronic_Index'].fillna(0)
    
    logger.info(f"[CHRONIC-AI] Max Score: {df_merged['Weighted_Chronic_Index'].max():.2f}")
    return df_merged

# ---------------------------------------------------------
# 3. SEASONALITY, STRESS & LOADING PROXY (AI Layer)
# ---------------------------------------------------------
def add_ai_features(df, events, data_end_date, logger): # <--- UPDATED to accept 'events'
    """
    Adds Seasonality (Sin/Cos), Stress Proxies, and NEW Loading Proxy.
    """
    logger.info("[AI-FEATURES] Generating Seasonality, Stress & Loading Proxy...")
    
    # A. Seasonality
    current_month = data_end_date.month
    df['Season_Sin'] = np.sin(2 * np.pi * current_month / 12)
    df['Season_Cos'] = np.cos(2 * np.pi * current_month / 12)
    
    # B. Customer Count (Log transform for skew)
    if 'Musteri_Sayisi' in df.columns:
        df['Musteri_Sayisi'] = df['Musteri_Sayisi'].fillna(0).clip(lower=0)
        # SAFETY FIX: use log1p
        df['Log_Musteri_Sayisi'] = np.log1p(df['Musteri_Sayisi']) 
        df['Log_Musteri_Sayisi'] = df['Log_Musteri_Sayisi'].replace([np.inf, -np.inf], 0)
    else:
        df['Log_Musteri_Sayisi'] = 0
        df['Musteri_Sayisi'] = 0
        
    # C. Environmental Stress (Urban/Rural)
    stress_cols = ['urban mv', 'urban lv', 'rural mv', 'rural lv', 'suburban mv', 'suburban lv']
    df['Environment_Stress_Index'] = 0.0
    
    found_any = False
    for col in stress_cols:
        if col in df.columns:
            found_any = True
            df[col] = df[col].fillna(0)
            weight = 1.5 if 'urban' in col else 1.0
            df['Environment_Stress_Index'] += df[col] * weight
            
    if not found_any:
        df['Environment_Stress_Index'] = 0

    # D. NEW: LOADING PROXY SCORE (The "Stolen Feature")
    # 1. Calculate Peak Hour Failure Ratio
    if not events.empty:
        events['Hour'] = events['Ariza_Baslangic_Zamani'].dt.hour
        # Peak hours defined as 17:00 - 21:00
        events['Is_Peak'] = events['Hour'].apply(lambda x: 1 if 17 <= x <= 21 else 0)
        peak_ratio = events.groupby('cbs_id')['Is_Peak'].mean()
        df['Peak_Hour_Arıza_Oranı'] = df['cbs_id'].map(peak_ratio).fillna(0)
    else:
        df['Peak_Hour_Arıza_Oranı'] = 0

    # 2. Composite Score (60% Customer Load + 40% Peak Hour Vulnerability)
    scaler = MinMaxScaler()
    # Reshape for scaler
    cust_norm = scaler.fit_transform(df[['Log_Musteri_Sayisi']].fillna(0))
    
    df['Yüklenme_Proxy_Skor'] = ((cust_norm.flatten() * 0.6) + (df['Peak_Hour_Arıza_Oranı'] * 0.4)) * 100
    logger.info(f"  > Loading Proxy Calculated. Mean Score: {df['Yüklenme_Proxy_Skor'].mean():.2f}")
        
    return df

# ---------------------------------------------------------
# 4. REPORTING & SANITY
# ---------------------------------------------------------
def generate_feature_distribution_report(features, logger, output_dir):
    """
    Logs basic stats for key features.
    """
    logger.info("[DISTRIBUTION] Generating simple report...")
    numeric_cols = ["Ekipman_Yasi_Gun", "Ariza_Sayisi", "Weighted_Chronic_Index", 
                    "Environment_Stress_Index", "Yüklenme_Proxy_Skor"] # Added new score
    
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
    
    dist_df = pd.DataFrame(summary)
    dist_path = os.path.join(output_dir, "feature_distribution_summary.csv")
    dist_df.to_csv(dist_path, index=False)
    logger.info(f"[DISTRIBUTION] Saved summary to {dist_path}")

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

def parse_numerics(df, logger):
    logger.info("[PARSING] Voltage & kVA...")
    if "kVA_Rating" in df.columns:
        df["kVA_Rating_Numeric"] = pd.to_numeric(df["kVA_Rating"], errors="coerce")
    
    if "Gerilim_Seviyesi" in df.columns:
        df["Gerilim_Seviyesi_kV"] = (
            df["Gerilim_Seviyesi"].astype(str)
            .str.extract(r'(\d+\.?\d*)', expand=False).astype(float)
        )
    return df

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
        
        # ID Normalization
        equipment["cbs_id"] = equipment["cbs_id"].astype(str).str.lower().str.strip()
        if not events.empty:
            events["cbs_id"] = events["cbs_id"].astype(str).str.lower().str.strip()

        # Date Parsing
        equipment["Kurulum_Tarihi"] = equipment["Kurulum_Tarihi"].apply(parse_date_safely)
        if not events.empty:
            events["Ariza_Baslangic_Zamani"] = events["Ariza_Baslangic_Zamani"].apply(parse_date_safely)

        # 1. Base Feature Set (Start with Equipment Master)
        # Explicitly keep Geo-columns for Reporting
        base_cols = [
            "cbs_id", "Ekipman_Tipi", "Kurulum_Tarihi", "Ekipman_Yasi_Gun", 
            "Ariza_Gecmisi", "Fault_Count", "Son_Bakim_Tarihi" # Added Son_Bakim_Tarihi
        ]
        
        # ADD GEO COLUMNS HERE TO ENSURE PASS-THROUGH
        geo_cols = ["Latitude", "Longitude", "Sehir", "Ilce", "Mahalle", "Musteri_Sayisi"]
        cols_to_keep = [c for c in base_cols + geo_cols if c in equipment.columns]
        
        features = equipment[cols_to_keep].copy()
        
        # Rename to Turkish Standard
        if "Fault_Count" in features.columns:
            features.rename(columns={"Fault_Count": "Ariza_Sayisi"}, inplace=True)
        
        # 2. IEEE Chronic Flags
        if not events.empty:
            chronic_df = compute_ieee_chronic_flags(events, data_end_date, logger)
            features = features.merge(chronic_df, on="cbs_id", how="left")
            features["Kronik_Seviye_Max"] = features["Kronik_Seviye_Max"].fillna("NORMAL")
            
        # 3. AI Weighted Score
        if not events.empty:
            features = calculate_weighted_chronic_score(features, events, data_end_date, logger)
            
        # 4. AI Seasonality & Stress & LOADING PROXY (UPDATED)
        features = add_ai_features(features, events, data_end_date, logger)
        
        # 5. ROBUST MAINTENANCE CALCULATION (NEW)
        # This fixes the missing 'Son_Bakim_Gun_Sayisi' error in Step 04
        logger.info("[FEAT] Calculating Maintenance Recency (Robust)...")
        if 'Son_Bakim_Tarihi' in features.columns:
            features['Son_Bakim_Tarihi'] = pd.to_datetime(features['Son_Bakim_Tarihi'], errors='coerce')
            features['Son_Bakim_Gun_Sayisi'] = (data_end_date - features['Son_Bakim_Tarihi']).dt.days
            # Fill NaTs with 9999 (Never maintained)
            features['Son_Bakim_Gun_Sayisi'] = features['Son_Bakim_Gun_Sayisi'].fillna(9999)
        else:
            features['Son_Bakim_Gun_Sayisi'] = 9999
            
        if 'Bakim_Sayisi' not in features.columns:
            features['Bakim_Sayisi'] = 0

        # 6. Numeric Parsing
        features = parse_numerics(features, logger)
        
        # 7. Final Sanity Fixes (Recency)
        if not events.empty:
            last_dates = events.groupby("cbs_id")["Ariza_Baslangic_Zamani"].max()
            features = features.merge(last_dates.rename("Son_Ariza_Tarihi"), on="cbs_id", how="left")
            features["Son_Ariza_Gun_Sayisi"] = (data_end_date - features["Son_Ariza_Tarihi"]).dt.days
            features.loc[features["Son_Ariza_Gun_Sayisi"].between(-5, 0), "Son_Ariza_Gun_Sayisi"] = 0
            features["Son_Ariza_Gun_Sayisi"] = features["Son_Ariza_Gun_Sayisi"].fillna(3650) # Fill NaNs
            
        # 8. Reports
        output_dir = os.path.dirname(FEATURE_OUTPUT_PATH)
        generate_feature_distribution_report(features, logger, output_dir)
        enhanced_sanity_checks(features, logger)
            
        # Save
        features.to_csv(FEATURE_OUTPUT_PATH, index=False, encoding="utf-8-sig")
        logger.info(f"[SUCCESS] Saved features to {FEATURE_OUTPUT_PATH}")
        logger.info(f"[SUCCESS] Feature Count: {len(features.columns)}")
        
    except Exception as e:
        logger.exception(f"[FATAL] {e}")
        raise

if __name__ == "__main__":
    main()