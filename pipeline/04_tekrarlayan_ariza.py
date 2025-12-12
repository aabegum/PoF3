"""
04_tekrarlayan_ariza.py (PoF3 v5.0 - Statistical Chronic Engine)

IMPROVEMENTS:
1. Poisson Risk Model: Calculates Prob(Faults >= 4) for next year based on Lambda.
   - Transforms static counts into forward-looking risk probabilities.
2. Dominant Cause Analysis: Identifies the #1 reason for failure per asset.
3. IEEE 1366 Classification: Standard Critical/High/Medium/Low logic.
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy.stats import poisson

# Ensure UTF-8 console
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.config import (
    INTERMEDIATE_PATHS, OUTPUT_DIR, LOG_DIR,
    FEATURE_OUTPUT_PATH,
    CHRONIC_WINDOW_DAYS, CHRONIC_THRESHOLD_EVENTS
)

STEP_NAME = "04_tekrarlayan_ariza"

# ==============================================================================
# LOGGING SETUP
# ==============================================================================
def setup_logger():
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"{STEP_NAME}_{ts}.log")
    
    logger = logging.getLogger(STEP_NAME)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    
    return logger

logger = setup_logger()

# ==============================================================================
# 1. LOAD DATA & DATES
# ==============================================================================
def load_data_end_date() -> pd.Timestamp:
    metadata_path = os.path.join(os.path.dirname(INTERMEDIATE_PATHS["fault_events_clean"]), "data_range_metadata.csv")
    
    if not os.path.exists(metadata_path):
        logger.warning(f"[WARN] Metadata not found at {metadata_path}. Using Today.")
        return pd.Timestamp.now().normalize()

    try:
        meta = pd.read_csv(metadata_path)
        val = meta.loc[meta["Parameter"] == "DATA_END_DATE", "Value"].values[0]
        dt = pd.to_datetime(val)
        logger.info(f"[INFO] Analysis Reference Date (DATA_END_DATE): {dt.date()}")
        return dt
    except Exception as e:
        logger.error(f"[ERROR] Could not read metadata: {e}")
        return pd.Timestamp.now().normalize()

def load_intermediate_data():
    logger.info("[STEP] Loading intermediate data...")
    
    events_path = INTERMEDIATE_PATHS["fault_events_clean"]
    equip_path = INTERMEDIATE_PATHS["equipment_master"]
    feat_path = FEATURE_OUTPUT_PATH
    
    if not os.path.exists(events_path): raise FileNotFoundError(f"Missing: {events_path}")
    if not os.path.exists(equip_path): raise FileNotFoundError(f"Missing: {equip_path}")
    
    events = pd.read_csv(events_path, parse_dates=["Ariza_Baslangic_Zamani"])
    equipment = pd.read_csv(equip_path, parse_dates=["Kurulum_Tarihi"])
    
    if os.path.exists(feat_path):
        features = pd.read_csv(feat_path)
    else:
        features = None
        
    # Normalize IDs
    events["cbs_id"] = events["cbs_id"].astype(str).str.lower().str.strip()
    equipment["cbs_id"] = equipment["cbs_id"].astype(str).str.lower().str.strip()
    if features is not None:
        features["cbs_id"] = features["cbs_id"].astype(str).str.lower().str.strip()
        
    return events, equipment, features

# ==============================================================================
# 2. CHRONIC FAULT LOGIC (IEEE + POISSON)
# ==============================================================================
def calculate_chronic_flags(events, data_end_date):
    logger.info("[STEP] Calculating Chronic Flags & Poisson Probabilities...")
    
    # Window: Last 365 Days
    window_start = data_end_date - timedelta(days=365)
    recent_events = events[events["Ariza_Baslangic_Zamani"] >= window_start].copy()
    
    # 1. IEEE 1366 Classification
    counts = recent_events.groupby("cbs_id").size().reset_index(name="Faults_Last_365d")
    
    def categorize(n):
        if n >= 4: return "KRITIK", 1, 1, 1, 1
        if n == 3: return "YUKSEK", 0, 1, 1, 1
        if n == 2: return "ORTA",   0, 0, 1, 1
        if n == 1: return "IZLEME", 0, 0, 0, 1
        return "NORMAL", 0, 0, 0, 0

    results = counts["Faults_Last_365d"].apply(lambda x: pd.Series(categorize(x)))
    results.columns = ["Kronik_Seviye_Max", "Kronik_Kritik", "Kronik_Yuksek", "Kronik_Orta", "Kronik_Izleme"]
    
    chronic_df = pd.concat([counts[["cbs_id", "Faults_Last_365d"]], results], axis=1)
    
    # 2. Poisson Risk Model (NEW)
    # Calculate Lambda (Annual Rate)
    # If no history, assume global average (e.g., 0.5 faults/year)
    # Here we use the observed count as the Lambda estimator for next year
    chronic_df['Lambda_Est'] = chronic_df['Faults_Last_365d'].replace(0, 0.2) # Small prior for stability
    
    # Probability of exceeding Critical Threshold (4 faults) next year
    # P(X >= 4) = 1 - P(X <= 3) = 1 - CDF(3)
    chronic_df['Prob_NextYear_Critical'] = 1 - poisson.cdf(3, chronic_df['Lambda_Est'])
    
    # Probability of at least 1 fault next year
    # P(X >= 1) = 1 - P(X = 0)
    chronic_df['Prob_NextYear_AnyFault'] = 1 - poisson.pmf(0, chronic_df['Lambda_Est'])
    
    logger.info("[STATS] Poisson Risk Analysis:")
    high_risk_poisson = (chronic_df['Prob_NextYear_Critical'] > 0.5).sum()
    logger.info(f"  > Assets with >50% chance of becoming Critical next year: {high_risk_poisson}")
        
    return chronic_df

# ==============================================================================
# 2b. CAUSE CODE ANALYTICS
# ==============================================================================
def calculate_dominant_cause(events):
    logger.info("[STEP] Analyzing Dominant Failure Causes...")
    if 'Ariza_Nedeni' not in events.columns:
        return pd.DataFrame(columns=['cbs_id', 'Dominant_Cause'])

    valid_events = events[events['Ariza_Nedeni'].notna() & (events['Ariza_Nedeni'] != 'UNKNOWN')].copy()
    cause_counts = valid_events.groupby(['cbs_id', 'Ariza_Nedeni']).size().reset_index(name='count')
    
    dominant = cause_counts.sort_values(['cbs_id', 'count'], ascending=[True, False]) \
                           .drop_duplicates(['cbs_id']) \
                           .rename(columns={'Ariza_Nedeni': 'Dominant_Cause'})
    
    return dominant[['cbs_id', 'Dominant_Cause']]

# ==============================================================================
# 3. STATISTICS AGGREGATION
# ==============================================================================
def build_statistics_table(equipment, events, data_end_date):
    logger.info("[STEP] Calculating global failure statistics...")
    
    stats = equipment[["cbs_id", "Ekipman_Tipi", "Kurulum_Tarihi"]].copy()
    
    agg = events.groupby("cbs_id").agg(
        Ilk_Ariza_Tarihi=("Ariza_Baslangic_Zamani", "min"),
        Son_Ariza_Tarihi=("Ariza_Baslangic_Zamani", "max"),
        Toplam_Ariza_Sayisi=("Ariza_Baslangic_Zamani", "count")
    ).reset_index()
    
    stats = stats.merge(agg, on="cbs_id", how="left")
    stats["Toplam_Ariza_Sayisi"] = stats["Toplam_Ariza_Sayisi"].fillna(0).astype(int)
    
    stats["Gozlem_Suresi_Gun"] = (data_end_date - stats["Kurulum_Tarihi"]).dt.days
    stats["Gozlem_Suresi_Gun"] = stats["Gozlem_Suresi_Gun"].clip(lower=1)
    stats["Gozlem_Suresi_Yil"] = stats["Gozlem_Suresi_Gun"] / 365.25
    stats["Lambda_Yillik_Ariza"] = stats["Toplam_Ariza_Sayisi"] / stats["Gozlem_Suresi_Yil"]
    
    return stats

# ==============================================================================
# 4. MERGE & SAVE
# ==============================================================================
def integrate_ensemble_results(stats_df):
    logger.info("[STEP] Integrating Ensemble Risk Models...")
    ensemble_path = os.path.join(OUTPUT_DIR, "ensemble_pof_final.csv")
    
    if not os.path.exists(ensemble_path):
        return stats_df
    
    ens_df = pd.read_csv(ensemble_path)
    ens_df["cbs_id"] = ens_df["cbs_id"].astype(str).str.lower().str.strip()
    
    rename_map = {
        "PoF_12Ay": "PoF_Ensemble_12Ay",
        "PoF_36Ay": "PoF_Ensemble_36Ay",
        "PoF_60Ay": "PoF_Ensemble_60Ay"
    }
    ens_df = ens_df.rename(columns=rename_map)
    
    merged = stats_df.merge(ens_df, on="cbs_id", how="left")
    return merged

def main():
    try:
        data_end_date = load_data_end_date()
        events, equipment, features = load_intermediate_data()
        
        # 1. Chronic Detection (IEEE + Poisson)
        chronic_flags = calculate_chronic_flags(events, data_end_date)
        
        # 2. Cause Analytics
        dominant_causes = calculate_dominant_cause(events)
        
        # 3. Stats
        stats_df = build_statistics_table(equipment, events, data_end_date)
        
        # 4. Merge All
        master = stats_df.merge(chronic_flags, on="cbs_id", how="left") \
                         .merge(dominant_causes, on="cbs_id", how="left")
        
        # Cleanup
        master["Kronik_Seviye_Max"] = master["Kronik_Seviye_Max"].fillna("NORMAL")
        master["Faults_Last_365d"] = master["Faults_Last_365d"].fillna(0).astype(int)
        master["Dominant_Cause"] = master["Dominant_Cause"].fillna("No Faults")
        master["Prob_NextYear_Critical"] = master["Prob_NextYear_Critical"].fillna(0.0)
        
        for col in ["Kronik_Kritik", "Kronik_Yuksek", "Kronik_Orta", "Kronik_Izleme"]:
            master[col] = master[col].fillna(0).astype(int)
            
        # 5. Integrate Step 03
        final_master = integrate_ensemble_results(master)
        
        # 6. Save
        risk_path = os.path.join(OUTPUT_DIR, "risk_equipment_master.csv")
        final_master.to_csv(risk_path, index=False, encoding="utf-8-sig")
        
        # Save Action List
        action_path = os.path.join(OUTPUT_DIR, "chronic_equipment_only.csv")
        action_df = final_master[final_master["Kronik_Seviye_Max"].isin(["KRITIK", "YUKSEK", "ORTA"])]
        action_df.to_csv(action_path, index=False, encoding="utf-8-sig")
        
        logger.info("="*60)
        logger.info("[SUCCESS] 04_tekrarlayan_ariza Completed (with Statistical Model).")
        logger.info(f"  > Master Risk Table: {risk_path}")
        logger.info("="*60)
        
    except Exception as e:
        logger.exception(f"[FATAL] {e}")

if __name__ == "__main__":
    main()