"""
04_tekrarlayan_ariza.py (PoF3 - Final Integration v4.3)

Updates:
  - Added 'Dominant Failure Cause' logic using 'Ariza_Nedeni'.
  - Calculates the most frequent failure reason for each asset.
  - Integrates this context into the Risk Master for field crews.
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

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
        logger.warning("[WARN] Feature file not found. Validation step will be skipped.")
        
    # Normalize IDs
    events["cbs_id"] = events["cbs_id"].astype(str).str.lower().str.strip()
    equipment["cbs_id"] = equipment["cbs_id"].astype(str).str.lower().str.strip()
    if features is not None:
        features["cbs_id"] = features["cbs_id"].astype(str).str.lower().str.strip()
        
    return events, equipment, features

# ==============================================================================
# 2. CHRONIC FAULT LOGIC (IEEE 1366)
# ==============================================================================
def calculate_chronic_flags(events, data_end_date):
    logger.info("[STEP] Calculating IEEE Chronic Flags...")
    
    window_start = data_end_date - timedelta(days=365)
    recent_events = events[events["Ariza_Baslangic_Zamani"] >= window_start].copy()
    
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
    
    logger.info("[STATS] Chronic Distribution:")
    for level in ["KRITIK", "YUKSEK", "ORTA", "IZLEME"]:
        cnt = (chronic_df["Kronik_Seviye_Max"] == level).sum()
        logger.info(f"  > {level}: {cnt} assets")
        
    return chronic_df

# ==============================================================================
# 2b. CAUSE CODE ANALYTICS (NEW)
# ==============================================================================
def calculate_dominant_cause(events):
    """
    Finds the most frequent 'Ariza_Nedeni' for each asset.
    Example: TR-101 -> 'Lightning' (3 faults)
    """
    logger.info("[STEP] Analyzing Dominant Failure Causes...")
    
    if 'Ariza_Nedeni' not in events.columns:
        logger.warning("[WARN] 'Ariza_Nedeni' column missing. Skipping cause analysis.")
        return pd.DataFrame(columns=['cbs_id', 'Dominant_Cause'])

    # Filter out empty/unknown causes if desired
    valid_events = events[events['Ariza_Nedeni'].notna() & (events['Ariza_Nedeni'] != 'UNKNOWN')].copy()
    
    # Count causes per ID
    cause_counts = valid_events.groupby(['cbs_id', 'Ariza_Nedeni']).size().reset_index(name='count')
    
    # Sort by count desc and take top 1
    # This gives the "Mode"
    dominant = cause_counts.sort_values(['cbs_id', 'count'], ascending=[True, False]) \
                           .drop_duplicates(['cbs_id']) \
                           .rename(columns={'Ariza_Nedeni': 'Dominant_Cause'})
    
    logger.info(f"[STATS] Calculated dominant causes for {len(dominant)} assets.")
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
# 4. MERGE STEP 03 RESULTS
# ==============================================================================
def integrate_ensemble_results(stats_df):
    logger.info("[STEP] Integrating Ensemble Risk Models...")
    
    ensemble_path = os.path.join(OUTPUT_DIR, "ensemble_pof_final.csv")
    
    if not os.path.exists(ensemble_path):
        logger.warning(f"[WARN] Ensemble file not found at {ensemble_path}.")
        return stats_df
    
    ens_df = pd.read_csv(ensemble_path)
    ens_df["cbs_id"] = ens_df["cbs_id"].astype(str).str.lower().str.strip()
    
    rename_map = {
        "PoF_12Ay": "PoF_Ensemble_12Ay",
        "PoF_3Ay":  "PoF_Ensemble_3Ay",
        "PoF_6Ay":  "PoF_Ensemble_6Ay",
        "PoF_24Ay": "PoF_Ensemble_24Ay"
    }
    ens_df = ens_df.rename(columns=rename_map)
    
    merged = stats_df.merge(ens_df, on="cbs_id", how="left")
    logger.info(f"[OK] Integrated risk scores for {len(ens_df)} assets.")
    return merged

# ==============================================================================
# 5. MAIN EXECUTION
# ==============================================================================
def main():
    try:
        data_end_date = load_data_end_date()
        events, equipment, features = load_intermediate_data()
        
        # 1. Chronic Detection
        chronic_flags = calculate_chronic_flags(events, data_end_date)
        
        # 2. Cause Analytics (NEW)
        dominant_causes = calculate_dominant_cause(events)
        
        # 3. Stats
        stats_df = build_statistics_table(equipment, events, data_end_date)
        
        # 4. Merge All (Stats + Chronic + Causes)
        master = stats_df.merge(chronic_flags, on="cbs_id", how="left") \
                         .merge(dominant_causes, on="cbs_id", how="left")
        
        # Clean up NaNs
        master["Kronik_Seviye_Max"] = master["Kronik_Seviye_Max"].fillna("NORMAL")
        master["Faults_Last_365d"] = master["Faults_Last_365d"].fillna(0).astype(int)
        master["Dominant_Cause"] = master["Dominant_Cause"].fillna("No Faults")
        
        for col in ["Kronik_Kritik", "Kronik_Yuksek", "Kronik_Orta", "Kronik_Izleme"]:
            master[col] = master[col].fillna(0).astype(int)
            
        # 5. Integrate Step 03 Models
        final_master = integrate_ensemble_results(master)
        
        # 6. Save
        risk_path = os.path.join(OUTPUT_DIR, "risk_equipment_master.csv")
        final_master.to_csv(risk_path, index=False, encoding="utf-8-sig")
        
        # Save Action List
        action_path = os.path.join(OUTPUT_DIR, "chronic_equipment_only.csv")
        action_df = final_master[final_master["Kronik_Seviye_Max"].isin(["KRITIK", "YUKSEK", "ORTA"])]
        action_df.to_csv(action_path, index=False, encoding="utf-8-sig")
        
        logger.info("="*60)
        logger.info("[SUCCESS] 04_tekrarlayan_ariza Completed.")
        logger.info(f"  > Master Risk Table: {risk_path}")
        logger.info("="*60)
        
    except Exception as e:
        logger.exception(f"[FATAL] {e}")

if __name__ == "__main__":
    main()