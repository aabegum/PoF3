"""
04b_risk_scoring.py (PoF3 - Advanced Risk Engine - Robust v4.3)

Purpose:
  Calculates the Consequence of Failure (CoF) and Total Risk Score.
  
Fixes:
  - RECOVERS 'Gerilim_Seviyesi' from Master file if missing (The Critical Fix).
  - Robust check for 'Gerilim_Seviyesi_kV'.
  - Defaults to Low Voltage logic if voltage data is truly missing.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Setup Paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from config.config import DATA_DIR, OUTPUT_DIR, LOG_DIR, INTERMEDIATE_PATHS

STEP_NAME = "04b_risk_scoring"

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------
COST_MAP = {
    "Trafo": 50000, "Ayırıcı": 15000, "Pano": 25000, 
    "Hat": 10000, "Direk": 3000, "Sigorta": 5000, "Other": 5000
}

# ------------------------------------------------------------------------------
# LOGGING
# ------------------------------------------------------------------------------
def setup_logger():
    os.makedirs(LOG_DIR, exist_ok=True)
    logger = logging.getLogger(STEP_NAME)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(os.path.join(LOG_DIR, f"{STEP_NAME}.log"), encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger

# ------------------------------------------------------------------------------
# 1. CALCULATE EXPECTED REPAIR TIME (MTTR Proxy)
# ------------------------------------------------------------------------------
def calculate_mttr_proxy(events_df, logger):
    logger.info("[CoF] Calculating Expected Repair Times (MTTR)...")
    
    if 'Kesinti_Suresi_Dakika' not in events_df.columns:
        if 'Ariza_Bitis_Zamani' in events_df.columns and 'Ariza_Baslangic_Zamani' in events_df.columns:
            events_df['start'] = pd.to_datetime(events_df['Ariza_Baslangic_Zamani'])
            events_df['end'] = pd.to_datetime(events_df['Ariza_Bitis_Zamani'])
            events_df['Kesinti_Suresi_Dakika'] = (events_df['end'] - events_df['start']).dt.total_seconds() / 60
    
    mttr_map = events_df.groupby('Ekipman_Tipi')['Kesinti_Suresi_Dakika'].median().to_dict()
    global_median = events_df['Kesinti_Suresi_Dakika'].median()
    
    logger.info(f"  > Global Median Repair Time: {global_median:.0f} mins")
    return mttr_map, global_median

# ------------------------------------------------------------------------------
# 2. CALCULATE COF (Financial + Operational)
# ------------------------------------------------------------------------------
def calculate_cof(df, mttr_map, global_mttr, logger):
    logger.info("[CoF] Calculating Consequence Scores...")
    
    # --- A. FINANCIAL CoF ---
    # Map cost based on equipment type, default to 5000 if unknown
    df['Cost_Base'] = df['Ekipman_Tipi'].map(COST_MAP).fillna(5000)
    
    # --- VOLTAGE MULTIPLIER LOGIC ---
    df['Voltage_Mult'] = 1.0 # Default to Low Voltage
    
    # Check if we have the specific kV column, if not try to parse 'Gerilim_Seviyesi'
    if 'Gerilim_Seviyesi_kV' not in df.columns:
        if 'Gerilim_Seviyesi' in df.columns:
             # Extract number from string (e.g., "34.5 kV" -> 34.5)
             df['Gerilim_Seviyesi_kV'] = df['Gerilim_Seviyesi'].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
        else:
             # Truly missing: Create dummy column with 0 (Low Voltage)
             df['Gerilim_Seviyesi_kV'] = 0.0
             logger.warning("  [WARN] 'Gerilim_Seviyesi' completely missing in dataframe. Using default 1.0x multiplier.")

    # Apply Logic based on kV
    # Medium Voltage (>1kV) -> 1.5x
    df.loc[df['Gerilim_Seviyesi_kV'] > 1, 'Voltage_Mult'] = 1.5
    # High Voltage (>30kV) -> 2.0x
    df.loc[df['Gerilim_Seviyesi_kV'] > 30, 'Voltage_Mult'] = 2.0
    
    # Calculate Financial Score
    df['CoF_Financial'] = df['Cost_Base'] * df['Voltage_Mult']
    
    # --- B. OPERATIONAL CoF ---
    # Handle customer count
    if 'Musteri_Sayisi' in df.columns:
        df['Musteri_Sayisi'] = df['Musteri_Sayisi'].fillna(0).clip(lower=1)
    else:
        df['Musteri_Sayisi'] = 1 # Fallback
        
    # Map Expected Repair Time
    df['Exp_Duration_Min'] = df['Ekipman_Tipi'].map(mttr_map).fillna(global_mttr)
    
    # Environmental Factor
    df['Env_Factor'] = 1.0
    if 'urban mv' in df.columns:
        df.loc[df['urban mv'] > 0, 'Env_Factor'] = 1.5
        
    # Raw Operational Score
    df['CoF_Operational_Raw'] = df['Musteri_Sayisi'] * df['Exp_Duration_Min'] * df['Env_Factor']
    
    # Normalize Scores (Log scale for operational due to high variance)
    scaler = MinMaxScaler(feature_range=(1, 100))
    
    if not df.empty:
        log_op = np.log1p(df[['CoF_Operational_Raw']])
        df['CoF_Operational_Score'] = scaler.fit_transform(log_op)
        df['CoF_Financial_Score'] = scaler.fit_transform(df[['CoF_Financial']])
    else:
        df['CoF_Operational_Score'] = 0
        df['CoF_Financial_Score'] = 0
    
    # Weighted Average: 70% Operational / 30% Financial
    df['CoF_Total_Score'] = (0.7 * df['CoF_Operational_Score']) + (0.3 * df['CoF_Financial_Score'])
    
    logger.info("  > CoF Calculation Complete.")
    return df

# ------------------------------------------------------------------------------
# 3. CALCULATE RISK & CATEGORIZE
# ------------------------------------------------------------------------------
def calculate_risk(df, logger):
    logger.info("[RISK] Calculating Final Risk Scores...")
    
    # Find the correct PoF column (Prefer 12 Month Ensemble)
    target_pof = 'PoF_Ensemble_12Ay'
    if target_pof not in df.columns:
        pofs = [c for c in df.columns if 'PoF' in c and '12Ay' in c]
        target_pof = pofs[0] if pofs else None
    
    if target_pof:
        logger.info(f"  > Using {target_pof} for Risk Calculation")
        df['Risk_Score'] = df[target_pof] * df['CoF_Total_Score']
        
        # Dynamic Percentile Bucketing (Ensures we always have 'Critical' assets)
        try:
            p90 = df['Risk_Score'].quantile(0.90)
            p70 = df['Risk_Score'].quantile(0.70)
            p40 = df['Risk_Score'].quantile(0.40)
        except:
            p90, p70, p40 = 20, 10, 5 # Fallback defaults
        
        def classify_risk(x):
            if x >= p90: return 'Critical'
            if x >= p70: return 'High'
            if x >= p40: return 'Medium'
            return 'Low'
            
        df['Risk_Class'] = df['Risk_Score'].apply(classify_risk)
        
        crit = (df['Risk_Class'] == 'Critical').sum()
        logger.info(f"  > Critical Risk Assets: {crit}")
        
    else:
        logger.warning("[WARN] No PoF column found. Risk Score = 0.")
        df['Risk_Score'] = 0
        df['Risk_Class'] = 'Unknown'
        
    return df

# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------
def main():
    logger = setup_logger()
    
    risk_master_path = os.path.join(OUTPUT_DIR, "risk_equipment_master.csv")
    events_path = os.path.join(DATA_DIR, "ara_ciktilar", "fault_events_clean.csv")
    
    if not os.path.exists(risk_master_path):
        logger.error("Risk Master not found. Run Step 04 first.")
        return
        
    df_risk = pd.read_csv(risk_master_path)
    df_events = pd.read_csv(events_path)
    
    # Normalize ID
    df_risk['cbs_id'] = df_risk['cbs_id'].astype(str).str.lower().str.strip()

    # ==========================================================================
    # 🔴 CRITICAL FIX: RECOVER MISSING DATA FROM MASTER
    # ==========================================================================
    # If 'Gerilim_Seviyesi' or 'Musteri_Sayisi' is missing from previous steps,
    # we go back to the source (equipment_master) and fetch it.
    if 'Gerilim_Seviyesi' not in df_risk.columns or 'Musteri_Sayisi' not in df_risk.columns:
        master_path = INTERMEDIATE_PATHS["equipment_master"]
        if os.path.exists(master_path):
            logger.info("[FIX] Recovering missing columns (Voltage/Customer) from Master File...")
            
            # Select columns to recover
            cols_to_load = ['cbs_id']
            if 'Gerilim_Seviyesi' not in df_risk.columns: cols_to_load.append('Gerilim_Seviyesi')
            if 'Musteri_Sayisi' not in df_risk.columns: cols_to_load.append('Musteri_Sayisi')
            
            # Load Master
            meta = pd.read_csv(master_path, usecols=cols_to_load)
            meta['cbs_id'] = meta['cbs_id'].astype(str).str.lower().str.strip()
            
            # Merge
            df_risk = df_risk.merge(meta, on='cbs_id', how='left')
            logger.info(f"  > Recovered columns: {cols_to_load[1:]}")
    # ==========================================================================
    
    # 2. Calc MTTR
    mttr_map, global_mttr = calculate_mttr_proxy(df_events, logger)
    
    # 3. Calc CoF
    df_risk = calculate_cof(df_risk, mttr_map, global_mttr, logger)
    
    # 4. Calc Risk
    df_risk = calculate_risk(df_risk, logger)
    
    # 5. Save
    df_risk.to_csv(risk_master_path, index=False, encoding="utf-8-sig")
    logger.info(f"[SUCCESS] Risk Master Updated: {risk_master_path}")

if __name__ == "__main__":
    main()