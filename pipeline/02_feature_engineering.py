"""
02_feature_engineering.py  (PoF3 Final Strict Version)

Purpose:
- Load equipment_master + fault_events_clean
- Enforce strict lowercase cbs_id
- Compute:
    MTBF
    Chronic flags
    Days since last fault
    Fault count
    Age (from equipment_master)
Outputs features_pof3.csv
"""

import os
import sys
import logging
from datetime import datetime
import numpy as np
import pandas as pd

# UTF-8 console
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.date_parser import parse_date_safely

# ------------------------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------------------------

try:
    from config.config import (
        ANALYSIS_DATE,
        INTERMEDIATE_PATHS,
        FEATURE_OUTPUT_PATH,
        CHRONIC_WINDOW_DAYS,
    )
except ImportError:
    ANALYSIS_DATE = pd.Timestamp.today().normalize()
    INTERMEDIATE_PATHS = {
        "fault_events_clean": "data/intermediate/fault_events_clean.csv",
        "equipment_master": "data/intermediate/equipment_master.csv",
    }
    FEATURE_OUTPUT_PATH = "data/intermediate/features_pof3.csv"
    CHRONIC_WINDOW_DAYS = 90


LOG_DIR = "logs"
STEP_NAME = "02_feature_engineering"


# ------------------------------------------------------------------------------------
# LOGGER
# ------------------------------------------------------------------------------------

def setup_logger(step_name):
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(LOG_DIR, f"{step_name}_{ts}.log")

    logger = logging.getLogger(step_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    logger.info("=" * 80)
    logger.info("02_feature_engineering - PoF3 Feature Engineering")
    logger.info("=" * 80)
    logger.info(f"Analysis date: {ANALYSIS_DATE}")
    logger.info("")
    return logger


# ------------------------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------------------------

def compute_mtbf_and_chronic(events, logger):
    events = events.sort_values(["cbs_id", "Ariza_Baslangic_Zamani"])

    mtbf_records = []
    chronic_records = []

    for cbs_id, grp in events.groupby("cbs_id"):
        times = grp["Ariza_Baslangic_Zamani"].dropna().sort_values().values

        if len(times) < 2:
            mtbf = np.nan
            chronic_flag = 0
        else:
            diffs = np.diff(times).astype("timedelta64[D]").astype(int)
            mtbf = float(np.mean(diffs))
            chronic_flag = int((diffs <= CHRONIC_WINDOW_DAYS).any())

        mtbf_records.append((cbs_id, mtbf))
        chronic_records.append((cbs_id, chronic_flag))

    mtbf_df = pd.DataFrame(mtbf_records, columns=["cbs_id", "MTBF_Gun"])
    chronic_df = pd.DataFrame(chronic_records, columns=["cbs_id", "Kronik_90g_Flag"])

    return mtbf_df.merge(chronic_df, on="cbs_id", how="outer")


# ------------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------------

def main():
    logger = setup_logger(STEP_NAME)

    try:
        eq_path = INTERMEDIATE_PATHS["equipment_master"]
        events_path = INTERMEDIATE_PATHS["fault_events_clean"]

        equipment = pd.read_csv(eq_path, encoding="utf-8-sig")
        events = pd.read_csv(events_path, encoding="utf-8-sig") if os.path.exists(events_path) else pd.DataFrame()

        # enforce strict lowercase schema
        if "cbs_id" not in equipment.columns:
            raise ValueError("FATAL: equipment_master must have column 'cbs_id'. Fix Step 01.")
        if not events.empty and "cbs_id" not in events.columns:
            raise ValueError("FATAL: fault_events_clean must have 'cbs_id'. Fix Step 01.")

        # parse dates
        if not events.empty:
            events["Ariza_Baslangic_Zamani"] = events["Ariza_Baslangic_Zamani"].apply(parse_date_safely)

        equipment["Kurulum_Tarihi"] = equipment["Kurulum_Tarihi"].apply(parse_date_safely)
        analysis_dt = pd.to_datetime(ANALYSIS_DATE)

        # base features
        features = equipment[[
            "cbs_id",
            "Ekipman_Tipi",
            "Kurulum_Tarihi",
            "Ekipman_Yasi_Gun",
            "Has_Ariza_Gecmisi",
            "Fault_Count",
            "Ilk_Ariza_Tarihi",
        ]].copy()

        # last fault
        if not events.empty:
            lastfault = (
                events.groupby("cbs_id")["Ariza_Baslangic_Zamani"]
                .max()
                .rename("Son_Ariza_Tarihi")
            )
            features = features.merge(lastfault, on="cbs_id", how="left")
            features["Son_Ariza_Tarihi"] = features["Son_Ariza_Tarihi"].apply(parse_date_safely)
            features["Son_Ariza_Gun_Sayisi"] = (analysis_dt - features["Son_Ariza_Tarihi"]).dt.days
        else:
            features["Son_Ariza_Tarihi"] = pd.NaT
            features["Son_Ariza_Gun_Sayisi"] = np.nan

        # MTBF + chronic
        if not events.empty:
            mtbf_df = compute_mtbf_and_chronic(events, logger)
            features = features.merge(mtbf_df, on="cbs_id", how="left")
        else:
            features["MTBF_Gun"] = np.nan
            features["Kronik_90g_Flag"] = 0

        # sanitize
        features["Fault_Count"] = features["Fault_Count"].fillna(0).astype(int)
        features["Has_Ariza_Gecmisi"] = features["Has_Ariza_Gecmisi"].fillna(0).astype(int)
        features["Kronik_90g_Flag"] = features["Kronik_90g_Flag"].fillna(0).astype(int)

        # log summary
        logger.info("[SUMMARY]")
        logger.info(f"  Total equipment: {len(features)}")
        logger.info(f"  With any fault: {(features['Fault_Count'] > 0).sum()}")
        logger.info(f"  Chronic flags: {features['Kronik_90g_Flag'].sum()}")

        # save
        os.makedirs(os.path.dirname(FEATURE_OUTPUT_PATH), exist_ok=True)
        features.to_csv(FEATURE_OUTPUT_PATH, index=False, encoding="utf-8-sig")
        logger.info(f"[OK] Saved: {FEATURE_OUTPUT_PATH}")
        logger.info("[SUCCESS] 02_feature_engineering completed.")

    except Exception as e:
        logger.exception(f"[FATAL] 02_feature_engineering failed: {e}")
        raise


if __name__ == "__main__":
    main()
