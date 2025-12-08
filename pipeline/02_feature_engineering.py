"""
02_feature_engineering.py  (PoF3)

Purpose:
- Read:
    * equipment_master (one row per cbs_id)
    * fault_events_clean (fault-level)
- Compute deterministic, domain-driven features:
    * Total failures (lifetime)
    * Days since last failure
    * MTBF (mean time between failures)
    * Chronic risk flag (e.g., any gap <= 90 days)
- Output a clean feature table for PoF and chronic steps.
"""

import os
import sys
import logging
from datetime import datetime

# Add project root to Python path for utils import
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

# Import safe date parser
from utils.date_parser import parse_date_safely

# UTF-8 stdout for Windows
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

try:
    from config.config import (
        ANALYSIS_DATE,
        INTERMEDIATE_PATHS,     # same dict as step 01
        FEATURE_OUTPUT_PATH,    # e.g. "data/intermediate/features_pof3.csv"
        CHRONIC_WINDOW_DAYS,    # e.g. 90
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
# LOGGING
# ------------------------------------------------------------------------------------

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
    logger.info(f"{step_name} - PoF3 Feature Engineering")
    logger.info("=" * 80)
    logger.info(f"Analysis date: {ANALYSIS_DATE}")
    logger.info("")
    return logger


# ------------------------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------------------------

def compute_mtbf_and_chronic(events: pd.DataFrame,
                              logger: logging.Logger) -> pd.DataFrame:
    """
    Compute MTBF (days) and chronic flag for each equipment:
    - MTBF_days: mean time between failures (if >=2 faults)
    - Chronic_90d_Flag: 1 if any gap between consecutive faults <= CHRONIC_WINDOW_DAYS
    """
    logger.info("[STEP] Computing MTBF and chronic flags.")

    # Ensure sorted by time
    events_sorted = events.sort_values(["cbs_id", "Ariza_Baslangic_Zamani"]).copy()

    mtbf_list = []
    chronic_list = []

    for cbs_id, grp in events_sorted.groupby("cbs_id"):
        times = grp["Ariza_Baslangic_Zamani"].dropna().sort_values()
        times = times.values
        if len(times) < 2:
            mtbf_days = np.nan
            chronic_flag = 0
        else:
            diffs = np.diff(times).astype("timedelta64[D]").astype(int)
            mtbf_days = float(np.mean(diffs))
            chronic_flag = int(np.any(diffs <= CHRONIC_WINDOW_DAYS))

        mtbf_list.append((cbs_id, mtbf_days))
        chronic_list.append((cbs_id, chronic_flag))

    mtbf_df = pd.DataFrame(mtbf_list, columns=["cbs_id", "MTBF_Gun"])
    chronic_df = pd.DataFrame(chronic_list, columns=["cbs_id", "Tekrarlayan_Ariza_90g_Flag"])

    logger.info(f"[OK] MTBF computed for {len(mtbf_df):,} equipment.")
    return mtbf_df.merge(chronic_df, on="cbs_id", how="outer")


# ------------------------------------------------------------------------------------
# MAIN FEATURE ENGINEERING
# ------------------------------------------------------------------------------------

def main():
    logger = setup_logger(STEP_NAME)

    try:
        # -------------------------------------------------------------------------
        # Load intermediate inputs
        # -------------------------------------------------------------------------
        eq_path = INTERMEDIATE_PATHS["equipment_master"]
        events_path = INTERMEDIATE_PATHS["fault_events_clean"]

        if not os.path.exists(eq_path):
            raise FileNotFoundError(f"Equipment master not found: {eq_path}")
        if not os.path.exists(events_path):
            logger.warning(f"[WARN] Fault events file not found: {events_path} – will proceed with zero-fault features.")
            events = pd.DataFrame()
        else:
            events = pd.read_csv(events_path, encoding="utf-8-sig")
            events["Ariza_Baslangic_Zamani"] = events["Ariza_Baslangic_Zamani"].apply(parse_date_safely)
            # Normalize CBS_ID to lowercase for internal processing
            events.rename(columns={"CBS_ID": "cbs_id"}, inplace=True)

        equipment = pd.read_csv(eq_path, encoding="utf-8-sig")
        equipment["Kurulum_Tarihi"] = equipment["Kurulum_Tarihi"].apply(parse_date_safely)
        analysis_dt = pd.to_datetime(ANALYSIS_DATE)

        logger.info(f"[OK] Loaded equipment master: {len(equipment):,} rows.")
        logger.info(f"[OK] Loaded fault events: {len(events):,} rows.")

        # -------------------------------------------------------------------------
        # Base feature set from equipment_master
        # Note: Column names are in Turkish as saved by 01_data_processing
        # -------------------------------------------------------------------------
        features = equipment[[
            "CBS_ID",
            "Ekipman_Tipi",
            "Kurulum_Tarihi",
            "Ekipman_Yasi_Gun",
            "Ariza_Gecmisine_Sahip",
            "Toplam_Ariza_Sayisi",
            "Ilk_Ariza_Tarihi",
        ]].copy()

        # Normalize column names for internal processing
        features.rename(columns={
            "CBS_ID": "cbs_id",
            "Ariza_Gecmisine_Sahip": "Has_Ariza_Gecmisi",
            "Toplam_Ariza_Sayisi": "Fault_Count",
        }, inplace=True)

        # Days since last failure
        if not events.empty:
            last_fault = (
                events
                .groupby("cbs_id")["Ariza_Baslangic_Zamani"]
                .max()
                .rename("Son_Ariza_Tarihi")
            )
            features = features.merge(last_fault, on="cbs_id", how="left")
            features["Son_Ariza_Tarihi"] = features["Son_Ariza_Tarihi"].apply(parse_date_safely)
            features["Son_Ariza_Gun_Sayisi"] = (analysis_dt - features["Son_Ariza_Tarihi"]).dt.days
        else:
            features["Son_Ariza_Tarihi"] = pd.NaT
            features["Son_Ariza_Gun_Sayisi"] = np.nan

        # -------------------------------------------------------------------------
        # MTBF + Chronic Flag
        # -------------------------------------------------------------------------
        if not events.empty:
            mtbf_chronic = compute_mtbf_and_chronic(events, logger)
            features = features.merge(mtbf_chronic, on="cbs_id", how="left")
        else:
            features["MTBF_Gun"] = np.nan
            features["Tekrarlayan_Ariza_90g_Flag"] = 0

        # Fill some reasonable defaults
        features["Fault_Count"] = features["Fault_Count"].fillna(0).astype(int)
        features["Has_Ariza_Gecmisi"] = features["Has_Ariza_Gecmisi"].fillna(0).astype(int)
        features["Tekrarlayan_Ariza_90g_Flag"] = features["Tekrarlayan_Ariza_90g_Flag"].fillna(0).astype(int)

        # Sanity logging
        logger.info("")
        logger.info("[SUMMARY] Feature distributions:")
        logger.info(f"  Total equipment: {len(features):,}")
        logger.info(f"  With any fault: {(features['Fault_Count'] > 0).sum():,}")
        logger.info(f"  Chronic_90d_Flag = 1: {features['Tekrarlayan_Ariza_90g_Flag'].sum():,}")

        # -------------------------------------------------------------------------
        # Save feature table
        # -------------------------------------------------------------------------
        os.makedirs(os.path.dirname(FEATURE_OUTPUT_PATH), exist_ok=True)
        features.to_csv(FEATURE_OUTPUT_PATH, index=False, encoding="utf-8-sig")
        logger.info("")
        logger.info(f"[OK] Saved feature table to: {FEATURE_OUTPUT_PATH}")
        logger.info("[SUCCESS] 02_feature_engineering completed successfully.")
        logger.info("=" * 80)

    except Exception as e:
        logger.exception(f"[FATAL] 02_feature_engineering failed: {e}")
        raise


if __name__ == "__main__":
    main()
