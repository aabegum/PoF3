"""
01_data_processing.py  (PoF3 Final Strict Version)

Purpose:
- Load raw fault + healthy data
- Enforce strict data contract (lowercase cbs_id only)
- Parse dates, durations, equipment types
- Create:
    fault_events_clean.csv
    equipment_master.csv
    survival_base.csv
Internal outputs use strict lowercase column names.
Customer-facing Turkish outputs are produced separately.
"""

import os
import sys
import logging
from datetime import datetime
import numpy as np
import pandas as pd

# UTF-8 console safety
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Safe date parser
from utils.date_parser import parse_date_safely

# ------------------------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------------------------

try:
    from config.config import (
        ANALYSIS_DATE,
        DATA_PATHS,
        INTERMEDIATE_PATHS,
        MIN_EQUIPMENT_PER_CLASS,
    )
except ImportError:
    ANALYSIS_DATE = pd.Timestamp.today().normalize()
    DATA_PATHS = {
        "fault_data": "data/inputs/fault_merged_data.xlsx",
        "healthy_data": "data/inputs/health_merged_data.xlsx",
    }
    INTERMEDIATE_PATHS = {
        "fault_events_clean": "data/intermediate/fault_events_clean.csv",
        "healthy_equipment_clean": "data/intermediate/healthy_equipment_clean.csv",
        "equipment_master": "data/intermediate/equipment_master.csv",
        "survival_base": "data/intermediate/survival_base.csv",
    }
    MIN_EQUIPMENT_PER_CLASS = 30


LOG_DIR = "logs"
STEP_NAME = "01_data_processing"


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
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    logger.info("=" * 80)
    logger.info("01_data_processing - PoF3 Data Processing")
    logger.info("=" * 80)
    logger.info(f"Analysis date: {ANALYSIS_DATE}")
    logger.info("")
    return logger


# ------------------------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------------------------

def convert_duration_minutes(series: pd.Series, logger):
    s = pd.to_numeric(series, errors="coerce")
    med = s.median()

    logger.info(f"[INFO] Raw duration median: {med:.2f}")

    if med > 10000:
        logger.info("[INFO] Interpreting durations as milliseconds → converting to minutes.")
        return s / 60000.0
    return s


def extract_equipment_type(series, logger):
    s = series.astype(str).str.strip()
    cleaned = s.str.replace(" Arızaları", "", regex=False).str.strip()
    return cleaned


# ------------------------------------------------------------------------------------
# LOADERS
# ------------------------------------------------------------------------------------

def load_fault_data(logger):
    path = DATA_PATHS["fault_data"]
    logger.info(f"[STEP] Loading fault data from: {path}")

    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]

    required = [
        "cbs_id",
        "Şebeke Unsuru",
        "Sebekeye_Baglanma_Tarihi",
        "started at",
        "ended at",
        "duration time",
        "cause code",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"FATAL: Missing fault columns: {missing}")

    df = df[df["cbs_id"].notna()].copy()

    df.rename(columns={
        "Sebekeye_Baglanma_Tarihi": "Kurulum_Tarihi",
        "Şebeke Unsuru": "Ekipman_Tipi",
        "duration time": "Süre_Ham",
        "cause code": "Ariza_Nedeni",
    }, inplace=True)

    df["Kurulum_Tarihi"] = df["Kurulum_Tarihi"].apply(parse_date_safely)
    df["started at"] = df["started at"].apply(parse_date_safely)
    df["ended at"] = df["ended at"].apply(parse_date_safely)

    df["Süre_Dakika"] = convert_duration_minutes(df["Süre_Ham"], logger)
    df["Ekipman_Tipi"] = extract_equipment_type(df["Ekipman_Tipi"], logger)

    before = len(df)
    df = df[
        df["Kurulum_Tarihi"].notna()
        & df["started at"].notna()
        & df["Süre_Dakika"].notna()
    ]
    dropped = before - len(df)
    if dropped:
        logger.warning(f"[WARN] Dropped {dropped} fault rows with invalid dates/durations.")

    return df


def load_healthy_data(logger):
    path = DATA_PATHS["healthy_data"]
    logger.info(f"[STEP] Loading healthy equipment data from: {path}")

    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]

    # Handle cbs_id
    if "cbs_id" in df.columns:
        pass
    elif "ID" in df.columns:
        logger.warning("[WARN] Healthy data uses 'ID'. Mapping to cbs_id.")
        df.rename(columns={"ID": "cbs_id"}, inplace=True)
    else:
        raise ValueError("FATAL: Healthy equipment requires 'cbs_id' (or temporary 'ID').")

    required = ["cbs_id", "Sebekeye_Baglanma_Tarihi", "Şebeke Unsuru"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"FATAL: Missing healthy columns: {missing}")

    df.rename(columns={
        "Sebekeye_Baglanma_Tarihi": "Kurulum_Tarihi",
        "Şebeke Unsuru": "Ekipman_Tipi",
    }, inplace=True)

    df["Kurulum_Tarihi"] = df["Kurulum_Tarihi"].apply(parse_date_safely)
    df["Ekipman_Tipi"] = extract_equipment_type(df["Ekipman_Tipi"], logger)

    df = df[df["Kurulum_Tarihi"].notna() & df["cbs_id"].notna()]

    return df


# ------------------------------------------------------------------------------------
# BUILD TABLES
# ------------------------------------------------------------------------------------

def build_fault_events(df_fault):
    return df_fault[[
        "cbs_id",
        "Ekipman_Tipi",
        "Kurulum_Tarihi",
        "started at",
        "ended at",
        "Süre_Dakika",
        "Ariza_Nedeni",
    ]].rename(columns={
        "started at": "Ariza_Baslangic_Zamani",
        "ended at": "Ariza_Bitis_Zamani",
        "Süre_Dakika": "Kesinti_Suresi_Dakika",
    })


def build_equipment_master(df_fault, df_healthy, logger):
    # Fault equipment
    fault_part = (
        df_fault
        .groupby("cbs_id")
        .agg(
            Kurulum_Tarihi=("Kurulum_Tarihi", "min"),
            Ekipman_Tipi=("Ekipman_Tipi", lambda x: x.mode().iloc[0]),
            Fault_Count=("cbs_id", "size"),
            Ilk_Ariza_Tarihi=("started at", "min"),
        )
        .reset_index()
    )

    # Healthy only equipment
    healthy_part = df_healthy.groupby("cbs_id").agg(
        Kurulum_Tarihi=("Kurulum_Tarihi", "min"),
        Ekipman_Tipi=("Ekipman_Tipi", lambda x: x.mode().iloc[0]),
    ).reset_index()
    healthy_part["Fault_Count"] = 0
    healthy_part["Ilk_Ariza_Tarihi"] = pd.NaT

    all_eq = pd.concat([fault_part, healthy_part], ignore_index=True)
    all_eq = all_eq.sort_values("cbs_id").drop_duplicates("cbs_id")

    all_eq["Ekipman_Yasi_Gun"] = (
        pd.to_datetime(ANALYSIS_DATE) - all_eq["Kurulum_Tarihi"]
    ).dt.days.clip(lower=0)

    all_eq["Has_Ariza_Gecmisi"] = (all_eq["Fault_Count"] > 0).astype(int)
    all_eq["Has_Failed"] = all_eq["Ilk_Ariza_Tarihi"].notna().astype(int)

    # Rare class grouping
    counts = all_eq["Ekipman_Tipi"].value_counts()
    rare = counts[counts < MIN_EQUIPMENT_PER_CLASS].index.tolist()
    if rare:
        logger.info(f"[INFO] Grouping rare equipment types: {rare}")
        all_eq.loc[all_eq["Ekipman_Tipi"].isin(rare), "Ekipman_Tipi"] = "Other"

    return all_eq


def build_survival_base(eq, events, logger):
    # First-failure update
    first_fail = events.groupby("cbs_id")["Ariza_Baslangic_Zamani"].min()
    eq = eq.merge(first_fail.rename("Ilk_Ariza"), on="cbs_id", how="left")

    eq["Ilk_Ariza_Tarihi"] = eq["Ilk_Ariza"].fillna(eq["Ilk_Ariza_Tarihi"])
    eq.drop(columns=["Ilk_Ariza"], inplace=True)

    analysis_dt = pd.to_datetime(ANALYSIS_DATE)
    eq["event"] = eq["Ilk_Ariza_Tarihi"].notna().astype(int)

    eq["duration_days"] = np.where(
        eq["event"] == 1,
        (eq["Ilk_Ariza_Tarihi"] - eq["Kurulum_Tarihi"]).dt.days,
        (analysis_dt - eq["Kurulum_Tarihi"]).dt.days,
    )

    eq = eq[eq["duration_days"] > 0]

    too_long = (eq["duration_days"] > 21900).sum()
    if too_long:
        logger.warning(f"[WARN] Capping {too_long} long-duration equipment (>21900 days).")
        eq["duration_days"] = eq["duration_days"].clip(upper=21900)

    return eq


# ------------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------------

def main():
    logger = setup_logger(STEP_NAME)

    try:
        df_fault = load_fault_data(logger)
        df_healthy = load_healthy_data(logger)

        fault_events = build_fault_events(df_fault)
        equipment_master = build_equipment_master(df_fault, df_healthy, logger)
        survival_base = build_survival_base(equipment_master, fault_events, logger)

        os.makedirs("data/intermediate", exist_ok=True)

        # Internal lowercase outputs
        fault_events.to_csv(INTERMEDIATE_PATHS["fault_events_clean"], index=False, encoding="utf-8-sig")
        df_healthy.to_csv(INTERMEDIATE_PATHS["healthy_equipment_clean"], index=False, encoding="utf-8-sig")
        equipment_master.to_csv(INTERMEDIATE_PATHS["equipment_master"], index=False, encoding="utf-8-sig")
        survival_base.to_csv(INTERMEDIATE_PATHS["survival_base"], index=False, encoding="utf-8-sig")

        logger.info("[SUCCESS] 01_data_processing completed.")
    except Exception as e:
        logger.exception(f"[FATAL] 01_data_processing failed: {e}")
        raise


if __name__ == "__main__":
    main()
