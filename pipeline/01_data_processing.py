# ================================================================
# PoF3 - STEP 01: DATA PROCESSING
# Clean, strict, IEEE-ready preparation of fault + healthy dataset
# ================================================================

import os
import pandas as pd
from datetime import datetime
from utils.logger import get_logger
from config.config import DATA_PATH
import sys
sys.stdout.reconfigure(encoding="utf-8")

logger = get_logger("01_data_processing")


# ================================================================
#  CONFIGURATION
# ================================================================
ANALYSIS_DATE = datetime.today()       # Dynamic analysis date
MIN_CLASS_SAMPLES = 30                 # Rare class threshold


# ================================================================
#  STRICT COLUMN CONTRACT
# ================================================================
# Mapping Turkish column names to expected English names
TURKISH_TO_ENGLISH_MAP = {
    'Ekipman Sınıfı': 'equipment_class',
    'TESIS_TARIHI': 'install_date',
    'Sebekeye_Baglanma_Tarihi': 'install_date',
    'ID': 'cbs_id',
    'Equipment_Type': 'equipment_class'  # English version for healthy data
}

REQUIRED_FAULT_COLUMNS = {
    "cbs_id",
    "equipment_class",
    "started at",
    "ended at",
    "install_date",
    "duration time",
    "cause code"
}

REQUIRED_HEALTHY_COLUMNS = {
    "cbs_id",
    "equipment_class",
    "install_date"
}


# ------------------------------------------------------------
# 1. READ FAULT DATA WITH STRICT VALIDATION
# ------------------------------------------------------------
def load_fault_data():
    fault_path = os.path.join(DATA_PATH, "inputs", "fault_merged_data.xlsx")

    logger.info(f"Loading fault data: {fault_path}")
    df = pd.read_excel(fault_path, engine="openpyxl")
    
    # Rename Turkish columns to English equivalents only if they exist
    rename_dict = {}
    for turk_col, eng_col in TURKISH_TO_ENGLISH_MAP.items():
        if turk_col in df.columns:
            rename_dict[turk_col] = eng_col
    
    df.rename(columns=rename_dict, inplace=True)

    # STRICT CHECK
    missing = REQUIRED_FAULT_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"FATAL: Missing required fault columns: {missing}")

    logger.info(f"Fault records loaded: {len(df):,}")

    return df


# ------------------------------------------------------------
# 2. READ HEALTHY EQUIPMENT DATA STRICTLY
# ------------------------------------------------------------
def load_healthy_data():
    healthy_path = os.path.join(DATA_PATH, "inputs", "health_merged_data.xlsx")

    logger.info(f"Loading healthy data: {healthy_path}")
    df = pd.read_excel(healthy_path, engine="openpyxl")
    
    # Rename Turkish columns to English equivalents only if they exist
    rename_dict = {}
    for turk_col, eng_col in TURKISH_TO_ENGLISH_MAP.items():
        if turk_col in df.columns:
            rename_dict[turk_col] = eng_col
    
    df.rename(columns=rename_dict, inplace=True)

    missing = REQUIRED_HEALTHY_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"FATAL: Healthy equipment missing columns: {missing}")

    logger.info(f"Healthy equipment loaded: {len(df):,}")
    return df


# ------------------------------------------------------------
# 3. FIX DATE FORMATS
# ------------------------------------------------------------
def parse_dates(df):
    date_cols = ["started at", "ended at", "install_date"]

    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")
        if df[col].isna().sum() > 0:
            logger.warning(f"Date parsing failed for column {col} — coerced to NaT")

    return df


# ------------------------------------------------------------
# 4. FIX DURATION TIME (ms → minutes)
# ------------------------------------------------------------
def convert_duration(df):
    if "duration time" not in df.columns:
        return df

    median_val = df["duration time"].median()

    if median_val > 10000:  # Assume milliseconds
        logger.info("Detected 'duration time' in milliseconds → converting to minutes")
        df["duration_minutes"] = df["duration time"] / 60000.0
    else:
        logger.info("Detected 'duration time' already in minutes → keeping as is")
        df["duration_minutes"] = df["duration time"]

    return df


# ------------------------------------------------------------
# 5. CALCULATE AGE
# ------------------------------------------------------------
def compute_equipment_age(df):
    logger.info("Computing equipment age...")

    df["equipment_age_days"] = (ANALYSIS_DATE - df["install_date"]).dt.days
    df["equipment_age_days"] = df["equipment_age_days"].clip(lower=0)

    return df


# ------------------------------------------------------------
# 6. HANDLE RARE CLASSES (<30 samples)
# ------------------------------------------------------------
def normalize_equipment_classes(df):
    logger.info("Normalizing rare equipment classes...")

    class_counts = df["equipment_class"].value_counts()
    rare = class_counts[class_counts < MIN_CLASS_SAMPLES].index.tolist()

    if len(rare) > 0:
        logger.info(f"Grouping rare classes into 'Other': {rare}")
        df.loc[df["equipment_class"].isin(rare), "equipment_class"] = "Other"

    return df


# ------------------------------------------------------------
# 7. MERGE HEALTHY + FAULT
# ------------------------------------------------------------
def merge_fault_and_healthy(fault, healthy):
    logger.info("Merging fault and healthy datasets...")

    fault["event"] = 1
    healthy["event"] = 0

    # Healthy records have no start/end timestamps
    healthy["started at"] = pd.NaT
    healthy["ended at"] = pd.NaT
    healthy["duration_minutes"] = 0

    df = pd.concat([fault, healthy], ignore_index=True)
    logger.info(f"Merged dataset size: {len(df):,}")

    return df


# ------------------------------------------------------------
# 8. SAVE CLEAN OUTPUT
# ------------------------------------------------------------
def save_output(df):
    out_path = os.path.join(DATA_PATH, "intermediate", "clean_data.csv")
    df.to_csv(out_path, index=False, encoding="utf-8")
    logger.info(f"Saved cleaned dataset → {out_path}")


# ================================================================
#  MAIN EXECUTION FUNCTION
# ================================================================
def run():
    logger.info("==============================")
    logger.info("STEP 01: DATA PROCESSING START")
    logger.info("==============================")

    fault = load_fault_data()
    healthy = load_healthy_data()

    fault = parse_dates(fault)
    healthy = parse_dates(healthy)

    fault = convert_duration(fault)
    healthy["duration_minutes"] = 0  # Healthy entries have no outage duration

    fault = compute_equipment_age(fault)
    healthy = compute_equipment_age(healthy)

    fault = normalize_equipment_classes(fault)
    healthy = normalize_equipment_classes(healthy)

    final = merge_fault_and_healthy(fault, healthy)
    save_output(final)

    logger.info("STEP 01 COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    run()
