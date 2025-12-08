"""
01_data_processing.py  (PoF3)

Purpose:
- Load fault + healthy equipment data
- Enforce a strict data contract (no fuzzy column guessing)
- Clean timestamps, durations, IDs, equipment classes
- Build:
    * fault_events_clean: fault-level table
    * equipment_master: one row per equipment (cbs_id)
    * survival_base: time-to-event table for Cox / RSF

Notes:
- Only `cbs_id` is used as equipment ID (no fallbacks in faults)
- Healthy data may still use `ID` → we map to `cbs_id` with a warning
- Raw "duration time" is assumed to be milliseconds and converted to minutes
- Outputs are internal/technical; Turkish-friendly naming can happen at PoF/CoF layer
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

# Make console UTF-8 safe on Windows
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ------------------------------------------------------------------------------------
# CONFIG IMPORT WITH SAFE FALLBACKS
# ------------------------------------------------------------------------------------

try:
    from config.config import (
        ANALYSIS_DATE,          # pd.Timestamp or datetime.date
        DATA_PATHS,             # dict with "fault_data", "healthy_data"
        INTERMEDIATE_PATHS,     # dict with output file paths
        MIN_EQUIPMENT_PER_CLASS # int threshold, e.g. 30
    )
except ImportError:
    # Minimal defaults so the script is runnable while config is being designed
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
# LOGGING SETUP
# ------------------------------------------------------------------------------------

def setup_logger(step_name: str) -> logging.Logger:
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"{step_name}_{ts}.log")

    logger = logging.getLogger(step_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # File handler
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter("%(message)s")
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    logger.info("=" * 80)
    logger.info(f"{step_name} - PoF3 Data Processing")
    logger.info("=" * 80)
    logger.info(f"Analysis date: {ANALYSIS_DATE}")
    logger.info("")
    return logger


# ------------------------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------------------------

def parse_date(series: pd.Series, logger: logging.Logger, name: str) -> pd.Series:
    s = series.apply(parse_date_safely)
    missing = s.isna().sum()
    if missing > 0:
        logger.warning(f"[WARN] {name}: {missing} records could not be parsed as dates and will be dropped later.")
    return s


def convert_duration_minutes(series: pd.Series, logger: logging.Logger) -> pd.Series:
    """
    Convert raw 'duration time' values to minutes.
    - If median > 10,000 → assume milliseconds and divide by 60,000
    - Else → assume already minutes
    """
    s = pd.to_numeric(series, errors="coerce")
    med = s.median()
    logger.info(f"[INFO] Raw duration median (as given): {med:.2f}")

    if med > 10000:
        logger.info("[INFO] Detected millisecond-scale durations. Converting to minutes (value / 60000).")
        s = s / 60000.0
    else:
        logger.info("[INFO] Using duration as minutes (no scale conversion).")

    return s


def extract_equipment_type_from_sebeke_unsuru(series: pd.Series, logger: logging.Logger) -> pd.Series:
    """
    'Şebeke Unsuru' values like 'Ayırıcı Arızaları' → 'Ayırıcı'
    """
    s = series.astype(str).str.strip()
    s_clean = s.str.replace(" Arızaları", "", regex=False).str.strip()
    logger.info("[INFO] Extracted equipment types from 'Şebeke Unsuru'.")
    return s_clean


def apply_turkish_labels(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """
    Apply Turkish column labels to output tables for customer-facing deliverables.
    """
    # Define Turkish column mappings for each table
    column_mappings = {
        "fault_events": {
            "cbs_id": "CBS_ID",
            "Ekipman_Tipi": "Ekipman_Tipi",
            "Kurulum_Tarihi": "Kurulum_Tarihi",
            "Ariza_Baslangic_Zamani": "Ariza_Baslangic_Zamani",
            "Ariza_Bitis_Zamani": "Ariza_Bitis_Zamani",
            "Kesinti_Suresi_Dakika": "Kesinti_Suresi_Dakika",
            "Ariza_Nedeni": "Ariza_Nedeni",
        },
        "equipment_master": {
            "cbs_id": "CBS_ID",
            "Kurulum_Tarihi": "Kurulum_Tarihi",
            "Ekipman_Tipi": "Ekipman_Tipi",
            "Fault_Count": "Toplam_Ariza_Sayisi",
            "Ilk_Ariza_Tarihi": "Ilk_Ariza_Tarihi",
            "Ekipman_Yasi_Gun": "Ekipman_Yasi_Gun",
            "Has_Ariza_Gecmisi": "Ariza_Gecmisine_Sahip",
            "Has_Failed": "Ariza_Yapti_Mi",
        },
        "survival_base": {
            "cbs_id": "CBS_ID",
            "Ekipman_Tipi": "Ekipman_Tipi",
            "Kurulum_Tarihi": "Kurulum_Tarihi",
            "Ilk_Ariza_Tarihi": "Ilk_Ariza_Tarihi",
            "event": "Olay",
            "duration_days": "Sure_Gun",
            "Has_Failed": "Ariza_Yapti_Mi",
        }
    }
    
    # Get the appropriate mapping based on table name
    if table_name in column_mappings:
        # Create reverse mapping since we want to rename columns
        rename_dict = column_mappings[table_name]
        # Only rename columns that actually exist in the dataframe
        rename_dict = {k: v for k, v in rename_dict.items() if k in df.columns}
        if rename_dict:
            df_renamed = df.rename(columns=rename_dict)
            return df_renamed
    return df


# ------------------------------------------------------------------------------------
# LOADERS
# ------------------------------------------------------------------------------------

def load_fault_data(logger: logging.Logger) -> pd.DataFrame:
    path = DATA_PATHS["fault_data"]
    logger.info(f"[STEP] Loading fault data from: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fault data file not found: {path}")

    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]

    required_cols = [
        "cbs_id",
        "Şebeke Unsuru",
        "Sebekeye_Baglanma_Tarihi",
        "started at",
        "ended at",
        "duration time",
        "cause code",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"FATAL: Missing required fault columns: {missing}")

    logger.info(f"[OK] Loaded fault data: {len(df):,} rows, {len(df.columns)} columns")

    # Keep only rows with cbs_id
    df = df[df["cbs_id"].notna()].copy()
    logger.info(f"[OK] Fault records with cbs_id: {len(df):,}")

    # Rename / create internal columns
    df.rename(
        columns={
            "Sebekeye_Baglanma_Tarihi": "Kurulum_Tarihi",
            "Şebeke Unsuru": "Ekipman_Tipi",
            "duration time": "Süre_Ham",
            "cause code": "Ariza_Nedeni",
        },
        inplace=True,
    )

    # Parse dates
    df["Kurulum_Tarihi"] = parse_date(df["Kurulum_Tarihi"], logger, "Kurulum_Tarihi")
    df["started at"] = parse_date(df["started at"], logger, "started at")
    df["ended at"] = parse_date(df["ended at"], logger, "ended at")

    # Convert durations
    df["Süre_Dakika"] = convert_duration_minutes(df["Süre_Ham"], logger)

    # Drop records with invalid essential fields
    before = len(df)
    df = df[
        df["Kurulum_Tarihi"].notna()
        & df["started at"].notna()
        & df["Süre_Dakika"].notna()
    ].copy()
    dropped = before - len(df)
    if dropped > 0:
        logger.warning(f"[WARN] Dropped {dropped:,} fault records with missing Kurulum_Tarihi / started at / duration.")

    # Extract clean equipment type
    df["Ekipman_Tipi"] = extract_equipment_type_from_sebeke_unsuru(df["Ekipman_Tipi"], logger)

    logger.info(f"[OK] Final fault data for processing: {len(df):,} rows.")
    return df


def load_healthy_data(logger: logging.Logger) -> pd.DataFrame:
    path = DATA_PATHS["healthy_data"]
    logger.info(f"[STEP] Loading healthy equipment data from: {path}")
    if not os.path.exists(path):
        logger.warning(f"[WARN] Healthy equipment data file not found: {path}")
        return pd.DataFrame()

    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]

    # Handle ID → cbs_id transition with warning
    if "cbs_id" in df.columns:
        logger.info("[OK] Found 'cbs_id' in healthy data.")
    elif "ID" in df.columns:
        logger.warning("[WARN] Healthy data uses 'ID' instead of 'cbs_id' – mapping 'ID' → 'cbs_id' (DEPRECATED, fix at source).")
        df.rename(columns={"ID": "cbs_id"}, inplace=True)
    else:
        raise ValueError("FATAL: Healthy equipment data must contain 'cbs_id' (or temporary 'ID').")

    required_cols = [
        "cbs_id",
        "Sebekeye_Baglanma_Tarihi",
        "Şebeke Unsuru",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"FATAL: Missing required healthy equipment columns: {missing}")

    # Rename to internal
    df.rename(
        columns={
            "Sebekeye_Baglanma_Tarihi": "Kurulum_Tarihi",
            "Şebeke Unsuru": "Ekipman_Tipi",
        },
        inplace=True,
    )

    df["Kurulum_Tarihi"] = parse_date(df["Kurulum_Tarihi"], logger, "Kurulum_Tarihi")
    df["Ekipman_Tipi"] = extract_equipment_type_from_sebeke_unsuru(df["Ekipman_Tipi"], logger)

    before = len(df)
    df = df[df["Kurulum_Tarihi"].notna() & df["cbs_id"].notna()].copy()
    dropped = before - len(df)
    if dropped > 0:
        logger.warning(f"[WARN] Dropped {dropped:,} healthy records with missing Kurulum_Tarihi or cbs_id.")

    logger.info(f"[OK] Final healthy equipment records: {len(df):,}")
    return df


# ------------------------------------------------------------------------------------
# CORE PROCESSING
# ------------------------------------------------------------------------------------

def build_fault_events_table(df_fault: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Returns a fault-level table with clean columns for downstream steps.
    """
    cols = [
        "cbs_id",
        "Ekipman_Tipi",
        "Kurulum_Tarihi",
        "started at",
        "ended at",
        "Süre_Dakika",
        "Ariza_Nedeni",
    ]
    events = df_fault[cols].copy()
    events.rename(
        columns={
            "started at": "Ariza_Baslangic_Zamani",
            "ended at": "Ariza_Bitis_Zamani",
            "Süre_Dakika": "Kesinti_Suresi_Dakika",
        },
        inplace=True,
    )
    logger.info(f"[OK] Fault events table created: {len(events):,} records.")
    return events


def build_equipment_master(df_fault: pd.DataFrame,
                           df_healthy: pd.DataFrame,
                           logger: logging.Logger) -> pd.DataFrame:
    """
    One row per equipment (cbs_id)
    - install date
    - equipment type
    - has_failure_history
    - equipment_age_days
    """
    logger.info("[STEP] Building equipment master table.")

    # From fault data
    if not df_fault.empty:
        fault_equipment = (
            df_fault
            .groupby("cbs_id")
            .agg(
                Kurulum_Tarihi=("Kurulum_Tarihi", "min"),
                Ekipman_Tipi=("Ekipman_Tipi", lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]),
                Fault_Count=("cbs_id", "size"),
                Ilk_Ariza_Tarihi=("started at", "min"),
            )
            .reset_index()
        )
    else:
        fault_equipment = pd.DataFrame(columns=[
            "cbs_id", "Kurulum_Tarihi", "Ekipman_Tipi", "Fault_Count", "Ilk_Ariza_Tarihi"
        ])

    # From healthy data
    if not df_healthy.empty:
        healthy_equipment = (
            df_healthy
            .groupby("cbs_id")
            .agg(
                Kurulum_Tarihi=("Kurulum_Tarihi", "min"),
                Ekipman_Tipi=("Ekipman_Tipi", lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]),
            )
            .reset_index()
        )
        healthy_equipment["Fault_Count"] = 0
        healthy_equipment["Ilk_Ariza_Tarihi"] = pd.NaT
    else:
        healthy_equipment = pd.DataFrame(columns=fault_equipment.columns)

    # Outer merge
    all_eq = pd.concat([fault_equipment, healthy_equipment], ignore_index=True)
    all_eq = (
        all_eq
        .sort_values(["cbs_id", "Kurulum_Tarihi"])
        .drop_duplicates(subset=["cbs_id"], keep="first")
        .reset_index(drop=True)
    )

    logger.info(f"[OK] Equipment master initial size: {len(all_eq):,}")

    # Age
    all_eq["Ekipman_Yasi_Gun"] = (pd.to_datetime(ANALYSIS_DATE) - all_eq["Kurulum_Tarihi"]).dt.days
    all_eq["Ekipman_Yasi_Gun"] = all_eq["Ekipman_Yasi_Gun"].clip(lower=0)

    # Failure history flag
    all_eq["Has_Ariza_Gecmisi"] = (all_eq["Fault_Count"] > 0).astype(int)
    
    # First failure indicator
    all_eq["Has_Failed"] = all_eq["Ilk_Ariza_Tarihi"].notna().astype(int)

    # Handle rare equipment types
    counts = all_eq["Ekipman_Tipi"].value_counts()
    rare = counts[counts < MIN_EQUIPMENT_PER_CLASS].index.tolist()
    if rare:
        logger.info(f"[INFO] Grouping rare equipment types into 'Other': {rare}")
        all_eq.loc[all_eq["Ekipman_Tipi"].isin(rare), "Ekipman_Tipi"] = "Other"

    logger.info(f"[OK] Equipment master finalized: {len(all_eq):,} unique equipment.")
    return all_eq


def build_survival_base(equipment_master: pd.DataFrame,
                        fault_events: pd.DataFrame,
                        logger: logging.Logger) -> pd.DataFrame:
    """
    Build survival_base table:
    - cbs_id
    - Ekipman_Tipi
    - Kurulum_Tarihi
    - Ilk_Ariza_Tarihi
    - event (1=failed, 0=censored)
    - duration_days
    """
    logger.info("[STEP] Building survival base table.")

    # Make a copy to avoid modifying the original
    eq = equipment_master.copy()
    
    # First failure per equipment - update the Ilk_Ariza_Tarihi column
    if not fault_events.empty:
        first_fail = (
            fault_events
            .groupby("cbs_id")["Ariza_Baslangic_Zamani"]
            .min()
            .rename("Ilk_Ariza_Tarihi_New")
        )
        # Update the existing Ilk_Ariza_Tarihi with more recent data from fault events
        eq = eq.merge(first_fail, on="cbs_id", how="left")
        # Use the more specific first failure date if available, otherwise keep existing
        eq["Ilk_Ariza_Tarihi"] = eq["Ilk_Ariza_Tarihi_New"].fillna(eq["Ilk_Ariza_Tarihi"])
        eq = eq.drop(columns=["Ilk_Ariza_Tarihi_New"])
    # If fault_events is empty, we keep the existing Ilk_Ariza_Tarihi column as is

    eq["event"] = eq["Ilk_Ariza_Tarihi"].notna().astype(int)

    analysis_dt = pd.to_datetime(ANALYSIS_DATE)

    # Duration:
    #   If failed → install → first failure
    #   Else      → install → analysis date
    eq["duration_days"] = np.where(
        eq["event"] == 1,
        (eq["Ilk_Ariza_Tarihi"] - eq["Kurulum_Tarihi"]).dt.days,
        (analysis_dt - eq["Kurulum_Tarihi"]).dt.days,
    )

    before = len(eq)
    eq = eq[eq["duration_days"] > 0].copy()
    dropped = before - len(eq)
    if dropped > 0:
        logger.warning(f"[WARN] Dropped {dropped:,} equipment with non-positive duration_days.")

    # Cap extremely long durations (e.g., > 60 years)
    max_days = 60 * 365
    too_long = (eq["duration_days"] > max_days).sum()
    if too_long > 0:
        logger.warning(f"[WARN] Capping {too_long:,} equipment with duration > {max_days} days.")
        eq["duration_days"] = eq["duration_days"].clip(upper=max_days)

    logger.info(f"[OK] Survival base size: {len(eq):,}")
    return eq


# ------------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------------

def main():
    logger = setup_logger(STEP_NAME)
    try:
        logger.info("[STEP] 1. Loading data")
        df_fault = load_fault_data(logger)
        df_healthy = load_healthy_data(logger)

        logger.info("")
        logger.info("[STEP] 2. Building fault events table")
        fault_events = build_fault_events_table(df_fault, logger)

        logger.info("")
        logger.info("[STEP] 3. Building equipment master")
        equipment_master = build_equipment_master(df_fault, df_healthy, logger)

        logger.info("")
        logger.info("[STEP] 4. Building survival base")
        survival_base = build_survival_base(equipment_master, fault_events, logger)

        # Ensure intermediate directory exists
        os.makedirs(os.path.dirname(INTERMEDIATE_PATHS["fault_events_clean"]), exist_ok=True)

        logger.info("")
        logger.info("[STEP] 5. Saving intermediate outputs")
        
        # Create column normalization report
        columns_normalized_path = "data/intermediate/columns_normalized.txt"
        os.makedirs(os.path.dirname(columns_normalized_path), exist_ok=True)
        with open(columns_normalized_path, 'w', encoding='utf-8') as f:
            f.write("COLUMN NORMALIZATION REPORT\n")
            f.write("="*50 + "\n")
            f.write(f"Fault Events Columns ({len(fault_events.columns)}):\n")
            for col in fault_events.columns:
                f.write(f"  - {col}\n")
            f.write(f"\nEquipment Master Columns ({len(equipment_master.columns)}):\n")
            for col in equipment_master.columns:
                f.write(f"  - {col}\n")
            f.write(f"\nSurvival Base Columns ({len(survival_base.columns)}):\n")
            for col in survival_base.columns:
                f.write(f"  - {col}\n")
        logger.info(f"[OK] Saved column normalization report to: {columns_normalized_path}")

        # Save data types schema
        import json
        schema_path = "data/intermediate/schema.json"
        schema = {
            "fault_events_clean": {col: str(fault_events[col].dtype) for col in fault_events.columns},
            "equipment_master": {col: str(equipment_master[col].dtype) for col in equipment_master.columns},
            "survival_base": {col: str(survival_base[col].dtype) for col in survival_base.columns},
        }
        with open(schema_path, 'w', encoding='utf-8') as f:
            json.dump(schema, f, indent=2, ensure_ascii=False)
        logger.info(f"[OK] Saved schema to: {schema_path}")

        # Apply Turkish labels to output dataframes
        fault_events_turkish = apply_turkish_labels(fault_events, "fault_events")
        equipment_master_turkish = apply_turkish_labels(equipment_master, "equipment_master")
        survival_base_turkish = apply_turkish_labels(survival_base, "survival_base")

        fault_events_turkish.to_csv(
            INTERMEDIATE_PATHS["fault_events_clean"],
            index=False,
            encoding="utf-8-sig",
        )
        logger.info(f"[OK] Saved fault events to: {INTERMEDIATE_PATHS['fault_events_clean']}")

        if not df_healthy.empty:
            df_healthy.to_csv(
                INTERMEDIATE_PATHS["healthy_equipment_clean"],
                index=False,
                encoding="utf-8-sig",
            )
            logger.info(f"[OK] Saved healthy equipment to: {INTERMEDIATE_PATHS['healthy_equipment_clean']}")

        equipment_master_turkish.to_csv(
            INTERMEDIATE_PATHS["equipment_master"],
            index=False,
            encoding="utf-8-sig",
        )
        logger.info(f"[OK] Saved equipment master to: {INTERMEDIATE_PATHS['equipment_master']}")

        survival_base_turkish.to_csv(
            INTERMEDIATE_PATHS["survival_base"],
            index=False,
            encoding="utf-8-sig",
        )
        logger.info(f"[OK] Saved survival base to: {INTERMEDIATE_PATHS['survival_base']}")

        logger.info("")
        logger.info("[SUCCESS] 01_data_processing completed successfully.")
        logger.info("=" * 80)

    except Exception as e:
        logger.exception(f"[FATAL] 01_data_processing failed: {e}")
        raise


if __name__ == "__main__":
    main()
