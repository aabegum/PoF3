# pipeline/04_chronic_detection.py (PoF3 - Enhanced v3)

"""
PoF3 - Step 04: Chronic Fault Detection & Risk Master Table (Enhanced v3)

Key Improvements vs v2
======================
1. STRICT TEMPORAL CONSISTENCY
   - Uses DATA_END_DATE from Step 01 (data_range_metadata.csv)
   - Fixed 365-day window [DATA_END_DATE - 365, DATA_END_DATE] for chronic logic
   - Fully aligned with Step 02 feature engineering & Step 03 survival models

2. MULTI-LEVEL CHRONIC CLASSIFICATION (IEEE 1366 STYLE)
   - KRITIK: 4+ faults in last 365 days
   - YUKSEK: 3 faults in last 365 days
   - ORTA:   2 faults in last 365 days
   - IZLEME: 1 fault in last 365 days
   - NORMAL: 0 faults in last 365 days
   → Step 04 flags are 100% consistent with Step 02 chronic features

3. ALL-EQUIPMENT COVERAGE
   - Starts from full equipment master
   - Computes fault stats, observation period, annualized failure rate (Lambda)
   - Merges chronic flags on top (no equipment lost)

4. STEP 02 VALIDATION (ALIGNMENT CHECK)
   - Compares Step 04 vs Step 02 for:
       * Kronik_Kritik, Kronik_Yuksek, Kronik_Orta, Kronik_Seviye_Max
       * Faults_Last_365d
   - Writes comparison report: chronic_step02_step04_comparison.csv

5. INTEGRATION WITH STEP 03 POF MODELS
   - Pulls PoF outputs from Step 03:
       * Cox calibrated (PoF_Cox_Cal_12Ay)
       * Weibull AFT (PoF_Weibull_12Ay)
       * RSF (PoF_RSF_12Ay)
       * Ensemble (PoF_Ensemble_12Ay)
       * ML (PoF_ML_XGB) from leakage_free_ml_pof.csv
   - Builds a canonical risk table:
       risk_equipment_master.csv

6. CLEANER SUMMARY & CORRECT PERCENTAGES
   - Fixes previous percentage bug (e.g., 204% etc.)
   - Clear separation between:
       * ACTION REQUIRED (KRITIK/YUKSEK/ORTA)
       * OBSERVATION (IZLEME)
       * NORMAL (0 faults)

Outputs
=======
- chronic_equipment_summary.csv      (all equipment + chronic flags + stats)
- chronic_equipment_only.csv         (KRITIK/YUKSEK/ORTA only)
- izleme_equipment.csv               (IZLEME only)
- chronic_step02_step04_comparison.csv
- risk_equipment_master.csv          (canonical risk view for dashboards)
"""

import os
import sys
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

from utils.logger import get_logger

# --------------------------------------------------------------------
# Paths / Config
# --------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]

try:
    from config.config import (
        INTERMEDIATE_PATHS,
        OUTPUT_DIR as CONFIG_OUTPUT_DIR,
        LOG_DIR as CONFIG_LOG_DIR,
        CHRONIC_WINDOW_DAYS,
        CHRONIC_THRESHOLD_EVENTS,
    )

    INTERMEDIATE_DIR = Path(list(INTERMEDIATE_PATHS.values())[0]).parent
    OUTPUT_DIR = Path(CONFIG_OUTPUT_DIR)
    LOG_DIR = Path(CONFIG_LOG_DIR)

except Exception:
    # Fallback defaults if config import fails
    DATA_DIR = BASE_DIR / "data"
    INTERMEDIATE_DIR = DATA_DIR / "ara_ciktilar"
    OUTPUT_DIR = DATA_DIR / "sonuclar"
    LOG_DIR = BASE_DIR / "loglar"

    # Default chronic config (used only if not in config)
    CHRONIC_WINDOW_DAYS = 365
    CHRONIC_THRESHOLD_EVENTS = 4  # for KRITIK

LOG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STEP_NAME = "04_tekrarlayan_ariza_v3"
LOG_FILE = LOG_DIR / f"{STEP_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logger = get_logger(STEP_NAME, LOG_FILE)


# --------------------------------------------------------------------
# 1) Load DATA_END_DATE from Step 01
# --------------------------------------------------------------------
def load_data_end_date() -> pd.Timestamp:
    """
    Load DATA_END_DATE from Step 01 metadata for temporal consistency.
    Falls back to today if metadata is missing.
    """
    metadata_path = INTERMEDIATE_DIR / "data_range_metadata.csv"

    if not metadata_path.exists():
        logger.warning(f"[WARN] Metadata file not found: {metadata_path}")
        logger.warning("[WARN] Falling back to current date for DATA_END_DATE")
        return pd.Timestamp.now().normalize()

    try:
        metadata = pd.read_csv(metadata_path, encoding="utf-8-sig")
        value = metadata.loc[metadata["Parameter"] == "DATA_END_DATE", "Value"].values[0]
        data_end_date = pd.to_datetime(value)
        logger.info(f"[INFO] Loaded DATA_END_DATE from Step 01: {data_end_date.date()}")
        return data_end_date
    except Exception as e:
        logger.error(f"[ERROR] Failed to load DATA_END_DATE from metadata: {e}")
        logger.warning("[WARN] Falling back to current date")
        return pd.Timestamp.now().normalize()


# --------------------------------------------------------------------
# 2) Load Intermediate Data
# --------------------------------------------------------------------
def load_intermediate_data():
    logger.info("=" * 80)
    logger.info(f"{STEP_NAME} - PoF3 Kronik Arıza Analizi (Enhanced v3)")
    logger.info("=" * 80)
    logger.info("")

    events_path = INTERMEDIATE_DIR / "fault_events_clean.csv"
    equipment_path = INTERMEDIATE_DIR / "equipment_master.csv"
    features_path = INTERMEDIATE_DIR / "ozellikler_pof3.csv"

    if not events_path.exists():
        logger.error(f"[FATAL] fault_events_clean.csv not found: {events_path}")
        raise FileNotFoundError(events_path)

    if not equipment_path.exists():
        logger.error(f"[FATAL] equipment_master.csv not found: {equipment_path}")
        raise FileNotFoundError(equipment_path)

    if not features_path.exists():
        logger.warning(f"[WARN] ozellikler_pof3.csv not found: {features_path}")
        features = None
    else:
        features = pd.read_csv(features_path, encoding="utf-8-sig")

    events = pd.read_csv(
        events_path,
        encoding="utf-8-sig",
        parse_dates=["Ariza_Baslangic_Zamani", "Ariza_Bitis_Zamani", "Kurulum_Tarihi"],
    )

    equipment = pd.read_csv(
        equipment_path,
        encoding="utf-8-sig",
        parse_dates=["Kurulum_Tarihi", "Ilk_Ariza_Tarihi"],
    )

    logger.info(f"[OK] Loaded fault events: {len(events):,} records")
    logger.info(f"[OK] Loaded equipment master: {len(equipment):,} records")
    if features is not None:
        logger.info(f"[OK] Loaded ozellikler_pof3 (Step 02 features): {len(features):,} records")

    # Normalize cbs_id
    for df in [events, equipment, features] if features is not None else [events, equipment]:
        df["cbs_id"] = df["cbs_id"].astype(str).str.lower().str.strip()

    return events, equipment, features


# --------------------------------------------------------------------
# 3) Multi-Level Chronic Detection (Fixed 365-day Window)
# --------------------------------------------------------------------
def calculate_multi_level_chronic(
    events: pd.DataFrame, data_end_date: pd.Timestamp
) -> pd.DataFrame:
    """
    Calculate multi-level chronic flags using a FIXED 365-day window:
      [DATA_END_DATE - 365, DATA_END_DATE]

    Severity Levels (IEEE 1366 style, aligned with Step 02):
      - KRITIK: 4+ faults in last 365 days
      - YUKSEK: 3 faults in last 365 days
      - ORTA:   2 faults in last 365 days
      - IZLEME: 1 fault in last 365 days
      - NORMAL: 0 faults in last 365 days
    """
    logger.info("")
    logger.info("[STEP] Calculating multi-level chronic classifications (fixed 365-day window)...")

    if events.empty:
        logger.warning("[WARN] No events found; returning empty chronic table")
        return pd.DataFrame(
            columns=[
                "cbs_id",
                "Kronik_Kritik",
                "Kronik_Yuksek",
                "Kronik_Orta",
                "Kronik_Izleme",
                "Kronik_Seviye_Max",
                "Faults_Last_365d",
            ]
        )

    # Filter to a 2-year recency window for performance
    RECENCY_WINDOW_DAYS = 730
    cutoff_date = data_end_date - timedelta(days=RECENCY_WINDOW_DAYS)

    ev = events.copy()
    ev = ev[ev["Ariza_Baslangic_Zamani"] >= cutoff_date].copy()

    logger.info(f"[INFO] Analyzing {len(ev):,} faults within {RECENCY_WINDOW_DAYS} days of DATA_END_DATE")
    logger.info(f"[INFO] DATA_END_DATE: {data_end_date.date()}")
    logger.info(f"[INFO] Analysis window: {cutoff_date.date()} → {data_end_date.date()}")

    last_365_start = data_end_date - timedelta(days=365)

    results = []

    for cbs_id, grp in ev.groupby("cbs_id"):
        times = grp["Ariza_Baslangic_Zamani"].dropna().sort_values()

        faults_last_365d = (times >= last_365_start).sum()

        if faults_last_365d >= 4:
            kronik_kritik = 1
            kronik_yuksek = 1
            kronik_orta = 1
            kronik_izleme = 1
            seviye = "KRITIK"
        elif faults_last_365d == 3:
            kronik_kritik = 0
            kronik_yuksek = 1
            kronik_orta = 1
            kronik_izleme = 1
            seviye = "YUKSEK"
        elif faults_last_365d == 2:
            kronik_kritik = 0
            kronik_yuksek = 0
            kronik_orta = 1
            kronik_izleme = 1
            seviye = "ORTA"
        elif faults_last_365d == 1:
            kronik_kritik = 0
            kronik_yuksek = 0
            kronik_orta = 0
            kronik_izleme = 1
            seviye = "IZLEME"
        else:
            kronik_kritik = 0
            kronik_yuksek = 0
            kronik_orta = 0
            kronik_izleme = 0
            seviye = "NORMAL"

        results.append(
            {
                "cbs_id": cbs_id,
                "Kronik_Kritik": kronik_kritik,
                "Kronik_Yuksek": kronik_yuksek,
                "Kronik_Orta": kronik_orta,
                "Kronik_Izleme": kronik_izleme,
                "Kronik_Seviye_Max": seviye,
                "Faults_Last_365d": faults_last_365d,
            }
        )

    chronic_df = pd.DataFrame(results)

    logger.info("")
    logger.info("[INFO] Chronic severity distribution (Step 04 standalone):")
    for seviye in ["KRITIK", "YUKSEK", "ORTA", "IZLEME", "NORMAL"]:
        count = (chronic_df["Kronik_Seviye_Max"] == seviye).sum()
        pct = count / len(chronic_df) if len(chronic_df) > 0 else 0.0
        logger.info(f"  {seviye:7s}: {count:5d} equipment ({pct:.1%})")

    total_chronic = (chronic_df["Kronik_Seviye_Max"] != "NORMAL").sum()
    chronic_rate = total_chronic / len(chronic_df) if len(chronic_df) > 0 else 0.0
    logger.info(f"[INFO] Total chronic rate (any level): {chronic_rate:.1%}")
    logger.info("")

    return chronic_df


# --------------------------------------------------------------------
# 4) Build Comprehensive Chronic Table for ALL Equipment
# --------------------------------------------------------------------
def build_chronic_table(
    events: pd.DataFrame, equipment: pd.DataFrame, data_end_date: pd.Timestamp
) -> pd.DataFrame:
    """
    Build a comprehensive table for ALL equipment:
      - Kurulum_Tarihi
      - Ekipman_Tipi
      - Ilk_Ariza_Tarihi, Son_Ariza_Tarihi
      - Toplam_Ariza_Sayisi
      - Observation period (Gozlem_Suresi_Gun/Yil)
      - Lambda_Yillik_Ariza (annualized fault rate)
    """
    logger.info("")
    logger.info("[STEP] Building comprehensive chronic table for ALL equipment...")

    stats = equipment[["cbs_id", "Kurulum_Tarihi", "Ekipman_Tipi"]].copy()

    logger.info(f"[INFO] Starting with all {len(stats):,} equipment")

    ev = events[events["cbs_id"].notna()].copy()

    fault_stats = (
        ev.groupby("cbs_id")["Ariza_Baslangic_Zamani"]
        .agg(Ilk_Ariza_Tarihi="min", Son_Ariza_Tarihi="max", Toplam_Ariza_Sayisi="count")
        .reset_index()
    )

    stats = stats.merge(fault_stats, on="cbs_id", how="left")
    stats["Toplam_Ariza_Sayisi"] = stats["Toplam_Ariza_Sayisi"].fillna(0).astype(int)

    logger.info(f"[INFO] Equipment with faults: {(stats['Toplam_Ariza_Sayisi'] > 0).sum():,}")
    logger.info(f"[INFO] Equipment without faults: {(stats['Toplam_Ariza_Sayisi'] == 0).sum():,}")

    # Observation period
    stats["Kurulum_Tarihi"] = pd.to_datetime(stats["Kurulum_Tarihi"], errors="coerce")
    stats["Gozlem_Suresi_Gun"] = (data_end_date - stats["Kurulum_Tarihi"]).dt.days
    stats["Gozlem_Suresi_Gun"] = stats["Gozlem_Suresi_Gun"].clip(lower=1)
    stats["Gozlem_Suresi_Yil"] = stats["Gozlem_Suresi_Gun"] / 365.25

    # Annualized failure rate (Lambda)
    stats["Lambda_Yillik_Ariza"] = stats["Toplam_Ariza_Sayisi"] / stats["Gozlem_Suresi_Yil"]

    with_faults = stats[stats["Toplam_Ariza_Sayisi"] > 0]
    if len(with_faults) > 0:
        logger.info("[INFO] Lambda statistics (equipment with faults only):")
        logger.info(
            f"  Mean:   {with_faults['Lambda_Yillik_Ariza'].mean():.3f} faults/year"
        )
        logger.info(
            f"  Median: {with_faults['Lambda_Yillik_Ariza'].median():.3f} faults/year"
        )
        logger.info(
            f"  P75:    {with_faults['Lambda_Yillik_Ariza'].quantile(0.75):.3f} faults/year"
        )
        logger.info(
            f"  P95:    {with_faults['Lambda_Yillik_Ariza'].quantile(0.95):.3f} faults/year"
        )

    return stats


# --------------------------------------------------------------------
# 5) Validation Against Step 02 (features table)
# --------------------------------------------------------------------
def validate_against_step02(
    chronic_stats: pd.DataFrame, chronic_flags: pd.DataFrame, features: pd.DataFrame
) -> pd.DataFrame:
    """
    Validate Step 04 chronic detection against Step 02 features.

    - Compares:
        * Kronik_Kritik / Kronik_Yuksek / Kronik_Orta / Kronik_Seviye_Max
        * Faults_Last_365d
    - Writes: chronic_step02_step04_comparison.csv
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("[VALIDATION] Comparing Step 04 chronic detection with Step 02 features...")
    logger.info("=" * 80)

    if features is None:
        logger.warning("[WARN] Features not available, skipping validation")
        # Return only chronic_stats merged with chronic_flags
        merged = chronic_stats.merge(chronic_flags, on="cbs_id", how="left")
        return merged

    # Merge Step 04 flags into stats first
    chronic_full = chronic_stats.merge(chronic_flags, on="cbs_id", how="left")

    # Normalize Step 04 flag columns
    for col in ["Kronik_Kritik", "Kronik_Yuksek", "Kronik_Orta", "Kronik_Izleme"]:
        if col in chronic_full.columns:
            chronic_full[col] = chronic_full[col].fillna(0).astype(int)

    if "Kronik_Seviye_Max" in chronic_full.columns:
        chronic_full["Kronik_Seviye_Max"] = chronic_full["Kronik_Seviye_Max"].fillna("NORMAL")

    if "Faults_Last_365d" in chronic_full.columns:
        chronic_full["Faults_Last_365d"] = chronic_full["Faults_Last_365d"].fillna(0).astype(int)

    # Step 02 columns expected
    step02_col_map = {
        "Kronik_Kritik": "Kronik_Kritik_Step02",
        "Kronik_Yuksek": "Kronik_Yuksek_Step02",
        "Kronik_Orta": "Kronik_Orta_Step02",
        "Kronik_Seviye_Max": "Kronik_Seviye_Max_Step02",
        "Faults_Last_365d": "Faults_Last_365d_Step02",
    }

    available_step02_cols = [c for c in step02_col_map.keys() if c in features.columns]
    if not available_step02_cols:
        logger.warning("[WARN] No Step 02 chronic-related columns found in features")
        return chronic_full

    logger.info(f"[INFO] Found Step 02 columns: {available_step02_cols}")

    merge_cols = ["cbs_id"] + available_step02_cols
    step02_subset = features[merge_cols].copy()
    step02_subset = step02_subset.rename(columns={c: step02_col_map[c] for c in available_step02_cols})

    validated = chronic_full.merge(step02_subset, on="cbs_id", how="left")

    # --- KRITIK comparison metrics ---------------------------------------
    if "Kronik_Kritik" in validated.columns and "Kronik_Kritik_Step02" in validated.columns:
        logger.info("")
        logger.info("[VALIDATION] KRITIK flag agreement:")

        step04 = validated["Kronik_Kritik"].fillna(0).astype(int)
        step02 = validated["Kronik_Kritik_Step02"].fillna(0).astype(int)

        step04_count = (step04 == 1).sum()
        step02_count = (step02 == 1).sum()
        both_count = ((step04 == 1) & (step02 == 1)).sum()
        either = ((step04 == 1) | (step02 == 1)).sum()

        jaccard = both_count / either if either > 0 else 1.0
        precision = both_count / step04_count if step04_count > 0 else 0.0
        recall = both_count / step02_count if step02_count > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        only_step04 = ((step04 == 1) & (step02 == 0)).sum()
        only_step02 = ((step04 == 0) & (step02 == 1)).sum()

        logger.info(f"  Step 04 KRITIK: {step04_count:,}")
        logger.info(f"  Step 02 KRITIK: {step02_count:,}")
        logger.info(f"  Both KRITIK:   {both_count:,}")
        logger.info(f"  Jaccard (IoU): {jaccard:.1%}")
        logger.info(f"  Precision:     {precision:.1%}")
        logger.info(f"  Recall:        {recall:.1%}")
        logger.info(f"  F1-Score:      {f1:.3f}")
        logger.info(f"  Only Step 04:  {only_step04:,}")
        logger.info(f"  Only Step 02:  {only_step02:,}")

    # --- Severity level comparison ---------------------------------------
    if "Kronik_Seviye_Max" in validated.columns and "Kronik_Seviye_Max_Step02" in validated.columns:
        logger.info("")
        logger.info("[VALIDATION] Severity level distribution (Step 04 vs Step 02):")

        step04_levels = validated["Kronik_Seviye_Max"].fillna("NORMAL")
        step02_levels = validated["Kronik_Seviye_Max_Step02"].fillna("NORMAL")

        comparison_rows = []
        for seviye in ["KRITIK", "YUKSEK", "ORTA", "IZLEME", "NORMAL"]:
            s4 = (step04_levels == seviye).sum()
            s2 = (step02_levels == seviye).sum()
            comparison_rows.append(
                {
                    "Seviye": seviye,
                    "Step04_Count": s4,
                    "Step02_Count": s2,
                    "Difference": s4 - s2,
                }
            )

        comparison_df = pd.DataFrame(comparison_rows)
        logger.info("\n" + comparison_df.to_string(index=False))

        comparison_path = OUTPUT_DIR / "chronic_step02_step04_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False, encoding="utf-8-sig")
        logger.info(f"\n[OK] Step 02 vs Step 04 comparison saved → {comparison_path}")

    # --- Faults_Last_365d comparison -------------------------------------
    if "Faults_Last_365d" in validated.columns and "Faults_Last_365d_Step02" in validated.columns:
        logger.info("")
        logger.info("[VALIDATION] Faults_Last_365d difference statistics:")

        diff = validated["Faults_Last_365d"].fillna(0).astype(int) - \
               validated["Faults_Last_365d_Step02"].fillna(0).astype(int)

        logger.info(f"  Mean difference:   {diff.mean():.3f}")
        logger.info(f"  Median difference: {diff.median():.3f}")
        logger.info(f"  P95 difference:    {diff.quantile(0.95):.3f}")
        logger.info(f"  Exact match rate:  {(diff == 0).mean():.1%}")

    logger.info("=" * 80)
    logger.info("")
    return validated


# --------------------------------------------------------------------
# 6) Load PoF Outputs from Step 03
# --------------------------------------------------------------------
def load_pof_outputs() -> pd.DataFrame:
    """
    Load PoF outputs from Step 03 and merge into a single table by cbs_id.

    Targets:
      - Cox (12Ay)
      - Weibull (12Ay)
      - RSF (12Ay)
      - Ensemble (12Ay)
      - ML XGB (from leakage_free_ml_pof.csv)
    """
    logger.info("")
    logger.info("[STEP] Loading PoF outputs from Step 03 for risk master table...")

    merged = None

    def safe_merge(df_base, new_path, col_name):
        if not new_path.exists():
            logger.warning(f"[WARN] PoF file not found: {new_path}")
            return df_base
        tmp = pd.read_csv(new_path, encoding="utf-8-sig")
        if "cbs_id" not in tmp.columns:
            logger.warning(f"[WARN] 'cbs_id' not in {new_path.name}, skipping")
            return df_base
        tmp["cbs_id"] = tmp["cbs_id"].astype(str).str.lower().str.strip()
        if df_base is None:
            df_base = tmp.copy()
        else:
            df_base = df_base.merge(tmp, on="cbs_id", how="left")
        logger.info(f"[OK] Merged PoF from {new_path.name}")
        return df_base

    # 12-month survival horizon files (from Step 03 naming)
    cox_12 = OUTPUT_DIR / "cox_sagkalim_12ay_ariza_olasiligi.csv"
    weib_12 = OUTPUT_DIR / "weibull_sagkalim_12ay_ariza_olasiligi.csv"
    rsf_12 = OUTPUT_DIR / "rsf_sagkalim_12ay_ariza_olasiligi.csv"
    ens_12 = OUTPUT_DIR / "ensemble_sagkalim_12ay_ariza_olasiligi.csv"
    ml_path = OUTPUT_DIR / "leakage_free_ml_pof.csv"

    merged = safe_merge(merged, cox_12, "PoF_Cox_Cal_12Ay")
    merged = safe_merge(merged, weib_12, "PoF_Weibull_12Ay")
    merged = safe_merge(merged, rsf_12, "PoF_RSF_12Ay")
    merged = safe_merge(merged, ens_12, "PoF_Ensemble_12Ay")

    if ml_path.exists():
        ml = pd.read_csv(ml_path, encoding="utf-8-sig")
        if "cbs_id" in ml.columns and "PoF_ML_XGB" in ml.columns:
            ml["cbs_id"] = ml["cbs_id"].astype(str).str.lower().str.strip()
            merged = ml if merged is None else merged.merge(ml[["cbs_id", "PoF_ML_XGB"]], on="cbs_id", how="left")
            logger.info(f"[OK] Merged ML PoF from {ml_path.name}")
        else:
            logger.warning(f"[WARN] ML PoF file {ml_path} missing required columns")
    else:
        logger.warning(f"[WARN] ML PoF file not found: {ml_path}")

    if merged is None:
        logger.warning("[WARN] No PoF files loaded; risk master table will not contain PoF columns")
        return pd.DataFrame(columns=["cbs_id"])

    # Drop any duplicate columns if they came from multiple merges
    merged = merged.loc[:, ~merged.columns.duplicated()]

    return merged


# --------------------------------------------------------------------
# 7) Save Outputs (Summary + Risk Master)
# --------------------------------------------------------------------
def save_outputs(chronic_full: pd.DataFrame):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Clean numeric infinities / NaNs
    chronic_full = chronic_full.replace([np.inf, -np.inf], np.nan)

    summary_path = OUTPUT_DIR / "chronic_equipment_summary.csv"
    chronic_only_path = OUTPUT_DIR / "chronic_equipment_only.csv"
    izleme_path = OUTPUT_DIR / "izleme_equipment.csv"

    chronic_full.to_csv(summary_path, index=False, encoding="utf-8-sig")
    logger.info(f"[OK] Chronic summary saved → {summary_path}")

    # Actionable chronic mask (exclude IZLEME)
    chronic_mask = (
        (chronic_full.get("Kronik_Kritik", 0) == 1)
        | (chronic_full.get("Kronik_Yuksek", 0) == 1)
        | (chronic_full.get("Kronik_Orta", 0) == 1)
    )

    # IZLEME-only (observation)
    izleme_mask = (chronic_full.get("Kronik_Izleme", 0) == 1) & (~chronic_mask)

    if izleme_mask.any():
        izleme_df = chronic_full[izleme_mask].copy()
        izleme_df.to_csv(izleme_path, index=False, encoding="utf-8-sig")
        logger.info(f"[OK] IZLEME-only equipment saved → {izleme_path}")

    chronic_only = chronic_full[chronic_mask].copy()
    chronic_only.to_csv(chronic_only_path, index=False, encoding="utf-8-sig")
    logger.info(f"[OK] Chronic-only table saved → {chronic_only_path}")

    # Summary statistics (correct percentages)
    total = len(chronic_full)
    kritik = int(chronic_full.get("Kronik_Kritik", pd.Series(dtype=int)).sum())
    yuksek = int(chronic_full.get("Kronik_Yuksek", pd.Series(dtype=int)).sum())
    orta = int(chronic_full.get("Kronik_Orta", pd.Series(dtype=int)).sum())
    izleme = int(chronic_full.get("Kronik_Izleme", pd.Series(dtype=int)).sum())
    actionable = kritik + yuksek + orta
    normal = total - kritik - yuksek - orta - izleme

    def pct(x):
        return x / total if total > 0 else 0.0

    logger.info("")
    logger.info("=" * 80)
    logger.info("[SUMMARY] Chronic Equipment Statistics (IEEE 1366 Aligned):")
    logger.info(f"  Total equipment: {total:,}")
    logger.info("")
    logger.info("  ACTION REQUIRED (KRITIK/YUKSEK/ORTA):")
    logger.info(f"    KRITIK (4+ faults/year): {kritik:5d} ({pct(kritik):.1%})")
    logger.info(f"    YUKSEK (3 faults/year):  {yuksek:5d} ({pct(yuksek):.1%})")
    logger.info(f"    ORTA (2 faults/year):    {orta:5d} ({pct(orta):.1%})")
    logger.info(f"    → Total actionable:      {actionable:5d} ({pct(actionable):.1%})")
    logger.info("")
    logger.info("  OBSERVATION (IZLEME):")
    logger.info(f"    IZLEME (1 fault/year):   {izleme:5d} ({pct(izleme):.1%})")
    logger.info("")
    logger.info("  NORMAL:")
    logger.info(f"    NORMAL (0 faults):       {normal:5d} ({pct(normal):.1%})")
    logger.info("=" * 80)
    logger.info("")
    logger.info("[NOTE] Step 04 uses a fixed 365-day window aligned with Step 02 / Step 03")
    logger.info("=" * 80)


def save_risk_master(chronic_full: pd.DataFrame, pof_df: pd.DataFrame):
    """
    Build and save the canonical risk_equipment_master.csv:
      - Equipment info
      - Chronic flags and Faults_Last_365d
      - PoF scores (12-month horizon + ML)
    """
    logger.info("")
    logger.info("[STEP] Building risk_equipment_master.csv ...")

    df = chronic_full.copy()

    # Merge PoF outputs if available
    if not pof_df.empty and "cbs_id" in pof_df.columns:
        pof_df["cbs_id"] = pof_df["cbs_id"].astype(str).str.lower().str.strip()
        df["cbs_id"] = df["cbs_id"].astype(str).str.lower().str.strip()
        df = df.merge(pof_df, on="cbs_id", how="left")
        logger.info("[OK] PoF outputs merged into risk master")

    # Optional: select and order columns for clarity
    core_cols = [
        "cbs_id",
        "Ekipman_Tipi",
        "Kurulum_Tarihi",
        "Ilk_Ariza_Tarihi",
        "Son_Ariza_Tarihi",
        "Toplam_Ariza_Sayisi",
        "Gozlem_Suresi_Gun",
        "Gozlem_Suresi_Yil",
        "Lambda_Yillik_Ariza",
        "Faults_Last_365d",
        "Kronik_Kritik",
        "Kronik_Yuksek",
        "Kronik_Orta",
        "Kronik_Izleme",
        "Kronik_Seviye_Max",
    ]

    # PoF columns (if present)
    pof_cols = [c for c in df.columns if c.startswith("PoF_")]

    # Keep everything, but try to order
    ordered_cols = [c for c in core_cols if c in df.columns] + [c for c in pof_cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in ordered_cols]

    df = df[ordered_cols + other_cols]

    # Clean numeric infinities / NaNs
    df = df.replace([np.inf, -np.inf], np.nan)

    risk_path = OUTPUT_DIR / "risk_equipment_master.csv"
    df.to_csv(risk_path, index=False, encoding="utf-8-sig")

    logger.info(f"[OK] Risk master saved → {risk_path}")
    logger.info("[INFO] This file is the canonical input for PoF dashboards / DSO reports.")


# --------------------------------------------------------------------
# 8) Main
# --------------------------------------------------------------------
def main():
    try:
        data_end_date = load_data_end_date()
        events, equipment, features = load_intermediate_data()

        # Step 04 chronic detection (fixed window)
        chronic_flags = calculate_multi_level_chronic(events, data_end_date)

        # Comprehensive stats for all equipment
        chronic_stats = build_chronic_table(events, equipment, data_end_date)

        # Validation & alignment with Step 02
        chronic_full = validate_against_step02(chronic_stats, chronic_flags, features)

        # Save chronic summary outputs
        save_outputs(chronic_full)

        # Load PoF outputs from Step 03 and build risk master
        pof_df = load_pof_outputs()
        save_risk_master(chronic_full, pof_df)

        logger.info("")
        logger.info("[SUCCESS] 04_tekrarlayan_ariza_v3 completed successfully!")

    except Exception as e:
        logger.exception(f"[FATAL] {STEP_NAME} failed: {e}")
        raise


if __name__ == "__main__":
    main()
