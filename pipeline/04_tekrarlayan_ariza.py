# pipeline/04_chronic_detection.py (Enhanced v2)

"""
Improvements:
- Loads DATA_END_DATE from Step 01 metadata
- Multi-level severity output (KRITIK/YUKSEK/ORTA/GOZLEM)
- Validation against Step 02 chronic flags
- Includes ALL equipment in analysis (not just faulted)
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.stdout.reconfigure(encoding='utf-8')

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd

from utils.logger import get_logger

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# --------------------------------------------------------------------
# Paths / Config
# --------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]

# Import configuration
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
    # Fallback
    DATA_DIR = BASE_DIR / "data"
    INTERMEDIATE_DIR = DATA_DIR / "ara_ciktilar"
    OUTPUT_DIR = DATA_DIR / "sonuclar"
    LOG_DIR = BASE_DIR / "loglar"
    CHRONIC_WINDOW_DAYS = 90
    CHRONIC_THRESHOLD_EVENTS = 3

LOG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / f"04_tekrarlayan_ariza_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logger = get_logger("04_tekrarlayan_ariza", LOG_FILE)

# Multi-level chronic definitions (aligned with Step 02)
CHRONIC_DEFINITIONS = {
    "KRITIK": {"window_days": 90, "min_faults": 3, "desc": "3+ arıza 90 günde - ACİL MÜDAHALE"},
    "YUKSEK": {"window_days": 365, "min_faults": 4, "desc": "4+ arıza 12 ayda - ÖNCELİKLİ"},
    "ORTA": {"window_days": 365, "min_faults": 3, "desc": "3+ arıza 12 ayda - TAKİP"},
    "GOZLEM": {"window_days": 180, "min_faults": 2, "desc": "2+ arıza 6 ayda - İZLEME"},
}


# --------------------------------------------------------------------
# Load DATA_END_DATE from Step 01
# --------------------------------------------------------------------
def load_data_end_date() -> pd.Timestamp:
    """
    Load DATA_END_DATE from Step 01 metadata for temporal consistency.
    """
    metadata_path = INTERMEDIATE_DIR / "data_range_metadata.csv"
    
    if not metadata_path.exists():
        logger.warning(f"[WARN] Metadata file not found: {metadata_path}")
        logger.warning("[WARN] Falling back to current date")
        return pd.Timestamp.now().normalize()
    
    try:
        metadata = pd.read_csv(metadata_path, encoding="utf-8-sig")
        data_end_date_str = metadata.loc[metadata["Parameter"] == "DATA_END_DATE", "Value"].values[0]
        data_end_date = pd.to_datetime(data_end_date_str)
        
        logger.info(f"[INFO] Loaded DATA_END_DATE from Step 01: {data_end_date.date()}")
        return data_end_date
    except Exception as e:
        logger.error(f"[ERROR] Failed to load DATA_END_DATE: {e}")
        logger.warning("[WARN] Falling back to current date")
        return pd.Timestamp.now().normalize()


# --------------------------------------------------------------------
# Data Loading
# --------------------------------------------------------------------
def load_intermediate_data():
    logger.info("=" * 80)
    logger.info("04_tekrarlayan_ariza - PoF3 Kronik Arıza Analizi (Enhanced v2)")
    logger.info("=" * 80)
    logger.info("")

    # Fault events
    events_path = INTERMEDIATE_DIR / "fault_events_clean.csv"
    if not events_path.exists():
        logger.error(f"[FATAL] fault_events_clean.csv not found: {events_path}")
        raise FileNotFoundError(events_path)

    equipment_path = INTERMEDIATE_DIR / "equipment_master.csv"
    if not equipment_path.exists():
        logger.error(f"[FATAL] equipment_master.csv not found: {equipment_path}")
        raise FileNotFoundError(equipment_path)

    features_path = INTERMEDIATE_DIR / "ozellikler_pof3.csv"
    if not features_path.exists():
        logger.warning(f"[WARN] ozellikler_pof3.csv not found: {features_path}")
        features = None
    else:
        features = pd.read_csv(features_path, encoding="utf-8-sig")

    events = pd.read_csv(
        events_path,
        encoding="utf-8-sig",
        parse_dates=["Ariza_Baslangic_Zamani", "Ariza_Bitis_Zamani", "Kurulum_Tarihi"]
    )
    equipment = pd.read_csv(
        equipment_path,
        encoding="utf-8-sig",
        parse_dates=["Kurulum_Tarihi", "Ilk_Ariza_Tarihi"]
    )

    logger.info(f"[OK] Loaded fault events: {len(events):,} records")
    logger.info(f"[OK] Loaded equipment master: {len(equipment):,} records")
    if features is not None:
        logger.info(f"[OK] Loaded ozellikler_pof3: {len(features):,} records")

    return events, equipment, features


# --------------------------------------------------------------------
# Multi-Level Chronic Detection
# --------------------------------------------------------------------
def calculate_multi_level_chronic(events: pd.DataFrame, data_end_date: pd.Timestamp) -> pd.DataFrame:
    """
    Calculate multi-level chronic flags using rolling window approach.
    Aligned with Step 02 logic.
    """
    logger.info("")
    logger.info("[STEP] Calculating multi-level chronic classifications...")
    
    if events.empty:
        return pd.DataFrame(columns=[
            "cbs_id", "Kronik_Kritik", "Kronik_Yuksek", "Kronik_Orta", 
            "Kronik_Gozlem", "Kronik_90g_Flag", "Kronik_Seviye_Max"
        ])
    
    # Only consider faults within 2 years of DATA_END_DATE for recency
    RECENCY_WINDOW_DAYS = 730
    cutoff_date = data_end_date - timedelta(days=RECENCY_WINDOW_DAYS)
    
    recent_events = events[events["Ariza_Baslangic_Zamani"] >= cutoff_date].copy()
    
    logger.info(f"[INFO] Analyzing {len(recent_events):,} faults within {RECENCY_WINDOW_DAYS} days of DATA_END_DATE")
    
    results = []
    
    for cbs_id, grp in recent_events.groupby("cbs_id"):
        times = grp["Ariza_Baslangic_Zamani"].sort_values().values
        n_faults = len(times)
        
        # Initialize flags
        kronik_kritik = 0
        kronik_yuksek = 0
        kronik_orta = 0
        kronik_gozlem = 0
        kronik_90g = 0
        
        if n_faults >= 2:
            # KRITIK: 3+ faults in 90 days
            for i in range(len(times) - 2):
                window = (times[i+2] - times[i]).astype('timedelta64[D]').astype(int)
                if window <= CHRONIC_DEFINITIONS["KRITIK"]["window_days"]:
                    kronik_kritik = 1
                    break
            
            # GOZLEM: 2+ faults in 180 days
            for i in range(len(times) - 1):
                window = (times[i+1] - times[i]).astype('timedelta64[D]').astype(int)
                if window <= CHRONIC_DEFINITIONS["GOZLEM"]["window_days"]:
                    kronik_gozlem = 1
                    break
            
            # 90g flag: Any consecutive faults within 90 days
            diffs = np.diff(times).astype("timedelta64[D]").astype(int)
            kronik_90g = int((diffs <= CHRONIC_WINDOW_DAYS).any())
        
        if n_faults >= 3:
            # ORTA: 3+ faults in 365 days
            for i in range(len(times) - 2):
                window = (times[i+2] - times[i]).astype('timedelta64[D]').astype(int)
                if window <= CHRONIC_DEFINITIONS["ORTA"]["window_days"]:
                    kronik_orta = 1
                    break
        
        if n_faults >= 4:
            # YUKSEK: 4+ faults in 365 days
            for i in range(len(times) - 3):
                window = (times[i+3] - times[i]).astype('timedelta64[D]').astype(int)
                if window <= CHRONIC_DEFINITIONS["YUKSEK"]["window_days"]:
                    kronik_yuksek = 1
                    break
        
        # Determine max severity
        if kronik_kritik == 1:
            seviye = "KRITIK"
        elif kronik_yuksek == 1:
            seviye = "YUKSEK"
        elif kronik_orta == 1:
            seviye = "ORTA"
        elif kronik_gozlem == 1:
            seviye = "GOZLEM"
        else:
            seviye = "NORMAL"
        
        results.append({
            "cbs_id": cbs_id,
            "Kronik_Kritik": kronik_kritik,
            "Kronik_Yuksek": kronik_yuksek,
            "Kronik_Orta": kronik_orta,
            "Kronik_Gozlem": kronik_gozlem,
            "Kronik_90g_Flag": kronik_90g,
            "Kronik_Seviye_Max": seviye
        })
    
    chronic_df = pd.DataFrame(results)
    
    # Log distribution
    logger.info("[INFO] Chronic severity distribution:")
    for seviye in ["KRITIK", "YUKSEK", "ORTA", "GOZLEM", "NORMAL"]:
        count = (chronic_df["Kronik_Seviye_Max"] == seviye).sum()
        pct = count / len(chronic_df) * 100 if len(chronic_df) > 0 else 0
        logger.info(f"  {seviye}: {count:,} equipment ({pct:.1f}%)")
    
    return chronic_df


# --------------------------------------------------------------------
# Build Comprehensive Chronic Table
# --------------------------------------------------------------------
def build_chronic_table(events: pd.DataFrame, equipment: pd.DataFrame, data_end_date: pd.Timestamp) -> pd.DataFrame:
    """
    Build comprehensive chronic analysis including ALL equipment.
    """
    logger.info("")
    logger.info("[STEP] Building comprehensive chronic table for ALL equipment...")

    # Start with ALL equipment
    stats = equipment[["cbs_id", "Kurulum_Tarihi", "Ekipman_Tipi"]].copy()
    
    logger.info(f"[INFO] Starting with all {len(stats):,} equipment")
    
    # Basic fault statistics
    ev = events[events["cbs_id"].notna()].copy()
    
    fault_stats = ev.groupby("cbs_id")["Ariza_Baslangic_Zamani"].agg(
        Ilk_Ariza_Tarihi="min",
        Son_Ariza_Tarihi="max",
        Toplam_Ariza_Sayisi="count",
    ).reset_index()
    
    # Left join to keep ALL equipment
    stats = stats.merge(fault_stats, on="cbs_id", how="left")
    stats["Toplam_Ariza_Sayisi"] = stats["Toplam_Ariza_Sayisi"].fillna(0).astype(int)
    
    logger.info(f"[INFO] Equipment with faults: {(stats['Toplam_Ariza_Sayisi'] > 0).sum():,}")
    logger.info(f"[INFO] Equipment without faults: {(stats['Toplam_Ariza_Sayisi'] == 0).sum():,}")
    
    # Recent faults (last 12 months)
    window_start = data_end_date - timedelta(days=365)
    recent_mask = ev["Ariza_Baslangic_Zamani"] >= window_start
    recent_counts = ev[recent_mask].groupby("cbs_id")["Ariza_Baslangic_Zamani"].count().rename("Son12Ay_Ariza_Sayisi")
    
    stats = stats.merge(recent_counts, on="cbs_id", how="left")
    stats["Son12Ay_Ariza_Sayisi"] = stats["Son12Ay_Ariza_Sayisi"].fillna(0).astype(int)
    
    # Observation period (from installation to DATA_END_DATE)
    stats["Kurulum_Tarihi"] = pd.to_datetime(stats["Kurulum_Tarihi"], errors="coerce")
    stats["Gozlem_Suresi_Gun"] = (data_end_date - stats["Kurulum_Tarihi"]).dt.days.clip(lower=1)
    stats["Gozlem_Suresi_Yil"] = stats["Gozlem_Suresi_Gun"] / 365.25
    
    # Annual failure rate (Lambda)
    stats["Lambda_Yillik_Ariza"] = stats["Toplam_Ariza_Sayisi"] / stats["Gozlem_Suresi_Yil"]
    
    # Log statistics
    equipment_with_faults = stats[stats["Toplam_Ariza_Sayisi"] > 0]
    if len(equipment_with_faults) > 0:
        logger.info(f"[INFO] Lambda statistics (equipment with faults only):")
        logger.info(f"  Mean: {equipment_with_faults['Lambda_Yillik_Ariza'].mean():.3f} faults/year")
        logger.info(f"  Median: {equipment_with_faults['Lambda_Yillik_Ariza'].median():.3f} faults/year")
        logger.info(f"  P75: {equipment_with_faults['Lambda_Yillik_Ariza'].quantile(0.75):.3f} faults/year")
        logger.info(f"  P95: {equipment_with_faults['Lambda_Yillik_Ariza'].quantile(0.95):.3f} faults/year")
    
    return stats


# --------------------------------------------------------------------
# Validation Against Step 02
# --------------------------------------------------------------------
def validate_against_step02(chronic_stats: pd.DataFrame, features: pd.DataFrame, logger) -> pd.DataFrame:
    """
    Validate Step 04 chronic flags against Step 02 feature flags.
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("[VALIDATION] Comparing Step 04 chronic detection with Step 02...")
    logger.info("=" * 80)
    
    if features is None:
        logger.warning("[WARN] Features not available, skipping validation")
        return chronic_stats
    
    # Expected Step 02 chronic columns
    step02_cols = {
        "Kronik_Kritik": "Kronik_Kritik_Step02",
        "Kronik_Yuksek": "Kronik_Yuksek_Step02",
        "Kronik_Orta": "Kronik_Orta_Step02",
        "Kronik_Seviye_Max": "Kronik_Seviye_Max_Step02"
    }
    
    available_cols = [col for col in step02_cols.keys() if col in features.columns]
    
    if not available_cols:
        logger.warning("[WARN] No Step 02 chronic columns found in features")
        return chronic_stats
    
    logger.info(f"[INFO] Found Step 02 columns: {available_cols}")
    
    # Merge Step 02 flags
    merge_cols = ["cbs_id"] + available_cols
    step02_subset = features[merge_cols].copy()
    
    # Rename to avoid collision
    rename_dict = {col: step02_cols[col] for col in available_cols}
    step02_subset.rename(columns=rename_dict, inplace=True)
    
    validated = chronic_stats.merge(step02_subset, on="cbs_id", how="left")
    
    # Comparison statistics
    logger.info("")
    logger.info("[VALIDATION] Agreement Analysis:")
    
    # Overall chronic detection comparison
    if "Kronik_Kritik" in chronic_stats.columns and "Kronik_Kritik_Step02" in validated.columns:
        step04_kritik = validated["Kronik_Kritik"].fillna(0)
        step02_kritik = validated["Kronik_Kritik_Step02"].fillna(0)
        
        agreement = (step04_kritik == step02_kritik).sum()
        total = len(validated)
        
        step04_count = step04_kritik.sum()
        step02_count = step02_kritik.sum()
        both_count = ((step04_kritik == 1) & (step02_kritik == 1)).sum()
        
        logger.info(f"  KRITIK flags:")
        logger.info(f"    Step 04 detected: {step04_count:,} equipment")
        logger.info(f"    Step 02 detected: {step02_count:,} equipment")
        logger.info(f"    Both flagged: {both_count:,} equipment")
        logger.info(f"    Agreement rate: {agreement/total:.1%}")
        
        # Unique to each step
        only_step04 = ((step04_kritik == 1) & (step02_kritik == 0)).sum()
        only_step02 = ((step04_kritik == 0) & (step02_kritik == 1)).sum()
        
        logger.info(f"    Only Step 04: {only_step04:,} equipment")
        logger.info(f"    Only Step 02: {only_step02:,} equipment")
    
    # Severity level comparison
    if "Kronik_Seviye_Max" in chronic_stats.columns and "Kronik_Seviye_Max_Step02" in validated.columns:
        logger.info("")
        logger.info("  Severity level distribution:")
        
        step04_dist = validated["Kronik_Seviye_Max"].fillna("NORMAL").value_counts()
        step02_dist = validated["Kronik_Seviye_Max_Step02"].fillna("NORMAL").value_counts()
        
        comparison_data = []
        for seviye in ["KRITIK", "YUKSEK", "ORTA", "GOZLEM", "NORMAL"]:
            step04_c = step04_dist.get(seviye, 0)
            step02_c = step02_dist.get(seviye, 0)
            comparison_data.append({
                "Seviye": seviye,
                "Step04_Count": step04_c,
                "Step02_Count": step02_c,
                "Difference": step04_c - step02_c
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        logger.info(f"\n{comparison_df.to_string(index=False)}")
        
        # Save comparison report
        comparison_path = OUTPUT_DIR / "chronic_step02_step04_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False, encoding="utf-8-sig")
        logger.info(f"\n[OK] Comparison report saved → {comparison_path}")
    
    logger.info("=" * 80)
    logger.info("")
    
    return validated


# --------------------------------------------------------------------
# Main Processing
# --------------------------------------------------------------------
def save_outputs(chronic_full: pd.DataFrame):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = OUTPUT_DIR / "chronic_equipment_summary.csv"
    chronic_only_path = OUTPUT_DIR / "chronic_equipment_only.csv"
    
    # Save full summary
    chronic_full.to_csv(summary_path, index=False, encoding="utf-8-sig")
    logger.info(f"[OK] Chronic summary saved → {summary_path}")

    # Save chronic-only (any chronic flag)
    chronic_mask = (
        (chronic_full.get("Kronik_Kritik", 0) == 1) |
        (chronic_full.get("Kronik_Yuksek", 0) == 1) |
        (chronic_full.get("Kronik_Orta", 0) == 1) |
        (chronic_full.get("Kronik_Gozlem", 0) == 1)
    )
    chronic_only = chronic_full[chronic_mask].copy()
    chronic_only.to_csv(chronic_only_path, index=False, encoding="utf-8-sig")
    logger.info(f"[OK] Chronic-only table saved → {chronic_only_path}")

    # Summary statistics
    total = len(chronic_full)
    kritik = chronic_full.get("Kronik_Kritik", pd.Series([0])).sum()
    yuksek = chronic_full.get("Kronik_Yuksek", pd.Series([0])).sum()
    orta = chronic_full.get("Kronik_Orta", pd.Series([0])).sum()
    gozlem = chronic_full.get("Kronik_Gozlem", pd.Series([0])).sum()
    any_chronic = len(chronic_only)
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("[SUMMARY] Chronic Equipment Statistics:")
    logger.info(f"  Total equipment: {total:,}")
    logger.info(f"  KRITIK: {kritik:,} ({kritik/total:.1%})")
    logger.info(f"  YUKSEK: {yuksek:,} ({yuksek/total:.1%})")
    logger.info(f"  ORTA: {orta:,} ({orta/total:.1%})")
    logger.info(f"  GOZLEM: {gozlem:,} ({gozlem/total:.1%})")
    logger.info(f"  Any chronic: {any_chronic:,} ({any_chronic/total:.1%})")
    logger.info("=" * 80)


def main():
    try:
        # Load DATA_END_DATE
        data_end_date = load_data_end_date()
        
        # Load data
        events, equipment, features = load_intermediate_data()
        
        # Calculate multi-level chronic flags (Step 04)
        chronic_flags = calculate_multi_level_chronic(events, data_end_date)
        
        # Build comprehensive table with ALL equipment
        chronic_stats = build_chronic_table(events, equipment, data_end_date)
        
        # Merge chronic flags (left join to keep all equipment)
        chronic_full = chronic_stats.merge(chronic_flags, on="cbs_id", how="left")
        
        # Fill NaN for equipment without chronic flags
        for col in ["Kronik_Kritik", "Kronik_Yuksek", "Kronik_Orta", "Kronik_Gozlem", "Kronik_90g_Flag"]:
            if col in chronic_full.columns:
                chronic_full[col] = chronic_full[col].fillna(0).astype(int)
        
        if "Kronik_Seviye_Max" in chronic_full.columns:
            chronic_full["Kronik_Seviye_Max"] = chronic_full["Kronik_Seviye_Max"].fillna("NORMAL")
        
        # Validate against Step 02
        chronic_full = validate_against_step02(chronic_full, features, logger)
        
        # Save outputs
        save_outputs(chronic_full)
        
        logger.info("")
        logger.info("[SUCCESS] 04_tekrarlayan_ariza completed successfully!")
        
    except Exception as e:
        logger.exception(f"[FATAL] 04_tekrarlayan_ariza failed: {e}")
        raise


if __name__ == "__main__":
    main()