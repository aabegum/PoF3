"""
01_veri_isleme.py (PoF3 - Enhanced v2)

Improvements:
- Reversed temporal validation order (missing data filter → temporal checks)
- Detailed missingness analysis report
- Maintenance data completeness logging
- Fixed equipment master deduplication to prioritize fault records
- Outlier detection for fault duration
- Age calculation using DATA_END_DATE instead of ANALYSIS_DATE

Amaç:
- Ham arıza + sağlam veri yükleme
- Zorunlu veri sözleşmesi (cbs_id küçük harf zorunlu)
- Tarih, süre, ekipman tipi temizliği
- Bakım ve ekipman niteliklerini (MARKA, gerilim, kVA, vb.) ekipman_master'a agregasyon
"""

import os
import sys
import logging
from datetime import datetime
from scipy import stats
import numpy as np
import pandas as pd

# Ensure project root in path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# UTF-8 güvenli çıktı
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Tarih parser
from utils.date_parser import parse_date_safely

# Config içe aktar
from config.config import (
    ANALYSIS_DATE,
    DATA_PATHS,
    INTERMEDIATE_PATHS,
    OUTPUT_PATHS,
    MIN_EQUIPMENT_PER_CLASS,
    LOG_DIR,
)


# ---------------------------------------------------------
# LOGGING
# ---------------------------------------------------------

def setup_logger() -> logging.Logger:
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"01_veri_isleme_{ts}.log")

    logger = logging.getLogger("01_veri_isleme")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # Dosya logu
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

    # Konsol logu
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    logger.info("=" * 80)
    logger.info("01_veri_isleme - PoF3 Veri İşleme (Enhanced v2)")
    logger.info("=" * 80)
    logger.info(f"Analiz Tarihi: {ANALYSIS_DATE}")
    logger.info("")
    return logger


# ---------------------------------------------------------
# Yardımcı fonksiyonlar
# ---------------------------------------------------------

def convert_duration_minutes(series: pd.Series, logger: logging.Logger) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    med = s.median()

    logger.info(f"[INFO] Süre medyanı (ham): {med}")

    if med > 10000:
        logger.info("[INFO] Süreler milisaniye → dakikaya dönüşüyor.")
        return s / 60000.0
    return s


def clean_equipment_type(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    return (
        s.str.replace(" Arızaları", "", regex=False)
         .str.replace(" Ariza", "", regex=False)
         .str.strip()
    )


def _rename_maintenance_and_attributes(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Arıza / sağlam verisindeki bakım ve ekipman nitelik kolonlarını
    iç standarda çevirir.
    """
    col_map = {
        "Bakım Sayısı": "Bakim_Sayisi",
        "Geçmiş İş Emri Tipleri": "Bakim_Is_Emri_Tipleri",
        "İlk Bakım İş Emri Tarihi": "Ilk_Bakim_Tarihi",
        "Son Bakım İş Emri Tarihi": "Son_Bakim_Tarihi", # Correct Excel Name
        "Son Bakım İş Emri Tipi": "Son_Bakim_Tipi",
        "Son Bakımdan İtibaren Geçen Gün Sayısı": "Son_Bakimdan_Gecen_Gun",
        "MARKA": "Marka",
        "component_voltage": "Gerilim_Seviyesi",
        "voltage_level": "Gerilim_Sinifi",
        "kVA_Rating": "kVA_Rating",
    }

    # FIX: Update this list to match the ACTUAL Excel headers
    expected_important = [
        "Bakım Sayısı", 
        "Son Bakım İş Emri Tarihi", # Updated from "Son Bakım"
        "MARKA", 
        "kVA_Rating"
    ]
    
    missing_important = [c for c in expected_important if c not in df.columns]
    if missing_important:
        logger.warning(f"[MAINTENANCE] Eksik önemli kolonlar: {missing_important}")

    # Renaming logic remains the same
    to_rename = {k: v for k, v in col_map.items() if k in df.columns}
    if to_rename:
        df = df.rename(columns=to_rename)
        logger.info(f"[MAINTENANCE] Renamed {len(to_rename)} maintenance/attribute columns")

    # Date parsing logic remains the same
    for date_col in ["Ilk_Bakim_Tarihi", "Son_Bakim_Tarihi"]:
        if date_col in df.columns:
            df[date_col] = df[date_col].apply(parse_date_safely)

    # Numeric conversion logic remains the same
    if "Bakim_Sayisi" in df.columns:
        df["Bakim_Sayisi"] = pd.to_numeric(df["Bakim_Sayisi"], errors="coerce")
    if "Son_Bakimdan_Gecen_Gun" in df.columns:
        df["Son_Bakimdan_Gecen_Gun"] = pd.to_numeric(df["Son_Bakimdan_Gecen_Gun"], errors="coerce")

    return df


def generate_missingness_report(df: pd.DataFrame, logger: logging.Logger, 
                                data_type: str, output_dir: str) -> None:
    """
    Detaylı eksik veri analizi raporu oluşturur.
    """
    logger.info(f"[MISSINGNESS] Generating detailed report for {data_type}...")
    
    # Overall missingness logic remains the same...
    total_rows = len(df)
    missing_summary = []
    
    for col in df.columns:
        missing_count = df[col].isna().sum()
        missing_pct = (missing_count / total_rows) * 100
        missing_summary.append({
            "Column": col,
            "Missing_Count": missing_count,
            "Missing_Percentage": round(missing_pct, 2),
            "Data_Type": str(df[col].dtype)
        })
    
    missing_df = pd.DataFrame(missing_summary).sort_values("Missing_Percentage", ascending=False)
    
    # Critical columns logic remains the same...
    critical_cols = ["Kurulum_Tarihi", "started at", "Süre_Ham", "Ekipman_Tipi"]
    critical_missing = missing_df[missing_df["Column"].isin(critical_cols)]
    
    if not critical_missing.empty:
        logger.warning(f"[MISSINGNESS] Critical columns with missing data:")
        for _, row in critical_missing.iterrows():
            logger.warning(f"  - {row['Column']}: {row['Missing_Count']} ({row['Missing_Percentage']}%)")
    
    # FIX: Add include_groups=False to silence the FutureWarning
    if "Ekipman_Tipi" in df.columns:
        equipment_missing = df.groupby("Ekipman_Tipi").apply(
            lambda x: pd.Series({
                "Total_Count": len(x),
                "Missing_Installation": x["Kurulum_Tarihi"].isna().sum(),
                "Missing_Started": x["started at"].isna().sum() if "started at" in x.columns else 0,
                "Missing_Duration": x["Süre_Ham"].isna().sum() if "Süre_Ham" in x.columns else 0
            }),
            include_groups=False  # <--- THIS IS THE FIX
        ).reset_index()
        
        equipment_missing_path = os.path.join(output_dir, f"{data_type}_missing_by_equipment.csv")
        equipment_missing.to_csv(equipment_missing_path, index=False, encoding="utf-8-sig")
        logger.info(f"[MISSINGNESS] Equipment-wise report → {equipment_missing_path}")
    
    # Save overall report logic remains the same...
    missing_path = os.path.join(output_dir, f"{data_type}_missingness_report.csv")
    missing_df.to_csv(missing_path, index=False, encoding="utf-8-sig")
    logger.info(f"[MISSINGNESS] Overall report → {missing_path}")


def detect_duration_outliers(df: pd.DataFrame, logger: logging.Logger, output_dir: str) -> pd.DataFrame:
    """
    ENHANCED: Uses Median Absolute Deviation (MAD) on Log-Scale.
    Why: Fault durations are Log-Normal. IQR is too aggressive and kills valid long repairs.
    This method only removes physically impossible data errors (e.g., > 6 Sigma).
    """
    logger.info("[OUTLIER] Starting Robust Duration Diagnostics...")
    
    duration_col = "Süre_Dakika"
    if duration_col not in df.columns:
        logger.warning("[OUTLIER] Duration column not found, skipping.")
        return df
    
    # 1. Negative Duration Check (Absolute Physics Violation)
    neg_mask = df[duration_col] <= 0
    if neg_mask.sum() > 0:
        logger.warning(f"[PHYSICS] Dropping {neg_mask.sum()} records with duration <= 0 minutes")
        df = df[~neg_mask].copy()

    # 2. Log-Normal Transformation
    # We add 1 minute (np.log1p) to avoid log(0) errors
    log_durations = np.log1p(df[duration_col])
    
    # 3. Robust Statistics (Median & MAD)
    # MAD is like Standard Deviation but ignores crazy outliers
    median_log = log_durations.median()
    mad_log = stats.median_abs_deviation(log_durations)
    
    # If MAD is 0 (e.g., 50% of data is exactly the same), fallback to a small epsilon
    if mad_log == 0:
        mad_log = 1e-6

    # Standard Sigma equivalent for MAD is 1.4826
    sigma_robust = mad_log * 1.4826
    
    # 4. Calculate Cutoffs (6 Sigma is very safe - only deletes errors)
    upper_limit_log = median_log + (6 * sigma_robust)
    
    # Convert back to minutes for human readability
    upper_limit_min = np.expm1(upper_limit_log)
    median_min = np.expm1(median_log)
    
    logger.info(f"[OUTLIER] Robust Statistics (Log-Normal Assumed):")
    logger.info(f"  - Median Duration: {median_min:.1f} min")
    logger.info(f"  - Upper Cutoff (6σ): {upper_limit_min:.1f} min ({upper_limit_min/60/24:.1f} days)")
    
    # 5. Identify and Flag Outliers
    outlier_mask = log_durations > upper_limit_log
    n_outliers = outlier_mask.sum()
    
    if n_outliers > 0:
        logger.warning(f"[OUTLIER] Detected {n_outliers} extreme outliers (> {upper_limit_min:.0f} min)")
        
        # Save report before dropping
        outlier_records = df[outlier_mask].copy()
        if not outlier_records.empty:
            outlier_path = os.path.join(output_dir, "duration_outliers_report.csv")
            outlier_records.to_csv(outlier_path, index=False, encoding="utf-8-sig")
            logger.info(f"[OUTLIER] Saved report to {outlier_path}")
        
        # Drop them (Filtering)
        df = df[~outlier_mask].copy()
    else:
        logger.info("[OUTLIER] No extreme outliers detected.")
        
    return df

# ---------------------------------------------------------
# Veri Yükleme
# ---------------------------------------------------------

def load_fault_data(logger: logging.Logger) -> pd.DataFrame:
    path = DATA_PATHS["fault_data"]
    logger.info(f"[STEP] Arıza verisi yükleniyor: {path}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"FATAL: Arıza dosyası bulunamadı: {path}")

    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]
    maintenance_cols_to_keep = [
            "Bakım Sayısı", 
            "Son Bakım İş Emri Tarihi", 
            "İlk Bakım İş Emri Tarihi",
            "Son Bakım İş Emri Tipi",
            "MARKA", 
            "kVA_Rating", 
            "component_voltage", 
            "voltage_level"
        ]
# UPDATE: Add new columns to the list of columns to keep
    # We use .get() or check existence to avoid errors if some are missing in future files
# Mevcutsa listeye ekle
    found_maint_cols = [c for c in maintenance_cols_to_keep if c in df.columns]

    # Extra cols (Geo/Customer) + Maintenance Cols
    extra_cols_to_keep = []
    from config.config import EXTRA_FAULT_COLS, COLUMN_MAPPING
    
    for col in EXTRA_FAULT_COLS:
        if col in df.columns:
            extra_cols_to_keep.append(col)
            
    # Hepsini birleştir
    cols_to_use = [
        "cbs_id", "Şebeke Unsuru", "Sebekeye_Baglanma_Tarihi",
        "started at", "ended at", "duration time", "cause code"
    ] + extra_cols_to_keep + found_maint_cols  # <-- BURADA EKLENDİ
    
    # Filter df
    df = df[cols_to_use].copy()
    
    # Rename standard columns (Existing code)
    rename_map = {
        "Şebeke Unsuru": "Ekipman_Tipi",
        "Sebekeye_Baglanma_Tarihi": "Kurulum_Tarihi",
        "cause code": "Ariza_Nedeni",
        "duration time": "Süre_Ham",
    }
    # Add the extra column renames
    for old_col in extra_cols_to_keep:
        if old_col in COLUMN_MAPPING:
            rename_map[old_col] = COLUMN_MAPPING[old_col]
            
    df.rename(columns=rename_map, inplace=True)
    original_count = len(df)
    logger.info(f"[INFO] Orijinal arıza kayıtları: {original_count:,}")

    # cbs_id zorunlu
    df = df[df["cbs_id"].notna()].copy()
    df["cbs_id"] = df["cbs_id"].astype(str).str.lower().str.strip()

    # Temel kolon rename
    df.rename(columns={
        "Şebeke Unsuru": "Ekipman_Tipi",
        "Sebekeye_Baglanma_Tarihi": "Kurulum_Tarihi",
        "cause code": "Ariza_Nedeni",
        "duration time": "Süre_Ham",
    }, inplace=True)

    # ============================================
    # MISSINGNESS ANALYSIS (BEFORE cleaning)
    # ============================================
    output_dir = os.path.dirname(INTERMEDIATE_PATHS["fault_events_clean"])
    os.makedirs(output_dir, exist_ok=True)
    generate_missingness_report(df, logger, "fault_raw", output_dir)

    # Bakım + ekipman nitelikleri rename / tip düzeltme
    df = _rename_maintenance_and_attributes(df, logger)

    # ============================================
    # STEP 1: FILTER MISSING CRITICAL FIELDS FIRST
    # ============================================
    logger.info("")
    logger.info("[FILTER] Step 1: Removing records with missing critical fields...")
    before_missing = len(df)
    
    # Tarih parse (önce parse et, sonra filtrele)
    df["Kurulum_Tarihi"] = df["Kurulum_Tarihi"].apply(parse_date_safely)
    df["started at"] = df["started at"].apply(parse_date_safely)
    df["ended at"] = df["ended at"].apply(parse_date_safely)
    
    # Süre ve ekipman tipi
    df["Süre_Dakika"] = convert_duration_minutes(df["Süre_Ham"], logger)
    df["Ekipman_Tipi"] = clean_equipment_type(df["Ekipman_Tipi"])
    
    # Filter missing critical fields
    df = df[
        df["Kurulum_Tarihi"].notna() &
        df["started at"].notna() &
        df["ended at"].notna() &
        df["Süre_Dakika"].notna()
    ].copy()
    
    dropped_missing = before_missing - len(df)
    if dropped_missing > 0:
        logger.warning(f"[FILTER] {dropped_missing:,} records dropped due to missing critical fields")
        logger.info(f"[FILTER] Remaining records: {len(df):,} ({100*len(df)/original_count:.1f}% of original)")

    # ============================================
    # STEP 2: TEMPORAL VALIDATION (on clean data)
    # ============================================
    logger.info("")
    logger.info("[FILTER] Step 2: Temporal validation on clean data...")
    
    temporal_df = df.copy()
    temporal_df["ended_minus_started"] = (
        temporal_df["ended at"] - temporal_df["started at"]
    ).dt.total_seconds() / 60

    temporal_df["invalid_order"] = temporal_df["ended at"] < temporal_df["started at"]
    temporal_df["invalid_before_install"] = temporal_df["started at"] < temporal_df["Kurulum_Tarihi"]

    # Süre uyuşmazlığı (±5 dk tolerans)
    temporal_df["duration_mismatch"] = np.abs(
        temporal_df["ended_minus_started"] - temporal_df["Süre_Dakika"]
    ) > 5
    
    logger.info(f"[TEMPORAL] ended < started: {temporal_df['invalid_order'].sum()}")
    logger.info(f"[TEMPORAL] started < installation date: {temporal_df['invalid_before_install'].sum()}")
    logger.info(f"[TEMPORAL] duration mismatch > 5 min: {temporal_df['duration_mismatch'].sum()}")

    # Temporal issues report
    issue_mask = (
        temporal_df["invalid_order"] |
        temporal_df["invalid_before_install"] |
        temporal_df["duration_mismatch"]
    )

    temporal_issues = temporal_df[issue_mask].copy()
    if not temporal_issues.empty:
        reasons = []
        for _, row in temporal_issues.iterrows():
            r = []
            if row.get("invalid_order", False):
                r.append("ended<started")
            if row.get("invalid_before_install", False):
                r.append("started<install")
            if row.get("duration_mismatch", False):
                r.append("duration_mismatch")
            reasons.append("|".join(r))
        temporal_issues["temporal_issue_reason"] = reasons

        temporal_report_path = os.path.join(output_dir, "temporal_issues_report.csv")
        temporal_issues.to_csv(temporal_report_path, index=False, encoding="utf-8-sig")
        logger.info(f"[TEMPORAL] Issue report saved → {temporal_report_path}")

    # Drop temporal bad records
    before_temporal = len(df)
    temporal_bad = (
        temporal_df["invalid_order"] |
        temporal_df["invalid_before_install"] |
        temporal_df["duration_mismatch"]
    )
    
    df = df[~temporal_bad].copy()
    dropped_temporal = before_temporal - len(df)
    if dropped_temporal > 0:
        logger.warning(f"[FILTER] {dropped_temporal} records dropped due to temporal issues")

    # Yardımcı kolonları temizle
    for col in ["ended_minus_started", "invalid_order", "invalid_before_install", "duration_mismatch"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True, errors="ignore")

    # ============================================
    # STEP 3: OUTLIER DETECTION
    # ============================================
    df = detect_duration_outliers(df, logger, output_dir)

    logger.info(f"[INFO] Final fault records after cleaning: {len(df):,} ({100*len(df)/original_count:.1f}% of original)")
    logger.info("")
    
    return df


def load_healthy_data(logger: logging.Logger) -> pd.DataFrame:
    path = DATA_PATHS["healthy_data"]
    logger.info(f"[STEP] Sağlam verisi yükleniyor: {path}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"FATAL: Sağlam dosyası bulunamadı: {path}")

    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]

    original_count = len(df)
    logger.info(f"[INFO] Orijinal sağlam ekipman kayıtları: {original_count:,}")

    # cbs_id kontrolü
    if "cbs_id" not in df.columns:
        if "ID" in df.columns:
            logger.warning("[WARN] Sağlam veri 'ID' kolonunu kullanıyor → 'cbs_id' olarak değiştirildi.")
            df.rename(columns={"ID": "cbs_id"}, inplace=True)
        else:
            raise ValueError("FATAL: Sağlam veri 'cbs_id' (veya geçici 'ID') içermeli.")

    df["cbs_id"] = df["cbs_id"].astype(str).str.lower().str.strip()

    df.rename(columns={
        "Şebeke Unsuru": "Ekipman_Tipi",
        "Sebekeye_Baglanma_Tarihi": "Kurulum_Tarihi",
    }, inplace=True)

    # Missingness analysis
    output_dir = os.path.dirname(INTERMEDIATE_PATHS["fault_events_clean"])
    generate_missingness_report(df, logger, "healthy_raw", output_dir)

    # Bakım + ekipman nitelikleri rename / tip düzeltme
    df = _rename_maintenance_and_attributes(df, logger)

    df["Kurulum_Tarihi"] = df["Kurulum_Tarihi"].apply(parse_date_safely)
    df["Ekipman_Tipi"] = clean_equipment_type(df["Ekipman_Tipi"])

    before_filter = len(df)
    df = df[df["Kurulum_Tarihi"].notna() & df["cbs_id"].notna()].copy()
    dropped = before_filter - len(df)
    if dropped > 0:
        logger.warning(f"[FILTER] {dropped} healthy records dropped due to missing Kurulum_Tarihi or cbs_id")
    
    logger.info(f"[INFO] Final healthy records: {len(df):,} ({100*len(df)/original_count:.1f}% of original)")
    logger.info("")

    return df


# ---------------------------------------------------------
# Maintenance Data Completeness Logging
# ---------------------------------------------------------

def log_maintenance_completeness(df: pd.DataFrame, logger: logging.Logger, source: str) -> None:
    """
    Bakım verisi tamlık analizi ve loglama.
    """
    logger.info(f"[MAINTENANCE COMPLETENESS] Analyzing {source} data...")
    
    maint_cols = {
        "Bakim_Sayisi": "Maintenance Count",
        "Son_Bakim_Tarihi": "Last Maintenance Date",
        "Ilk_Bakim_Tarihi": "First Maintenance Date",
        "Son_Bakim_Tipi": "Last Maintenance Type",
        "Marka": "Brand",
        "kVA_Rating": "kVA Rating",
        "Gerilim_Seviyesi": "Voltage Level"
    }
    
    total = len(df)
    for col_name, col_label in maint_cols.items():
        if col_name in df.columns:
            non_missing = df[col_name].notna().sum()
            completeness = (non_missing / total) * 100
            logger.info(f"  - {col_label}: {non_missing:,}/{total:,} ({completeness:.1f}%)")
            
            # Ekipman tipine göre breakdown
            if "Ekipman_Tipi" in df.columns and non_missing > 0:
                by_type = df.groupby("Ekipman_Tipi")[col_name].apply(
                    lambda x: f"{x.notna().sum()}/{len(x)} ({100*x.notna().sum()/len(x):.1f}%)"
                )
                logger.info(f"    By equipment type:")
                for eq_type, stat in by_type.items():
                    logger.info(f"      * {eq_type}: {stat}")
        else:
            logger.warning(f"  - {col_label}: COLUMN NOT FOUND")
    
    # Bakım sayısı analizi
    if "Bakim_Sayisi" in df.columns:
        has_maintenance = df["Bakim_Sayisi"].notna() & (df["Bakim_Sayisi"] > 0)
        logger.info(f"[MAINTENANCE] Equipment with maintenance records: {has_maintenance.sum():,}/{total:,} ({100*has_maintenance.sum()/total:.1f}%)")
    
    logger.info("")


# ---------------------------------------------------------
# Tablo İnşası
# ---------------------------------------------------------

def build_fault_events(df_fault: pd.DataFrame) -> pd.DataFrame:
    return df_fault[[
        "cbs_id", "Ekipman_Tipi", "Kurulum_Tarihi",
        "started at", "ended at", "Süre_Dakika", "Ariza_Nedeni"
    ]].rename(columns={
        "started at": "Ariza_Baslangic_Zamani",
        "ended at": "Ariza_Bitis_Zamani",
        "Süre_Dakika": "Kesinti_Suresi_Dakika",
    })


def _aggregate_equipment_block(df: pd.DataFrame, logger: logging.Logger, source: str) -> pd.DataFrame:
    """
    fault / sağlam kaynaktan ekipman bazlı agregasyon.
    Bakım ve ekipman niteliklerini de mümkün olduğunca toplar.
    """
    if df.empty:
        return pd.DataFrame(columns=["cbs_id"])

    agg_dict = {
        "Kurulum_Tarihi": ("Kurulum_Tarihi", "min"),
        "Ekipman_Tipi": ("Ekipman_Tipi", lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]),
    }

    if source == "fault":
            agg_dict.update({
                "Fault_Count": ("cbs_id", "size"),
                "Ilk_Ariza_Tarihi": ("started at", "min"),
                "Son_Ariza_Tarihi": ("started at", "max"),
            })
            
            # NEW: Aggregate Geo & Consequence Data
            # Strategy: 
            # - Location: Take 'first' (assuming it doesn't move)
            # - Customer Count: Take 'max' (Worst case scenario for risk)
            # - Urban/Rural: Take 'max' (If it was ever Urban, treat as Urban)
            
            # Check which renamed columns exist in the df
            possible_geo_cols = ["Latitude", "Longitude", "Sehir", "Ilce", "Mahalle"]
            for col in possible_geo_cols:
                if col in df.columns:
                    agg_dict[col] = (col, "first")
                    
            if "Musteri_Sayisi" in df.columns:
                agg_dict["Musteri_Sayisi"] = ("Musteri_Sayisi", "max")
                
            # For the raw urban/rural columns (if they exist)
            env_cols = ["urban mv", "urban lv", "rural mv", "rural lv", "suburban mv", "suburban lv"]
            for col in env_cols:
                if col in df.columns:
                    agg_dict[col] = (col, "max")

    # Opsiyonel bakım/nitelik kolonları
    opt_cols = [
        "Bakim_Sayisi",
        "Bakim_Is_Emri_Tipleri",
        "Ilk_Bakim_Tarihi",
        "Son_Bakim_Tarihi",
        "Son_Bakim_Tipi",
        "Son_Bakimdan_Gecen_Gun",
        "Marka",
        "Gerilim_Seviyesi",
        "Gerilim_Sinifi",
        "kVA_Rating",
    ]

    for col in opt_cols:
        if col in df.columns:
            if col in ["Bakim_Sayisi", "Son_Bakimdan_Gecen_Gun", "kVA_Rating"]:
                agg_dict[col] = (col, "max")
            elif col in ["Ilk_Bakim_Tarihi"]:
                agg_dict[col] = (col, "min")
            elif col in ["Son_Bakim_Tarihi"]:
                agg_dict[col] = (col, "max")
            else:
                agg_dict[col] = (
                    col,
                    lambda x: x.mode().iloc[0] if not x.mode().empty else x.dropna().iloc[0]
                    if x.dropna().size > 0 else np.nan
                )

    grouped = df.groupby("cbs_id").agg(**agg_dict).reset_index()
    logger.info(f"[INFO] '{source}' kaynağından ekipman agregasyonu: {len(grouped)} satır")
    return grouped


def build_equipment_master(df_fault: pd.DataFrame,
                           df_healthy: pd.DataFrame,
                           logger: logging.Logger,
                           data_end_date: pd.Timestamp) -> pd.DataFrame:
    """
    FIXED: Prioritize fault records over healthy records during deduplication.
    AGE CALCULATION: Use DATA_END_DATE (last fault date) instead of ANALYSIS_DATE.
    """
    fault_part = _aggregate_equipment_block(df_fault, logger, source="fault")

    healthy_part = _aggregate_equipment_block(df_healthy, logger, source="healthy")
    if not healthy_part.empty:
        if "Fault_Count" not in healthy_part.columns:
            healthy_part["Fault_Count"] = 0
        if "Ilk_Ariza_Tarihi" not in healthy_part.columns:
            healthy_part["Ilk_Ariza_Tarihi"] = pd.NaT
        if "Son_Ariza_Tarihi" not in healthy_part.columns:
            healthy_part["Son_Ariza_Tarihi"] = pd.NaT

    all_eq = pd.concat([fault_part, healthy_part], ignore_index=True)
    
    # ============================================
    # FIXED: Prioritize fault records explicitly
    # ============================================
    logger.info("[DEDUP] Deduplicating equipment records (prioritizing fault records)...")
    before_dedup = len(all_eq)
    
    # Sort by Fault_Count descending, then by Kurulum_Tarihi ascending
    # This ensures fault records come first, and among fault records, earliest installation is kept
    all_eq = all_eq.sort_values(
        ["cbs_id", "Fault_Count", "Kurulum_Tarihi"], 
        ascending=[True, False, True]
    ).drop_duplicates("cbs_id", keep="first")
    
    dropped_dupes = before_dedup - len(all_eq)
    if dropped_dupes > 0:
        logger.info(f"[DEDUP] Dropped {dropped_dupes} duplicate cbs_id records (kept fault records)")
# NEW: Flag missing Location/Consequence data
    # This helps us identify "Ghost Assets" (we know they exist, but not where)
    if "Latitude" in all_eq.columns:
        all_eq["Location_Known"] = all_eq["Latitude"].notna().astype(int)
    else:
        all_eq["Location_Known"] = 0
        
    # Default Customer Count to Median if missing (or 0, depending on preference)
    # For Risk Analysis, 0 is dangerous. Let's flag it.
    if "Musteri_Sayisi" in all_eq.columns:
         # Fill with -1 to indicate 'Unknown' for now
         all_eq["Musteri_Sayisi"] = all_eq["Musteri_Sayisi"].fillna(-1)
    # ============================================
    # AGE CALCULATION: Use DATA_END_DATE instead of ANALYSIS_DATE
    # ============================================
    logger.info(f"[AGE] Calculating equipment age using DATA_END_DATE: {data_end_date.date()}")
    
    # For equipment WITH faults: age = (last fault date - installation date)
    # For equipment WITHOUT faults: age = (data_end_date - installation date)
    all_eq["Ekipman_Yasi_Gun"] = np.where(
        all_eq["Fault_Count"] > 0,
        (all_eq["Son_Ariza_Tarihi"] - all_eq["Kurulum_Tarihi"]).dt.days,
        (data_end_date - all_eq["Kurulum_Tarihi"]).dt.days
    )
    
    # Clip negative ages to 0
    all_eq["Ekipman_Yasi_Gun"] = all_eq["Ekipman_Yasi_Gun"].clip(lower=0)
    
    all_eq["Ariza_Gecmisi"] = (all_eq["Fault_Count"] > 0).astype(int)

    # Log age statistics
    logger.info(f"[AGE] Equipment age statistics (days):")
    logger.info(f"  - Mean: {all_eq['Ekipman_Yasi_Gun'].mean():.0f}")
    logger.info(f"  - Median: {all_eq['Ekipman_Yasi_Gun'].median():.0f}")
    logger.info(f"  - Min: {all_eq['Ekipman_Yasi_Gun'].min():.0f}")
    logger.info(f"  - Max: {all_eq['Ekipman_Yasi_Gun'].max():.0f}")

    # Nadir sınıfları grupla
    counts = all_eq["Ekipman_Tipi"].value_counts()
    rare = counts[counts < MIN_EQUIPMENT_PER_CLASS].index.tolist()

    if rare:
        logger.info(f"[INFO] Nadir ekipman sınıfları 'Diger' altına alındı: {rare}")
        all_eq.loc[all_eq["Ekipman_Tipi"].isin(rare), "Ekipman_Tipi"] = "Diger"

    return all_eq


def build_survival_base(eq: pd.DataFrame,
                        events: pd.DataFrame,
                        logger: logging.Logger,
                        data_end_date: pd.Timestamp) -> pd.DataFrame:
    """
    FIXED: Use DATA_END_DATE for censoring instead of ANALYSIS_DATE.
    """
    first_fail = events.groupby("cbs_id")["Ariza_Baslangic_Zamani"].min()
    eq = eq.merge(first_fail.rename("Ilk_Ariza"), on="cbs_id", how="left")

    eq["Ilk_Ariza_Tarihi"] = eq["Ilk_Ariza"].fillna(eq.get("Ilk_Ariza_Tarihi"))
    eq.drop(columns=["Ilk_Ariza"], inplace=True)

    eq["event"] = eq["Ilk_Ariza_Tarihi"].notna().astype(int)

    # Use DATA_END_DATE for censoring
    eq["duration_days"] = np.where(
        eq["event"] == 1,
        (eq["Ilk_Ariza_Tarihi"] - eq["Kurulum_Tarihi"]).dt.days,
        (data_end_date - eq["Kurulum_Tarihi"]).dt.days,
    )

    eq = eq[eq["duration_days"] > 0].copy()

    # Extreme duration handling
    extreme_threshold = 60 * 365  # 60 years
    too_long = (eq["duration_days"] > extreme_threshold).sum()
    
    if too_long:
        logger.warning(f"[WARN] {too_long} kayıt 60 yıldan uzun süreye sahip - detay rapor oluşturuluyor.")
        
        # Create detailed report for extreme cases
        extreme_records = eq[eq["duration_days"] > extreme_threshold].copy()
        extreme_records["duration_years"] = extreme_records["duration_days"] / 365.25
        
        extreme_path = os.path.join(
            os.path.dirname(INTERMEDIATE_PATHS["survival_base"]),
            "extreme_duration_report.csv"
        )
        extreme_records.to_csv(extreme_path, index=False, encoding="utf-8-sig")
        logger.info(f"[WARN] Extreme duration report → {extreme_path}")
        
        # Clip to 60 years
        eq["duration_days"] = eq["duration_days"].clip(upper=extreme_threshold)
        logger.info(f"[WARN] Durations clipped to {extreme_threshold} days (60 years)")

    logger.info(f"[SURVIVAL] Final survival base: {len(eq):,} equipment records")
    logger.info(f"[SURVIVAL] Event rate: {eq['event'].mean():.2%} ({eq['event'].sum():,} failures)")
    
    return eq


# ---------------------------------------------------------
# Ana İşlem
# ---------------------------------------------------------

def main():
    logger = setup_logger()

    try:
        df_fault = load_fault_data(logger)
        df_healthy = load_healthy_data(logger)

        # ---------------------------------------------------------
        # DATA_START_DATE and DATA_END_DATE Auto-Detection
        # ---------------------------------------------------------
        logger.info("")
        logger.info("[DATA RANGE DETECTION] Detecting temporal boundaries from fault data...")

        DATA_START_DATE = df_fault["started at"].min()
        DATA_END_DATE = df_fault["started at"].max()
        data_span_days = (DATA_END_DATE - DATA_START_DATE).days
        data_span_years = data_span_days / 365.25

        logger.info(f"[DATA RANGE] DATA_START_DATE = {DATA_START_DATE.date()}")
        logger.info(f"[DATA RANGE] DATA_END_DATE   = {DATA_END_DATE.date()}")
        logger.info(f"[DATA RANGE] Data span       = {data_span_years:.2f} years ({data_span_days:,} days)")
        logger.info("")

        # Validation: Minimum 2 years required for ML training
        MIN_REQUIRED_YEARS = 2.0
        if data_span_years < MIN_REQUIRED_YEARS:
            logger.error(f"[FATAL] Insufficient data span: {data_span_years:.2f} years < {MIN_REQUIRED_YEARS} years")
            logger.error("[FATAL] At least 2 years of fault data required for reliable PoF models")
            raise ValueError(f"Insufficient data: {data_span_years:.2f} years < {MIN_REQUIRED_YEARS} years required")

        logger.info(f"[OK] Data span validation passed: {data_span_years:.2f} years >= {MIN_REQUIRED_YEARS} years")

        # Save metadata for downstream scripts
        metadata = pd.DataFrame({
            "Parameter": ["DATA_START_DATE", "DATA_END_DATE", "DATA_SPAN_DAYS", "DATA_SPAN_YEARS"],
            "Value": [
                DATA_START_DATE.strftime("%Y-%m-%d"),
                DATA_END_DATE.strftime("%Y-%m-%d"),
                str(data_span_days),
                f"{data_span_years:.2f}"
            ]
        })

        metadata_path = os.path.join(
            os.path.dirname(INTERMEDIATE_PATHS["fault_events_clean"]),
            "data_range_metadata.csv"
        )
        metadata.to_csv(metadata_path, index=False, encoding="utf-8-sig")
        logger.info(f"[INFO] Data range metadata saved → {metadata_path}")
        logger.info("")
        
        # ---------------------------------------------------------
        # MAINTENANCE DATA COMPLETENESS ANALYSIS
        # ---------------------------------------------------------
        logger.info("=" * 80)
        logger.info("MAINTENANCE DATA COMPLETENESS ANALYSIS")
        logger.info("=" * 80)
        log_maintenance_completeness(df_fault, logger, "FAULT")
        log_maintenance_completeness(df_healthy, logger, "HEALTHY")
        logger.info("=" * 80)
        logger.info("")

        # ---------------------------------------------------------
        # Build tables with DATA_END_DATE
        # ---------------------------------------------------------
        fault_events = build_fault_events(df_fault)
        equipment_master = build_equipment_master(df_fault, df_healthy, logger, DATA_END_DATE)
        survival_base = build_survival_base(equipment_master, fault_events, logger, DATA_END_DATE)

        # --- Teknik çıktılar ---
        os.makedirs(os.path.dirname(INTERMEDIATE_PATHS["fault_events_clean"]), exist_ok=True)

        fault_events.to_csv(INTERMEDIATE_PATHS["fault_events_clean"], index=False, encoding="utf-8-sig")
        df_healthy.to_csv(INTERMEDIATE_PATHS["healthy_equipment_clean"], index=False, encoding="utf-8-sig")
        equipment_master.to_csv(INTERMEDIATE_PATHS["equipment_master"], index=False, encoding="utf-8-sig")
        survival_base.to_csv(INTERMEDIATE_PATHS["survival_base"], index=False, encoding="utf-8-sig")

        # --- Türkiye EDAŞ için müşteri-facing çıktılar ---
        fault_events.to_csv(OUTPUT_PATHS["ariza_kayitlari"], index=False, encoding="utf-8-sig")
        equipment_master.to_csv(OUTPUT_PATHS["ekipman_listesi"], index=False, encoding="utf-8-sig")
        survival_base.to_csv(OUTPUT_PATHS["sagkalim_taban"], index=False, encoding="utf-8-sig")
        df_healthy.to_csv(OUTPUT_PATHS["saglam_ekipman_listesi"], index=False, encoding="utf-8-sig")

        logger.info("")
        logger.info("=" * 80)
        logger.info("[SUCCESS] 01_veri_isleme tamamlandı (Enhanced v2)")
        logger.info("=" * 80)
        logger.info("")
        logger.info("SUMMARY:")
        logger.info(f"  - Fault events: {len(fault_events):,}")
        logger.info(f"  - Equipment master: {len(equipment_master):,}")
        logger.info(f"  - Survival base: {len(survival_base):,}")
        logger.info(f"  - Healthy equipment: {len(df_healthy):,}")
        logger.info("")

    except Exception as e:
        logger.exception(f"[FATAL] 01_veri_isleme hatası: {e}")
        raise


if __name__ == "__main__":
    main()