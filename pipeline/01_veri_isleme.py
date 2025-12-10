"""
01_veri_isleme.py (PoF3 - Nihai Türkçe Uyumlu Sürüm)

Amaç:
- Ham arıza + sağlam veri yükleme
- Zorunlu veri sözleşmesi (cbs_id küçük harf zorunlu)
- Tarih, süre, ekipman tipi temizliği
- Bakım ve ekipman niteliklerini (MARKA, gerilim, kVA, vb.) ekipman_master'a agregasyon
- Aşağıdaki teknik çıktıları üretir:
    data/ara_ciktilar/fault_events_clean.csv
    data/ara_ciktilar/healthy_equipment_clean.csv
    data/ara_ciktilar/equipment_master.csv
    data/ara_ciktilar/survival_base.csv
    data/ara_ciktilar/temporal_issues_report.csv

Aşağıdaki müşteri-facing (Türkçe) çıktıları üretir:
    data/sonuclar/ariza_kayitlari.csv
    data/sonuclar/ekipman_listesi.csv
    data/sonuclar/sagkalim_taban.csv
    data/sonuclar/saglam_ekipman_listesi.csv
"""

import os
import sys
import logging
from datetime import datetime

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

# Proje kökünü path'e ekle
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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
    logger.info("01_veri_isleme - PoF3 Veri İşleme")
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
    iç standarda çevirir. Eksik olanları sessizce atlar.
    """
    col_map = {
        "Bakım Sayısı": "Bakim_Sayisi",
        "Geçmiş İş Emri Tipleri": "Bakim_Is_Emri_Tipleri",
        "İlk Bakım İş Emri Tarihi": "Ilk_Bakim_Tarihi",
        "Son Bakım": "Son_Bakim_Tarihi",
        "Son Bakım İş Emri Tipi": "Son_Bakim_Tipi",
        "Son Bakımdan İtibaren Geçen Gün Sayısı": "Son_Bakimdan_Gecen_Gun",
        "MARKA": "Marka",
        "component voltage": "Gerilim_Seviyesi",
        "voltage_level": "Gerilim_Sinifi",
        "kVA_Rating": "kVA_Rating",
    }

    # Mevcut olanları renameliyoruz
    to_rename = {k: v for k, v in col_map.items() if k in df.columns}
    if to_rename:
        df = df.rename(columns=to_rename)

    # Tarih tipine çevrilecek kolonlar
    for date_col in ["Ilk_Bakim_Tarihi", "Son_Bakim_Tarihi"]:
        if date_col in df.columns:
            df[date_col] = df[date_col].apply(parse_date_safely)

    # Sayısal kolonlar
    if "Bakim_Sayisi" in df.columns:
        df["Bakim_Sayisi"] = pd.to_numeric(df["Bakim_Sayisi"], errors="coerce")
    if "Son_Bakimdan_Gecen_Gun" in df.columns:
        df["Son_Bakimdan_Gecen_Gun"] = pd.to_numeric(df["Son_Bakimdan_Gecen_Gun"], errors="coerce")

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

    required = [
        "cbs_id", "Şebeke Unsuru", "Sebekeye_Baglanma_Tarihi",
        "started at", "ended at", "duration time", "cause code"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"FATAL: Arıza verisinde eksik kolonlar var: {missing}")

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

    # Bakım + ekipman nitelikleri rename / tip düzeltme
    df = _rename_maintenance_and_attributes(df, logger)

    # Tarih parse
    df["Kurulum_Tarihi"] = df["Kurulum_Tarihi"].apply(parse_date_safely)
    df["started at"] = df["started at"].apply(parse_date_safely)
    df["ended at"] = df["ended at"].apply(parse_date_safely)

    # Süre ve ekipman tipi
    df["Süre_Dakika"] = convert_duration_minutes(df["Süre_Ham"], logger)
    df["Ekipman_Tipi"] = clean_equipment_type(df["Ekipman_Tipi"])

    # -------------------------------------------------------------
    # Temporal validation checks + rapor
    # -------------------------------------------------------------
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

    # Sorunlu satırlar için rapor
    issue_mask = (
        temporal_df["invalid_order"] |
        temporal_df["invalid_before_install"] |
        temporal_df["duration_mismatch"] |
        temporal_df["Kurulum_Tarihi"].isna() |
        temporal_df["started at"].isna() |
        temporal_df["Süre_Dakika"].isna()
    )

    temporal_issues = temporal_df[issue_mask].copy()
    if not temporal_issues.empty:
        # Issue nedeni kolonu
        reasons = []
        for _, row in temporal_issues.iterrows():
            r = []
            if row.get("invalid_order", False):
                r.append("ended<started")
            if row.get("invalid_before_install", False):
                r.append("started<install")
            if row.get("duration_mismatch", False):
                r.append("duration_mismatch")
            if pd.isna(row.get("Kurulum_Tarihi")):
                r.append("missing_install")
            if pd.isna(row.get("started at")):
                r.append("missing_started")
            if pd.isna(row.get("Süre_Dakika")):
                r.append("missing_duration")
            reasons.append("|".join(r))
        temporal_issues["temporal_issue_reason"] = reasons

        temporal_report_path = os.path.join(
            os.path.dirname(INTERMEDIATE_PATHS["fault_events_clean"]),
            "temporal_issues_report.csv",
        )
        os.makedirs(os.path.dirname(temporal_report_path), exist_ok=True)
        temporal_issues.to_csv(temporal_report_path, index=False, encoding="utf-8-sig")
        logger.info(f"[INFO] Temporal issue report saved → {temporal_report_path}")

    # Önce saf temporal tutarsızlıkları at
    before_temporal = len(df)
    temporal_bad = (
        temporal_df["invalid_order"] |
        temporal_df["invalid_before_install"] |
        temporal_df["duration_mismatch"]
    )
    df = df[~temporal_bad].copy()
    dropped_temporal = before_temporal - len(df)
    if dropped_temporal > 0:
        logger.warning(f"[WARN] Zaman tutarsızlığı olan {dropped_temporal} satır atıldı.")

    # Sonra eksik zorunlu alanlara göre filtrele
    before_missing = len(df)
    df = df[
        df["Kurulum_Tarihi"].notna() &
        df["started at"].notna() &
        df["Süre_Dakika"].notna()
    ].copy()
    dropped_missing = before_missing - len(df)
    if dropped_missing > 0:
        logger.warning(f"[WARN] Tarih/süre eksikliği nedeniyle {dropped_missing} satır atıldı.")

    # Yardımcı kolonları temizle (analize gitmesin)
    for col in ["ended_minus_started", "invalid_order", "invalid_before_install", "duration_mismatch"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True, errors="ignore")

    return df


def load_healthy_data(logger: logging.Logger) -> pd.DataFrame:
    path = DATA_PATHS["healthy_data"]
    logger.info(f"[STEP] Sağlam verisi yükleniyor: {path}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"FATAL: Sağlam dosyası bulunamadı: {path}")

    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]

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

    # Bakım + ekipman nitelikleri rename / tip düzeltme
    df = _rename_maintenance_and_attributes(df, logger)

    df["Kurulum_Tarihi"] = df["Kurulum_Tarihi"].apply(parse_date_safely)
    df["Ekipman_Tipi"] = clean_equipment_type(df["Ekipman_Tipi"])

    df = df[df["Kurulum_Tarihi"].notna() & df["cbs_id"].notna()].copy()

    return df


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
        })

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
                # string ve kategori kolonlar için ilk non-null veya mode
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
                           logger: logging.Logger) -> pd.DataFrame:
    fault_part = _aggregate_equipment_block(df_fault, logger, source="fault")

    healthy_part = _aggregate_equipment_block(df_healthy, logger, source="healthy")
    if not healthy_part.empty:
        # Sağlam tarafta arıza bilgisi yok
        if "Fault_Count" not in healthy_part.columns:
            healthy_part["Fault_Count"] = 0
        if "Ilk_Ariza_Tarihi" not in healthy_part.columns:
            healthy_part["Ilk_Ariza_Tarihi"] = pd.NaT

    all_eq = pd.concat([fault_part, healthy_part], ignore_index=True)
    # fault kısmı öncelikli kalsın
    all_eq = all_eq.sort_values(["cbs_id", "Kurulum_Tarihi"]).drop_duplicates("cbs_id", keep="first")

    # Yaş ve arıza geçmişi
    analysis_dt = pd.to_datetime(ANALYSIS_DATE)
    all_eq["Ekipman_Yasi_Gun"] = (analysis_dt - all_eq["Kurulum_Tarihi"]).dt.days.clip(lower=0)
    all_eq["Ariza_Gecmisi"] = (all_eq["Fault_Count"] > 0).astype(int)

    # Nadir sınıfları grupla
    counts = all_eq["Ekipman_Tipi"].value_counts()
    rare = counts[counts < MIN_EQUIPMENT_PER_CLASS].index.tolist()

    if rare:
        logger.info(f"[INFO] Nadir ekipman sınıfları 'Other' altına alındı: {rare}")
        all_eq.loc[all_eq["Ekipman_Tipi"].isin(rare), "Ekipman_Tipi"] = "Other"

    return all_eq


def build_survival_base(eq: pd.DataFrame,
                        events: pd.DataFrame,
                        logger: logging.Logger) -> pd.DataFrame:
    first_fail = events.groupby("cbs_id")["Ariza_Baslangic_Zamani"].min()
    eq = eq.merge(first_fail.rename("Ilk_Ariza"), on="cbs_id", how="left")

    eq["Ilk_Ariza_Tarihi"] = eq["Ilk_Ariza"].fillna(eq.get("Ilk_Ariza_Tarihi"))
    eq.drop(columns=["Ilk_Ariza"], inplace=True)

    analysis_dt = pd.to_datetime(ANALYSIS_DATE)

    eq["event"] = eq["Ilk_Ariza_Tarihi"].notna().astype(int)

    eq["duration_days"] = np.where(
        eq["event"] == 1,
        (eq["Ilk_Ariza_Tarihi"] - eq["Kurulum_Tarihi"]).dt.days,
        (analysis_dt - eq["Kurulum_Tarihi"]).dt.days,
    )

    eq = eq[eq["duration_days"] > 0].copy()

    too_long = (eq["duration_days"] > 60 * 365).sum()
    if too_long:
        logger.warning(f"[WARN] 60 yıldan uzun süreli {too_long} kayıt kesildi.")
        eq["duration_days"] = eq["duration_days"].clip(upper=60 * 365)

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

        # Save metadata for Step 03
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

        fault_events = build_fault_events(df_fault)
        equipment_master = build_equipment_master(df_fault, df_healthy, logger)
        survival_base = build_survival_base(equipment_master, fault_events, logger)

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

        logger.info("[SUCCESS] 01_veri_isleme tamamlandı.")

    except Exception as e:
        logger.exception(f"[FATAL] 01_veri_isleme hatası: {e}")
        raise


if __name__ == "__main__":
    main()
