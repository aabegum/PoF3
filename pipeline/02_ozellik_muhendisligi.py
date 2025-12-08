"""
02_ozellik_muhendisligi.py  (PoF3 - Genişletilmiş Türkçe Özellik Mühendisliği)

Amaç:
- 01_veri_isleme çıktıları olan:
    data/ara_ciktilar/equipment_master.csv
    data/ara_ciktilar/fault_events_clean.csv
  dosyalarını yükler.
- cbs_id şemasını zorunlu olarak küçük harf ve string yapar.
- Aşağıdaki temel özellikleri üretir:
    - Ekipman_Tipi
    - Ekipman_Yasi_Gun
    - Ariza_Gecmisi
    - Ariza_Sayisi (Fault_Count)
    - Ilk_Ariza_Tarihi
    - Son_Ariza_Tarihi
    - Son_Ariza_Gun_Sayisi
    - MTBF_Gun
    - Kronik_90g_Flag (IEEE 1366 - 90g penceresi)
- Bakım ve ekipman niteliklerine dayalı ek özellikler:
    - Bakim_Sayisi
    - Bakim_Var_Mi
    - Ilk_Bakim_Tarihi
    - Son_Bakim_Tarihi
    - Son_Bakim_Tipi
    - Son_Bakimdan_Gecen_Gun
    - Marka, Gerilim_Seviyesi, Gerilim_Sinifi, kVA_Rating

Çıktı:
    data/ara_ciktilar/ozellikler_pof3.csv
ve
    data/ara_ciktilar/ozellik_sanity_report.csv
"""

import os
import sys
import logging
from datetime import datetime

import numpy as np
import pandas as pd

# UTF-8 konsol
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Proje kökü
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.date_parser import parse_date_safely
from config.config import (
    ANALYSIS_DATE,
    INTERMEDIATE_PATHS,
    FEATURE_OUTPUT_PATH,
    CHRONIC_WINDOW_DAYS,
    LOG_DIR,
)


STEP_NAME = "02_ozellik_muhendisligi"


# ---------------------------------------------------------
# LOGGING
# ---------------------------------------------------------

def setup_logger() -> logging.Logger:
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"{STEP_NAME}_{ts}.log")

    logger = logging.getLogger(STEP_NAME)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    logger.info("=" * 80)
    logger.info(f"{STEP_NAME} - PoF3 Özellik Mühendisliği (Genişletilmiş)")
    logger.info("=" * 80)
    logger.info(f"Analiz Tarihi: {ANALYSIS_DATE}")
    logger.info("")
    return logger


# ---------------------------------------------------------
# MTBF & Kronik Hesaplama
# ---------------------------------------------------------

def compute_mtbf_and_chronic(events: pd.DataFrame,
                             logger: logging.Logger) -> pd.DataFrame:
    """
    MTBF (Mean Time Between Failures) ve 90 günlük kronik flag hesaplar.
    """
    if events.empty:
        return pd.DataFrame(columns=["cbs_id", "MTBF_Gun", "Kronik_90g_Flag"])

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

    out = mtbf_df.merge(chronic_df, on="cbs_id", how="outer")
    return out


# ---------------------------------------------------------
# ANA İŞLEM
# ---------------------------------------------------------

def main():
    logger = setup_logger()

    try:
        # -------------------------------------------------
        # 1) Verileri yükle
        # -------------------------------------------------
        eq_path = INTERMEDIATE_PATHS["equipment_master"]
        events_path = INTERMEDIATE_PATHS["fault_events_clean"]

        logger.info(f"[STEP] equipment_master yükleniyor: {eq_path}")
        equipment = pd.read_csv(eq_path, encoding="utf-8-sig")

        logger.info(f"[STEP] fault_events_clean yükleniyor: {events_path}")
        events = pd.read_csv(events_path, encoding="utf-8-sig") if os.path.exists(events_path) else pd.DataFrame()

        # cbs_id şeması
        equipment["cbs_id"] = equipment["cbs_id"].astype(str).str.lower().str.strip()
        if not events.empty:
            events["cbs_id"] = events["cbs_id"].astype(str).str.lower().str.strip()

        # Tarih parse
        equipment["Kurulum_Tarihi"] = equipment["Kurulum_Tarihi"].apply(parse_date_safely)
        if "Ilk_Ariza_Tarihi" in equipment.columns:
            equipment["Ilk_Ariza_Tarihi"] = equipment["Ilk_Ariza_Tarihi"].apply(parse_date_safely)

        for col in ["Ilk_Bakim_Tarihi", "Son_Bakim_Tarihi"]:
            if col in equipment.columns:
                equipment[col] = equipment[col].apply(parse_date_safely)

        if not events.empty:
            events["Ariza_Baslangic_Zamani"] = events["Ariza_Baslangic_Zamani"].apply(parse_date_safely)

        analysis_dt = pd.to_datetime(ANALYSIS_DATE)

        # -------------------------------------------------
        # 2) Temel özellik seti
        # -------------------------------------------------
        base_cols = [
            "cbs_id",
            "Ekipman_Tipi",
            "Kurulum_Tarihi",
            "Ekipman_Yasi_Gun",
            "Ariza_Gecmisi",
            "Fault_Count",
        ]
        if "Ilk_Ariza_Tarihi" in equipment.columns:
            base_cols.append("Ilk_Ariza_Tarihi")

        existing_base = [c for c in base_cols if c in equipment.columns]
        features = equipment[existing_base].copy()

        # Türkçe isimlendirme: Fault_Count → Ariza_Sayisi
        if "Fault_Count" in features.columns:
            features.rename(columns={"Fault_Count": "Ariza_Sayisi"}, inplace=True)

        # -------------------------------------------------
        # 3) Son arıza bilgisi
        # -------------------------------------------------
        if not events.empty:
            lastfault = (
                events.groupby("cbs_id")["Ariza_Baslangic_Zamani"]
                .max()
                .rename("Son_Ariza_Tarihi")
            )
            features = features.merge(lastfault, on="cbs_id", how="left")
            features["Son_Ariza_Tarihi"] = features["Son_Ariza_Tarihi"].apply(parse_date_safely)
            features["Son_Ariza_Gun_Sayisi"] = (
                analysis_dt - features["Son_Ariza_Tarihi"]
            ).dt.days
        else:
            features["Son_Ariza_Tarihi"] = pd.NaT
            features["Son_Ariza_Gun_Sayisi"] = np.nan

        # -------------------------------------------------
        # 4) MTBF + kronik (90g) flag
        # -------------------------------------------------
        if not events.empty:
            mtbf_df = compute_mtbf_and_chronic(events, logger)
            features = features.merge(mtbf_df, on="cbs_id", how="left")
        else:
            features["MTBF_Gun"] = np.nan
            features["Kronik_90g_Flag"] = 0

        # -------------------------------------------------
        # 5) Bakım & ekipman nitelikleri (equipment_master içinden)
        # -------------------------------------------------
        # Bu kolonlar 01_veri_isleme içinde agregasyonla üretilmiş durumda
        bakim_cols = [
            "Bakim_Sayisi",
            "Bakim_Is_Emri_Tipleri",
            "Ilk_Bakim_Tarihi",
            "Son_Bakim_Tarihi",
            "Son_Bakim_Tipi",
            "Son_Bakimdan_Gecen_Gun",
        ]
        attr_cols = [
            "Marka",
            "Gerilim_Seviyesi",
            "Gerilim_Sinifi",
            "kVA_Rating",
        ]

        # Önce bu kolonları equipment'tan çek
        extra_cols = [c for c in bakim_cols + attr_cols if c in equipment.columns]
        if extra_cols:
            extra_df = equipment[["cbs_id"] + extra_cols].copy()
            features = features.merge(extra_df, on="cbs_id", how="left")

        # Bakım sayısı ve bayraklar
        if "Bakim_Sayisi" in features.columns:
            features["Bakim_Sayisi"] = features["Bakim_Sayisi"].fillna(0).astype(int)
            features["Bakim_Var_Mi"] = (features["Bakim_Sayisi"] > 0).astype(int)
        else:
            features["Bakim_Sayisi"] = 0
            features["Bakim_Var_Mi"] = 0

        # Son_Bakimdan_Gecen_Gun yoksa Son_Bakim_Tarihi üzerinden hesapla
        if "Son_Bakimdan_Gecen_Gun" in features.columns:
            # Mevcut değerleri koru; NaN olanları tarih üzerinden doldur
            mask_nan = features["Son_Bakimdan_Gecen_Gun"].isna()
            if "Son_Bakim_Tarihi" in features.columns:
                tmp_days = (analysis_dt - features["Son_Bakim_Tarihi"]).dt.days
                features.loc[mask_nan, "Son_Bakimdan_Gecen_Gun"] = tmp_days[mask_nan]
        else:
            if "Son_Bakim_Tarihi" in features.columns:
                features["Son_Bakimdan_Gecen_Gun"] = (
                    analysis_dt - features["Son_Bakim_Tarihi"]
                ).dt.days
            else:
                features["Son_Bakimdan_Gecen_Gun"] = np.nan

        # -------------------------------------------------
        # 6) Tip düzeltmeleri & doldurmalar
        # -------------------------------------------------
        if "Ariza_Sayisi" in features.columns:
            features["Ariza_Sayisi"] = features["Ariza_Sayisi"].fillna(0).astype(int)
        if "Ariza_Gecmisi" in features.columns:
            features["Ariza_Gecmisi"] = features["Ariza_Gecmisi"].fillna(0).astype(int)
        if "Kronik_90g_Flag" in features.columns:
            features["Kronik_90g_Flag"] = features["Kronik_90g_Flag"].fillna(0).astype(int)

        # -------------------------------------------------
        # 7) Özet log
        # -------------------------------------------------
        logger.info("[OZET]")
        logger.info(f"Toplam ekipman: {len(features)}")
        if "Ariza_Sayisi" in features.columns:
            logger.info(f"Arıza geçmişi olan ekipman: {(features['Ariza_Sayisi'] > 0).sum()}")
        if "Kronik_90g_Flag" in features.columns:
            logger.info(f"Kronik (90g) flag sayısı: {features['Kronik_90g_Flag'].sum()}")
        if "Bakim_Sayisi" in features.columns:
            logger.info(f"Bakım kaydı olan ekipman: {(features['Bakim_Sayisi'] > 0).sum()}")

        # -------------------------------------------------
        # 8) SANITY CHECK – otomatik kalite kontrol
        # -------------------------------------------------
        logger.info("")
        logger.info("[SANITY CHECK] Özellik seti kalite kontrolü başlıyor...")

        sanity_issues = []

        def add_issue(msg):
            sanity_issues.append(msg)
            logger.warning(f"[SANITY] {msg}")

        # Negatif yaş
        if "Ekipman_Yasi_Gun" in features.columns:
            neg_yas = (features["Ekipman_Yasi_Gun"] < 0).sum()
            if neg_yas > 0:
                add_issue(f"Negatif ekipman yaşı tespit edildi: {neg_yas} kayıt.")

            zero_age = (features["Ekipman_Yasi_Gun"] == 0).sum()
            if zero_age > 0:
                add_issue(f"Yaşı 0 gün olan ekipman sayısı: {zero_age}")

        # Arıza sayısı
        if "Ariza_Sayisi" in features.columns:
            fault_neg = (features["Ariza_Sayisi"] < 0).sum()
            if fault_neg > 0:
                add_issue(f"Negatif arıza sayısı bulunan kayıtlar: {fault_neg}")

        # MTBF
        if "MTBF_Gun" in features.columns:
            mtbf_neg = (features["MTBF_Gun"] < 0).sum()
            if mtbf_neg > 0:
                add_issue(f"Negatif MTBF değeri tespit edildi: {mtbf_neg}")

            mtbf_large = (features["MTBF_Gun"] > 36500).sum()
            if mtbf_large > 0:
                add_issue(f"MTBF > 100 yıl olan kayıtlar: {mtbf_large}")

        # Son arızadan geçen gün
        if "Son_Ariza_Gun_Sayisi" in features.columns:
            neg_since_last = (features["Son_Ariza_Gun_Sayisi"] < 0).sum()
            if neg_since_last > 0:
                add_issue(f"Negatif 'Son_Ariza_Gun_Sayisi' tespit edildi: {neg_since_last}")

        # Bakım kolonlarındaki NaN sayıları
        for col in [
            "Bakim_Sayisi",
            "Ilk_Bakim_Tarihi",
            "Son_Bakim_Tarihi",
            "Son_Bakimdan_Gecen_Gun",
            "Son_Bakim_Tipi",
        ]:
            if col in features.columns:
                missing = features[col].isna().sum()
                logger.info(f"[SANITY] {col}: NaN sayısı = {missing}")

        # Sanity raporu csv
        sanity_report_path = os.path.join(
            os.path.dirname(FEATURE_OUTPUT_PATH),
            "ozellik_sanity_report.csv"
        )

        if sanity_issues:
            sanity_df = pd.DataFrame({"issue": sanity_issues})
        else:
            sanity_df = pd.DataFrame({"issue": ["NO ISSUES DETECTED"]})

        sanity_df.to_csv(sanity_report_path, index=False, encoding="utf-8-sig")
        logger.info(f"[OK] Sanity kontrol raporu kaydedildi → {sanity_report_path}")
        logger.info("[SANITY CHECK] Tamamlandı.")
        logger.info("")

        # -------------------------------------------------
        # 9) Kaydet
        # -------------------------------------------------
        os.makedirs(os.path.dirname(FEATURE_OUTPUT_PATH), exist_ok=True)
        features.to_csv(FEATURE_OUTPUT_PATH, index=False, encoding="utf-8-sig")
        logger.info(f"[OK] Özellik seti kaydedildi → {FEATURE_OUTPUT_PATH}")
        logger.info(f"[SUCCESS] {STEP_NAME} tamamlandı.")

    except Exception as e:
        logger.exception(f"[FATAL] {STEP_NAME} hata verdi: {e}")
        raise


if __name__ == "__main__":
    main()
