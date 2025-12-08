# pipeline/04_chronic_detection.py

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
sys.stdout.reconfigure(encoding='utf-8')

# Add project root to Python path for utils import
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd

from utils.logger import get_logger

# --------------------------------------------------------------------
# Paths / Config
# --------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
INPUT_DIR = DATA_DIR / "inputs"
INTERMEDIATE_DIR = DATA_DIR / "intermediate"
OUTPUT_DIR = DATA_DIR / "outputs"
LOG_DIR = BASE_DIR / "logs"

LOG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / f"04_chronic_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logger = get_logger("04_chronic_detection", LOG_FILE)

# Esnek ama deterministic analiz tarihi
ANALYSIS_DATE = datetime(2025, 12, 4)

# Kronik tanım parametreleri (gerekirse config.py'ye taşınabilir)
CHRONIC_WINDOW_DAYS = 365          # Son 12 ay
CHRONIC_MIN_FAULTS_IEEE = 3        # IEEE benzeri: 12 ayda en az 3 arıza
CHRONIC_LAMBDA_ALERT = 2.0         # λ > 2 arıza / yıl ise raporlanır (flag değil, uyarı)


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------
def load_intermediate_data():
    logger.info("=" * 80)
    logger.info("04_chronic_detection - PoF3 Kronik Arıza Analizi")
    logger.info("=" * 80)
    logger.info("")

    # Fault events
    events_path = INTERMEDIATE_DIR / "fault_events_clean.csv"
    if not events_path.exists():
        logger.error(f"[FATAL] fault_events_clean.csv bulunamadı: {events_path}")
        raise FileNotFoundError(events_path)

    equipment_path = INTERMEDIATE_DIR / "equipment_master.csv"
    if not equipment_path.exists():
        logger.error(f"[FATAL] equipment_master.csv bulunamadı: {equipment_path}")
        raise FileNotFoundError(equipment_path)

    features_path = INTERMEDIATE_DIR / "features_pof3.csv"
    if not features_path.exists():
        logger.warning(f"[WARN] features_pof3.csv bulunamadı: {features_path}")
        features = None
    else:
        features = pd.read_csv(features_path)

    events = pd.read_csv(
        events_path,
        parse_dates=["Ariza_Baslangic_Zamani", "Ariza_Bitis_Zamani", "Kurulum_Tarihi"]
    )
    equipment = pd.read_csv(
        equipment_path,
        parse_dates=["Kurulum_Tarihi", "Ilk_Ariza_Tarihi"]
    )

    logger.info(f"[OK] Yüklenen fault events: {len(events):,} kayıt")
    logger.info(f"[OK] Yüklenen equipment master: {len(equipment):,} kayıt")
    if features is not None:
        logger.info(f"[OK] Yüklenen features_pof3: {len(features):,} kayıt")

    return events, equipment, features


def build_chronic_table(events: pd.DataFrame, equipment: pd.DataFrame) -> pd.DataFrame:
    logger.info("")
    logger.info("[STEP] Ekipman bazında kronik metriklerin hesaplanması")

    # Beklenen kolonlar
    required_event_cols = ["CBS_ID", "Ariza_Baslangic_Zamani"]
    for col in required_event_cols:
        if col not in events.columns:
            raise KeyError(f"[FATAL] fault_events_clean içerisinde '{col}' kolonu bulunamadı.")

    required_eq_cols = ["CBS_ID", "Kurulum_Tarihi", "Ekipman_Tipi"]
    for col in required_eq_cols:
        if col not in equipment.columns:
            raise KeyError(f"[FATAL] equipment_master içerisinde '{col}' kolonu bulunamadı.")

    # Çalışma kopyaları
    ev = events.copy()
    eq = equipment.copy()

    # Eksik CBS_ID filtrele
    ev = ev.dropna(subset=["CBS_ID"])
    eq = eq.dropna(subset=["CBS_ID"])

    # Gözlem penceresi
    window_start = ANALYSIS_DATE - timedelta(days=CHRONIC_WINDOW_DAYS)

    # Temel arıza istatistikleri
    logger.info("[INFO] Ekipman bazında temel arıza istatistikleri hesaplanıyor...")
    grp = ev.groupby("CBS_ID")

    stats = grp["Ariza_Baslangic_Zamani"].agg(
        Ilk_Ariza_Tarihi=("min"),
        Son_Ariza_Tarihi=("max"),
        Toplam_Ariza_Sayisi=("count"),
    ).reset_index()

    # Son 12 ay arıza sayısı
    recent_mask = ev["Ariza_Baslangic_Zamani"] >= window_start
    recent_counts = (
        ev[recent_mask]
        .groupby("CBS_ID")["Ariza_Baslangic_Zamani"]
        .count()
        .rename("Son12Ay_Ariza_Sayisi")
    )

    stats = stats.merge(
        recent_counts,
        on="CBS_ID",
        how="left"
    )
    stats["Son12Ay_Ariza_Sayisi"] = stats["Son12Ay_Ariza_Sayisi"].fillna(0).astype(int)

    # Kurulum tarihi + gözlem süresi (yıl)
    eq_tmp = eq[["CBS_ID", "Kurulum_Tarihi", "Ekipman_Tipi"]].copy()
    stats = stats.merge(eq_tmp, on="CBS_ID", how="left")

    # Gözlem süresi (Kurulum → Analiz tarihi veya Son Arıza / Analiz tarihi)
    stats["Kurulum_Tarihi"] = pd.to_datetime(stats["Kurulum_Tarihi"], errors="coerce")
    stats["Son_Ariza_Tarihi"] = pd.to_datetime(stats["Son_Ariza_Tarihi"], errors="coerce")

    # Gözlem sonu: min(AnalizTarihi, Son_Ariza)
    stats["Gozlem_Sonu"] = stats["Son_Ariza_Tarihi"].fillna(ANALYSIS_DATE)
    stats["Gozlem_Sonu"] = stats["Gozlem_Sonu"].apply(
        lambda d: min(d, ANALYSIS_DATE)
    )

    # Gözlem süresi (yıl)
    stats["Gozlem_Suresi_Gun"] = (
        stats["Gozlem_Sonu"] - stats["Kurulum_Tarihi"]
    ).dt.days

    stats["Gozlem_Suresi_Gun"] = stats["Gozlem_Suresi_Gun"].clip(lower=1)
    stats["Gozlem_Suresi_Yil"] = stats["Gozlem_Suresi_Gun"] / 365.25

    # Yıllık arıza oranı λ
    stats["Lambda_Yillik_Ariza"] = stats["Toplam_Ariza_Sayisi"] / stats["Gozlem_Suresi_Yil"]

    logger.info(
        f"[INFO] Ortalama λ: {stats['Lambda_Yillik_Ariza'].mean():.3f} arıza/yıl "
        f"(Median: {stats['Lambda_Yillik_Ariza'].median():.3f})"
    )

    # IEEE-benzeri kronik flag: son 12 ayda ≥ 3 arıza
    stats["Kronik_IEEE_Flag"] = (
        (stats["Son12Ay_Ariza_Sayisi"] >= CHRONIC_MIN_FAULTS_IEEE).astype(int)
    )

    chronic_rate = stats["Kronik_IEEE_Flag"].mean()
    logger.info(f"[INFO] Kronik_IEEE_Flag oranı: {chronic_rate:.3%}")
    if chronic_rate > 0.15:
        logger.warning("[WARN] Kronik oranı %15 üzeri – eşikleri gözden geçirmeniz gerekebilir.")
    elif chronic_rate < 0.03:
        logger.warning("[WARN] Kronik oranı %3 altında – model fazla katı olabilir.")

    return stats


def merge_with_feature_flags(chronic_stats: pd.DataFrame, features: pd.DataFrame | None) -> pd.DataFrame:
    if features is None:
        logger.info("[INFO] features_pof3 bulunamadı; sadece IEEE kronik flag kullanılacak.")
        chronic_stats["Kronik_90G_Flag"] = 0
        chronic_stats["Kronik_Birlesik_Flag"] = chronic_stats["Kronik_IEEE_Flag"]
        return chronic_stats

    if "CBS_ID" not in features.columns:
        logger.warning("[WARN] features_pof3 içinde CBS_ID kolonu yok; merge atlanıyor.")
        chronic_stats["Kronik_90G_Flag"] = 0
        chronic_stats["Kronik_Birlesik_Flag"] = chronic_stats["Kronik_IEEE_Flag"]
        return chronic_stats

    # Kolon adı feature tarafında farklı olabilir
    flag_cols = [c for c in features.columns if "Chronic" in c or "Kronik" in c]
    if not flag_cols:
        logger.info("[INFO] features_pof3 içerisinde kronik flag kolonu bulunamadı.")
        chronic_stats["Kronik_90G_Flag"] = 0
        chronic_stats["Kronik_Birlesik_Flag"] = chronic_stats["Kronik_IEEE_Flag"]
        return chronic_stats

    chronic_flag_col = flag_cols[0]
    logger.info(f"[INFO] Kronik feature flag kolonu olarak '{chronic_flag_col}' kullanılacak.")

    feat_subset = features[["CBS_ID", chronic_flag_col]].copy()
    feat_subset.rename(columns={chronic_flag_col: "Kronik_90G_Flag"}, inplace=True)
    feat_subset["Kronik_90G_Flag"] = feat_subset["Kronik_90G_Flag"].fillna(0).astype(int)

    merged = chronic_stats.merge(feat_subset, on="CBS_ID", how="left")
    merged["Kronik_90G_Flag"] = merged["Kronik_90G_Flag"].fillna(0).astype(int)

    # Birleşik flag: IEEE veya 90g kronik ise 1
    merged["Kronik_Birlesik_Flag"] = (
        ((merged["Kronik_IEEE_Flag"] == 1) | (merged["Kronik_90G_Flag"] == 1)).astype(int)
    )

    merged_rate = merged["Kronik_Birlesik_Flag"].mean()
    logger.info(f"[INFO] Kronik_Birlesik_Flag oranı: {merged_rate:.3%}")

    return merged


def save_outputs(chronic_full: pd.DataFrame):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = OUTPUT_DIR / "chronic_equipment_summary.csv"
    chronic_only_path = OUTPUT_DIR / "chronic_equipment_only.csv"

    chronic_full.to_csv(summary_path, index=False)
    logger.info(f"[OK] Kronik özet tablo kaydedildi: {summary_path}")

    chronic_only = chronic_full[chronic_full["Kronik_Birlesik_Flag"] == 1].copy()
    chronic_only.to_csv(chronic_only_path, index=False)
    logger.info(f"[OK] Sadece kronik ekipman tablosu kaydedildi: {chronic_only_path}")

    logger.info(
        f"[SUMMARY] Toplam ekipman: {len(chronic_full):,} | "
        f"Kronik ekipman (birleşik): {len(chronic_only):,} "
        f"({len(chronic_only)/len(chronic_full):.1%})"
    )


def main():
    try:
        events, equipment, features = load_intermediate_data()
        chronic_stats = build_chronic_table(events, equipment)
        chronic_full = merge_with_feature_flags(chronic_stats, features)
        save_outputs(chronic_full)
        logger.info("[SUCCESS] 04_chronic_detection başarıyla tamamlandı.")
    except Exception as e:
        logger.exception(f"[FATAL] 04_chronic_detection hata ile sonlandı: {e}")
        raise


if __name__ == "__main__":
    main()
