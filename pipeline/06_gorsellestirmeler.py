"""
06_gorsellestirmeler.py
PoF3 – Türkçe Görselleştirme Modülü

Amaç:
- Özellik seti (ozellikler_pof3.csv) üzerinden:
    * Ekipman yaşı dağılımı
    * Ekipman tipi dağılımı
    * MTBF dağılımı
    * Kronik flag dağılımı
- Survival base üzerinden:
    * En büyük 3 ekipman tipi için KM eğrileri
- Risk skorları üzerinden:
    * Risk skoru dağılımı
    * Risk sınıfı dağılımı

Çıktı:
    gorseller/plots_survival/01_ekipman_yasi_dagilimi.png
    gorseller/plots_survival/02_ekipman_tipi_dagilimi.png
    gorseller/plots_chronic/01_mtbf_dagilimi.png
    gorseller/plots_chronic/02_kronik_flag_dagilimi.png
    gorseller/plots_survival/04_km_en_buyuk_3_tur.png
    gorseller/plots_risk/01_risk_skoru_dagilimi.png
    gorseller/plots_risk/02_risk_sinifi_dagilimi.png
"""

import os
import sys
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# Kaplan–Meier için
from lifelines import KaplanMeierFitter

# UTF-8 konsol
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Proje kökü
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ----------------------------------------------------------------------
# CONFIG: config.config varsa kullan, yoksa makul varsayılanlar
# ----------------------------------------------------------------------
try:
    from config.config import (
        FEATURE_OUTPUT_PATH,
        INTERMEDIATE_PATHS,
        OUTPUT_DIR,
        LOG_DIR,
    )
except Exception:
    FEATURE_OUTPUT_PATH = os.path.join("data", "ara_ciktilar", "ozellikler_pof3.csv")
    INTERMEDIATE_PATHS = {
        "survival_base": os.path.join("data", "ara_ciktilar", "survival_base.csv"),
    }
    OUTPUT_DIR = os.path.join("data", "sonuclar")
    LOG_DIR = "logs"


STEP_NAME = "06_gorsellestirmeler"
PLOTS_ROOT_DIR = os.path.join("gorseller")


# ----------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------
def setup_logger() -> logging.Logger:
    os.makedirs(LOG_DIR, exist_ok=True)
    from datetime import datetime

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
    logger.info(f"{STEP_NAME} - PoF3 Görselleştirme Modülü")
    logger.info("=" * 80)
    return logger


logger = setup_logger()


# ----------------------------------------------------------------------
# Yardımcı: güvenli veri yükleme
# ----------------------------------------------------------------------
def safe_read_csv(path: str, desc: str) -> pd.DataFrame:
    if not os.path.exists(path):
        logger.warning(f"[WARN] {desc} dosyası bulunamadı: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path, encoding="utf-8-sig")
    logger.info(f"[OK] {desc} yüklendi: {len(df):,} satır")
    return df


# ----------------------------------------------------------------------
# Yardımcı: figürü kaydet ve kapat
# ----------------------------------------------------------------------
def save_figure(fig, out_path: str, desc: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"[OK] {desc} kaydedildi: {out_path}")


# ----------------------------------------------------------------------
# 1) Özellik tabanlı görseller
# ----------------------------------------------------------------------
def plot_feature_based_charts(features: pd.DataFrame):
    if features.empty:
        logger.warning("[WARN] Özellik seti boş – feature tabanlı grafikler atlanıyor.")
        return

    # 1. Ekipman yaşı dağılımı (gün)
    if "Ekipman_Yasi_Gun" in features.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        data = features["Ekipman_Yasi_Gun"].dropna()
        ax.hist(data, bins=40)
        ax.set_title("Ekipman Yaşı Dağılımı")
        ax.set_xlabel("Ekipman Yaşı (gün)")
        ax.set_ylabel("Ekipman Sayısı")
        out_path = os.path.join(
            PLOTS_ROOT_DIR, "plots_survival", "01_ekipman_yasi_dagilimi.png"
        )
        save_figure(fig, out_path, "Ekipman yaşı dağılım grafiği")
    else:
        logger.warning("[WARN] 'Ekipman_Yasi_Gun' kolonu yok – yaş dağılımı çizilemedi.")

    # 2. Ekipman tipi dağılımı
    if "Ekipman_Tipi" in features.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        counts = features["Ekipman_Tipi"].value_counts().sort_values(ascending=False)
        counts.plot(kind="bar", ax=ax)
        ax.set_title("Ekipman Tipi Dağılımı")
        ax.set_xlabel("Ekipman Tipi")
        ax.set_ylabel("Ekipman Sayısı")
        out_path = os.path.join(
            PLOTS_ROOT_DIR, "plots_survival", "02_ekipman_tipi_dagilimi.png"
        )
        save_figure(fig, out_path, "Ekipman tipi dağılım grafiği")
    else:
        logger.warning("[WARN] 'Ekipman_Tipi' kolonu yok – ekipman tipi grafiği çizilemedi.")

    # 3. MTBF dağılımı
    if "MTBF_Gun" in features.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        data = features["MTBF_Gun"].replace([np.inf, -np.inf], np.nan).dropna()
        if len(data) > 0:
            ax.hist(data, bins=40)
            ax.set_title("MTBF Dağılımı")
            ax.set_xlabel("MTBF (gün)")
            ax.set_ylabel("Ekipman Sayısı")
            out_path = os.path.join(
                PLOTS_ROOT_DIR, "plots_chronic", "01_mtbf_dagilimi.png"
            )
            save_figure(fig, out_path, "MTBF dağılım grafiği")
        else:
            plt.close(fig)
            logger.warning("[WARN] MTBF_Gun kolonunda geçerli değer yok – grafik atlandı.")
    else:
        logger.warning("[WARN] 'MTBF_Gun' kolonu yok – MTBF grafiği çizilemedi.")

    # 4. Kronik flag dağılımı (Kronik_90g_Flag)
    if "Kronik_90g_Flag" in features.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        counts = features["Kronik_90g_Flag"].value_counts().sort_index()
        counts.index = counts.index.map({0: "Kronik Değil", 1: "Kronik"})
        counts.plot(kind="bar", ax=ax)
        ax.set_title("Kronik (90g) Flag Dağılımı")
        ax.set_xlabel("Durum")
        ax.set_ylabel("Ekipman Sayısı")
        out_path = os.path.join(
            PLOTS_ROOT_DIR, "plots_chronic", "02_kronik_flag_dagilimi.png"
        )
        save_figure(fig, out_path, "Kronik flag dağılım grafiği")
    else:
        logger.warning("[WARN] 'Kronik_90g_Flag' kolonu yok – kronik grafiği çizilemedi.")


# ----------------------------------------------------------------------
# 2) Survival (KM) grafikleri
# ----------------------------------------------------------------------
def plot_km_curves(survival_base: pd.DataFrame):
    if survival_base.empty:
        logger.warning("[WARN] survival_base boş – KM eğrileri çizilemeyecek.")
        return

    required = {"duration_days", "event", "Ekipman_Tipi"}
    missing = required - set(survival_base.columns)
    if missing:
        logger.warning(f"[WARN] KM için eksik kolon(lar): {missing} – KM atlanıyor.")
        return

    df = survival_base.copy()
    df = df[df["duration_days"] > 0].copy()

    # En çok gözleme sahip 3 ekipman tipi
    top_types = (
        df["Ekipman_Tipi"].value_counts().head(3).index.tolist()
    )
    if not top_types:
        logger.warning("[WARN] KM için yeterli ekipman tipi yok.")
        return

    kmf = KaplanMeierFitter()
    fig, ax = plt.subplots(figsize=(8, 6))

    for eq_type in top_types:
        sub = df[df["Ekipman_Tipi"] == eq_type]
        if sub.empty:
            continue
        kmf.fit(
            durations=sub["duration_days"],
            event_observed=sub["event"],
            label=str(eq_type),
        )
        kmf.plot_survival_function(ax=ax)

    ax.set_title("En Büyük 3 Ekipman Tipi İçin KM Sağkalım Eğrileri")
    ax.set_xlabel("Süre (gün)")
    ax.set_ylabel("Sağkalım Olasılığı")

    out_path = os.path.join(
        PLOTS_ROOT_DIR, "plots_survival", "04_km_en_buyuk_3_tur.png"
    )
    save_figure(fig, out_path, "KM ekipman tipi grafiği")


# ----------------------------------------------------------------------
# 3) Risk tabanlı grafikler
# ----------------------------------------------------------------------
def plot_risk_charts(risk_df: pd.DataFrame):
    if risk_df.empty:
        logger.warning("[WARN] risk_skorlari veri seti boş – risk grafikleri atlanıyor.")
        return

    # 1. Risk skoru dağılımı
    if "Risk_Skoru" in risk_df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        data = risk_df["Risk_Skoru"].replace([np.inf, -np.inf], np.nan).dropna()
        ax.hist(data, bins=40)
        ax.set_title("Risk Skoru Dağılımı")
        ax.set_xlabel("Risk Skoru (PoF * CoF)")
        ax.set_ylabel("Ekipman Sayısı")
        out_path = os.path.join(
            PLOTS_ROOT_DIR, "plots_risk", "01_risk_skoru_dagilimi.png"
        )
        save_figure(fig, out_path, "Risk skoru dağılım grafiği")
    else:
        logger.warning("[WARN] 'Risk_Skoru' kolonu yok – risk skoru grafiği çizilemedi.")

    # 2. Risk sınıfı dağılımı
    if "Risk_Sinifi" in risk_df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        counts = risk_df["Risk_Sinifi"].value_counts().reindex(
            ["Düşük", "Orta", "Yüksek", "Çok Yüksek"]
        )
        counts = counts.fillna(0)
        counts.plot(kind="bar", ax=ax)
        ax.set_title("Risk Sınıfı Dağılımı")
        ax.set_xlabel("Risk Sınıfı")
        ax.set_ylabel("Ekipman Sayısı")
        out_path = os.path.join(
            PLOTS_ROOT_DIR, "plots_risk", "02_risk_sinifi_dagilimi.png"
        )
        save_figure(fig, out_path, "Risk sınıfı dağılım grafiği")
    else:
        logger.warning("[WARN] 'Risk_Sinifi' kolonu yok – risk sınıfı grafiği çizilemedi.")


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    # 1) Özellik seti
    features = safe_read_csv(FEATURE_OUTPUT_PATH, "Özellik seti (ozellikler_pof3)")

    # 2) Survival base
    survival_path = INTERMEDIATE_PATHS.get(
        "survival_base", os.path.join("data", "ara_ciktilar", "survival_base.csv")
    )
    survival_base = safe_read_csv(survival_path, "survival_base")

    # 3) Risk skorları
    risk_path = os.path.join(OUTPUT_DIR, "risk_skorlari.csv")
    risk_df = safe_read_csv(risk_path, "risk_skorlari")

    logger.info("")
    logger.info("[STEP] Özellik tabanlı grafikler üretiliyor...")
    plot_feature_based_charts(features)

    logger.info("[STEP] Survival (KM) grafikleri üretiliyor...")
    plot_km_curves(survival_base)

    logger.info("[STEP] Risk tabanlı grafikler üretiliyor...")
    plot_risk_charts(risk_df)

    logger.info("[SUCCESS] 06_gorsellestirmeler başarıyla tamamlandı.")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
