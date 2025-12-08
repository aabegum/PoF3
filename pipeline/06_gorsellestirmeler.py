# pipeline/06_visualizations.py

from pathlib import Path
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt

from utils.logger import get_logger
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

sys.stdout.reconfigure(encoding='utf-8')

# --------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
INTERMEDIATE_DIR = DATA_DIR / "intermediate"
OUTPUT_DIR = DATA_DIR / "outputs"
VISUALS_DIR = BASE_DIR / "visuals"
LOG_DIR = BASE_DIR / "logs"

(LOG_DIR).mkdir(parents=True, exist_ok=True)
(VISUALS_DIR / "plots_survival").mkdir(parents=True, exist_ok=True)
(VISUALS_DIR / "plots_chronic").mkdir(parents=True, exist_ok=True)
(VISUALS_DIR / "plots_risk").mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / f"06_visualizations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logger = get_logger("06_visualizations", LOG_FILE)


# --------------------------------------------------------------------
# 01 – Data Processing görselleri
# --------------------------------------------------------------------
def plot_age_and_type_distributions():
    eq_path = INTERMEDIATE_DIR / "equipment_master.csv"
    if not eq_path.exists():
        logger.warning(f"[WARN] equipment_master.csv bulunamadı: {eq_path}")
        return

    df = pd.read_csv(eq_path)
    # Ekipman yaş histogramı
    if "Ekipman_Yasi_Gun" in df.columns:
        plt.figure(figsize=(8, 5))
        df["Ekipman_Yasi_Gun"].dropna().plot.hist(bins=40)
        plt.xlabel("Ekipman Yaşı (gün)")
        plt.ylabel("Ekipman Sayısı")
        plt.title("Ekipman Yaşı Dağılımı")
        out = VISUALS_DIR / "plots_survival" / "01_ekipman_yasi_dagilimi.png"
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
        logger.info(f"[OK] Ekipman yaşı dağılım grafiği kaydedildi: {out}")

    # Ekipman tipi dağılımı
    type_col = "Ekipman_Tipi" if "Ekipman_Tipi" in df.columns else None
    if type_col:
        plt.figure(figsize=(9, 5))
        df[type_col].value_counts().sort_values(ascending=False).plot.bar()
        plt.xlabel("Ekipman Tipi")
        plt.ylabel("Ekipman Sayısı")
        plt.title("Ekipman Tipi Dağılımı")
        plt.xticks(rotation=45, ha="right")
        out = VISUALS_DIR / "plots_survival" / "02_ekipman_tipi_dagilimi.png"
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
        logger.info(f"[OK] Ekipman tipi dağılım grafiği kaydedildi: {out}")


# --------------------------------------------------------------------
# 02 – Feature Engineering görselleri
# --------------------------------------------------------------------
def plot_mtbf_and_chronic():
    feat_path = INTERMEDIATE_DIR / "features_pof3.csv"
    if not feat_path.exists():
        logger.warning(f"[WARN] features_pof3.csv bulunamadı: {feat_path}")
        return

    df = pd.read_csv(feat_path)

    # MTBF dağılımı
    if "MTBF_Gun" in df.columns:
        plt.figure(figsize=(8, 5))
        df["MTBF_Gun"].dropna().clip(upper=3650).plot.hist(bins=40)
        plt.xlabel("MTBF (gün)")
        plt.ylabel("Ekipman Sayısı")
        plt.title("MTBF Dağılımı (3650 gün ile kırpılmış)")
        out = VISUALS_DIR / "plots_chronic" / "01_mtbf_dagilimi.png"
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
        logger.info(f"[OK] MTBF dağılım grafiği kaydedildi: {out}")

    # Kronik flag oranı
    kronik_cols = [c for c in df.columns if "Kronik" in c or "Chronic" in c]
    if kronik_cols:
        col = kronik_cols[0]
        plt.figure(figsize=(5, 5))
        df[col].fillna(0).astype(int).value_counts().sort_index().plot.bar()
        plt.xlabel("Kronik Flag")
        plt.ylabel("Ekipman Sayısı")
        plt.title(f"Kronik Flag Dağılımı ({col})")
        plt.xticks(ticks=[0, 1], labels=["0 - Normal", "1 - Kronik"])
        out = VISUALS_DIR / "plots_chronic" / "02_kronik_flag_dagilimi.png"
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
        logger.info(f"[OK] Kronik flag dağılım grafiği kaydedildi: {out}")


# --------------------------------------------------------------------
# 03 – Survival görselleri (Kaplan-Meier)
# --------------------------------------------------------------------
def plot_km_curves():
    try:
        from lifelines import KaplanMeierFitter
    except ImportError:
        logger.warning("[WARN] lifelines yüklü değil, Kaplan-Meier grafikleri atlanacak.")
        return

    surv_path = INTERMEDIATE_DIR / "survival_base.csv"
    if not surv_path.exists():
        logger.warning(f"[WARN] survival_base.csv bulunamadı: {surv_path}")
        return

    df = pd.read_csv(surv_path)
    required = ["Sure_Gun", "Olay"]
    for col in required:
        if col not in df.columns:
            logger.warning(f"[WARN] survival_base içerisinde '{col}' kolonu yok; KM grafikleri atlanıyor.")
            return

    if "Ekipman_Tipi" not in df.columns:
        logger.warning("[WARN] survival_base içerisinde 'Ekipman_Tipi' yok; tek KM eğrisi çizilecek.")
        plt.figure(figsize=(8, 5))
        kmf = KaplanMeierFitter()
        kmf.fit(durations=df["Sure_Gun"], event_observed=df["Olay"])
        kmf.plot()
        plt.xlabel("Süre (gün)")
        plt.ylabel("Sağkalım Olasılığı")
        plt.title("Kaplan-Meier Eğrisi (Tüm Ekipmanlar)")
        out = VISUALS_DIR / "plots_survival" / "03_km_tum_ekipmanlar.png"
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
        logger.info(f"[OK] KM eğrisi kaydedildi: {out}")
        return

    # En çok gözlenen ilk 3 ekipman tipi
    top_types = (
        df["Ekipman_Tipi"].value_counts().head(3).index.tolist()
    )

    plt.figure(figsize=(9, 6))
    kmf = KaplanMeierFitter()

    for etype in top_types:
        sub = df[df["Ekipman_Tipi"] == etype]
        if len(sub) < 30:
            continue
        kmf.fit(durations=sub["Sure_Gun"], event_observed=sub["Olay"], label=etype)
        kmf.plot(ci_show=False)

    plt.xlabel("Süre (gün)")
    plt.ylabel("Sağkalım Olasılığı")
    plt.title("Kaplan-Meier Eğrileri (En Büyük 3 Ekipman Tipi)")
    out = VISUALS_DIR / "plots_survival" / "04_km_en_buyuk_3_tur.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    logger.info(f"[OK] KM ekipman tipi grafiği kaydedildi: {out}")


# --------------------------------------------------------------------
# 05 – Risk görselleri
# --------------------------------------------------------------------
def plot_risk_distribution():
    risk_path = OUTPUT_DIR / "pof3_risk_table.csv"
    if not risk_path.exists():
        logger.warning(f"[WARN] pof3_risk_table.csv bulunamadı: {risk_path}")
        return

    df = pd.read_csv(risk_path)
    if "Risk_Skoru" not in df.columns:
        logger.warning("[WARN] Risk_Skoru kolonu bulunamadı; risk görselleri atlanıyor.")
        return

    # Risk_Skoru histogramı
    plt.figure(figsize=(8, 5))
    df["Risk_Skoru"].dropna().clip(upper=df["Risk_Skoru"].quantile(0.99)).plot.hist(bins=40)
    plt.xlabel("Risk Skoru (PoF * CoF)")
    plt.ylabel("Ekipman Sayısı")
    plt.title("Risk Skoru Dağılımı (99. persentil ile kırpılmış)")
    out = VISUALS_DIR / "plots_risk" / "01_risk_skoru_dagilimi.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    logger.info(f"[OK] Risk skoru dağılım grafiği kaydedildi: {out}")

    # Risk sınıfı bar chart
    if "Risk_Sinifi" in df.columns:
        plt.figure(figsize=(6, 5))
        df["Risk_Sinifi"].value_counts().reindex(["DÜŞÜK", "ORTA", "YÜKSEK", "BİLİNMİYOR"]).plot.bar()
        plt.xlabel("Risk Sınıfı")
        plt.ylabel("Ekipman Sayısı")
        plt.title("Risk Sınıfı Dağılımı")
        out = VISUALS_DIR / "plots_risk" / "02_risk_sinifi_dagilimi.png"
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
        logger.info(f"[OK] Risk sınıfı dağılım grafiği kaydedildi: {out}")


def main():
    logger.info("=" * 80)
    logger.info("06_visualizations - PoF3 Görselleştirme Modülü")
    logger.info("=" * 80)

    try:
        plot_age_and_type_distributions()
        plot_mtbf_and_chronic()
        plot_km_curves()
        plot_risk_distribution()
        logger.info("[SUCCESS] 06_visualizations başarıyla tamamlandı.")
    except Exception as e:
        logger.exception(f"[FATAL] 06_visualizations hata ile sonlandı: {e}")
        raise


if __name__ == "__main__":
    main()
