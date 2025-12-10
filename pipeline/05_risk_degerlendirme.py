"""
05_risk_degerlendirme.py  (PoF3 - Risk Değerlendirmesi)

Amaç:
- Sağkalım tabanlı PoF çıktıları (özellikle 12 ay ufku)
- CoF (Consequence of Failure) skorları
- Özellik seti (ozellikler_pof3)

üzerinden ekipman bazında risk skoru üretmek:

    Risk_Skoru = PoF_12Ay * CoF

Ayrıca:
- Statik arıza eğilimi skoru (varsa) ile zenginleştirme
- Risk sınıflandırması (Düşük / Orta / Yüksek / Kritik)
- Özet tablolar (ekipman tipi bazında risk dağılımı)

Girdiler (varsayılan dizinler):
    data/sonuclar/cox_sagkalim_12ay_ariza_olasiligi.csv
    data/sonuclar/rsf_sagkalim_12ay_ariza_olasiligi.csv (opsiyonel)
    data/sonuclar/statik_ariza_egilim_skoru.csv (opsiyonel)
    data/sonuclar/cof_pof3.csv  (veya benzer isimli CoF dosyaları)
    data/ara_ciktilar/ozellikler_pof3.csv

Çıktılar:
    data/sonuclar/risk_skorlari_pof3.csv
    data/sonuclar/risk_skoru_ozet_ekipman_tipi.csv
Log:
    loglar/05_risk_degerlendirme_YYYYMMDD_HHMMSS.log
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# UTF-8 konsol
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Proje kökü
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ----------------------------------------------------------------------
# CONFIG ve varsayılan yollar
# ----------------------------------------------------------------------
try:
    from config.config import ANALYSIS_DATE  # type: ignore
except Exception:
    ANALYSIS_DATE = datetime.today().date()

# LOG klasörü -> özellikle "loglar"
try:
    from config.config import LOG_DIR as CONFIG_LOG_DIR  # type: ignore
    LOG_DIR = Path(CONFIG_LOG_DIR)
except Exception:
    LOG_DIR = PROJECT_ROOT / "loglar"

RESULTS_DIR = PROJECT_ROOT / "data" / "sonuclar"
INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "ara_ciktilar"

POF_12M_FILES = [
    "cox_sagkalim_12ay_ariza_olasiligi.csv",   # yeni Türkçe isim
    "pof_cox_12m.csv",                          # eski PoF2 tarzı
]

COF_FILES = [
    "cof_pof3.csv",
    "cof_degerleri.csv",
    "cof_input.csv",
    "cof.csv",
]

OZELLIKLER_FILE = INTERMEDIATE_DIR / "ozellikler_pof3.csv"
STATIK_POF_FILE = RESULTS_DIR / "statik_ariza_egilim_skoru.csv"

STEP_NAME = "05_risk_degerlendirme"


# ----------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------
def setup_logger() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"{STEP_NAME}_{ts}.log"

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
    logger.info(f"{STEP_NAME} - PoF3 Risk Değerlendirmesi")
    logger.info("=" * 80)
    logger.info(f"Analiz Tarihi: {ANALYSIS_DATE}")
    logger.info("")
    return logger


# ----------------------------------------------------------------------
# Yardımcı fonksiyonlar
# ----------------------------------------------------------------------
def detect_id_column(df: pd.DataFrame) -> str:
    candidates = [c for c in df.columns if c.lower() in ["cbs_id", "id", "ekipman_id"]]
    if candidates:
        return candidates[0]
    # fallback: ilk kolon
    return df.columns[0]


def detect_pof_column(df: pd.DataFrame) -> str:
    # PoF kolonu: isimde "pof" veya "olas" geçen numerik kolon
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    for c in numeric_cols:
        name = c.lower()
        if "pof" in name or "olas" in name or "cox" in name or "rsf" in name:
            return c
    # fallback: ikinci kolon
    return numeric_cols[0] if len(numeric_cols) >= 1 else df.columns[-1]


def detect_cof_column(df: pd.DataFrame) -> str:
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    for c in numeric_cols:
        name = c.lower()
        if "cof" in name or "maliyet" in name or "risk" in name:
            return c
    # fallback: son numerik kolon
    return numeric_cols[-1] if numeric_cols else df.columns[-1]


def load_first_existing(base_dir: Path, candidates: List[str]) -> Tuple[Path, pd.DataFrame]:
    for name in candidates:
        path = base_dir / name
        if path.exists():
            df = pd.read_csv(path, encoding="utf-8-sig")
            return path, df
    raise FileNotFoundError(
        f"Hiçbir dosya bulunamadı. Denenen dosyalar: "
        f"{[str(base_dir / n) for n in candidates]}"
    )


# ----------------------------------------------------------------------
# VERİ YÜKLEME
# ----------------------------------------------------------------------
def load_pof_12m(logger: logging.Logger) -> Tuple[pd.DataFrame, str]:
    """12 aylık PoF (öncelikle Cox) dosyasını yükler."""
    path, pof_df = load_first_existing(RESULTS_DIR, POF_12M_FILES)
    logger.info(f"[OK] PoF (12 ay) dosyası bulundu: {path}")

    id_col = detect_id_column(pof_df)
    pof_col = detect_pof_column(pof_df)
    logger.info(f"[INFO] PoF ID kolonu: {id_col}, PoF skoru kolonu: {pof_col}")

    # Normalize isim
    pof_df = pof_df[[id_col, pof_col]].copy()
    pof_df.rename(columns={id_col: "cbs_id", pof_col: "PoF_12Ay"}, inplace=True)
    pof_df["cbs_id"] = pof_df["cbs_id"].astype(str).str.lower().str.strip()

    return pof_df, str(path)


def load_cof(logger: logging.Logger) -> Tuple[pd.DataFrame, str]:
    """CoF giriş dosyasını yükler; birkaç olası isim dener."""
    tried_paths = []
    for name in COF_FILES:
        path = RESULTS_DIR / name
        tried_paths.append(str(path))
        if path.exists():
            cof_df = pd.read_csv(path, encoding="utf-8-sig")
            logger.info(f"[OK] CoF dosyası bulundu: {path}")

            id_col = detect_id_column(cof_df)
            cof_col = detect_cof_column(cof_df)
            logger.info(f"[INFO] CoF ID kolonu: {id_col}, CoF skoru kolonu: {cof_col}")

            cof_df = cof_df[[id_col, cof_col]].copy()
            cof_df.rename(columns={id_col: "cbs_id", cof_col: "CoF"}, inplace=True)
            cof_df["cbs_id"] = cof_df["cbs_id"].astype(str).str.lower().str.strip()

            return cof_df, str(path)

    raise FileNotFoundError(
        f"Hiçbir CoF dosyası bulunamadı. Denenen dosya isimleri: {COF_FILES}"
    )


def load_features(logger: logging.Logger) -> pd.DataFrame:
    if not OZELLIKLER_FILE.exists():
        raise FileNotFoundError(f"Özellik seti bulunamadı: {OZELLIKLER_FILE}")

    feat = pd.read_csv(OZELLIKLER_FILE, encoding="utf-8-sig")
    logger.info(f"[OK] Özellik seti (ozellikler_pof3) yüklendi: {len(feat):,} satır")

    if "cbs_id" not in feat.columns:
        raise KeyError("ozellikler_pof3 içinde 'cbs_id' kolonu bulunamadı.")

    feat["cbs_id"] = feat["cbs_id"].astype(str).str.lower().str.strip()

    # Risk çıktısında anlamlı olacak çekirdek kolonlar
    keep_cols = [
        "cbs_id",
        "Ekipman_Tipi",
        "Ekipman_Yasi_Gun",
        "Ariza_Gecmisi",
        "Ariza_Sayisi",
        "Kronik_90g_Flag",
        "Bakim_Sayisi",
        "Bakim_Var_Mi",
        "Son_Bakimdan_Gecen_Gun",
        "Gerilim_Seviyesi",
        "kVA_Rating",
        "Marka",
    ]
    existing = [c for c in keep_cols if c in feat.columns]
    return feat[existing].copy()


def load_statik_pof(logger: logging.Logger) -> Optional[pd.DataFrame]:
    if not STATIK_POF_FILE.exists():
        logger.info("[INFO] statik_ariza_egilim_skoru.csv bulunamadı. Statik PoF opsiyonel ve atlanacak.")
        return None

    df = pd.read_csv(STATIK_POF_FILE, encoding="utf-8-sig")
    logger.info(f"[OK] Statik PoF dosyası yüklendi: {STATIK_POF_FILE} ({len(df):,} satır)")

    if "cbs_id" not in df.columns:
        id_col = detect_id_column(df)
        df.rename(columns={id_col: "cbs_id"}, inplace=True)

    # Statik PoF kolonunu bul
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    statik_col = None
    for c in numeric_cols:
        if "pof" in c.lower() or "skor" in c.lower() or "score" in c.lower():
            statik_col = c
            break
    if statik_col is None and len(numeric_cols) >= 1:
        statik_col = numeric_cols[-1]  # fallback

    if statik_col is None:
        logger.warning("[WARN] Statik PoF kolonunu tespit edemedim. Dosya atlanacak.")
        return None

    df = df[["cbs_id", statik_col]].copy()
    df.rename(columns={statik_col: "PoF_Statik"}, inplace=True)
    df["cbs_id"] = df["cbs_id"].astype(str).str.lower().str.strip()
    return df


# ----------------------------------------------------------------------
# RİSK HESABI
# ----------------------------------------------------------------------
def assign_risk_classes(risk_series: pd.Series) -> pd.Series:
    """
    Risk_Skoru için yüzdeliklere göre sınıflar üretir:
        0-50p  -> Düşük
        50-80p -> Orta
        80-95p -> Yüksek
        95-100p-> Kritik
    """
    valid = risk_series.dropna()
    if valid.empty:
        return pd.Series(["Bilinmiyor"] * len(risk_series), index=risk_series.index)

    q50 = valid.quantile(0.50)
    q80 = valid.quantile(0.80)
    q95 = valid.quantile(0.95)

    def _cls(x):
        if pd.isna(x):
            return "Bilinmiyor"
        if x <= q50:
            return "Düşük"
        elif x <= q80:
            return "Orta"
        elif x <= q95:
            return "Yüksek"
        else:
            return "Kritik"

    return risk_series.apply(_cls)


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    logger = setup_logger()

    try:
        # --------------------------------------------------------------
        # 1) Girdileri yükle
        # --------------------------------------------------------------
        logger.info("[STEP] PoF, CoF ve özellik seti yükleniyor...")

        pof_12m_df, pof_path = load_pof_12m(logger)
        cof_df, cof_path = load_cof(logger)
        feat_df = load_features(logger)
        statik_df = load_statik_pof(logger)

        logger.info("")
        logger.info(f"[INFO] PoF_12Ay dosyası: {pof_path}")
        logger.info(f"[INFO] CoF dosyası:     {cof_path}")
        logger.info(f"[INFO] Özellik seti:   {OZELLIKLER_FILE}")
        if statik_df is not None:
            logger.info(f"[INFO] Statik PoF dosyası: {STATIK_POF_FILE}")
        logger.info("")

        # --------------------------------------------------------------
        # 2) Merge – cbs_id üzerinden birleşik risk datası
        # --------------------------------------------------------------
        logger.info("[STEP] PoF + CoF + özellik seti birleştiriliyor...")

        df = pof_12m_df.merge(cof_df, on="cbs_id", how="inner")
        logger.info(f"[OK] PoF + CoF merge: {len(df):,} satır")

        df = df.merge(feat_df, on="cbs_id", how="left")
        logger.info(f"[OK] Özellik seti ile merge sonrası satır sayısı: {len(df):,}")

        if statik_df is not None:
            df = df.merge(statik_df, on="cbs_id", how="left")
            logger.info("[OK] Statik PoF sütunu (PoF_Statik) risk datasına eklendi.")

        # --------------------------------------------------------------
        # 3) Risk skoru hesapla
        # --------------------------------------------------------------
        logger.info("")
        logger.info("[STEP] Risk skoru hesaplanıyor (Risk_Skoru = PoF_12Ay * CoF)...")

        df["Risk_Skoru"] = df["PoF_12Ay"] * df["CoF"]

        # Basit sanity
        logger.info(f"[INFO] PoF_12Ay min/median/max: {df['PoF_12Ay'].min():.4f} / "
                    f"{df['PoF_12Ay'].median():.4f} / {df['PoF_12Ay'].max():.4f}")
        logger.info(f"[INFO] CoF min/median/max:      {df['CoF'].min():.2f} / "
                    f"{df['CoF'].median():.2f} / {df['CoF'].max():.2f}")
        logger.info(f"[INFO] Risk_Skoru min/median/max: {df['Risk_Skoru'].min():.4f} / "
                    f"{df['Risk_Skoru'].median():.4f} / {df['Risk_Skoru'].max():.4f}")

        # --------------------------------------------------------------
        # 4) Risk sınıfları
        # --------------------------------------------------------------
        logger.info("")
        logger.info("[STEP] Risk sınıfları atanıyor...")

        df["Risk_Sinifi"] = assign_risk_classes(df["Risk_Skoru"])

        sınıf_sayım = df["Risk_Sinifi"].value_counts(dropna=False)
        logger.info("[INFO] Risk sınıfları dağılımı:")
        for cls, cnt in sınıf_sayım.items():
            logger.info(f"    {cls:8s}: {cnt:5d} ekipman")

        # --------------------------------------------------------------
        # 5) ÇIKTILAR
        # --------------------------------------------------------------
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        # Ana risk tablosu
        risk_out_path = RESULTS_DIR / "risk_skorlari_pof3.csv"
        df.to_csv(risk_out_path, index=False, encoding="utf-8-sig")
        logger.info(f"[OK] Risk skorları çıktısı kaydedildi: {risk_out_path}")

        # Ekipman tipi bazında özet
        if "Ekipman_Tipi" in df.columns:
            summary = (
                df.groupby("Ekipman_Tipi")
                .agg(
                    Adet=("cbs_id", "count"),
                    Ortalama_Risk=("Risk_Skoru", "mean"),
                    Medyan_Risk=("Risk_Skoru", "median"),
                )
                .reset_index()
            )

            summary_path = RESULTS_DIR / "risk_skoru_ozet_ekipman_tipi.csv"
            summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
            logger.info(f"[OK] Ekipman tipi bazlı risk özeti kaydedildi: {summary_path}")
        else:
            logger.warning("[WARN] Ekipman_Tipi kolonu yok; ekipman tipi bazlı özet üretilemedi.")

        logger.info("")
        logger.info("[SUCCESS] 05_risk_degerlendirme başarıyla tamamlandı.")
        logger.info("=" * 80)

    except Exception as e:
        logger.exception(f"[FATAL] 05_risk_degerlendirme hata ile sonlandı: {e}")
        raise


if __name__ == "__main__":
    main()
