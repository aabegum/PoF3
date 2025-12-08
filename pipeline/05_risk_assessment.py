# pipeline/05_risk_assessment.py

import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to Python path for utils import
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd

from utils.logger import get_logger

# --------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
INPUT_DIR = DATA_DIR / "inputs"
OUTPUT_DIR = DATA_DIR / "outputs"
LOG_DIR = BASE_DIR / "logs"

LOG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / f"05_risk_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logger = get_logger("05_risk_assessment", LOG_FILE)

# Risk kolon isimleri (config.py ile de uyumlu)
COF_COL_DEFAULT = "CoF_Skoru"
RISK_COL_DEFAULT = "Risk_Skoru"


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------
def detect_id_column(df: pd.DataFrame) -> str:
    candidates = ["CBS_ID", "cbs_id", "Ekipman_ID", "Ekipman Id", "Ekipman ID"]
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"ID kolonu bulunamadı. Adaylar: {candidates}")


def detect_pof_column(df: pd.DataFrame) -> str:
    # 12 aylık PoF kolonu: isim henüz sabit değil, esnek tespit
    pof_candidates = [c for c in df.columns if "pof" in c.lower()]
    if not pof_candidates:
        # fallback: 12 ay diye isimlenmiş kolon var mı
        pof_candidates = [c for c in df.columns if "12" in c and df[c].dtype != "O"]

    if not pof_candidates:
        raise KeyError("PoF kolonu tespit edilemedi (isimde 'pof' geçen kolon bulunamadı).")

    if len(pof_candidates) > 1:
        logger.warning(f"[WARN] Birden fazla PoF kolonu bulundu: {pof_candidates}. "
                       f"İlk kolon kullanılacak: {pof_candidates[0]}")

    return pof_candidates[0]


def load_pof_12m():
    logger.info("=" * 80)
    logger.info("05_risk_assessment - PoF3 Risk Değerlendirmesi")
    logger.info("=" * 80)
    logger.info("")

    pof_path = OUTPUT_DIR / "pof_cox_12m.csv"
    if not pof_path.exists():
        logger.error(f"[FATAL] pof_cox_12m.csv bulunamadı: {pof_path}")
        raise FileNotFoundError(pof_path)

    pof = pd.read_csv(pof_path)
    logger.info(f"[OK] 12M PoF dosyası yüklendi: {len(pof):,} ekipman")

    id_col = detect_id_column(pof)
    pof_col = detect_pof_column(pof)

    logger.info(f"[INFO] ID kolonu: {id_col}")
    logger.info(f"[INFO] PoF kolonu: {pof_col}")

    return pof, id_col, pof_col


def load_cof(id_col: str) -> tuple[pd.DataFrame | None, str | None]:
    # CoF veri kaynağı: önce CSV, sonra Excel denenir
    candidates = [
        INPUT_DIR / "cof_data.csv",
        INPUT_DIR / "cof_data.xlsx",
        OUTPUT_DIR / "cof_scores.csv"
    ]

    kof_path = None
    for path in candidates:
        if path.exists():
            kof_path = path
            break

    if kof_path is None:
        logger.warning("[WARN] CoF dosyası bulunamadı. Risk_Skoru PoF'e eşit olacak (CoF=1 varsayımı).")
        return None, None

    logger.info(f"[OK] CoF verisi bulundu: {kof_path}")

    if kof_path.suffix.lower() == ".csv":
        cof = pd.read_csv(kof_path)
    else:
        cof = pd.read_excel(kof_path)

    if id_col not in cof.columns:
        # aynı ID kolonunu bulmaya çalış
        alt_id = detect_id_column(cof)
        if alt_id != id_col:
            logger.warning(
                f"[WARN] CoF dosyasında ID kolonu '{alt_id}'. PoF tarafındaki '{id_col}' ile eşleşecek."
            )
            cof.rename(columns={alt_id: id_col}, inplace=True)

    # CoF kolonu tespiti
    cof_candidates = [c for c in cof.columns if "cof" in c.lower() or "risk" in c.lower()]
    if not cof_candidates:
        logger.warning(
            f"[WARN] CoF skoru için kolon bulunamadı. Yeni kolon '{COF_COL_DEFAULT}' üretilecek ve 1 atanacak."
        )
        cof[COF_COL_DEFAULT] = 1.0
        cof_col = COF_COL_DEFAULT
    else:
        if len(cof_candidates) > 1:
            logger.warning(f"[WARN] Birden fazla CoF adayı bulundu: {cof_candidates}. "
                           f"İlk kolon kullanılacak: {cof_candidates[0]}")
        cof_col = cof_candidates[0]

    logger.info(f"[INFO] CoF kolonu: {cof_col}")
    return cof[[id_col, cof_col]].copy(), cof_col


def categorize_risk(risk_values: pd.Series) -> pd.Series:
    """
    Basit 3 seviyeli risk sınıflandırması:
    - DÜŞÜK: 0–alt_eşik
    - ORTA : alt_eşik–üst_eşik
    - YÜKSEK: üst_eşik–max
    Eşikler default: 0.02 ve 0.05 (PoF*CoF ~ 2% ve 5%).
    """
    # Eğer değerler çok küçükse quantile tabanlı da yapılabilir
    alt_esik = 0.02
    ust_esik = 0.05

    def _cat(x):
        if pd.isna(x):
            return "BİLİNMİYOR"
        if x < alt_esik:
            return "DÜŞÜK"
        elif x < ust_esik:
            return "ORTA"
        else:
            return "YÜKSEK"

    return risk_values.apply(_cat)


def main():
    try:
        pof, id_col, pof_col = load_pof_12m()
        cof, cof_col = load_cof(id_col)

        df = pof.copy()

        if cof is None:
            # CoF yoksa tüm CoF=1
            df[COF_COL_DEFAULT] = 1.0
            used_cof_col = COF_COL_DEFAULT
            logger.info(f"[INFO] CoF verisi bulunamadığı için tüm ekipmanlara {COF_COL_DEFAULT}=1 atandı.")
        else:
            df = df.merge(cof, on=id_col, how="left")
            used_cof_col = cof_col
            # Eksik CoF'leri medyan ile doldur
            missing = df[used_cof_col].isna().sum()
            if missing > 0:
                med = df[used_cof_col].median()
                df[used_cof_col] = df[used_cof_col].fillna(med)
                logger.warning(
                    f"[WARN] {missing} ekipmanda CoF eksikti. Medyan değer ({med:.3f}) ile dolduruldu."
                )

        # Risk_Skoru hesapla
        risk_col = RISK_COL_DEFAULT
        df[risk_col] = df[pof_col] * df[used_cof_col]

        logger.info(
            f"[INFO] Risk_Skoru istatistikleri - min: {df[risk_col].min():.4f}, "
            f"mean: {df[risk_col].mean():.4f}, max: {df[risk_col].max():.4f}"
        )

        # Risk sınıfları
        df["Risk_Sinifi"] = categorize_risk(df[risk_col])

        # Çıktılar
        risk_table_path = OUTPUT_DIR / "pof3_risk_table.csv"
        df.to_csv(risk_table_path, index=False)
        logger.info(f"[OK] Ekipman bazlı risk tablosu kaydedildi: {risk_table_path}")

        # Ekipman tipine göre özet
        type_col = None
        for cand in ["Ekipman_Tipi", "Equipment_Type"]:
            if cand in df.columns:
                type_col = cand
                break

        if type_col:
            summary = (
                df.groupby([type_col, "Risk_Sinifi"])[risk_col]
                .agg(Ortalama_Risk=("mean"), Ekipman_Sayisi=("count"))
                .reset_index()
            )
            summary_path = OUTPUT_DIR / "pof3_risk_summary_by_type.csv"
            summary.to_csv(summary_path, index=False)
            logger.info(f"[OK] Ekipman tipi bazlı risk özeti kaydedildi: {summary_path}")
        else:
            logger.warning("[WARN] Ekipman_Tipi kolonu bulunamadı; tip bazlı özet oluşturulamadı.")

        logger.info("[SUCCESS] 05_risk_assessment başarıyla tamamlandı.")

    except Exception as e:
        logger.exception(f"[FATAL] 05_risk_assessment hata ile sonlandı: {e}")
        raise


if __name__ == "__main__":
    main()
