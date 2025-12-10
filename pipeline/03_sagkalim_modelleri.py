"""
03_sagkalim_modelleri.py (PoF3 - Sağkalım Modelleri)

Amaç:
- Step 01 → survival_base
- Step 02 → ozellikler_pof3
üzerinden:

  1) Cox Proportional Hazards modeli ile sağkalım analizi
  2) (Varsa) Random Survival Forest (RSF) modeli
  3) XGBoost + CatBoost ile statik arıza eğilimi (ML tabanlı PoF skoru)

Girdi:
    INTERMEDIATE_PATHS["survival_base"]
    FEATURE_OUTPUT_PATH  (örn: data/ara_ciktilar/ozellikler_pof3.csv)

Çıktılar (müşteri-facing, Türkçe):

    Cox PoF (sağkalım tabanlı ufuklar):
        data/sonuclar/cox_sagkalim_3ay_ariza_olasiligi.csv
        data/sonuclar/cox_sagkalim_6ay_ariza_olasiligi.csv
        data/sonuclar/cox_sagkalim_12ay_ariza_olasiligi.csv

    RSF PoF (varsa sksurv):
        data/sonuclar/rsf_sagkalim_3ay_ariza_olasiligi.csv
        data/sonuclar/rsf_sagkalim_6ay_ariza_olasiligi.csv
        data/sonuclar/rsf_sagkalim_12ay_ariza_olasiligi.csv

    Statik arıza eğilimi (ML tabanlı PoF):
        data/sonuclar/statik_ariza_egilim_skoru.csv

Model dosyaları (iç kullanım):
    <proje_koku>/modeller/pof_ml_xgb.json
    <proje_koku>/modeller/pof_ml_catboost.cbm
"""

# ----------------------------------------------------------------------
# Standard library imports
# ----------------------------------------------------------------------
import os
import sys
import logging
import warnings
from datetime import datetime, timedelta

# ----------------------------------------------------------------------
# Third-party imports
# ----------------------------------------------------------------------
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Project root setup
# ----------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# UTF-8 console
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# Project imports
# ----------------------------------------------------------------------
from utils.date_parser import parse_date_safely

# Cox PH
from lifelines import CoxPHFitter

# RSF (optional)
try:
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.util import Surv
    HAS_RSF = True
except Exception:
    HAS_RSF = False

# ML libraries (optional)
try:
    import xgboost as xgb
    from catboost import CatBoostClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, average_precision_score
    HAS_ML = True
except Exception:
    HAS_ML = False

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
try:
    from config.config import (
        ANALYSIS_DATE,
        INTERMEDIATE_PATHS,
        FEATURE_OUTPUT_PATH,
        OUTPUT_DIR,
        RESULT_PATHS,
        SURVIVAL_HORIZONS,
        SURVIVAL_HORIZONS_MONTHS,
        ML_REF_DAYS_BEFORE_ANALYSIS,
        ML_PREDICTION_WINDOW_DAYS,
        MIN_EQUIPMENT_PER_CLASS,
        RANDOM_STATE,
        LOG_DIR,
    )
except ImportError:
    # Fallback defaults
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    ANALYSIS_DATE = pd.Timestamp.today().normalize()
    INTERMEDIATE_PATHS = {
        "survival_base": os.path.join(DATA_DIR, "ara_ciktilar", "survival_base.csv"),
        "fault_events_clean": os.path.join(DATA_DIR, "ara_ciktilar", "fault_events_clean.csv"),
        "equipment_master": os.path.join(DATA_DIR, "ara_ciktilar", "equipment_master.csv"),
    }
    FEATURE_OUTPUT_PATH = os.path.join(DATA_DIR, "ara_ciktilar", "ozellikler_pof3.csv")
    OUTPUT_DIR = os.path.join(DATA_DIR, "sonuclar")
    RESULT_PATHS = {"POF": os.path.join(DATA_DIR, "sonuclar")}
    SURVIVAL_HORIZONS = [3, 6, 12, 24]
    SURVIVAL_HORIZONS_MONTHS = [3, 6, 12]
    ML_REF_DAYS_BEFORE_ANALYSIS = 365
    ML_PREDICTION_WINDOW_DAYS = 365
    MIN_EQUIPMENT_PER_CLASS = 30
    RANDOM_STATE = 42
    LOG_DIR = os.path.join(PROJECT_ROOT, "loglar")

STEP_NAME = "03_sagkalim_modelleri"
MODELS_DIR = os.path.join(PROJECT_ROOT, "modeller")


# ----------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------
def setup_logger(step_name: str) -> logging.Logger:
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"{step_name}_{ts}.log")

    logger = logging.getLogger(step_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    logger.info("=" * 80)
    logger.info(f"{step_name} - PoF3 Sağkalım Modelleri")
    logger.info("=" * 80)
    logger.info("")
    return logger


# ----------------------------------------------------------------------
# YARDIMCI: kolon isimlerini normalize et
# ----------------------------------------------------------------------
def normalize_columns(df_surv: pd.DataFrame, df_feat: pd.DataFrame, logger: logging.Logger):
    """
    Survival ve özellik tablolarında olası eski/İngilizce kolon isimlerini
    PoF3 Türkçe standardına çevirir.
    """

    # survival_base
    surv = df_surv.rename(
        columns={
            "CBS_ID": "cbs_id",
            "Sure_Gun": "duration_days",
            "Olay": "event",
        }
    )

    # features (ozellikler_pof3)
    feat = df_feat.rename(
        columns={
            "CBS_ID": "cbs_id",
            "Fault_Count": "Ariza_Sayisi",
            "Has_Ariza_Gecmisi": "Ariza_Gecmisi",
            "Kronik_90g_Flag": "Kronik_Flag_90g",
            "Gerilim": "Gerilim_Seviyesi",
            "MARKA": "Marka",
            "MARKA ": "Marka",
            "component voltage": "Gerilim_Seviyesi",
            "voltage_level": "Gerilim_Seviyesi",
            "voltage_level ": "Gerilim_Seviyesi",
        }
    )

    # event, duration_days zorunlu
    if "cbs_id" not in surv.columns:
        raise KeyError("survival_base içinde 'cbs_id' kolonu bulunamadı.")
    if "duration_days" not in surv.columns:
        raise KeyError("survival_base içinde 'duration_days' kolonu bulunamadı.")
    if "event" not in surv.columns:
        raise KeyError("survival_base içinde 'event' kolonu bulunamadı.")

    # Ariza_Sayisi fallback
    if "Ariza_Sayisi" not in feat.columns and "Fault_Count" in feat.columns:
        feat["Ariza_Sayisi"] = feat["Fault_Count"]

    # Ariza_Gecmisi fallback
    if "Ariza_Gecmisi" not in feat.columns and "Has_Ariza_Gecmisi" in feat.columns:
        feat["Ariza_Gecmisi"] = feat["Has_Ariza_Gecmisi"]

    # Kronik flag fallback
    if "Kronik_Flag_90g" not in feat.columns and "Kronik_90g_Flag" in feat.columns:
        feat["Kronik_Flag_90g"] = feat["Kronik_90g_Flag"]

    logger.info("[OK] Survival ve özellik kolon isimleri normalize edildi.")
    return surv, feat


# ----------------------------------------------------------------------
# Cox / RSF için veri hazırlığı
# ----------------------------------------------------------------------
def prepare_cox_data(df_full: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Cox / RSF için veri hazırlığı:
    - duration_days > 0 filtre
    - Nadir Ekipman_Tipi sınıflarını "Other" altında toplama
    - Çekirdek özellik seti + bakım/ekipman nitelikleri
    - Kategorik değişkenleri (Ekipman_Tipi) one-hot kodlama
    """

    required = ["duration_days", "event", "Ekipman_Tipi", "Ekipman_Yasi_Gun", "cbs_id"]
    missing = [c for c in required if c not in df_full.columns]
    if missing:
        raise KeyError(f"Sağkalım birleşik datasında eksik kolon(lar) var: {missing}")

    df = df_full.copy()

    # Negatif / sıfır süreyi at
    before = len(df)
    df = df[df["duration_days"] > 0].copy()
    dropped = before - len(df)
    if dropped > 0:
        logger.warning(f"[WARN] duration_days <= 0 olan {dropped:,} kayıt atıldı.")

    # Nadir ekipman tiplerini grupla
    counts = df["Ekipman_Tipi"].value_counts()
    rare = counts[counts < MIN_EQUIPMENT_PER_CLASS].index.tolist()
    if rare:
        logger.info(f"[INFO] Nadir ekipman tipleri 'Other' altında toplanıyor: {list(rare)}")
        df.loc[df["Ekipman_Tipi"].isin(rare), "Ekipman_Tipi"] = "Other"

    logger.info(
        f"[INFO] Ekipman tipleri: {df['Ekipman_Tipi'].nunique()} sınıf - "
        f"{sorted(df['Ekipman_Tipi'].unique())}"
    )

    # ----------- Özellik seti (çekirdek + bakım/ekipman) -----------------
    base_features = [
        "Ekipman_Yasi_Gun",
        "MTBF_Gun",
        "Ariza_Sayisi",
        "Kronik_Flag_90g",
        "Son_Ariza_Gun_Sayisi",
    ]

    maintenance_features = [
        "Bakim_Sayisi",
        "Bakim_Var_Mi",  # Binary indicator: has maintenance record
        "Son_Bakimdan_Gecen_Gun",
    ]

    chronic_features = [
        "Kronik_Kritik",  # Multi-level chronic flags
        "Kronik_Yuksek",
        "Kronik_Orta",
        "Kronik_Gozlem",
    ]

    equipment_features = [
        "kVA_Rating",
    ]

    # Varsa Gerilim_Seviyesi'ni sayısal hale getirmeye çalış (örn. kV içeren string)
    if "Gerilim_Seviyesi" in df.columns:
        try:
            # "34.5 kV" → 34.5 gibi
            df["Gerilim_Seviyesi_Sayisal"] = (
                df["Gerilim_Seviyesi"]
                .astype(str)
                .str.extract(r"([\d\.,]+)", expand=False)
                .str.replace(",", ".", regex=False)
                .astype(float)
            )
            equipment_features.append("Gerilim_Seviyesi_Sayisal")
            logger.info("[INFO] Gerilim_Seviyesi_Sayisal türetildi.")
        except Exception:
            logger.warning("[WARN] Gerilim_Seviyesi sayısallaştırılamadı, Cox'ta kullanılmayacak.")

    candidate_features = base_features + maintenance_features + chronic_features + equipment_features

    feature_cols = ["Ekipman_Tipi"] + [c for c in candidate_features if c in df.columns]

    # Zorunlu kolonlar + hedefler
    required_cols = feature_cols + ["duration_days", "event", "cbs_id"]
    df = df[required_cols].copy()

    # Sayısal özelliklerde NaN → median (veya 0 eğer median de NaN ise)
    # SPECIAL HANDLING: Maintenance features should NOT use median imputation
    numeric_cols = [
        c
        for c in df.columns
        if c not in ["Ekipman_Tipi", "cbs_id"]
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    # Maintenance columns: never maintained ≠ maintained at median time
    maintenance_cols = ["Son_Bakimdan_Gecen_Gun", "Ilk_Bakim_Tarihi"]

    for col in numeric_cols:
        if df[col].isna().any():
            # Special handling for maintenance features
            if any(m in col for m in maintenance_cols):
                df[col] = df[col].fillna(9999)  # Sentinel: never maintained
                logger.info(f"[INFO] {col} kolonundaki NaN değerler 9999 ile dolduruldu (hiç bakım yapılmamış).")
            else:
                med = df[col].median()
                # Eğer median de NaN ise (tüm değerler NaN), 0 kullan
                if pd.isna(med):
                    df[col] = df[col].fillna(0)
                    logger.info(f"[INFO] {col} kolonundaki NaN değerler 0 ile dolduruldu (median bulunamadı).")
                else:
                    df[col] = df[col].fillna(med)
                    logger.info(f"[INFO] {col} kolonundaki NaN değerler median={med:.2f} ile dolduruldu.")

    # cbs_id yedeği
    cbs_ids = df[["cbs_id"]].copy()

    # One-hot: Ekipman_Tipi
    df = pd.get_dummies(df, columns=["Ekipman_Tipi"], drop_first=True)

    # cbs_id tekrar güvence altına al
    if "cbs_id" not in df.columns:
        df["cbs_id"] = cbs_ids["cbs_id"].values

    # Son kontrol: hala NaN var mı?
    nan_counts = df.isna().sum()
    if nan_counts.any():
        logger.warning("[WARN] Bazı kolonlarda hala NaN değerler var:")
        for col in nan_counts[nan_counts > 0].index:
            logger.warning(f"  {col}: {nan_counts[col]} NaN")
        # Tüm kalan NaN'ları 0 ile doldur
        df = df.fillna(0)
        logger.info("[INFO] Kalan tüm NaN değerler 0 ile dolduruldu.")

    # Sabit kolonları tespit et ve çıkar (varyans = 0)
    numeric_features = [c for c in df.columns if c not in ["cbs_id", "duration_days", "event"]]
    constant_cols = []
    for col in numeric_features:
        if df[col].std() == 0:
            constant_cols.append(col)

    if constant_cols:
        logger.warning(f"[WARN] Sabit değerli kolonlar tespit edildi ve çıkarılacak: {constant_cols}")
        df = df.drop(columns=constant_cols)

    logger.info(f"[OK] Cox/RSF verisi hazır: {len(df):,} satır, {len(df.columns)} kolon")
    return df


# ----------------------------------------------------------------------
# Cox modeli
# ----------------------------------------------------------------------
def fit_cox_model(df_cox: pd.DataFrame, logger: logging.Logger) -> CoxPHFitter:
    logger.info("[STEP] Cox Proportional Hazards modeli eğitiliyor")

    train_df = df_cox.drop(columns=["cbs_id"]).copy()

    cph = CoxPHFitter(penalizer=0.01)
    try:
        cph.fit(train_df, duration_col="duration_days", event_col="event")
        logger.info("[OK] Cox modeli başarıyla eğitildi.")
        logger.info(f"[INFO] Concordance index: {cph.concordance_index_:.3f}")
    except Exception as e:
        logger.exception(f"[FATAL] Cox model eğitimi başarısız: {e}")
        raise

    return cph


def compute_pof_from_cox(
    cph: CoxPHFitter,
    df_cox: pd.DataFrame,
    horizons_days,
    logger: logging.Logger,
) -> dict:
    """
    Cox modeli üzerinden farklı ufuklar için PoF hesaplar.
    Dönüş: {days: pd.Series(index=cbs_id, values=PoF)}
    """
    logger.info("[STEP] Cox modeli ile PoF hesaplanıyor")

    cbs_ids = df_cox["cbs_id"].copy()
    drop_cols = ["duration_days", "event", "cbs_id"]
    X = df_cox.drop(columns=drop_cols).copy()

    results = {}

    for days in horizons_days:
        # Convert days to months for labeling
        months = round(days / 30)
        logger.info(f"  Ufuk: {days} gün (~{months} ay)")

        try:
            surv = cph.predict_survival_function(X, times=[days]).T
            pof = 1.0 - surv[days]
            pof.index = cbs_ids.values
            results[m] = pof
            logger.info(f"    Ortalama PoF: {pof.mean():.3f}, Maks: {pof.max():.3f}")
        except Exception as e:
            logger.error(f"    [ERROR] {m} ay için PoF hesaplanamadı: {e}")

    logger.info(f"[OK] Cox modeli için {len(results)} ufuk hesaplandı.")
    return results


# ----------------------------------------------------------------------
# RSF modeli
# ----------------------------------------------------------------------
def fit_rsf_model(df_cox: pd.DataFrame, logger: logging.Logger):
    if not HAS_RSF:
        logger.warning(
            "[WARN] sksurv kütüphanesi yüklü değil. RSF modeli atlanacak. "
            "pip install scikit-survival ile kurulabilir."
        )
        return None

    logger.info("[STEP] Random Survival Forest (RSF) modeli eğitiliyor")

    work = df_cox.copy()
    work = work[work["duration_days"] > 0].copy()

    y = Surv.from_arrays(
        event=work["event"].astype(bool),
        time=work["duration_days"].astype(float),
    )
    X = work.drop(columns=["event", "duration_days", "cbs_id"])

    try:
        rsf = RandomSurvivalForest(
            n_estimators=400,
            min_samples_split=20,
            min_samples_leaf=5,
            max_features="sqrt",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        rsf.fit(X, y)
        logger.info("[OK] RSF modeli başarıyla eğitildi.")

        # Feature importance
        logger.info("[RSF] Feature importance hesaplanıyor...")
        try:
            importance = rsf.feature_importances_
            importance_df = pd.DataFrame({
                "feature": X.columns,
                "importance": importance
            }).sort_values("importance", ascending=False)

            logger.info("[RSF] Top 10 önemli özellikler:")
            for _, row in importance_df.head(10).iterrows():
                logger.info(f"  {row['feature']:30s}: {row['importance']:.4f}")

            # Save importance
            importance_path = os.path.join(RESULT_PATHS["POF"], "rsf_feature_importance.csv")
            importance_df.to_csv(importance_path, index=False, encoding="utf-8-sig")
            logger.info(f"[RSF] Feature importance kaydedildi → {importance_path}")
        except Exception as e:
            logger.warning(f"[WARN] RSF feature importance hesaplanamadı: {e}")

        return rsf
    except Exception as e:
        logger.exception(f"[FATAL] RSF modeli eğitimi başarısız: {e}")
        return None


def compute_pof_from_rsf(
    rsf_model,
    df_cox: pd.DataFrame,
    horizons_days,
    logger: logging.Logger,
) -> dict:
    if rsf_model is None:
        logger.warning("[WARN] RSF modeli None, PoF hesaplanmayacak.")
        return {}

    logger.info("[STEP] RSF modeli ile PoF hesaplanıyor")

    work = df_cox.copy()
    cbs_ids = work["cbs_id"].copy()
    X = work.drop(columns=["duration_days", "event", "cbs_id"])

    results = {}

    surv_fns = rsf_model.predict_survival_function(X)

    for days in horizons_days:
        # Convert days to months for labeling
        months = round(days / 30)
        logger.info(f"  RSF ufuk: {days} gün (~{months} ay)")

        try:
            pof_vals = np.array([1.0 - fn(days) for fn in surv_fns])
            series = pd.Series(pof_vals, index=cbs_ids.values)
            results[m] = series

            logger.info(
                f"    Ortalama RSF PoF: {series.mean():.3f}, Maks: {series.max():.3f}"
            )
        except Exception as e:
            logger.error(f"    [ERROR] RSF {m} ay için PoF hesaplanamadı: {e}")

    return results


# ----------------------------------------------------------------------
# ML tabanlı Leakage-Free PoF (XGBoost + CatBoost)
# ----------------------------------------------------------------------
def build_leakage_free_ml_dataset(logger):
    logger.info("")
    logger.info("[ML] Leakage-free veri seti oluşturuluyor...")

    # ---------------------------------------------------------
    # Load DATA_END_DATE from Step 01 metadata
    # ---------------------------------------------------------
    metadata_path = os.path.join(
        os.path.dirname(INTERMEDIATE_PATHS["fault_events_clean"]),
        "data_range_metadata.csv"
    )

    if not os.path.exists(metadata_path):
        logger.warning(f"[WARN] Data range metadata not found: {metadata_path}")
        logger.warning("[WARN] Falling back to ANALYSIS_DATE from config")
        analysis_dt = pd.to_datetime(ANALYSIS_DATE)
    else:
        metadata = pd.read_csv(metadata_path, encoding="utf-8-sig")
        DATA_END_DATE_str = metadata.loc[metadata["Parameter"] == "DATA_END_DATE", "Value"].iloc[0]
        DATA_END_DATE = pd.to_datetime(DATA_END_DATE_str)
        DATA_START_DATE_str = metadata.loc[metadata["Parameter"] == "DATA_START_DATE", "Value"].iloc[0]
        DATA_START_DATE = pd.to_datetime(DATA_START_DATE_str)

        logger.info(f"[ML] DATA_START_DATE (detected) = {DATA_START_DATE.date()}")
        logger.info(f"[ML] DATA_END_DATE (detected)   = {DATA_END_DATE.date()}")

        analysis_dt = DATA_END_DATE

    # ---------------------------------------------------------
    # Calculate valid T_ref
    # ---------------------------------------------------------
    # T_ref should be positioned such that:
    # 1. We have at least 2 years of training data before T_ref
    # 2. We have 12 months of prediction window that doesn't exceed DATA_END_DATE
    #
    # Therefore: T_ref = DATA_END_DATE - 12 months (365 days)

    ref_date = analysis_dt - timedelta(days=ML_PREDICTION_WINDOW_DAYS)
    window_end = ref_date + timedelta(days=ML_PREDICTION_WINDOW_DAYS)

    logger.info("")
    logger.info(f"[ML] Referans tarih (T_ref) = {ref_date.date()}")
    logger.info(f"[ML] Tahmin penceresi = {ref_date.date()} → {window_end.date()}")

    # Validation check
    if os.path.exists(metadata_path):
        training_span_days = (ref_date - DATA_START_DATE).days
        training_span_years = training_span_days / 365.25

        logger.info(f"[ML] Training data span = {training_span_years:.2f} years ({training_span_days:,} days)")

        MIN_TRAIN_YEARS = 2.0
        if training_span_years < MIN_TRAIN_YEARS:
            logger.error(f"[FATAL] Insufficient training data: {training_span_years:.2f} years < {MIN_TRAIN_YEARS} years")
            logger.error(f"[FATAL] T_ref = {ref_date.date()} leaves insufficient historical data")
            raise ValueError(f"Insufficient training data: {training_span_years:.2f} years")

        logger.info(f"[OK] Training data validation passed: {training_span_years:.2f} years >= {MIN_TRAIN_YEARS} years")
        logger.info("")

    # ----------------------------------------------
    # 1) Load input sources
    # ----------------------------------------------
    events = pd.read_csv(INTERMEDIATE_PATHS["fault_events_clean"], encoding="utf-8-sig")
    eq = pd.read_csv(INTERMEDIATE_PATHS["equipment_master"], encoding="utf-8-sig")

    events.rename(columns={"CBS_ID": "cbs_id"}, inplace=True)
    eq.rename(columns={"CBS_ID": "cbs_id"}, inplace=True)

    events["Ariza_Baslangic_Zamani"] = events["Ariza_Baslangic_Zamani"].apply(parse_date_safely)
    eq["Kurulum_Tarihi"] = eq["Kurulum_Tarihi"].apply(parse_date_safely)

    events["cbs_id"] = events["cbs_id"].astype(str).str.lower().str.strip()
    eq["cbs_id"] = eq["cbs_id"].astype(str).str.lower().str.strip()

    # ----------------------------------------------
    # 2) Split past & future faults
    # ----------------------------------------------
    past = events[events["Ariza_Baslangic_Zamani"] <= ref_date].copy()
    future = events[
        (events["Ariza_Baslangic_Zamani"] > ref_date)
        & (events["Ariza_Baslangic_Zamani"] <= window_end)
    ].copy()

    logger.info(f"[ML] Geçmiş arıza sayısı: {len(past):,}")
    logger.info(f"[ML] Pencere içi arıza sayısı: {len(future):,}")

    # ----------------------------------------------
    # 3) Build past-features
    # ----------------------------------------------
    rows = []
    for cid, grp in past.groupby("cbs_id"):
        times = grp["Ariza_Baslangic_Zamani"].dropna().sort_values().values
        cnt = len(times)

        if cnt < 2:
            mtbf = np.nan
        else:
            diffs = np.diff(times).astype("timedelta64[D]").astype(int)
            mtbf = float(np.mean(diffs))

        last = times[-1] if cnt > 0 else None
        days_since_last = (ref_date - last).days if last is not None else np.nan

        rows.append((cid, cnt, mtbf, days_since_last))

    past_feat = pd.DataFrame(
        rows,
        columns=[
            "cbs_id",
            "Ariza_Sayisi_Gecmis",
            "MTBF_Gun_Gecmis",
            "Son_Ariza_Gun_Sayisi_Gecmis",
        ]
    )

    # ----------------------------------------------
    # 4) Build labels (future window)
    # ----------------------------------------------
    if future.empty:
        target = pd.DataFrame({"cbs_id": eq["cbs_id"], "Label_Ariza_Pencere": 0})
    else:
        tmp = future.groupby("cbs_id").size().rename("Label_Ariza_Pencere").reset_index()
        tmp["Label_Ariza_Pencere"] = 1
        target = tmp

    # ----------------------------------------------
    # 5) Base equipment snapshot at T_ref
    # ----------------------------------------------
    base = eq.copy()
    base = base[base["Kurulum_Tarihi"] <= ref_date].copy()

    base["Ekipman_Yasi_Gun_ML"] = (ref_date - base["Kurulum_Tarihi"]).dt.days.clip(lower=0)

    if "Gerilim_Seviyesi" not in eq.columns:
        base["Gerilim_Seviyesi"] = "UNKNOWN"
    if "Marka" not in eq.columns:
        base["Marka"] = "UNKNOWN"

    # ----------------------------------------------
    # 6) Merge everything
    # ----------------------------------------------
    df = (
        base[["cbs_id", "Ekipman_Tipi", "Gerilim_Seviyesi", "Marka", "Ekipman_Yasi_Gun_ML"]]
        .merge(past_feat, on="cbs_id", how="left")
        .merge(target, on="cbs_id", how="left")
    )

    df["Label_Ariza_Pencere"] = df["Label_Ariza_Pencere"].fillna(0).astype(int)
    df["Ariza_Sayisi_Gecmis"] = df["Ariza_Sayisi_Gecmis"].fillna(0).astype(int)
    df["MTBF_Gun_Gecmis"] = df["MTBF_Gun_Gecmis"].fillna(df["MTBF_Gun_Gecmis"].median())
    df["Son_Ariza_Gun_Sayisi_Gecmis"] = df["Son_Ariza_Gun_Sayisi_Gecmis"].fillna(
        df["Ekipman_Yasi_Gun_ML"]
    )

    logger.info(f"[ML] ML veri seti: {len(df):,} ekipman")
    logger.info(f"[ML] Pozitif label sayısı: {df['Label_Ariza_Pencere'].sum():,}")

    return df
def train_leakage_free_ml_models(df, logger):
    logger.info("")
    logger.info("[STEP] Leakage-free ML PoF modeli eğitiliyor (XGBoost + CatBoost)")

    y = df["Label_Ariza_Pencere"]
    id_col = "cbs_id"

    # Separate inputs
    numeric_cols = [
        "Ekipman_Yasi_Gun_ML",
        "Ariza_Sayisi_Gecmis",
        "MTBF_Gun_Gecmis",
        "Son_Ariza_Gun_Sayisi_Gecmis",
    ]
    cat_cols = ["Ekipman_Tipi", "Gerilim_Seviyesi", "Marka"]

    # Add binary/ordinal features if available
    binary_cols = []
    if "Bakim_Var_Mi" in df.columns:
        binary_cols.append("Bakim_Var_Mi")
    if "Kronik_90g_Flag" in df.columns:
        binary_cols.append("Kronik_90g_Flag")

    # Combine all features
    all_feature_cols = numeric_cols + binary_cols + cat_cols

    X_num = df[numeric_cols].copy()
    X_bin = df[binary_cols].copy() if binary_cols else pd.DataFrame(index=df.index)
    X_cat = df[cat_cols].astype(str)

    # XGBoost: numerical + binary + one-hot categorical
    X_xgb = pd.concat([X_num, X_bin, pd.get_dummies(X_cat, drop_first=True)], axis=1)

    # CatBoost: numerical + binary + categorical
    X_catb = pd.concat([X_num, X_bin, X_cat], axis=1)
    cat_idx = [X_catb.columns.get_loc(c) for c in cat_cols]

    from sklearn.model_selection import train_test_split
    X_train_xgb, X_test_xgb, y_train, y_test = train_test_split(
        X_xgb, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )

    X_train_cat, X_test_cat, _, _ = train_test_split(
        X_catb, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )

    # ------------------- XGBoost -------------------
    logger.info("[ML] XGBoost eğitiliyor...")
    model_xgb = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
    )
    model_xgb.fit(X_train_xgb, y_train)

    proba_xgb = model_xgb.predict_proba(X_test_xgb)[:, 1]
    auc_xgb = roc_auc_score(y_test, proba_xgb)
    ap_xgb = average_precision_score(y_test, proba_xgb)

    logger.info(f"[ML] XGBoost AUC={auc_xgb:.3f}, AP={ap_xgb:.3f}")

    # ------------------- CatBoost -------------------
    logger.info("[ML] CatBoost eğitiliyor...")
    model_cat = CatBoostClassifier(
        iterations=400,
        depth=4,
        learning_rate=0.05,
        eval_metric="AUC",
        loss_function="Logloss",
        random_state=RANDOM_STATE,
        verbose=False,
    )
    model_cat.fit(
        X_train_cat,
        y_train,
        cat_features=cat_idx,
        eval_set=(X_test_cat, y_test),
        verbose=False,
    )

    proba_cat = model_cat.predict_proba(X_test_cat)[:, 1]
    auc_cat = roc_auc_score(y_test, proba_cat)
    ap_cat = average_precision_score(y_test, proba_cat)

    logger.info(f"[ML] CatBoost AUC={auc_cat:.3f}, AP={ap_cat:.3f}")

    # ------------------- Advanced Features -------------------
    RESULTS = RESULT_PATHS["POF"]
    os.makedirs(RESULTS, exist_ok=True)

    # 1. Temporal Cross-Validation
    logger.info("")
    logger.info("[ADVANCED] Running temporal cross-validation...")
    try:
        from utils.ml_advanced import temporal_cross_validation

        cv_results = temporal_cross_validation(
            X=X_xgb,
            y=y,
            model_fn=lambda: xgb.XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE, eval_metric="logloss"
            ),
            n_splits=3,
            logger=logger
        )

        # Save CV results
        cv_df = pd.DataFrame({
            "metric": ["AUC"] * 3 + ["AP"] * 3,
            "fold": [1, 2, 3, 1, 2, 3],
            "score": cv_results["auc_scores"] + cv_results["ap_scores"]
        })
        cv_path = os.path.join(RESULTS, "temporal_cv_scores.csv")
        cv_df.to_csv(cv_path, index=False, encoding="utf-8-sig")
        logger.info(f"[ADVANCED] Temporal CV scores saved → {cv_path}")
    except Exception as e:
        logger.warning(f"[WARN] Temporal CV failed: {e}")

    # 2. SHAP Feature Importance
    logger.info("")
    logger.info("[ADVANCED] Computing SHAP feature importance...")
    try:
        from utils.ml_advanced import compute_shap_importance

        shap_df = compute_shap_importance(model_xgb, X_xgb, max_samples=1000, logger=logger)
        if not shap_df.empty:
            shap_path = os.path.join(RESULTS, "shap_feature_importance.csv")
            shap_df.to_csv(shap_path, index=False, encoding="utf-8-sig")
            logger.info(f"[ADVANCED] SHAP importance saved → {shap_path}")
    except Exception as e:
        logger.warning(f"[WARN] SHAP computation failed: {e}")

    # ------------------- Save full-model outputs -------------------
    out_df = pd.DataFrame({
        "cbs_id": df["cbs_id"],
        "PoF_ML_XGB": model_xgb.predict_proba(X_xgb)[:, 1],
        "PoF_ML_CatBoost": model_cat.predict_proba(X_catb)[:, 1],
    })
    out_path = os.path.join(RESULTS, "leakage_free_ml_pof.csv")
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    logger.info(f"[ML] Leakage-free ML PoF skorları kaydedildi → {out_path}")

    # Save models
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_xgb.save_model(os.path.join(MODELS_DIR, "leakage_free_pof_xgb.json"))
    model_cat.save_model(os.path.join(MODELS_DIR, "leakage_free_pof_catboost.cbm"))

    logger.info(f"[ML] Modeller kaydedildi → {MODELS_DIR}")

# ----------------------------------------------------------------------
# ANA AKIŞ
# ----------------------------------------------------------------------
def main():
    logger = setup_logger(STEP_NAME)

    try:
        # --------------------------------------------------------------
        # 1) Veri yükleme
        # --------------------------------------------------------------
        logger.info("[STEP] survival_base ve ozellikler_pof3 dosyaları yükleniyor")

        surv_path = INTERMEDIATE_PATHS.get("survival_base")
        if surv_path is None:
            raise KeyError("INTERMEDIATE_PATHS['survival_base'] tanımlı değil (config.config).")

        if not os.path.exists(surv_path):
            raise FileNotFoundError(f"Survival base bulunamadı: {surv_path}")

        if FEATURE_OUTPUT_PATH is None or not os.path.exists(FEATURE_OUTPUT_PATH):
            raise FileNotFoundError(f"Özellik tablosu (FEATURE_OUTPUT_PATH) bulunamadı: {FEATURE_OUTPUT_PATH}")

        survival_base = pd.read_csv(surv_path, encoding="utf-8-sig")
        features = pd.read_csv(FEATURE_OUTPUT_PATH, encoding="utf-8-sig")

        survival_base, features = normalize_columns(survival_base, features, logger)

        logger.info(f"[OK] survival_base yüklendi: {len(survival_base):,} satır")
        logger.info(f"[OK] ozellikler_pof3 yüklendi: {len(features):,} satır")

        # --------------------------------------------------------------
        # 2) Survival + özellik merge
        # --------------------------------------------------------------
        # survival_base ve features'da ortak kolonlar var (Ekipman_Tipi, Ekipman_Yasi_Gun vb.)
        # features'dan alalım çünkü daha zengin (MTBF_Gun, Son_Ariza_Gun_Sayisi, vb.)

        # survival_base'den sadece zorunlu survival kolonları + cbs_id alalım
        surv_cols = ["cbs_id", "event", "duration_days"]
        df_surv = survival_base[surv_cols].copy()

        # features'dan ise tüm özellikleri alalım
        merge_cols = [
            "cbs_id",
            "Ekipman_Tipi",
            "Ekipman_Yasi_Gun",
            "MTBF_Gun",
            "Ariza_Sayisi",
            "Ariza_Gecmisi",
            "Kronik_90g_Flag",
            "Kronik_Kritik",  # Multi-level chronic flags
            "Kronik_Yuksek",
            "Kronik_Orta",
            "Kronik_Gozlem",
            "Kronik_Seviye_Max",
            "Son_Ariza_Gun_Sayisi",
            "Bakim_Sayisi",
            "Bakim_Var_Mi",  # Binary maintenance indicator
            "Ilk_Bakim_Tarihi",
            "Son_Bakim_Tarihi",
            "Son_Bakimdan_Gecen_Gun",
            "Son_Bakim_Tipi",
            "Bakim_Is_Emri_Tipleri",
            "Gerilim_Seviyesi",
            "kVA_Rating",
            "Marka",
        ]
        cols_in_features = [c for c in merge_cols if c in features.columns]
        feat_sub = features[cols_in_features].copy()

        df_full = df_surv.merge(feat_sub, on="cbs_id", how="left")
        logger.info(f"[OK] Survival + özellik birleşik veri seti: {len(df_full):,} satır")
        logger.info("")

        # --------------------------------------------------------------
        # 3) Cox / RSF için veri hazırlığı
        # --------------------------------------------------------------
        df_cox = prepare_cox_data(df_full, logger)
        logger.info("")

        # --------------------------------------------------------------
        # 4) Cox modeli + PoF
        # --------------------------------------------------------------
        cph = fit_cox_model(df_cox, logger)
        logger.info("")

        # Use actual days from config and convert to months for labeling
        pof_cox = compute_pof_from_cox(cph, df_cox, [int(d/30) for d in SURVIVAL_HORIZONS], logger)
        logger.info("")

        # --------------------------------------------------------------
        # 5) Cox çıktılarını kaydet (Türkçe dosya adları)
        # --------------------------------------------------------------
        logger.info("[STEP] Cox PoF çıktılarını kaydetme")
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        for days, series in pof_cox.items():
            # Convert days to months for labeling
            months = round(days / 30)
            out_df = pd.DataFrame(
                {
                    "cbs_id": series.index,
                    f"PoF_Cox_{months}Ay": series.values,
                }
            )
            out_path = os.path.join(OUTPUT_DIR, f"cox_sagkalim_{months}ay_ariza_olasiligi.csv")
            out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
            logger.info(f"[OK] {out_path}")

        # --------------------------------------------------------------
        # 6) RSF modeli + PoF
        # --------------------------------------------------------------
        logger.info("")
        logger.info("[STEP] Random Survival Forest (RSF) adımı")
        rsf = fit_rsf_model(df_cox, logger)
        rsf_pof = compute_pof_from_rsf(rsf, df_cox, [int(d/30) for d in SURVIVAL_HORIZONS], logger)

        if rsf_pof:
            logger.info("[STEP] RSF PoF çıktılarını kaydetme")
            for days, series in rsf_pof.items():
                # Convert days to months for labeling
                months = round(days / 30)
                out_df = pd.DataFrame(
                    {
                        "cbs_id": series.index,
                        f"PoF_RSF_{months}Ay": series.values,
                    }
                )
                out_path = os.path.join(OUTPUT_DIR, f"rsf_sagkalim_{months}ay_ariza_olasiligi.csv")
                out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
                logger.info(f"[OK] {out_path}")
        else:
            logger.info("[INFO] RSF çıktısı üretilmedi (RSF modeli yok veya hata).")

        # --------------------------------------------------------------
        # 7) ML tabanlı statik PoF (XGBoost + CatBoost)
        # --------------------------------------------------------------
        logger.info("")
        logger.info("[STEP] Leakage-free ML PoF hesaplama başlıyor")

        df_ml = build_leakage_free_ml_dataset(logger)
        train_leakage_free_ml_models(df_ml, logger)

        # --------------------------------------------------------------
        # 8) Survival Curves and Advanced Visualizations
        # --------------------------------------------------------------
        logger.info("")
        logger.info("[STEP] Generating survival curves and visualizations...")

        try:
            from utils.survival_plotting import (
                plot_survival_curves_by_class,
                plot_cox_coefficients,
                plot_feature_importance_comparison
            )
            from config.config import VISUAL_DIR

            # Plot survival curves
            surv_plot_path = os.path.join(VISUAL_DIR, "survival_curves_by_class.png")
            plot_survival_curves_by_class(
                df=df_full,
                output_path=surv_plot_path,
                logger=logger
            )

            # Plot Cox coefficients
            if cph is not None:
                cox_plot_path = os.path.join(VISUAL_DIR, "cox_coefficients.png")
                plot_cox_coefficients(
                    cox_model=cph,
                    output_path=cox_plot_path,
                    logger=logger
                )

            # Plot feature importance comparison (if both available)
            rsf_imp_path = os.path.join(OUTPUT_DIR, "rsf_feature_importance.csv")
            shap_imp_path = os.path.join(OUTPUT_DIR, "shap_feature_importance.csv")
            if os.path.exists(rsf_imp_path) and os.path.exists(shap_imp_path):
                rsf_imp = pd.read_csv(rsf_imp_path, encoding="utf-8-sig")
                shap_imp = pd.read_csv(shap_imp_path, encoding="utf-8-sig")
                comp_plot_path = os.path.join(VISUAL_DIR, "feature_importance_comparison.png")
                plot_feature_importance_comparison(
                    rsf_importance=rsf_imp,
                    shap_importance=shap_imp,
                    output_path=comp_plot_path,
                    logger=logger
                )

        except Exception as e:
            logger.warning(f"[WARN] Visualization generation failed: {e}")

        logger.info("")
        logger.info("[SUCCESS] 03_sagkalim_modelleri başarıyla tamamlandı.")
        logger.info("=" * 80)

    except Exception as e:
        logger.exception(f"[FATAL] 03_sagkalim_modelleri başarısız: {e}")
        raise


if __name__ == "__main__":
    main()
