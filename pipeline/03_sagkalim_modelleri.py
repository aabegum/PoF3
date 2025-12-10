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

import os
import sys
import logging
from datetime import datetime

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Proje kökü ve UTF-8 konsol
# ----------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# Opsiyonel ML kütüphaneleri
# ----------------------------------------------------------------------
try:
    import xgboost as xgb
    from catboost import CatBoostClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, average_precision_score
    HAS_ML = True
except Exception:
    HAS_ML = False

# Cox PH
from lifelines import CoxPHFitter

# RSF (opsiyonel)
try:
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.util import Surv
    HAS_RSF = True
except Exception:
    HAS_RSF = False

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
try:
    from config.config import (
        INTERMEDIATE_PATHS,
        FEATURE_OUTPUT_PATH,
        OUTPUT_DIR,
        SURVIVAL_HORIZONS_MONTHS,
        MIN_EQUIPMENT_PER_CLASS,
        RANDOM_STATE,
        LOG_DIR,
    )
except Exception:
    # Fallback (çok gerekirse)
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    INTERMEDIATE_PATHS = {
        "survival_base": os.path.join(DATA_DIR, "ara_ciktilar", "survival_base.csv"),
    }
    FEATURE_OUTPUT_PATH = os.path.join(DATA_DIR, "ara_ciktilar", "ozellikler_pof3.csv")
    OUTPUT_DIR = os.path.join(DATA_DIR, "sonuclar")
    SURVIVAL_HORIZONS_MONTHS = [3, 6, 12]
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
        "Son_Bakimdan_Gecen_Gun",
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

    candidate_features = base_features + maintenance_features + equipment_features

    feature_cols = ["Ekipman_Tipi"] + [c for c in candidate_features if c in df.columns]

    # Zorunlu kolonlar + hedefler
    required_cols = feature_cols + ["duration_days", "event", "cbs_id"]
    df = df[required_cols].copy()

    # Sayısal özelliklerde NaN → median (veya 0 eğer median de NaN ise)
    numeric_cols = [
        c
        for c in df.columns
        if c not in ["Ekipman_Tipi", "cbs_id"]
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    for col in numeric_cols:
        if df[col].isna().any():
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
    horizons_months,
    logger: logging.Logger,
) -> dict:
    """
    Cox modeli üzerinden farklı ufuklar için PoF hesaplar.
    Dönüş: {ay: pd.Series(index=cbs_id, values=PoF)}
    """
    logger.info("[STEP] Cox modeli ile PoF hesaplanıyor")

    cbs_ids = df_cox["cbs_id"].copy()
    drop_cols = ["duration_days", "event", "cbs_id"]
    X = df_cox.drop(columns=drop_cols).copy()

    results = {}

    for m in horizons_months:
        days = m * 30
        logger.info(f"  Ufuk: {m} ay ({days} gün)")

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
        return rsf
    except Exception as e:
        logger.exception(f"[FATAL] RSF modeli eğitimi başarısız: {e}")
        return None


def compute_pof_from_rsf(
    rsf_model,
    df_cox: pd.DataFrame,
    horizons_months,
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

    for m in horizons_months:
        days = m * 30
        logger.info(f"  RSF ufuk: {m} ay ({days} gün)")

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
# ML tabanlı statik PoF (XGBoost + CatBoost)
# ----------------------------------------------------------------------
def train_ml_pof_models(df_full: pd.DataFrame, logger: logging.Logger) -> None:
    """
    ML tabanlı PoF modeli (XGBoost + CatBoost).

    Hedef: event (0/1) → ekipman bugüne kadar en az bir kez arıza yapmış mı?
    - Zaman ufkuna bağlı olmayan "statik arıza eğilimi" skoru üretir.
    - Sağkalım modellerini tamamlayıcı olarak kullanılır.
    """
    if not HAS_ML:
        logger.warning(
            "[WARN] XGBoost/CatBoost bulunamadı, ML PoF modeli atlanıyor. "
            "pip install xgboost catboost komutlarını kontrol edin."
        )
        return

    if "event" not in df_full.columns:
        logger.warning("[WARN] ML PoF modeli için 'event' kolonu zorunlu. ML adımı atlanıyor.")
        return

    logger.info("")
    logger.info("[STEP] ML tabanlı statik PoF modelleri (XGBoost + CatBoost) eğitiliyor")

    data = df_full.copy()

    y = data["event"].astype(int)
    if y.nunique() < 2:
        logger.warning("[WARN] event tek sınıflı, ML modeli eğitilmeyecek.")
        return

    id_col = "cbs_id"
    drop_cols = {id_col, "event", "duration_days"}

    # Kategorik özellikler (native CatBoost için)
    cat_cols = []
    for cand in ["Ekipman_Tipi", "Gerilim_Seviyesi", "Marka"]:
        if cand in data.columns:
            cat_cols.append(cand)

    # Sayısal özellikler (kategorik olanları hariç tut)
    numeric_cols = [
        c
        for c in data.columns
        if c not in drop_cols
        and c not in cat_cols
        and pd.api.types.is_numeric_dtype(data[c])
    ]

    logger.info(f"[INFO] ML - sayısal özellikler: {numeric_cols}")
    logger.info(f"[INFO] ML - kategorik özellikler: {cat_cols}")

    # XGBoost: sayısal + one-hot kategorik
    X_xgb = data[numeric_cols].copy()
    if cat_cols:
        # Her kategorik kolonu tek tek get_dummies ile işle
        for col in cat_cols:
            dummies = pd.get_dummies(data[[col]], drop_first=True, prefix=col)
            X_xgb = pd.concat([X_xgb, dummies], axis=1)

    # CatBoost: sayısal + kategorik direkt
    feature_cols_cat = numeric_cols + cat_cols
    X_cat = data[feature_cols_cat].copy()

    # CatBoost için kategorik kolonları string'e çevir
    for col in cat_cols:
        if col in X_cat.columns:
            X_cat[col] = X_cat[col].astype(str)

    cat_indices = [feature_cols_cat.index(c) for c in cat_cols]

    X_xgb_train, X_xgb_test, y_train, y_test = train_test_split(
        X_xgb,
        y,
        test_size=0.3,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_cat_train, X_cat_test, _, _ = train_test_split(
        X_cat,
        y,
        test_size=0.3,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # -------------------- XGBoost --------------------
    logger.info("[STEP] XGBoost PoF modelini eğitme")
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    xgb_model.fit(X_xgb_train, y_train)
    y_proba_test = xgb_model.predict_proba(X_xgb_test)[:, 1]
    auc_xgb = roc_auc_score(y_test, y_proba_test)
    ap_xgb = average_precision_score(y_test, y_proba_test)
    logger.info(f"[OK] XGBoost test AUC: {auc_xgb:.3f}, AP: {ap_xgb:.3f}")

    xgb_proba_all = xgb_model.predict_proba(X_xgb)[:, 1]

    # -------------------- CatBoost --------------------
    logger.info("[STEP] CatBoost PoF modelini eğitme")
    cat_model = CatBoostClassifier(
        iterations=200,
        depth=4,
        learning_rate=0.05,
        l2_leaf_reg=3.0,
        loss_function="Logloss",
        eval_metric="AUC",
        verbose=False,
        random_state=RANDOM_STATE,
    )

    cat_model.fit(
        X_cat_train,
        y_train,
        cat_features=cat_indices,
        eval_set=(X_cat_test, y_test),
        verbose=False,
    )
    y_proba_cat_test = cat_model.predict_proba(X_cat_test)[:, 1]
    auc_cat = roc_auc_score(y_test, y_proba_cat_test)
    ap_cat = average_precision_score(y_test, y_proba_cat_test)
    logger.info(f"[OK] CatBoost test AUC: {auc_cat:.3f}, AP: {ap_cat:.3f}")

    cat_proba_all = cat_model.predict_proba(X_cat)[:, 1]

    # -------------------- Çıktıları kaydet --------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    out_df = pd.DataFrame(
        {
            id_col: data[id_col].values,
            "PoF_ML_XGB": xgb_proba_all,
            "PoF_ML_CatBoost": cat_proba_all,
        }
    )
    out_path = os.path.join(OUTPUT_DIR, "statik_ariza_egilim_skoru.csv")
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info(f"[OK] Statik arıza eğilimi skorları kaydedildi: {out_path}")

    # Model dosyaları
    os.makedirs(MODELS_DIR, exist_ok=True)
    xgb_model.save_model(os.path.join(MODELS_DIR, "pof_ml_xgb.json"))
    cat_model.save_model(os.path.join(MODELS_DIR, "pof_ml_catboost.cbm"))
    logger.info(f"[OK] ML modelleri kaydedildi → {MODELS_DIR}")


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
            "Kronik_Flag_90g",
            "Son_Ariza_Gun_Sayisi",
            "Bakim_Sayisi",
            "Ilk_Bakim_Tarihi",
            "Son_Bakim_Tarihi",
            "Son_Bakimdan_Gecen_Gun",
            "Son_Bakim_Tipi",
            "Gecmis_Bakim_Tipleri",
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

        pof_cox = compute_pof_from_cox(cph, df_cox, SURVIVAL_HORIZONS_MONTHS, logger)
        logger.info("")

        # --------------------------------------------------------------
        # 5) Cox çıktılarını kaydet (Türkçe dosya adları)
        # --------------------------------------------------------------
        logger.info("[STEP] Cox PoF çıktılarını kaydetme")
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        for m, series in pof_cox.items():
            out_df = pd.DataFrame(
                {
                    "cbs_id": series.index,
                    f"PoF_Cox_{m}Ay": series.values,
                }
            )
            out_path = os.path.join(OUTPUT_DIR, f"cox_sagkalim_{m}ay_ariza_olasiligi.csv")
            out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
            logger.info(f"[OK] {out_path}")

        # --------------------------------------------------------------
        # 6) RSF modeli + PoF
        # --------------------------------------------------------------
        logger.info("")
        logger.info("[STEP] Random Survival Forest (RSF) adımı")
        rsf = fit_rsf_model(df_cox, logger)
        rsf_pof = compute_pof_from_rsf(rsf, df_cox, SURVIVAL_HORIZONS_MONTHS, logger)

        if rsf_pof:
            logger.info("[STEP] RSF PoF çıktılarını kaydetme")
            for m, series in rsf_pof.items():
                out_df = pd.DataFrame(
                    {
                        "cbs_id": series.index,
                        f"PoF_RSF_{m}Ay": series.values,
                    }
                )
                out_path = os.path.join(OUTPUT_DIR, f"rsf_sagkalim_{m}ay_ariza_olasiligi.csv")
                out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
                logger.info(f"[OK] {out_path}")
        else:
            logger.info("[INFO] RSF çıktısı üretilmedi (RSF modeli yok veya hata).")

        # --------------------------------------------------------------
        # 7) ML tabanlı statik PoF (XGBoost + CatBoost)
        # --------------------------------------------------------------
        logger.info("")
        logger.info("[STEP] ML tabanlı statik PoF (XGBoost + CatBoost)")
        train_ml_pof_models(df_full, logger)

        logger.info("")
        logger.info("[SUCCESS] 03_sagkalim_modelleri başarıyla tamamlandı.")
        logger.info("=" * 80)

    except Exception as e:
        logger.exception(f"[FATAL] 03_sagkalim_modelleri başarısız: {e}")
        raise


if __name__ == "__main__":
    main()
