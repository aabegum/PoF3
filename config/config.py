import os
from datetime import datetime

# ============================
# GLOBAL PATH SETTINGS
# ============================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_PATH = os.path.join(BASE_DIR, "data")
INPUT_PATH = os.path.join(DATA_PATH, "inputs")
INTERMEDIATE_PATH = os.path.join(DATA_PATH, "intermediate")
OUTPUT_PATH = os.path.join(DATA_PATH, "outputs")

VISUAL_PATH = os.path.join(BASE_DIR, "visuals")
LOG_PATH = os.path.join(BASE_DIR, "logs")

# Ensure dirs exist
for p in [INPUT_PATH, INTERMEDIATE_PATH, OUTPUT_PATH, VISUAL_PATH, LOG_PATH]:
    os.makedirs(p, exist_ok=True)


# ============================
# ANALYSIS PARAMETERS
# ============================

# Dynamic or fixed analysis date
ANALYSIS_DATE = datetime.today()

# Prediction horizons (for survival analysis or ML)
SURVIVAL_HORIZONS = [90, 180, 365]   # 3M, 6M, 12M

# Minimum number of equipment needed to keep class
MIN_EQUIPMENT_PER_CLASS = 30


# ============================
# OUTPUT NAMING (TURKISH)
# ============================

OUTPUT_FILES = {
    "clean_data":          "01_temiz_veri.csv",
    "features":            "02_ozellikler.csv",
    "pof_survival":        "03_pof_hayatta_kalma.csv",
    "pof_rsf":             "03_pof_rsf.csv",
    "chronic":             "04_kronik_arizalar.csv",
    "risk":                "05_risk_skorlari.csv",
    "survival_plot":       "hayatta_kalma_egrileri.png",
    "chronic_plot":        "kronik_sicaklik_haritasi.png",
    "risk_matrix":         "risk_matrisi.png",
}


# ============================
# IEEE CHRONIC FAILURE SETTINGS
# ============================

CHRONIC_THRESHOLD_EVENTS = 3          # IEEE 1366
CHRONIC_WINDOW_DAYS = 365             # 12 months
CHRONIC_MIN_RATE = 1.5                # failures/year flag


# ============================
# ML SETTINGS (Hybrid Mode)
# ============================

USE_ML = True
ML_MODELS = ["XGBoost", "CatBoost"]
RANDOM_STATE = 42
TEST_SIZE = 0.3

# Feature selection thresholds
FEATURE_IMPORTANCE_MIN = 0.01


# ============================
# TURKISH LABELS FOR OUTPUTS
# ============================

TURKISH_LABELS = {
    "cbs_id": "CBS_ID",
    "equipment_class": "Ekipman_Sinifi",
    "install_date": "Kurulum_Tarihi",
    "equipment_age_days": "Ekipman_Yasi_Gun",
    "duration_minutes": "Kesinti_Suresi_Dakika",
    "event": "Ariza_Olayi",
    "pof": "Ariza_Olasiligi",
    "risk": "Risk_Skoru"
}
