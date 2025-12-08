"""
PoF3 Configuration File
=======================

Centralized configuration for all pipeline steps.
This file provides consistent paths, parameters, and settings across the entire PoF3 analysis pipeline.

Usage:
    from config.config import ANALYSIS_DATE, DATA_PATHS, INTERMEDIATE_PATHS, etc.
    or
    from config.config import CONFIG  # For dictionary-style access
"""
import os
import pandas as pd

# ======================================================
# Dizin Yapısı
# ======================================================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR = os.path.join(BASE_DIR, "data")
INPUT_DIR = os.path.join(DATA_DIR, "girdiler")
INTERMEDIATE_DIR = os.path.join(DATA_DIR, "ara_ciktilar")
OUTPUT_DIR = os.path.join(DATA_DIR, "sonuclar")

VISUAL_DIR = os.path.join(BASE_DIR, "gorseller")

# LOG directory (01_veri_isleme expects LOGLAR_DIR)
LOG_DIR = os.path.join(BASE_DIR, "loglar")

# Create directories
for d in [INPUT_DIR, INTERMEDIATE_DIR, OUTPUT_DIR, VISUAL_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)


# ======================================================
# Analiz Parametreleri
# ======================================================

ANALYSIS_DATE = pd.Timestamp.today().normalize()
MIN_EQUIPMENT_PER_CLASS = 30


# ======================================================
# Veri Girdi Yolları
# ======================================================

DATA_PATHS = {
    "fault_data": os.path.join(INPUT_DIR, "ariza_final.xlsx"),
    "healthy_data": os.path.join(INPUT_DIR, "saglam_final.xlsx"),
}


# ======================================================
# Ara Çıktılar (Step 01 → Step 02 → Step 03)
# ======================================================

INTERMEDIATE_PATHS = {
    "fault_events_clean": os.path.join(INTERMEDIATE_DIR, "fault_events_clean.csv"),
    "healthy_equipment_clean": os.path.join(INTERMEDIATE_DIR, "healthy_equipment_clean.csv"),
    "equipment_master": os.path.join(INTERMEDIATE_DIR, "equipment_master.csv"),
    "survival_base": os.path.join(INTERMEDIATE_DIR, "survival_base.csv"),

    # Step 02 Türkçe çıktı
    "ozellikler_pof3": os.path.join(INTERMEDIATE_DIR, "ozellikler_pof3.csv"),
}

FEATURE_OUTPUT_PATH = INTERMEDIATE_PATHS["ozellikler_pof3"]


# ======================================================
# Nihai Çıktılar (Müşteri-facing)
# ======================================================

OUTPUT_PATHS = {
    # Step 01 Turkish outputs
    "ariza_kayitlari": os.path.join(OUTPUT_DIR, "ariza_kayitlari.csv"),
    "ekipman_listesi": os.path.join(OUTPUT_DIR, "ekipman_listesi.csv"),
    "sagkalim_taban": os.path.join(OUTPUT_DIR, "sagkalim_taban.csv"),
    "saglam_ekipman_listesi": os.path.join(OUTPUT_DIR, "saglam_ekipman_listesi.csv"),

    # Step 03
    "pof_cox": os.path.join(OUTPUT_DIR, "03_pof_cox_survival.csv"),
    "pof_rsf": os.path.join(OUTPUT_DIR, "03_pof_rsf.csv"),

    # Step 04
    "chronic_failures": os.path.join(OUTPUT_DIR, "04_chronic_failures.csv"),

    # Step 05
    "risk_scores": os.path.join(OUTPUT_DIR, "05_risk_scores.csv"),
    "risk_matrix": os.path.join(OUTPUT_DIR, "05_risk_matrix.csv"),
}


# ======================================================
# Görselleştirme yolları
# ======================================================

VISUAL_PATHS = {
    "survival_curves": os.path.join(VISUAL_DIR, "survival_curves_by_class.png"),
    "chronic_heatmap": os.path.join(VISUAL_DIR, "chronic_failure_heatmap.png"),
    "risk_matrix": os.path.join(VISUAL_DIR, "risk_matrix.png"),
    "feature_importance": os.path.join(VISUAL_DIR, "feature_importance.png"),
}


# (rest of config unchanged…)

# ============================
# SURVIVAL ANALYSIS SETTINGS
# ============================

# Prediction horizons in days for survival analysis
# These represent the time windows for which we predict Probability of Failure (PoF)
SURVIVAL_HORIZONS = [90, 180, 365]  # 3 months, 6 months, 12 months

# Convert to months for labeling (used in some scripts)
SURVIVAL_HORIZONS_MONTHS = [3, 6, 12]


# ============================
# CHRONIC FAILURE DETECTION
# (IEEE 1366 Standard)
# ============================

# Minimum number of failures to flag as chronic
CHRONIC_THRESHOLD_EVENTS = 3

# Time window for chronic failure analysis (days)
CHRONIC_WINDOW_DAYS = 90  # Changed from 365 to 90 based on your MTBF analysis

# Minimum failure rate threshold (failures per year)
CHRONIC_MIN_RATE = 1.5


# ============================
# MACHINE LEARNING SETTINGS
# (For hybrid PoF models - XGBoost/RSF)
# ============================

USE_ML = True
ML_MODELS = ["XGBoost", "CatBoost", "RandomSurvivalForest"]
RANDOM_STATE = 42
TEST_SIZE = 0.3

# Feature selection
FEATURE_IMPORTANCE_MIN = 0.01  # Minimum importance threshold

# RSF parameters
RSF_N_ESTIMATORS = 100
RSF_MIN_SAMPLES_SPLIT = 10
RSF_MIN_SAMPLES_LEAF = 5


# ============================
# RISK ASSESSMENT SETTINGS
# ============================

# PoF Categories (Probability of Failure)
POF_THRESHOLDS = {
    "Low": 0.25,
    "Medium": 0.50,
    "High": 0.75,
    "Very High": 1.0,
}

# CoF Categories (Consequence of Failure)
# To be defined based on customer impact, outage duration, etc.
COF_THRESHOLDS = {
    "Low": 1,
    "Medium": 2,
    "High": 3,
    "Critical": 4,
}

# Risk Matrix (PoF × CoF)
RISK_MATRIX = {
    "Low": {"Low": "Low", "Medium": "Low", "High": "Medium", "Critical": "Medium"},
    "Medium": {"Low": "Low", "Medium": "Medium", "High": "High", "Critical": "High"},
    "High": {"Low": "Medium", "Medium": "High", "High": "High", "Critical": "Critical"},
    "Very High": {"Low": "Medium", "Medium": "High", "High": "Critical", "Critical": "Critical"},
}


# ============================
# LOGGING SETTINGS
# ============================

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# ============================
# DICTIONARY-STYLE CONFIG
# (For backward compatibility with scripts using CONFIG["key"])
# ============================

CONFIG = {
    # Paths
    "paths": {
        "base": BASE_DIR,
        "data": DATA_DIR,
        "inputs": INPUT_DIR,
        "intermediate": INTERMEDIATE_DIR,
        "outputs": OUTPUT_DIR,
        "visuals": VISUAL_DIR,
        "logs": LOG_DIR,
    },

    # Analysis parameters
    "analysis_date": ANALYSIS_DATE,
    "min_class_size": MIN_EQUIPMENT_PER_CLASS,

    # Survival analysis
    "survival_horizons": SURVIVAL_HORIZONS_MONTHS,

    # Chronic detection
    "chronic": {
        "threshold_events": CHRONIC_THRESHOLD_EVENTS,
        "window_days": CHRONIC_WINDOW_DAYS,
        "min_rate": CHRONIC_MIN_RATE,
    },

    # ML settings
    "ml": {
        "use_ml": USE_ML,
        "models": ML_MODELS,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
    },

    # Risk assessment
    "risk": {
        "pof_thresholds": POF_THRESHOLDS,
        "cof_thresholds": COF_THRESHOLDS,
        "risk_matrix": RISK_MATRIX,
    },
}


# ============================
# VALIDATION
# ============================

def validate_config():
    """Validate that all required input files and directories exist."""
    errors = []

    # Check input files
    for name, path in DATA_PATHS.items():
        if not os.path.exists(path):
            errors.append(f"Missing input file: {name} at {path}")

    # Check directories
    required_dirs = [INPUT_DIR, INTERMEDIATE_DIR, OUTPUT_DIR, VISUAL_DIR, LOG_DIR]
    for directory in required_dirs:
        if not os.path.exists(directory):
            errors.append(f"Missing directory: {directory}")

    if errors:
        print("Configuration Warnings:")
        for error in errors:
            print(f"  - {error}")
        return False

    return True


def print_config_summary():
    """Print a summary of the current configuration."""
    print("=" * 80)
    print("PoF3 Configuration Summary")
    print("=" * 80)
    print(f"Base Directory:     {BASE_DIR}")
    print(f"Analysis Date:      {ANALYSIS_DATE}")
    print(f"Min Class Size:     {MIN_EQUIPMENT_PER_CLASS}")
    print(f"Survival Horizons:  {SURVIVAL_HORIZONS} days")
    print(f"Chronic Window:     {CHRONIC_WINDOW_DAYS} days")
    print(f"Use ML Models:      {USE_ML}")
    print("=" * 80)


if __name__ == "__main__":
    # When run directly, validate and print config
    print_config_summary()
    print()
    if validate_config():
        print("[OK] Configuration is valid!")
    else:
        print("[WARN] Configuration has warnings (see above)")
