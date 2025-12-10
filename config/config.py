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

# IMPORTANT: ANALYSIS_DATE will be overridden by DATA_END_DATE detected in Step 01
# This ensures ML predictions don't extend beyond available data
ANALYSIS_DATE = pd.Timestamp.today().normalize()

MIN_EQUIPMENT_PER_CLASS = 30

# Data validation parameters
MIN_DATA_SPAN_YEARS = 2.0  # Minimum years of fault data required
MIN_TRAIN_YEARS = 2.0      # Minimum years of training data before T_ref


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
# Nihai Çıktılar (Müşteri-facing - Turkish Names)
# ======================================================

OUTPUT_PATHS = {
    # Step 01 Turkish outputs
    "ariza_kayitlari": os.path.join(OUTPUT_DIR, "ariza_kayitlari.csv"),
    "ekipman_listesi": os.path.join(OUTPUT_DIR, "ekipman_listesi.csv"),
    "sagkalim_taban": os.path.join(OUTPUT_DIR, "sagkalim_taban.csv"),
    "saglam_ekipman_listesi": os.path.join(OUTPUT_DIR, "saglam_ekipman_listesi.csv"),

    # Step 03 - Cox Survival Model Outputs (updated to match actual config horizons)
    "cox_3ay": os.path.join(OUTPUT_DIR, "cox_sagkalim_3ay_ariza_olasiligi.csv"),
    "cox_6ay": os.path.join(OUTPUT_DIR, "cox_sagkalim_6ay_ariza_olasiligi.csv"),
    "cox_12ay": os.path.join(OUTPUT_DIR, "cox_sagkalim_12ay_ariza_olasiligi.csv"),

    # Step 03 - RSF Survival Model Outputs (updated to match actual config horizons)
    "rsf_3ay": os.path.join(OUTPUT_DIR, "rsf_sagkalim_3ay_ariza_olasiligi.csv"),
    "rsf_6ay": os.path.join(OUTPUT_DIR, "rsf_sagkalim_6ay_ariza_olasiligi.csv"),
    "rsf_12ay": os.path.join(OUTPUT_DIR, "rsf_sagkalim_12ay_ariza_olasiligi.csv"),

    # Step 03 - Leakage-Free ML Model Outputs
    "leakage_free_ml_pof": os.path.join(OUTPUT_DIR, "leakage_free_ml_pof.csv"),

    # Step 04 - Chronic Equipment Analysis
    "chronic_summary": os.path.join(OUTPUT_DIR, "chronic_equipment_summary.csv"),
    "chronic_only": os.path.join(OUTPUT_DIR, "chronic_equipment_only.csv"),

    # Documentation
    "readme": os.path.join(OUTPUT_DIR, "OKUBBENI.txt"),
}

# Legacy aliases for backward compatibility
RESULT_PATHS = {
    "POF": OUTPUT_DIR,
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
SURVIVAL_HORIZONS = [90, 180, 365, 730]  # 3, 6, 12, 24 months

# Convert to months for labeling (used in some scripts)
SURVIVAL_HORIZONS_MONTHS = [3, 6, 12, 24]

ML_REF_DAYS_BEFORE_ANALYSIS = 365
ML_PREDICTION_WINDOW_DAYS   = 365
RANDOM_STATE                 = 42

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
