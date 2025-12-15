# PoF3 - Technical Documentation
## Developer & Operations Guide

**Version:** 3.1
**Last Updated:** December 2025
**Python:** 3.12.6
**License:** Enterprise

---

## 📋 Table of Contents

1. [System Architecture](#system-architecture)
2. [Installation & Setup](#installation--setup)
3. [Pipeline Stages](#pipeline-stages)
4. [Configuration](#configuration)
5. [Running the Pipeline](#running-the-pipeline)
6. [Output Files](#output-files)
7. [Testing](#testing)
8. [Deployment](#deployment)
9. [Troubleshooting](#troubleshooting)
10. [Development Guidelines](#development-guidelines)
11. [API Reference](#api-reference)
12. [Performance Optimization](#performance-optimization)

---

## 🏗️ System Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      PoF3 Pipeline                          │
│                                                             │
│  Input Data (Excel)                                         │
│  ├── ariza_final.xlsx (Fault events)                        │
│  └── saglam_final.xlsx (Healthy equipment)                  │
│                    ↓                                         │
│  Stage 01: Data Processing (01_veri_isleme.py)              │
│  ├── Turkish date parsing                                   │
│  ├── Outlier detection (MAD on log scale)                   │
│  ├── Auto-detect DATA_END_DATE                              │
│  └── Missingness reporting                                  │
│                    ↓                                         │
│  Stage 02: Feature Engineering (02_ozellik_muhendisligi.py) │
│  ├── IEEE 1366 chronic flags                                │
│  ├── Weighted chronic index                                 │
│  ├── Age, MTBF, seasonality features                        │
│  └── Stress indicators                                      │
│                    ↓                                         │
│  Stage 03: Survival Models (03_sagkalim_modelleri.py)       │
│  ├── Cox Proportional Hazards                               │
│  ├── Random Survival Forest (RSF)                           │
│  ├── XGBoost + CatBoost Ensemble                            │
│  ├── Temporal Cross-Validation (3-fold)                     │
│  ├── SHAP Feature Importance                                │
│  └── Survival curve visualization                           │
│                    ↓                                         │
│  Stage 04: Chronic Detection (04_tekrarlayan_ariza.py)      │
│  ├── IEEE 1366 rolling window (365 days)                    │
│  ├── Poisson probability                                    │
│  └── Equipment categorization                               │
│                    ↓                                         │
│  Stage 04b: CoF Scoring (04b_risk_scoring.py)               │
│  ├── Equipment cost mapping                                 │
│  ├── Voltage multipliers                                    │
│  ├── Customer impact                                        │
│  └── MTTR estimation                                        │
│                    ↓                                         │
│  Stage 05: Risk Assessment (05_risk_degerlendirme.py)       │
│  ├── Merge PoF × CoF                                        │
│  ├── Risk classification (4-tier)                           │
│  └── Equipment type summaries                               │
│                    ↓                                         │
│  Stage 05: Reporting (05_raporlama_ve_gorsellestirme.py)    │
│  ├── Action lists (urgent, CAPEX, maintenance)              │
│  ├── Visualizations (PNG charts)                            │
│  ├── Excel report                                           │
│  └── PowerPoint presentation (optional)                     │
│                    ↓                                         │
│  Output Deliverables                                        │
│  ├── CSV files (risk_skorlari_pof3.csv, etc.)               │
│  ├── PNG visualizations                                     │
│  ├── Excel report                                           │
│  └── PowerPoint deck                                        │
└─────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
PoF3/
├── config/
│   └── config.py                 # Centralized configuration
├── pipeline/
│   ├── 01_veri_isleme.py         # Data processing
│   ├── 02_ozellik_muhendisligi.py # Feature engineering
│   ├── 03_sagkalim_modelleri.py  # Survival models (MAIN)
│   ├── 04_tekrarlayan_ariza.py   # Chronic detection
│   ├── 04b_risk_scoring.py       # CoF calculation
│   ├── 05_risk_degerlendirme.py  # Risk assessment
│   └── 05_raporlama_ve_gorsellestirme.py # Reporting
├── orchestrator/
│   └── run_pipeline.py           # Master runner
├── utils/
│   ├── logger.py                 # Logging utilities
│   ├── ml_advanced.py            # Advanced ML functions
│   ├── survival_plotting.py      # Visualization utilities
│   ├── data_processing.py        # Data utilities
│   ├── data_validation.py        # Validation utilities
│   ├── date_parser.py            # Turkish date parsing
│   └── translations.py           # Turkish localization
├── data/
│   ├── girdiler/                 # Input files
│   │   ├── ariza_final.xlsx
│   │   └── saglam_final.xlsx
│   ├── ara_ciktilar/             # Intermediate outputs
│   └── sonuclar/                 # Final results
├── gorseller/                    # Visualizations
├── loglar/                       # Execution logs
├── modeller/                     # Trained models (gitignored)
├── tests/                        # Unit tests (empty - TODO)
├── docs/                         # Documentation
├── requirements.txt              # Python dependencies
├── BENIOKU_MUSTERI.md            # Customer guide
├── BENIOKU_TEKNIK.md             # Technical guide (this file)
├── IMPLEMENTATION_SUMMARY.md     # Feature summary
├── QUICK_REFERENCE.md            # Quick start guide
└── README.md                     # Project overview
```

---
## 4.1 Model Kalibrasyonu ve Temel Oranlar (Base Rates)

PoF3 modeli, "dengesiz veri" (imbalanced data) yanılgısına düşmemek için DSOs (Dağıtım Şirketleri) için geçerli olan **gerçekçi yıllık arıza oranlarına** göre kalibre edilmiştir. Model çıktıları aşağıdaki endüstri standartları ile uyumlu olacak şekilde denetlenir:

| Varlık Tipi | Beklenen Yıllık Arıza Oranı | Kalibrasyon Notu |
|:---|:---|:---|    
| **Güç Trafosu** | %0.5 – %5.0 | Yaş ve yüklenme durumuna duyarlı |
| **Kesici (Breaker)** | %3.0 – %8.0 | Bakım geçmişi ve mekanik aşınma odaklı |
| **Ayırıcı (Switch)** | %5.0 – %12.0 | Çevresel faktörler (korozyon/nem) ağırlıklı |
| **Hatlar (OH/UG)** | %0.5 – %15.0 | Hava durumu ve dış etkenler (kazı vb.) |
| **Sigortalar** | %15.0 – %30.0 | Operasyonel "sigorta atması" dahil |
| **Direkler** | %0.1 – %3.0 | Sadece fiziksel/yapısal bütünlük kaybı |

**Not:** Model, her varlık tipi için ayrı ayrı eğitilmiş (stratified) ve bu taban oranlara göre doğrulanmıştır (walk-forward validation).
## 🔧 Installation & Setup

### Prerequisites

- **Python:** 3.12.6+ (3.10+ should work)
- **OS:** Windows 10/11, Linux, macOS
- **RAM:** 4GB minimum, 8GB recommended
- **Disk:** 2GB for environment + data

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd PoF3
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/macOS
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Optional Dependencies:**

```bash
# For SHAP feature importance
pip install shap>=0.43.0

# For PowerPoint generation
pip install python-pptx>=0.6.21

# For testing
pip install pytest pytest-cov
```

### Step 4: Verify Installation

```bash
python -c "import pandas, lifelines, xgboost, catboost; print('OK')"
```

### Step 5: Prepare Input Data

Place input files in `data/girdiler/`:
- `ariza_final.xlsx` - Fault events
- `saglam_final.xlsx` - Healthy equipment

**Required Columns:**

**ariza_final.xlsx:**
- `cbs_id` - Equipment ID
- `Ariza_Baslangic_Zamani` - Fault start datetime
- `Ekipman_Tipi` - Equipment type
- `Sure_Saat` - Duration (hours)

**saglam_final.xlsx:**
- `cbs_id` - Equipment ID
- `Ekipman_Tipi` - Equipment type
- Optional: voltage, location, manufacturer

---

## 🚀 Running the Pipeline

### Full Pipeline (Recommended)

```bash
python orchestrator/run_pipeline.py
```

**Output:**
- Master log: `loglar/pipeline_master_YYYYMMDD_HHMMSS.log`
- Execution summary in console
- All CSV/PNG outputs in `data/sonuclar/` and `gorseller/`

**Expected Runtime:** 2-5 minutes (depends on data size)

### Individual Stages (Development)

```bash
# Stage 01: Data Processing
python pipeline/01_veri_isleme.py

# Stage 02: Feature Engineering
python pipeline/02_ozellik_muhendisligi.py

# Stage 03: Survival Models (most time-intensive)
python pipeline/03_sagkalim_modelleri.py

# Stage 04: Chronic Detection
python pipeline/04_tekrarlayan_ariza.py

# Stage 04b: CoF Scoring
python pipeline/04b_risk_scoring.py

# Stage 05: Risk Assessment
python pipeline/05_risk_degerlendirme.py

# Stage 05: Reporting
python pipeline/05_raporlama_ve_gorsellestirme.py
```

### Debug Mode

```bash
# Enable verbose logging
export LOG_LEVEL=DEBUG  # Linux/macOS
set LOG_LEVEL=DEBUG     # Windows

python orchestrator/run_pipeline.py
```

---

## ⚙️ Configuration

### config/config.py

**Key Sections:**

#### 1. Directory Paths

```python
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INPUT_DIR = DATA_DIR / "girdiler"
INTERMEDIATE_DIR = DATA_DIR / "ara_ciktilar"
OUTPUT_DIR = DATA_DIR / "sonuclar"
VISUALIZATIONS_DIR = PROJECT_ROOT / "gorseller"
LOG_DIR = PROJECT_ROOT / "loglar"
MODEL_DIR = PROJECT_ROOT / "modeller"
```

#### 2. Analysis Parameters

```python
ANALYSIS_DATE = None  # Auto-detected from DATA_END_DATE
MIN_DATA_SPAN_YEARS = 2.0  # Minimum historical data
MIN_TRAIN_YEARS = 2.0      # Training data before T_ref
```

#### 3. Survival Horizons

```python
SURVIVAL_HORIZONS = [90, 180, 365, 730]  # days
SURVIVAL_HORIZONS_MONTHS = [3, 6, 12, 24]  # labels
```

**To add new horizon:**
```python
SURVIVAL_HORIZONS = [90, 180, 365, 730, 1095]  # Add 36 months
SURVIVAL_HORIZONS_MONTHS = [3, 6, 12, 24, 36]
```

#### 4. Chronic Detection Settings

```python
CHRONIC_THRESHOLD_EVENTS = 3  # IEEE 1366: ≥4 events = chronic
CHRONIC_WINDOW_DAYS = 90      # Optimized from 365
CHRONIC_MIN_RATE = 1.5        # failures/year
```

#### 5. ML Settings

```python
USE_ML = True
RANDOM_STATE = 42
RSF_N_ESTIMATORS = 100
RSF_MIN_SAMPLES_SPLIT = 10
RSF_MIN_SAMPLES_LEAF = 5
```

#### 6. Risk Matrix

```python
RISK_MATRIX = {
    'DÜŞÜK': {'pof_max': 0.3, 'cof_max': 3.0},
    'ORTA': {'pof_max': 0.6, 'cof_max': 6.0},
    'YÜKSEK': {'pof_max': 0.8, 'cof_max': 8.0},
    'KRİTİK': {'pof_max': 1.0, 'cof_max': 10.0}
}
```

---

## 🔍 Pipeline Stages

### Stage 01: Data Processing

**File:** `pipeline/01_veri_isleme.py`

**Purpose:** Load and clean raw fault + healthy equipment data

**Key Functions:**

1. **`load_and_validate_fault_data()`**
   - Loads `ariza_final.xlsx`
   - Validates required columns
   - Parses Turkish dates (mixed formats)
   - Handles duration outliers (MAD on log scale)

2. **`load_and_validate_healthy_data()`**
   - Loads `saglam_final.xlsx`
   - Normalizes equipment IDs
   - Extracts metadata

3. **`create_survival_base()`**
   - Creates target variable: `event` (1=fault, 0=censored)
   - Calculates `duration_days`
   - Merges fault + healthy equipment

**Outputs:**
- `fault_events_clean.csv`
- `healthy_equipment_clean.csv`
- `equipment_master.csv`
- `survival_base.csv`
- `data_range_metadata.csv`

**Performance:** ~45 seconds

---

### Stage 02: Feature Engineering

**File:** `pipeline/02_ozellik_muhendisligi.py`

**Purpose:** Create predictive features

**Key Features:**

1. **IEEE 1366 Chronic Flags**
   ```python
   kronik_flag = (ariza_sayisi_365gun >= 4) & (poisson_p < 0.05)
   ```

2. **Weighted Chronic Index**
   ```python
   chronic_index = w1*rate_30d + w2*rate_90d + w3*rate_365d
   ```

3. **Age Features**
   - `ekipman_yasi_gun` - Equipment age in days
   - `son_ariza_sonrasi_gun` - Days since last fault

4. **MTBF (Mean Time Between Failures)**
   ```python
   MTBF = total_operational_days / fault_count
   ```

5. **Seasonality Features**
   ```python
   month_sin = sin(2π * month / 12)
   month_cos = cos(2π * month / 12)
   ```

6. **Stress Indicators**
   - Maintenance frequency
   - Fault severity
   - Equipment type risk profile

**Outputs:**
- `ozellikler_pof3.csv` - Feature matrix

**Performance:** ~3 seconds

---

### Stage 03: Survival Models

**File:** `pipeline/03_sagkalim_modelleri.py`

**Purpose:** Train survival models and generate PoF predictions

**Models:**

#### 1. Cox Proportional Hazards

```python
from lifelines import CoxPHFitter

cox = CoxPHFitter()
cox.fit(df, duration_col='duration_days', event_col='event')

# Predict survival function
survival_func = cox.predict_survival_function(X_new)
pof_12m = 1 - survival_func.loc[365]  # 12-month PoF
```

**Outputs:**
- `cox_sagkalim_3ay_ariza_olasiligi.csv`
- `cox_sagkalim_6ay_ariza_olasiligi.csv`
- `cox_sagkalim_12ay_ariza_olasiligi.csv`
- `cox_sagkalim_24ay_ariza_olasiligi.csv`
- `cox_coefficients.png` - Hazard ratios

#### 2. Random Survival Forest (RSF)

```python
from sksurv.ensemble import RandomSurvivalForest

rsf = RandomSurvivalForest(
    n_estimators=100,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
rsf.fit(X, y)

# Feature importance
feature_importance = rsf.feature_importances_
```

**Outputs:**
- `rsf_sagkalim_3ay_ariza_olasiligi.csv`
- `rsf_sagkalim_6ay_ariza_olasiligi.csv`
- `rsf_sagkalim_12ay_ariza_olasiligi.csv`
- `rsf_sagkalim_24ay_ariza_olasiligi.csv`
- `rsf_feature_importance.csv`

#### 3. XGBoost + CatBoost Ensemble

```python
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Binary classification for each horizon
for horizon in [365, 1095, 1825]:  # 12, 36, 60 months
    y_binary = (duration_days <= horizon) & (event == 1)

    # XGBoost
    xgb = XGBClassifier(n_estimators=100, max_depth=6)
    xgb.fit(X, y_binary)

    # CatBoost
    cb = CatBoostClassifier(iterations=100, depth=6, verbose=0)
    cb.fit(X, y_binary)

    # Ensemble average
    pof = (xgb.predict_proba(X)[:, 1] + cb.predict_proba(X)[:, 1]) / 2
```

**Outputs:**
- `leakage_free_ml_pof.csv` - Ensemble predictions

#### 4. Advanced Features

**Temporal Cross-Validation:**

```python
from utils.ml_advanced import temporal_cross_validation

cv_results = temporal_cross_validation(
    X=X_xgb,
    y=y,
    model_fn=lambda: XGBClassifier(...),
    n_splits=3,
    logger=logger
)
# Output: temporal_cv_scores.csv
```

**SHAP Feature Importance:**

```python
from utils.ml_advanced import compute_shap_importance

shap_df = compute_shap_importance(
    model=xgb_model,
    X=X_xgb,
    max_samples=1000,
    logger=logger
)
# Output: shap_feature_importance.csv
```

**Visualizations:**

```python
from utils.survival_plotting import (
    plot_survival_curves_by_class,
    plot_cox_coefficients,
    plot_feature_importance_comparison
)

plot_survival_curves_by_class(km_by_type, ...)
# Output: survival_curves_by_class.png

plot_cox_coefficients(cox_model, ...)
# Output: cox_coefficients.png

plot_feature_importance_comparison(rsf_importance, shap_importance, ...)
# Output: feature_importance_comparison.png
```

**Performance:** ~95 seconds (includes temporal CV + SHAP)

---

### Stage 04: Chronic Detection

**File:** `pipeline/04_tekrarlayan_ariza.py`

**Purpose:** IEEE 1366 chronic equipment classification

**Logic:**

```python
# Rolling 365-day window
df['ariza_sayisi_365gun'] = df.groupby('cbs_id')['event'].rolling(
    window='365D', on='Ariza_Baslangic_Zamani'
).sum()

# Poisson probability (null hypothesis: normal failure rate)
expected_rate = 1.5  # failures/year
poisson_p = poisson.sf(ariza_sayisi - 1, expected_rate)

# Chronic flag
kronik_flag = (ariza_sayisi >= 4) & (poisson_p < 0.05)

# Risk categories
if ariza_sayisi >= 6:
    kategori = 'KRİTİK'
elif ariza_sayisi >= 4:
    kategori = 'YÜKSEK'
elif ariza_sayisi >= 3:
    kategori = 'ORTA'
else:
    kategori = 'DÜŞÜK'
```

**Outputs:**
- `chronic_equipment_summary.csv` - All equipment with flags
- `chronic_equipment_only.csv` - Filtered chronic only

**Performance:** ~2 seconds

---

### Stage 04b: CoF Scoring

**File:** `pipeline/04b_risk_scoring.py`

**Purpose:** Calculate Consequence of Failure (CoF)

**Formula:**

```python
CoF = equipment_cost × voltage_multiplier × customer_impact × mttr_factor

# Voltage multipliers
voltage_multipliers = {
    'Alçak Gerilim': 1.0,
    'Orta Gerilim': 1.5,
    'Yüksek Gerilim': 2.0
}

# Equipment cost (relative scale 1-10)
equipment_costs = {
    'Transformatör': 8.0,
    'Kesici': 7.0,
    'Ayırıcı': 5.0,
    'Sigorta Kutusu': 3.0,
    'Kablo': 6.0
}

# Customer impact (normalized)
customer_impact = min(1 + log10(customer_count + 1) / 4, 2.0)

# MTTR factor (hours)
mttr_factor = 1 + (mttr_hours / 24)  # Capped at reasonable values
```

**Outputs:**
- `cof_pof3.csv` - CoF scores per equipment

**Performance:** ~1 second

---

### Stage 05: Risk Assessment

**File:** `pipeline/05_risk_degerlendirme.py`

**Purpose:** Combine PoF × CoF into risk scores

**Logic:**

```python
# Merge PoF (12-month) with CoF
risk_df = pof_12m_df.merge(cof_df, on='cbs_id')

# Calculate risk score
risk_df['Risk_Score'] = risk_df['PoF_12M'] * risk_df['CoF']

# Risk classification
def classify_risk(row):
    pof = row['PoF_12M']
    cof = row['CoF']
    risk = row['Risk_Score']

    if risk >= 7.0:
        return 'KRİTİK'
    elif risk >= 5.0:
        return 'YÜKSEK'
    elif risk >= 3.0:
        return 'ORTA'
    else:
        return 'DÜŞÜK'

risk_df['Risk_Sinifi'] = risk_df.apply(classify_risk, axis=1)
```

**Outputs:**
- `risk_skorlari_pof3.csv` - Risk scores
- `risk_skoru_ozet_ekipman_tipi.csv` - Summary by type

**Performance:** ~2 seconds

---

### Stage 05: Reporting

**File:** `pipeline/05_raporlama_ve_gorsellestirme.py`

**Purpose:** Generate deliverables

**Phases:**

#### 1. Action Lists

```python
# Critical equipment (0-30 days)
urgent = risk_df[risk_df['Risk_Sinifi'] == 'KRİTİK']

# CAPEX planning (high risk equipment)
capex = risk_df[risk_df['Risk_Sinifi'].isin(['KRİTİK', 'YÜKSEK'])]

# Maintenance list (medium risk)
maintenance = risk_df[risk_df['Risk_Sinifi'] == 'ORTA']
```

#### 2. Visualizations

```python
import matplotlib.pyplot as plt

# Risk distribution
plt.figure(figsize=(10, 6))
risk_df['Risk_Sinifi'].value_counts().plot(kind='bar')
plt.title('Risk Sınıfı Dağılımı')
plt.savefig('gorseller/risk_distribution.png')

# Equipment type distribution
# Fault trends
# PoF by horizon
# etc.
```

#### 3. Excel Report

```python
import pandas as pd

with pd.ExcelWriter('data/sonuclar/PoF_Analysis.xlsx') as writer:
    summary_df.to_excel(writer, sheet_name='Özet', index=False)
    urgent_df.to_excel(writer, sheet_name='Acil Müdahale', index=False)
    capex_df.to_excel(writer, sheet_name='CAPEX Planı', index=False)
    # etc.
```

#### 4. PowerPoint (Optional)

```python
try:
    from pptx import Presentation

    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    title = slide.shapes.title
    title.text = "PoF3 Analiz Sonuçları"
    # Add charts, tables, etc.

    prs.save('data/sonuclar/PoF_Dashboard.pptx')
except ImportError:
    logger.warning("python-pptx not installed, skipping PowerPoint generation")
```

**Outputs:**
- CSV action lists
- PNG visualizations
- Excel report
- PowerPoint presentation (if library installed)

**Performance:** ~10 seconds

---

## 📊 Output Files

### CSV Files (data/sonuclar/)

| File | Rows | Columns | Description |
|------|------|---------|-------------|
| `risk_skorlari_pof3.csv` | All equipment | cbs_id, PoF_12M, CoF, Risk_Score, Risk_Sinifi | Main decision file |
| `risk_equipment_master.csv` | All equipment | Full metadata + risk | Comprehensive view |
| `chronic_equipment_summary.csv` | All equipment | chronic flags, Poisson p-value | IEEE 1366 analysis |
| `ensemble_pof_final.csv` | All equipment | ML ensemble predictions | Advanced PoF |
| `shap_feature_importance.csv` | Top features | feature, abs_importance | SHAP explainability |
| `rsf_feature_importance.csv` | All features | feature, importance | RSF rankings |
| `temporal_cv_scores.csv` | 3 folds | metric, fold, score | Model validation |

### Visualizations (gorseller/)

| File | Type | Description |
|------|------|-------------|
| `chronic_distribution.png` | Bar chart | Chronic equipment count |
| `equipment_distribution.png` | Pie chart | Equipment type breakdown |
| `fault_trends.png` | Line chart | Monthly fault trends |
| `feature_importance.png` | Horizontal bar | Top 15 features (SHAP) |
| `pof_by_horizon.png` | Box plot | PoF distribution by horizon |
| `survival_curves_by_class.png` | Kaplan-Meier | Survival by equipment type |
| `cox_coefficients.png` | Forest plot | Cox hazard ratios |

---

## 🧪 Testing

### Current Status: ⚠️ NO TESTS

**Critical Gap:** `/tests` directory is empty.

### Recommended Test Structure

```
tests/
├── __init__.py
├── conftest.py                    # Pytest fixtures
├── test_data_processing.py        # Stage 01 tests
├── test_feature_engineering.py    # Stage 02 tests
├── test_survival_models.py        # Stage 03 tests
├── test_chronic_detection.py      # Stage 04 tests
├── test_risk_scoring.py           # Stage 04b tests
├── test_utilities.py              # Utils tests
├── test_integration.py            # End-to-end tests
└── fixtures/
    ├── sample_fault_data.csv
    ├── sample_healthy_data.csv
    └── expected_outputs.csv
```

### Example Unit Test

```python
# tests/test_data_processing.py

import pytest
from pipeline.01_veri_isleme import load_and_validate_fault_data

def test_fault_data_loading(sample_fault_file):
    """Test fault data loading with valid input."""
    df = load_and_validate_fault_data(sample_fault_file)

    assert len(df) > 0
    assert 'cbs_id' in df.columns
    assert 'Ariza_Baslangic_Zamani' in df.columns
    assert df['cbs_id'].notna().all()

def test_outlier_detection():
    """Test MAD-based outlier detection."""
    # Implement test
    pass

def test_turkish_date_parsing():
    """Test multi-format date parsing."""
    from utils.date_parser import parse_mixed_dates

    dates = [
        "1.2.2021 16:59",
        "07-01-2024 21:17:45",
        "2021-02-01 14:30:00"
    ]
    parsed = parse_mixed_dates(dates)

    assert len(parsed) == 3
    assert all(pd.notna(parsed))
```

### Running Tests

```bash
# Install pytest
pip install pytest pytest-cov

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=pipeline --cov=utils --cov-report=html

# Run specific test file
pytest tests/test_data_processing.py -v
```

### Integration Test

```bash
# Full pipeline test with sample data
pytest tests/test_integration.py -v -s
```

---

## 🚢 Deployment

### Production Deployment Options

#### Option 1: Scheduled Batch Job

```bash
# Cron job (Linux)
# Run pipeline monthly on 1st day at 2 AM
0 2 1 * * cd /opt/PoF3 && /opt/PoF3/.venv/bin/python orchestrator/run_pipeline.py >> /var/log/pof3.log 2>&1

# Windows Task Scheduler
# Create scheduled task running:
C:\PoF3\.venv\Scripts\python.exe C:\PoF3\orchestrator\run_pipeline.py
```

#### Option 2: Docker Container

**Dockerfile:**

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Run pipeline
CMD ["python", "orchestrator/run_pipeline.py"]
```

**Build & Run:**

```bash
# Build image
docker build -t pof3:latest .

# Run container
docker run -v /path/to/data:/app/data pof3:latest
```

#### Option 3: FastAPI Wrapper (Future)

```python
# api/main.py (not yet implemented)

from fastapi import FastAPI, UploadFile
from pipeline.03_sagkalim_modelleri import predict_pof

app = FastAPI()

@app.post("/predict")
async def predict_equipment_pof(equipment_data: dict):
    """Predict PoF for single equipment."""
    pof_12m = predict_pof(equipment_data, horizon=365)
    return {"cbs_id": equipment_data['cbs_id'], "pof_12m": pof_12m}

@app.post("/batch-predict")
async def batch_predict(file: UploadFile):
    """Batch prediction from CSV."""
    # Implement batch prediction
    pass
```

**Run API:**

```bash
pip install fastapi uvicorn
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Environment Variables

```bash
# .env file (not tracked in git)
DATA_DIR=/opt/pof3/data
LOG_LEVEL=INFO
ENABLE_SHAP=true
ENABLE_PPTX=false
```

**Load in Python:**

```python
from dotenv import load_dotenv
import os

load_dotenv()
log_level = os.getenv('LOG_LEVEL', 'INFO')
```

---

## 🛠️ Troubleshooting

### Common Issues

#### Issue 1: SHAP Not Installed

**Error:**
```
[WARN] SHAP computation failed: No module named 'shap'
```

**Solution:**
```bash
pip install shap>=0.43.0
```

**Impact:** Pipeline continues, SHAP importance skipped.

---

#### Issue 2: Matplotlib Backend Error

**Error:**
```
RuntimeError: main thread is not in main loop
```

**Solution:**
```python
# Add to top of visualization scripts
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
```

---

#### Issue 3: Memory Error (Large Datasets)

**Error:**
```
MemoryError: Unable to allocate array
```

**Solution:**
```python
# config/config.py
RSF_N_ESTIMATORS = 50  # Reduce from 100
MAX_SHAP_SAMPLES = 500  # Reduce from 1000

# Or increase system RAM
```

---

#### Issue 4: Date Parsing Failures

**Error:**
```
ValueError: time data '...' does not match format
```

**Solution:**
Check `utils/date_parser.py` supports all date formats in your data.

```python
# Add new format to parse_mixed_dates()
formats = [
    '%d.%m.%Y %H:%M',
    '%d-%m-%Y %H:%M:%S',
    '%Y-%m-%d %H:%M:%S',
    '%d/%m/%Y %H:%M',
    '%Y%m%d %H%M%S'  # Add new format
]
```

---

#### Issue 5: Missing Input Files

**Error:**
```
FileNotFoundError: data/girdiler/ariza_final.xlsx
```

**Solution:**
Ensure input files exist:
```bash
ls -la data/girdiler/
# Should show: ariza_final.xlsx, saglam_final.xlsx
```

---

#### Issue 6: Temporal CV Fails

**Error:**
```
ValueError: Not enough data for 3-fold CV
```

**Solution:**
Reduce CV folds in `utils/ml_advanced.py`:
```python
def temporal_cross_validation(..., n_splits=2):  # Change from 3
```

---

### Debugging Steps

1. **Check Logs:**
   ```bash
   tail -f loglar/pipeline_master_*.log
   ```

2. **Validate Input Data:**
   ```python
   python -c "from config.config import validate_config; validate_config()"
   ```

3. **Run Individual Stages:**
   ```bash
   python pipeline/01_veri_isleme.py  # Isolate issue
   ```

4. **Enable Debug Mode:**
   ```python
   # In logger.py, change level
   logging.basicConfig(level=logging.DEBUG)
   ```

---

## 💻 Development Guidelines

### Code Style

**Follow PEP 8:**

```bash
# Install linters
pip install black isort flake8 mypy

# Format code
black pipeline/ utils/
isort pipeline/ utils/

# Check style
flake8 pipeline/ utils/

# Type check
mypy pipeline/ utils/
```

### Git Workflow

```bash
# Create feature branch
git checkout -b feature/add-new-model

# Make changes
# ...

# Commit with descriptive messages
git add .
git commit -m "feat: add Weibull survival model to Stage 03"

# Push and create PR
git push origin feature/add-new-model
```

### Adding New Features

**Example: Add Weibull Model to Stage 03**

1. **Update config/config.py:**
   ```python
   USE_WEIBULL = True
   ```

2. **Add function to pipeline/03_sagkalim_modelleri.py:**
   ```python
   from lifelines import WeibullAFTFitter

   def fit_weibull_model(df):
       weibull = WeibullAFTFitter()
       weibull.fit(df, duration_col='duration_days', event_col='event')
       return weibull
   ```

3. **Add output path to config:**
   ```python
   WEIBULL_POF_12M = OUTPUT_DIR / "weibull_sagkalim_12ay_ariza_olasiligi.csv"
   ```

4. **Update tests:**
   ```python
   def test_weibull_model():
       # Test implementation
       pass
   ```

5. **Update documentation:**
   - Add to BENIOKU_TEKNIK.md
   - Update IMPLEMENTATION_SUMMARY.md

---

## 📚 API Reference

### Utility Functions

#### utils/logger.py

```python
def setup_logger(name: str, log_file: str, level: int = logging.INFO) -> logging.Logger:
    """
    Setup logger with file and console handlers.

    Args:
        name: Logger name
        log_file: Path to log file
        level: Logging level (INFO, DEBUG, etc.)

    Returns:
        Configured logger instance
    """
```

#### utils/ml_advanced.py

```python
def temporal_cross_validation(
    X: pd.DataFrame,
    y: np.ndarray,
    model_fn: Callable,
    n_splits: int = 3,
    logger: logging.Logger = None
) -> dict:
    """
    Perform time-series cross-validation.

    Args:
        X: Feature matrix
        y: Target variable
        model_fn: Function returning model instance
        n_splits: Number of CV folds
        logger: Logger instance

    Returns:
        Dict with AUC and AP scores per fold
    """

def compute_shap_importance(
    model,
    X: pd.DataFrame,
    max_samples: int = 1000,
    logger: logging.Logger = None
) -> pd.DataFrame:
    """
    Compute SHAP feature importance.

    Args:
        model: Trained XGBoost/CatBoost model
        X: Feature matrix
        max_samples: Max samples for SHAP (performance)
        logger: Logger instance

    Returns:
        DataFrame with feature importance
    """
```

#### utils/date_parser.py

```python
def parse_mixed_dates(date_series: pd.Series) -> pd.Series:
    """
    Parse Turkish dates in mixed formats.

    Supported formats:
    - 1.2.2021 16:59
    - 07-01-2024 21:17:45
    - 2021-02-01 14:30:00
    - 01/02/2021 09:30

    Args:
        date_series: Pandas series with date strings

    Returns:
        Pandas series with datetime objects
    """
```

---

## ⚡ Performance Optimization

### Profiling

```python
# Add to top of script
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Your code here

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions
```

### Optimization Tips

1. **Reduce SHAP Samples:**
   ```python
   # utils/ml_advanced.py
   MAX_SHAP_SAMPLES = 500  # Instead of 1000
   ```

2. **Use Parallel Processing:**
   ```python
   from joblib import Parallel, delayed

   results = Parallel(n_jobs=-1)(
       delayed(process_equipment)(eq) for eq in equipment_list
   )
   ```

3. **Cache Intermediate Results:**
   ```python
   import joblib

   # Save
   joblib.dump(model, 'models/rsf_12m.pkl')

   # Load
   model = joblib.load('models/rsf_12m.pkl')
   ```

4. **Use Categorical Dtypes:**
   ```python
   df['Ekipman_Tipi'] = df['Ekipman_Tipi'].astype('category')
   ```

### Performance Benchmarks

| Dataset Size | Step 03 Time | Total Pipeline | RAM Usage |
|--------------|--------------|----------------|-----------|
| 1K equipment | ~30s | ~60s | <1GB |
| 6K equipment | ~95s | ~150s | 2GB |
| 50K equipment | ~8min | ~12min | 6GB |

---

## 📞 Support & Contribution

### Reporting Issues

Create GitHub issue with:
1. Error message (full stack trace)
2. Steps to reproduce
3. Input data sample (anonymized)
4. Python version, OS
5. Logs from `loglar/`

### Contributing

1. Fork repository
2. Create feature branch
3. Add tests for new features
4. Update documentation
5. Submit pull request

### Code Review Checklist

- [ ] Code follows PEP 8
- [ ] Tests added and passing
- [ ] Documentation updated
- [ ] Changelog updated
- [ ] No hardcoded paths/credentials
- [ ] Error handling implemented
- [ ] Logging added

---

## 📄 License

**Enterprise License**
© 2025 PoF3 Project. All rights reserved.

---

**Last Updated:** December 2025
**Version:** 3.1
**Maintainer:** Technical Team
