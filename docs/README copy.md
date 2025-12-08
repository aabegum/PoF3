PoF3 – Probability of Failure Analytics Pipeline

Hybrid CoxPH + Random Survival Forest + (Optional) ML Models
MRC Türkiye – Ar-Ge / Şebeke Yönetimi Analitiği

1. Overview

PoF3 is the next-generation asset-health analytics pipeline designed for electricity distribution companies.
It provides probabilistic failure risk estimates for each network component (Direk, Sigorta, Ayırıcı, Trafo, …) using a hybrid survival-analysis architecture:

Cox Proportional Hazards (CoxPH) – interpretable, statistically grounded baseline

Random Survival Forest (RSF) – nonlinear, non-parametric alternative

Optional ML layer – XGBoost & CatBoost for static PoF scoring (event likelihood)

The pipeline operates on structured datasets generated from:

fault_merged_data.xlsx

health_merged_data.xlsx

and produces:

Clean survival tables

Feature-engineered equipment attributes

PoF estimates at 3M / 6M / 12M horizons

Chronic equipment detection

Risk scoring (PoF × CoF)

2. Pipeline Structure
📦 PoF3
 ├── config/
 │    └── config.py
 ├── utils/
 │    ├── date_parser.py
 │    └── logger.py
 ├── pipeline/
 │    ├── 01_data_processing.py
 │    ├── 02_feature_engineering.py
 │    ├── 03_survival_models.py
 │    ├── 04_chronic_detection.py
 │    ├── 05_risk_assessment.py
 │    └── 06_visualizations.py
 ├── data/
 │    ├── inputs/
 │    ├── intermediate/
 │    └── outputs/
 └── README.md   ← (this file)

3. Data Flow
Step 01 – Data Processing

Creates the core technical datasets:

fault_events_clean.csv

equipment_master.csv

survival_base.csv

Key operations:

Strict schema enforcement

Safe date parsing (multiple formats: dd.mm.yyyy hh:mm, yyyy-mm-dd HH:MM:SS)

Duration normalization (auto-detect ms → minutes)

Equipment classification standardization

First-failure extraction

Lowercase cbs_id enforcement (PoF3-wide)

Step 02 – Feature Engineering

Outputs:
features_pof3.csv

Computed features:

Fault_Count

Has_Ariza_Gecmisi (Fault_Count > 0)

Ekipman_Yasi_Gun

MTBF_Gun (Mean Time Between Failures)

Tekrarlayan_Ariza_90g_Flag (Chronic indicator)

Son_Ariza_Gun_Sayisi

Ekipman_Tipi (clean, standardized)

Designed to be expanded with:

Voltage-level features

Marka / kVA rating

Maintenance history (Bakım Sayısı, Son Bakım, etc.)

Step 03 – Survival Modeling

Produces:

pof_cox_3m.csv

pof_cox_6m.csv

pof_cox_12m.csv

pof_rsf_3m.csv

pof_rsf_6m.csv

pof_rsf_12m.csv

Models used:

CoxPH: interpretable hazard model

RSF: nonlinear survival forest

(Optional) XGBoost / CatBoost: static ML PoF score

The script:

Merges survival_base + engineered features

Performs rare-category grouping

Encodes relevant features

Computes PoF at required horizons

Saves predictions to data/outputs/

Step 04 – Chronic Detection

Produces:

chronic_equipment_summary.csv

chronic_equipment_only.csv

Uses:

Inter-failure gap analysis

λ (failure rate) statistics

Multiple window sizes (default 90 days)

Identifies equipment that repeatedly fails within short windows.

Step 05 – Risk Assessment

Produces:

pof3_risk_table.csv

Calculates:

Risk = PoF × CoF


Defaults to CoF = 1 unless a CoF file is provided.

Outputs include:

cbs_id

PoF (12M)

CoF

Risk score

Step 06 – Visualizations

Provides:

Kaplan–Meier curves

CoxPH hazard ratios plot

RSF variable importance

PoF distribution histograms

Risk heatmaps

Useful for dashboards, internal audits, and R&D reporting.

4. Key Design Principles
✔ Strict Data Contract

Column names must match expected schema.
No fuzzy guessing, no ad-hoc renames.

✔ cbs_id everywhere

Lowercase, consistent, enforced across all steps.

✔ Deterministic & Reproducible

All intermediate files saved.
Full logs generated under /logs/.

✔ Hybrid Modeling

Both statistical and machine learning models supported.

✔ Expandable

Feature engineering is modular; new features (voltage, marka, bakım history) can be added easily.

5. Running the Pipeline
Run all steps manually
python pipeline/01_data_processing.py
python pipeline/02_feature_engineering.py
python pipeline/03_survival_models.py
python pipeline/04_chronic_detection.py
python pipeline/05_risk_assessment.py
python pipeline/06_visualizations.py

Outputs appear under:
data/intermediate/
data/outputs/
logs/

6. Inputs Required
File	Description
fault_merged_data.xlsx	OMS / WFM fault logs with timestamps & equipment IDs
health_merged_data.xlsx	Installed, healthy, never-failed assets
(optional) CoF.csv	Consequence of Failure scores
7. Outputs Summary
File	Description
pof_cox_*	Cox model PoF estimates
pof_rsf_*	RSF model PoF estimates
pof_ml_static.csv	(optional) ML-based risk score
chronic_equipment_only.csv	Chronic equipment list
pof3_risk_table.csv	Final risk scores
schema.json	Generated schema dictionary
columns_normalized.txt	Column tracking
8. Dependencies

Python 3.10+

pandas, numpy

lifelines

scikit-survival

xgboost (optional)

catboost (optional)

matplotlib, seaborn (for visualizations)

Install via:

pip install -r requirements.txt

9. Extending PoF3

You can easily integrate:

Maintenance history as predictive features

Transformer loading / SCADA time-series

GIS spatial indicators (rural/mv/lv segmentation)

Weather features (wind, temperature)

Reliability KPIs (SAIFI/SAIDI contributors)

The architecture is intentionally modular.

10. License & Ownership

This pipeline is part of the MRC Türkiye Ar-Ge portfolio.
Distribution or external publication requires approval from MRC management.

