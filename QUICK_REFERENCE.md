# PoF3 Pipeline - Quick Reference Guide

## 🚀 Running the Pipeline

```bash
# Full pipeline (all steps)
python orchestrator/run_pipeline.py

# Single step (for testing)
python pipeline/01_veri_isleme.py
python pipeline/02_ozellik_muhendisligi.py
python pipeline/03_sagkalim_modelleri.py
python pipeline/04_tekrarlayan_ariza.py
```

## 📊 Key Outputs

### Prediction Files (data/sonuclar/)
| File | Description | Horizons |
|------|-------------|----------|
| `cox_sagkalim_*ay_ariza_olasiligi.csv` | Cox PH model predictions | 3, 6, 12, 24 months |
| `rsf_sagkalim_*ay_ariza_olasiligi.csv` | Random Survival Forest predictions | 3, 6, 12, 24 months |
| `leakage_free_ml_pof.csv` | ML model predictions (XGBoost + CatBoost) | 12 months |
| `chronic_equipment_summary.csv` | Chronic failure analysis | All equipment |
| `chronic_equipment_only.csv` | Only chronic equipment | Filtered |

### Feature Importance Files
| File | Model | Format |
|------|-------|--------|
| `rsf_feature_importance.csv` | RSF | feature, importance |
| `shap_feature_importance.csv` | XGBoost | feature, abs_importance |
| `temporal_cv_scores.csv` | XGBoost | metric, fold, score |

### Visualizations (gorseller/)
- `survival_curves_by_class.png` - Kaplan-Meier curves by equipment type
- `cox_coefficients.png` - Cox model hazard ratios
- `feature_importance_comparison.png` - RSF vs SHAP importance

## 🔍 Troubleshooting

### Common Issues

**Issue**: SHAP not installed
```
[WARN] SHAP computation failed: No module named 'shap'
```
**Solution**:
```bash
pip install shap
```
**Impact**: Pipeline continues, SHAP importance skipped

---

**Issue**: Optional step fails (05, 06)
```
[SKIP] 05_risk_degerlendirme :    0.0 sn
```
**Solution**: This is normal if CoF data missing. Pipeline continues with core features.

---

**Issue**: Matplotlib backend error
```
[WARN] Visualization generation failed
```
**Solution**: Check matplotlib installation or use non-interactive backend
```python
import matplotlib
matplotlib.use('Agg')
```

## 📈 Performance Expectations

| Dataset Size | Pipeline Time | Memory Usage |
|--------------|---------------|--------------|
| ~6K equipment | 2-3 minutes | < 2GB RAM |
| ~50K equipment | 10-15 minutes | 4-6GB RAM |

### Step Breakdown
- **Step 01** (Data Processing): ~45s
- **Step 02** (Feature Engineering): ~3s
- **Step 03** (Survival Models + ML): ~95s ← Most time-intensive
- **Step 04** (Chronic Detection): ~2s

## 🎯 Configuration Quick Changes

### Add/Remove Prediction Horizons
**File**: `config/config.py`
```python
SURVIVAL_HORIZONS = [90, 180, 365, 730]  # days
SURVIVAL_HORIZONS_MONTHS = [3, 6, 12, 24]  # for labels
```

### Adjust Temporal CV Folds
**File**: `utils/ml_advanced.py`
```python
def temporal_cross_validation(..., n_splits=3):  # Change 3 to desired folds
```

### Change SHAP Sample Size
**File**: `pipeline/03_sagkalim_modelleri.py`
```python
shap_df = compute_shap_importance(model_xgb, X_xgb, max_samples=1000)  # Adjust 1000
```

## 📝 Log Files

### Master Log
`loglar/pipeline_master_YYYYMMDD_HHMMSS.log` - Complete pipeline execution trace

### Step Logs
- `loglar/01_veri_isleme_YYYYMMDD_HHMMSS.log`
- `loglar/02_ozellik_muhendisligi_YYYYMMDD_HHMMSS.log`
- `loglar/03_sagkalim_modelleri_YYYYMMDD_HHMMSS.log`
- `loglar/04_tekrarlayan_ariza_YYYYMMDD_HHMMSS.log`

## 🔧 Maintenance

### Disable Advanced Features

**Temporal CV**:
```python
# In pipeline/03_sagkalim_modelleri.py, comment out:
# cv_results = temporal_cross_validation(...)
```

**SHAP**:
```python
# Comment out:
# shap_df = compute_shap_importance(...)
```

**Visualizations**:
```python
# Comment out entire visualization section (Step 8)
```

### Enable Second Reference Window

**File**: `pipeline/03_sagkalim_modelleri.py`

Add after temporal CV:
```python
# 3. Second Reference Window
from utils.ml_advanced import train_second_reference_window
predictions_2, metrics_2 = train_second_reference_window(
    events=events, eq=eq, ref_date_1=ref_date,
    window_days=365, model_fn=lambda: xgb.XGBClassifier(...),
    numeric_cols=..., cat_cols=..., logger=logger
)
```

## 📞 Support

For detailed implementation information, see: [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)

---

**Version**: 3.1
**Last Updated**: 2025-12-10
