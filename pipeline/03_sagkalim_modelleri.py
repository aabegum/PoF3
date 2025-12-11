"""
03_hibrit_model.py (PoF3 v7 - Validated Edition)

UPDATES:
1. Leakage Guard: Added 'Source Leakage' columns (Lat/Lon/Musteri).
2. Validation: Added Calibration Curves and Temporal CV.
3. Analysis: Added Correlation Matrix export.
4. Stability: Preserves RSF/Weibull fixes.
"""

import os
import sys
import logging
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Statistical & ML Libraries
from lifelines import CoxPHFitter, WeibullAFTFitter
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.feature_selection import RFE
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Check for ML libs
try:
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.util import Surv
    HAS_RSF = True
except ImportError:
    HAS_RSF = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except ImportError:
    HAS_CAT = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config.config import (
    ANALYSIS_DATE, INTERMEDIATE_PATHS, FEATURE_OUTPUT_PATH, 
    OUTPUT_DIR, SURVIVAL_HORIZONS_MONTHS, LOG_DIR, RANDOM_STATE
)

STEP_NAME = "03_hibrit_model"

# ==============================================================================
# LOGGING
# ==============================================================================
def setup_logger():
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"{STEP_NAME}_{ts}.log")
    
    logger = logging.getLogger(STEP_NAME)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    
    print(f"\n📝 [DEBUG] LOG FILE: {os.path.abspath(log_path)}\n")
    return logger

# ==============================================================================
# ANALYSIS & VALIDATION TOOLS (NEW)
# ==============================================================================
def save_correlation_matrix(df, logger):
    logger.info("[ANALYSIS] Generating Correlation Matrix...")
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    
    out_path = os.path.join(OUTPUT_DIR, "feature_correlations.csv")
    corr.to_csv(out_path)
    logger.info(f"[ANALYSIS] Correlation matrix saved -> {out_path}")

def plot_calibration_curve_func(y_true, y_prob, model_name, logger):
    """
    Plots Predicted Probability vs Observed Frequency.
    Perfectly calibrated model lies on the 45-degree line.
    """
    logger.info(f"[VALIDATION] Generating Calibration Plot for {model_name}...")
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    
    brier = brier_score_loss(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker='o', label=f'{model_name} (Brier={brier:.3f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title(f'Calibration Curve: {model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_path = os.path.join(os.path.dirname(OUTPUT_DIR), "gorseller", f"calibration_{model_name.lower().replace(' ', '_')}.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    logger.info(f"[VALIDATION] Plot saved -> {out_path}")

def run_temporal_cv(model, X, y, logger):
    """
    Simulates time-series split cross-validation.
    """
    logger.info("[VALIDATION] Running Temporal Cross-Validation (3 Splits)...")
    tscv = TimeSeriesSplit(n_splits=3)
    
    scores = []
    fold = 1
    
    # Reset index to ensure integer indexing works for TSCV
    X_reset = X.reset_index(drop=True)
    y_reset = y.reset_index(drop=True)
    
    for train_index, test_index in tscv.split(X_reset):
        X_train, X_test = X_reset.iloc[train_index], X_reset.iloc[test_index]
        y_train, y_test = y_reset.iloc[train_index], y_reset.iloc[test_index]
        
        # Clone and fit
        if hasattr(model, 'get_params'): # Sklearn/XGBoost
            clone = model.__class__(**model.get_params())
            clone.fit(X_train, y_train)
            preds = clone.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, preds)
            logger.info(f"  > Fold {fold}: AUC = {auc:.4f}")
            scores.append(auc)
        
        fold += 1
        
    logger.info(f"[VALIDATION] Temporal CV Mean AUC: {np.mean(scores):.4f}")

# ==============================================================================
# DATA PREP
# ==============================================================================
def sanitize_data(df, logger):
    logger.info("[SANITIZE] Checking for Infinity and NaNs...")
    df = df.replace([np.inf, -np.inf], np.nan)
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0)
    
    for col in num_cols:
        if col not in ['cbs_id', 'event']:
            if df[col].max() > 1e9 or df[col].min() < -1e9:
                df[col] = df[col].clip(-1e9, 1e9)
    return df

def group_rare_categories(df, cat_cols, threshold=0.01):
    for col in cat_cols:
        counts = df[col].value_counts(normalize=True)
        rare_vals = counts[counts < threshold].index
        if len(rare_vals) > 0:
            df.loc[df[col].isin(rare_vals), col] = 'Other'
    return df

def run_feature_consensus(df, target_col='event', duration_col='duration_days', logger=None):
    if logger: logger.info("="*60)
    if logger: logger.info("[SELECTOR] Running Weighted Feature Consensus")
    
    cols_to_drop = ['cbs_id', target_col, duration_col]
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    y = df[target_col]
    X = X.fillna(0)
    X = X.loc[:, X.std() > 0]
    
    # 1. VIF
    if logger: logger.info("[SELECTOR] Step 1: VIF Analysis...")
    try:
        vif_data = pd.DataFrame()
        vif_data["feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
        passed_vif = vif_data[vif_data["VIF"] < 10]["feature"].tolist()
    except:
        passed_vif = X.columns.tolist()
        
    # 2. LASSO
    if logger: logger.info("[SELECTOR] Step 2: LASSO...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    try:
        lasso = LassoCV(cv=5, random_state=RANDOM_STATE).fit(X_scaled, y)
        coef = pd.Series(lasso.coef_, index=X.columns)
        selected_lasso = coef[coef.abs() > 0].index.tolist()
    except:
        selected_lasso = []
    
    # 3. RFE
    if logger: logger.info("[SELECTOR] Step 3: RFE...")
    try:
        model = LogisticRegression(solver='liblinear', random_state=RANDOM_STATE)
        rfe = RFE(model, n_features_to_select=15)
        rfe.fit(X_scaled, y)
        selected_rfe = X.columns[rfe.support_].tolist()
    except:
        selected_rfe = []
    
    consensus_scores = {}
    for col in X.columns:
        score = 0
        if col in selected_lasso: score += 2
        if col in selected_rfe: score += 2
        if col in passed_vif: score += 1
        consensus_scores[col] = score
        
    final_features = [col for col, score in consensus_scores.items() if score >= 3]
    
    # Force keep physics
    force_keep = ['Ekipman_Yasi_Gun', 'Gerilim_Seviyesi_kV', 'Bakim_Sayisi', 
                  'Son_Ariza_Gun_Sayisi', 'Ariza_Sayisi', 'Weighted_Chronic_Index']
    for fk in force_keep:
        if fk in X.columns and fk not in final_features:
            final_features.append(fk)
            
    if logger: logger.info(f"[SELECTOR] Kept {len(final_features)} features.")
    return ['cbs_id', target_col, duration_col] + final_features

def load_and_prep_data(logger):
    logger.info("[LOAD] Loading datasets...")
    surv_df = pd.read_csv(INTERMEDIATE_PATHS["survival_base"])
    feat_df = pd.read_csv(FEATURE_OUTPUT_PATH)
    
    full_df = surv_df[["cbs_id", "duration_days", "event"]].merge(feat_df, on="cbs_id", how="inner")
    full_df = full_df[full_df["duration_days"] > 0].copy()
    
    drop_cols = ["Kurulum_Tarihi", "Ilk_Ariza_Tarihi", "Son_Ariza_Tarihi", "Son_Bakim_Tarihi", "Ilk_Bakim_Tarihi"]
    full_df.drop(columns=[c for c in drop_cols if c in full_df.columns], inplace=True)
    
    cat_cols = [c for c in full_df.select_dtypes(include=['object']).columns if c != 'cbs_id']
    full_df = group_rare_categories(full_df, cat_cols, threshold=0.02)
    full_df = pd.get_dummies(full_df, columns=cat_cols, drop_first=True)
    full_df = sanitize_data(full_df, logger)
    
    final_cols = run_feature_consensus(full_df, logger=logger)
    full_df = full_df[final_cols]
    
    # Save Correlations
    save_correlation_matrix(full_df, logger)
    
    return full_df

# ==============================================================================
# MODELS
# ==============================================================================
def train_cox(df, logger):
    logger.info("[COX] Training...")
    try:
        cph = CoxPHFitter(penalizer=0.1)
        train_df = df.drop(columns=['cbs_id']).loc[:, df.drop(columns=['cbs_id']).std() > 0.001]
        cph.fit(train_df, duration_col='duration_days', event_col='event')
        logger.info(f"[COX] Concordance: {cph.concordance_index_:.4f}")
        return cph
    except Exception as e:
        logger.error(f"[COX] Failed: {e}")
        return None

def train_weibull(df, logger):
    logger.info("[WEIBULL] Training...")
    try:
        aft = WeibullAFTFitter(penalizer=0.1)
        train_df = df.drop(columns=['cbs_id']).loc[:, df.drop(columns=['cbs_id']).std() > 0.001]
        train_df['duration_days'] = train_df['duration_days'].replace(0, 0.1)
        aft.fit(train_df, duration_col='duration_days', event_col='event')
        try:
            rho = aft.params_['rho_'] if 'rho_' in aft.params_ else getattr(aft, 'rho_', None)
            logger.info(f"[WEIBULL] Concordance: {aft.concordance_index_:.4f}")
            if rho is not None:
                rho_val = rho.iloc[0] if isinstance(rho, pd.Series) else rho
                logger.info(f"[WEIBULL] Rho (Shape): {float(rho_val):.4f}")
        except:
            logger.info("[WEIBULL] Trained (diagnostics unavailable)")
        return aft
    except Exception as e:
        logger.error(f"[WEIBULL] Failed: {e}")
        return None

def train_rsf(df, logger):
    if not HAS_RSF: return None
    logger.info("[RSF] Training Survival Forest...")
    X = df.drop(columns=['cbs_id', 'duration_days', 'event']).astype('float32')
    y = Surv.from_arrays(event=df['event'].astype(bool), time=df['duration_days'])
    
    if not np.all(np.isfinite(X)): X = X.replace([np.inf, -np.inf], 0)
    
    rsf = RandomSurvivalForest(n_estimators=100, min_samples_leaf=20, n_jobs=-1, random_state=42)
    rsf.fit(X, y)
    logger.info(f"[RSF] Score: {rsf.score(X, y):.4f}")
    return rsf

def train_ml_models(df, logger):
    """
    Trains XGBoost using Stratified Split + Calibration Check + Temporal CV.
    """
    if not HAS_XGB: return None
    logger.info("="*60)
    logger.info("[ML] Training XGBoost with Leakage Guard")
    logger.info("="*60)
    
    # 1. EXPANDED LEAKAGE GUARD
    # We drop ALL columns identified by the Leakage Hunter
    leakage_cols = [
        'Son_Ariza_Gun_Sayisi', 'Faults_Last_365d', 'Weighted_Chronic_Index',
        'Ariza_Gecmisi', 'Ariza_Sayisi', 
        'Musteri_Sayisi', 'Log_Musteri_Sayisi', # Source Leakage
        'Latitude', 'Longitude' # Source Leakage
    ]
    
    cols_to_drop = ['cbs_id', 'duration_days', 'event'] + [c for c in leakage_cols if c in df.columns]
    
    X = df.drop(columns=cols_to_drop)
    y = df['event']
    
    logger.info(f"[ML] Training features ({X.shape[1]}): {list(X.columns[:10])}...")
    
    # 2. TRAIN/TEST SPLIT
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y)
    
    models = {}
    
    # XGBoost
    xgb_clf = xgb.XGBClassifier(n_estimators=100, max_depth=4, eval_metric='logloss', use_label_encoder=False)
    xgb_clf.fit(X_train, y_train)
    
    preds_test = xgb_clf.predict_proba(X_test)[:,1]
    auc_xgb = roc_auc_score(y_test, preds_test)
    logger.info(f"[ML] XGBoost Test AUC: {auc_xgb:.4f}")
    models['xgb'] = xgb_clf
    
    # 3. VALIDATION
    plot_calibration_curve_func(y_test, preds_test, "XGBoost", logger)
    run_temporal_cv(xgb_clf, X, y, logger)
    
    # CatBoost
    if HAS_CAT:
        from catboost import CatBoostClassifier
        cat_clf = CatBoostClassifier(iterations=100, depth=4, verbose=0, allow_writing_files=False)
        cat_clf.fit(X_train, y_train)
        models['cat'] = cat_clf
    
    return models

# ==============================================================================
# PREDICTION HELPERS
# ==============================================================================
def get_model_features(model, df, model_type='generic'):
    if model_type == 'ml':
        leakage_cols = [
            'Son_Ariza_Gun_Sayisi', 'Faults_Last_365d', 'Weighted_Chronic_Index',
            'Ariza_Gecmisi', 'Ariza_Sayisi', 
            'Musteri_Sayisi', 'Log_Musteri_Sayisi',
            'Latitude', 'Longitude'
        ]
        cols_to_drop = ['cbs_id', 'duration_days', 'event'] + [c for c in leakage_cols if c in df.columns]
        return df.drop(columns=cols_to_drop, errors='ignore')
        
    elif hasattr(model, 'params_'): # Lifelines
        if isinstance(model.params_.index, pd.MultiIndex):
            cols = [x[1] for x in model.params_.index if x[1] != 'Intercept']
        else:
            cols = [x for x in model.params_.index if x != 'Intercept']
        cols = list(set(cols))
        valid = [c for c in cols if c in df.columns]
        return df[valid]
    else:
        # RSF
        cols_to_drop = ['cbs_id', 'duration_days', 'event']
        return df.drop(columns=cols_to_drop, errors='ignore')

def get_hazard_from_model(model, df, model_type, horizons):
    if model is None: return {}
    
    X = get_model_features(model, df, model_type)
    hazards = {}
    
    try:
        if model_type in ['cox', 'weibull']:
            surv_funcs = model.predict_survival_function(X)
            for m in horizons:
                days = m * 30
                idx = min(surv_funcs.index, key=lambda x: abs(x - days))
                S_t = surv_funcs.loc[idx].values
                S_t = np.clip(S_t, 1e-6, 0.999999)
                hazards[m] = pd.Series(-np.log(S_t), index=df['cbs_id'])
                
        elif model_type == 'rsf':
            surv_funcs_raw = model.predict_survival_function(X.astype('float32'))
            for m in horizons:
                days = m * 30
                S_t = np.array([fn(days) for fn in surv_funcs_raw])
                S_t = np.clip(S_t, 1e-6, 0.999999)
                hazards[m] = pd.Series(-np.log(S_t), index=df['cbs_id'])
                
        elif model_type == 'ml':
            if isinstance(model, dict): pass
            else:
                P = model.predict_proba(X)[:, 1]
                for m in horizons:
                    scale = min(1.0, (m*30)/365.0)
                    S_t = 1 - (P * scale)
                    S_t = np.clip(S_t, 1e-6, 0.999999)
                    hazards[m] = pd.Series(-np.log(S_t), index=df['cbs_id'])
                
    except Exception as e:
        print(f"[WARN] Prediction failed for {model_type}: {e}")
        
    return hazards

def run_hazard_ensemble(df, cox_mod, wei_mod, rsf_mod, ml_mods, horizons, logger):
    logger.info("="*60)
    logger.info("[ENSEMBLE] Hazard Stacking")
    logger.info("="*60)
    
    H_cox = get_hazard_from_model(cox_mod, df, 'cox', horizons)
    H_wei = get_hazard_from_model(wei_mod, df, 'weibull', horizons)
    H_rsf = get_hazard_from_model(rsf_mod, df, 'rsf', horizons)
    
    if ml_mods:
        H_ml = get_hazard_from_model(ml_mods['xgb'], df, 'ml', horizons) 
        if 'cat' in ml_mods:
            H_cat = get_hazard_from_model(ml_mods['cat'], df, 'ml', horizons)
            for m in horizons:
                H_ml[m] = (H_ml[m] + H_cat[m]) / 2
    else:
        H_ml = {}
    
    final_results = pd.DataFrame({'cbs_id': df['cbs_id']})
    w_cox, w_wei, w_rsf, w_ml = 0.10, 0.15, 0.25, 0.50 
    
    for m in horizons:
        h_sum = np.zeros(len(df))
        
        if m in H_cox: h_sum += H_cox[m].values * w_cox
        if m in H_wei: h_sum += H_wei[m].values * w_wei
        if m in H_rsf: h_sum += H_rsf[m].values * w_rsf
        if m in H_ml:  h_sum += H_ml[m].values * w_ml
            
        final_results[f'PoF_{m}Ay'] = 1 - np.exp(-h_sum)

    if 'PoF_12Ay' in final_results.columns:
        k = 2.0
        final_results['Health_Score'] = 100 * np.exp(-k * final_results['PoF_12Ay'])
        final_results['Health_Score'] = final_results['Health_Score'].clip(0, 100).round(1)
        
        conditions = [
            (final_results['Health_Score'] >= 80),
            (final_results['Health_Score'] >= 60),
            (final_results['Health_Score'] >= 40),
            (final_results['Health_Score'] < 40)
        ]
        choices = ['Excellent', 'Good', 'Moderate', 'Critical']
        final_results['Health_Class'] = np.select(conditions, choices, default='Unknown')
        
        crit = (final_results['Health_Class'] == 'Critical').sum()
        logger.info(f"[ENSEMBLE] Critical Assets (Score < 40): {crit}")
        logger.info(f"[ENSEMBLE] Mean Health Score: {final_results['Health_Score'].mean():.1f}")

    out_path = os.path.join(OUTPUT_DIR, "ensemble_pof_final.csv")
    final_results.to_csv(out_path, index=False)
    logger.info(f"[SUCCESS] Saved Ensemble Output → {out_path}")

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    logger = setup_logger()
    try:
        df = load_and_prep_data(logger)
        
        cox_mod = train_cox(df, logger)
        wei_mod = train_weibull(df, logger)
        rsf_mod = train_rsf(df, logger)
        ml_mods = train_ml_models(df, logger)
        
        run_hazard_ensemble(df, cox_mod, wei_mod, rsf_mod, ml_mods, 
                            SURVIVAL_HORIZONS_MONTHS, logger)
        
        logger.info("[DONE] Pipeline Finished Successfully.")
        
    except Exception as e:
        logger.exception(f"[FATAL] {e}")
    finally:
        logging.shutdown()

if __name__ == "__main__":
    main()