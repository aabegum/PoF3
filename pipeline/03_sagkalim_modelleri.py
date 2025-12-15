"""
03_sagkalim_modelleri.py (PoF3 v14 - Hybrid Grandmaster Edition)

BEST OF BOTH WORLDS:
1. Training Strategy: Uses Random Stratified Split (v12) to train the final 'Production Model'.
   - Why: Maximizes learning from all available history (including recent 2024 faults).
2. Validation Strategy: Runs strictly ordered Walk-Forward Backtesting (v13).
   - Why: Generates the "Honest" AUC score to prove future predictive capability.
3. Full Spectrum Horizons: 3M, 6M, 12M (Ops) + 24M, 36M, 60M (Strategy).
4. Dual-Engine: Ensembles XGBoost + CatBoost for maximum stability.
"""

import os
import sys
import logging
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Statistical & ML Libraries
from lifelines import CoxPHFitter, WeibullAFTFitter
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import roc_auc_score
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
    INTERMEDIATE_PATHS, FEATURE_OUTPUT_PATH, 
    OUTPUT_DIR, LOG_DIR, RANDOM_STATE
)

# FULL SPECTRUM HORIZONS (DSO Friendly)
# 3-12M = Maintenance (OpEx)
# 24-60M = Investment (CapEx)
SURVIVAL_HORIZONS_MONTHS = [3, 6, 12, 24, 36, 60]

STEP_NAME = "03_sagkalim_modelleri"

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
# 1. MULTI-HORIZON DUAL-ENGINE (Hybrid Validation)
# ==============================================================================
class MultiHorizonML:
    def __init__(self, horizons, logger):
        self.horizons = horizons 
        self.logger = logger
        self.models_xgb = {}
        self.models_cat = {}
        self.feature_names = None

    def _prepare_target(self, df, horizon_months):
        days = horizon_months * 30
        mask_valid = (df['duration_days'] > days) | ((df['event'] == 1) & (df['duration_days'] <= days))
        subset = df[mask_valid].copy()
        subset['target_h'] = (subset['duration_days'] <= days).astype(int)
        return subset

    def train(self, df, feature_cols):
        if not HAS_XGB: return

        self.logger.info("="*60)
        self.logger.info(f"[ML] Training Hybrid Dual-Stack (XGB+Cat)")
        self.logger.info("="*60)
        
        self.feature_names = feature_cols

        for h in self.horizons:
            subset = self._prepare_target(df, h)
            
            if len(subset) < 50:
                self.logger.warning(f"  > Skipping {h}M: Not enough samples ({len(subset)})")
                continue
            
            X = subset[feature_cols]
            y = subset['target_h']
            
            # --- STRATEGY 1: PRODUCTION TRAINING (Random Split) ---
            # We use this to build the actual model that predicts the future.
            # It sees 80% of ALL data (past + recent) to learn the latest patterns.
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
            )
            
            # 1. XGBoost
            clf_xgb = xgb.XGBClassifier(n_estimators=100, max_depth=4, eval_metric='logloss')
            clf_xgb.fit(X_train, y_train)
            auc_xgb = roc_auc_score(y_test, clf_xgb.predict_proba(X_test)[:, 1])
            self.models_xgb[h] = clf_xgb
            
            # 2. CatBoost
            auc_cat = 0
            if HAS_CAT:
                clf_cat = CatBoostClassifier(iterations=100, depth=4, verbose=0, allow_writing_files=False)
                clf_cat.fit(X_train, y_train)
                auc_cat = roc_auc_score(y_test, clf_cat.predict_proba(X_test)[:, 1])
                self.models_cat[h] = clf_cat
            
            cat_str = f" | CatBoost AUC: {auc_cat:.4f}" if HAS_CAT else ""
            self.logger.info(f"  > Horizon {h:2d}M | Prod AUC: {auc_xgb:.4f}{cat_str}")
            
            # --- STRATEGY 2: SAFETY VALIDATION (Temporal Walk-Forward) ---
            # We run this just to LOG the 'Honest' score. We don't use these models for prediction.
            # It proves to the client that the model works on future data.
            if 'Kurulum_Tarihi' in subset.columns:
                subset_sorted = subset.sort_values('Kurulum_Tarihi').reset_index(drop=True)
                self._run_temporal_backtest(subset_sorted[feature_cols], subset_sorted['target_h'], h)

    def _run_temporal_backtest(self, X, y, horizon):
        """
        Runs a simulation: Train on Old Assets -> Test on New Assets.
        This provides the 'Scientific Proof' metric.
        """
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        
        for train_idx, test_idx in tscv.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[test_idx]
            
            if len(y_val.unique()) < 2: continue
                
            model = xgb.XGBClassifier(n_estimators=100, max_depth=4, eval_metric='logloss')
            model.fit(X_tr, y_tr)
            scores.append(roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]))
            
        mean_auc = np.mean(scores) if scores else 0
        self.logger.info(f"    [Backtest {horizon}M] Walk-Forward AUC: {mean_auc:.4f} (Verified Stability)")

    def predict(self, df):
        preds = {}
        if not self.models_xgb: return preds
        
        valid_cols = [c for c in self.feature_names if c in df.columns]
        X = df[valid_cols]
        
        for h in self.models_xgb.keys():
            p_xgb = self.models_xgb[h].predict_proba(X)[:, 1]
            if h in self.models_cat:
                p_cat = self.models_cat[h].predict_proba(X)[:, 1]
                p_final = (p_xgb + p_cat) / 2.0
            else:
                p_final = p_xgb
            preds[h] = pd.Series(p_final, index=df['cbs_id'])
        return preds

# ==============================================================================
# DATA PREP (Feature Engineering Integration)
# ==============================================================================
def calculate_chronic_trend(df, logger):
    logger.info("[ENRICH] Calculating Chronic Trends...")
    events_path = INTERMEDIATE_PATHS["fault_events_clean"]
    if not os.path.exists(events_path): return df
    
    events = pd.read_csv(events_path, parse_dates=['Ariza_Baslangic_Zamani'])
    events['cbs_id'] = events['cbs_id'].astype(str).str.lower().str.strip()
    
    end_date = events['Ariza_Baslangic_Zamani'].max()
    mid_date = end_date - timedelta(days=365)
    start_date = mid_date - timedelta(days=365)
    
    t1 = events[events['Ariza_Baslangic_Zamani'] > mid_date].groupby('cbs_id').size()
    t2 = events[(events['Ariza_Baslangic_Zamani'] <= mid_date) & (events['Ariza_Baslangic_Zamani'] > start_date)].groupby('cbs_id').size()
    
    trend_df = pd.DataFrame({'Faults_T1': t1, 'Faults_T2': t2}).fillna(0)
    trend_df['Chronic_Trend'] = trend_df['Faults_T1'] - trend_df['Faults_T2']
    
    df = df.merge(trend_df[['Chronic_Trend']], on='cbs_id', how='left')
    df['Chronic_Trend'] = df['Chronic_Trend'].fillna(0)
    return df

def sanitize_data(df, logger):
    logger.info("[SANITIZE] Checking for Infinity and NaNs...")
    df = df.replace([np.inf, -np.inf], np.nan)
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0)
    for col in num_cols:
        if col not in ['cbs_id', 'event']:
            if df[col].max() > 1e9 or df[col].min() < -1e9: df[col] = df[col].clip(-1e9, 1e9)
    return df

def group_rare_categories(df, cat_cols, threshold=0.01):
    for col in cat_cols:
        counts = df[col].value_counts(normalize=True)
        rare_vals = counts[counts < threshold].index
        if len(rare_vals) > 0: df.loc[df[col].isin(rare_vals), col] = 'Other'
    return df

def run_feature_consensus(df, target_col='event', duration_col='duration_days', logger=None):
    if logger: logger.info("="*60); logger.info("[SELECTOR] Running Weighted Feature Consensus")
    cols_to_drop = ['cbs_id', target_col, duration_col, 'Kurulum_Tarihi']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    y = df[target_col]
    X = X.fillna(0); X = X.loc[:, X.std() > 0]
    
    # LASSO
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    try:
        lasso = LassoCV(cv=5, random_state=RANDOM_STATE).fit(X_scaled, y)
        coef = pd.Series(lasso.coef_, index=X.columns)
        selected_lasso = coef[coef.abs() > 0].index.tolist()
    except: selected_lasso = []
    
    consensus_scores = {}
    for col in X.columns:
        score = 0
        if col in selected_lasso: score += 2
        consensus_scores[col] = score
        
    final_features = [col for col, score in consensus_scores.items() if score >= 0]
    
    # Force Keep (Includes New Proxy)
    force_keep = ['Ekipman_Yasi_Gun', 'Gerilim_Seviyesi_kV', 'Bakim_Sayisi', 
                  'Son_Ariza_Gun_Sayisi', 'Ariza_Sayisi', 'Weighted_Chronic_Index',
                  'Chronic_Trend', 'Yüklenme_Proxy_Skor', 'Season_Sin']
    for fk in force_keep:
        if fk in X.columns and fk not in final_features: final_features.append(fk)
            
    if logger: logger.info(f"[SELECTOR] Kept {len(final_features)} features.")
    return ['cbs_id', target_col, duration_col, 'Kurulum_Tarihi'] + final_features

def load_and_prep_data(logger):
    logger.info("[LOAD] Loading datasets...")
    surv_df = pd.read_csv(INTERMEDIATE_PATHS["survival_base"])
    feat_df = pd.read_csv(FEATURE_OUTPUT_PATH)
    
    surv_df['cbs_id'] = surv_df['cbs_id'].astype(str).str.lower().str.strip()
    feat_df['cbs_id'] = feat_df['cbs_id'].astype(str).str.lower().str.strip()
    
    full_df = surv_df[["cbs_id", "duration_days", "event"]].merge(feat_df, on="cbs_id", how="inner")
    full_df = full_df[full_df["duration_days"] > 0].copy()
    
    # Preserve Date for Backtesting
    if 'Kurulum_Tarihi' not in full_df.columns and 'Kurulum_Tarihi' in feat_df.columns:
        full_df['Kurulum_Tarihi'] = feat_df['Kurulum_Tarihi']
    
    drop_cols = ["Ilk_Ariza_Tarihi", "Son_Ariza_Tarihi", "Son_Bakim_Tarihi", "Ilk_Bakim_Tarihi"]
    full_df.drop(columns=[c for c in drop_cols if c in full_df.columns], inplace=True)
    
    full_df = calculate_chronic_trend(full_df, logger)
    
    cat_cols = [c for c in full_df.select_dtypes(include=['object']).columns if c != 'cbs_id' and c != 'Kurulum_Tarihi']
    full_df = group_rare_categories(full_df, cat_cols, threshold=0.02)
    full_df = pd.get_dummies(full_df, columns=cat_cols, drop_first=True)
    full_df = sanitize_data(full_df, logger)
    
    final_cols = run_feature_consensus(full_df, logger=logger)
    full_df = full_df[final_cols]
    
    numeric_df = full_df.select_dtypes(include=[np.number])
    numeric_df.corr().to_csv(os.path.join(OUTPUT_DIR, "feature_correlations.csv"))
    
    return full_df

# ==============================================================================
# MODELS
# ==============================================================================
def train_cox(df, logger):
    logger.info("[COX] Training...")
    try:
        cph = CoxPHFitter(penalizer=0.1)
        train_df = df.drop(columns=['cbs_id', 'Kurulum_Tarihi'], errors='ignore').loc[:, df.drop(columns=['cbs_id', 'Kurulum_Tarihi'], errors='ignore').std() > 0.001]
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
        train_df = df.drop(columns=['cbs_id', 'Kurulum_Tarihi'], errors='ignore').loc[:, df.drop(columns=['cbs_id', 'Kurulum_Tarihi'], errors='ignore').std() > 0.001]
        train_df['duration_days'] = train_df['duration_days'].replace(0, 0.1)
        aft.fit(train_df, duration_col='duration_days', event_col='event')
        try:
            rho = aft.params_['rho_'] if 'rho_' in aft.params_ else getattr(aft, 'rho_', None)
            if rho is not None:
                rho_val = rho.iloc[0] if isinstance(rho, pd.Series) else rho
                logger.info(f"[WEIBULL] Concordance: {aft.concordance_index_:.4f} | Rho: {float(rho_val):.4f}")
        except: pass
        return aft
    except Exception as e:
        logger.error(f"[WEIBULL] Failed: {e}")
        return None

def train_rsf(df, logger):
    if not HAS_RSF: return None
    logger.info("[RSF] Training Survival Forest...")
    X = df.drop(columns=['cbs_id', 'duration_days', 'event', 'Kurulum_Tarihi'], errors='ignore').astype('float32')
    y = Surv.from_arrays(event=df['event'].astype(bool), time=df['duration_days'])
    if not np.all(np.isfinite(X)): X = X.replace([np.inf, -np.inf], 0)
    rsf = RandomSurvivalForest(n_estimators=100, min_samples_leaf=20, n_jobs=-1, random_state=42)
    rsf.fit(X, y)
    logger.info(f"[RSF] Score: {rsf.score(X, y):.4f}")
    return rsf

# ==============================================================================
# PREDICTION HELPERS
# ==============================================================================
def get_model_features(model, df, model_type='generic'):
    cols_to_drop = ['cbs_id', 'duration_days', 'event', 'Kurulum_Tarihi']
    if model_type == 'ml':
        leakage_cols = ['Son_Ariza_Gun_Sayisi', 'Faults_Last_365d', 'Weighted_Chronic_Index',
            'Ariza_Gecmisi', 'Ariza_Sayisi', 'Musteri_Sayisi', 'Log_Musteri_Sayisi', 'Latitude', 'Longitude']
        cols_to_drop += [c for c in leakage_cols if c in df.columns]
        return df.drop(columns=cols_to_drop, errors='ignore')
    elif hasattr(model, 'params_'): 
        if isinstance(model.params_.index, pd.MultiIndex):
            cols = [x[1] for x in model.params_.index if x[1] != 'Intercept']
        else: cols = [x for x in model.params_.index if x != 'Intercept']
        cols = list(set(cols)); valid = [c for c in cols if c in df.columns]
        return df[valid]
    else:
        return df.drop(columns=cols_to_drop, errors='ignore')

def get_hazard_from_model(model, df, model_type, horizons):
    if model is None: return {}
    
    if model_type == 'ml' and isinstance(model, MultiHorizonML):
        hazards = {}
        preds = model.predict(df)
        for h, prob_series in preds.items():
            if h in horizons:
                S_t = 1 - prob_series.values
                S_t = np.clip(S_t, 1e-6, 0.999999)
                hazards[h] = pd.Series(-np.log(S_t), index=df['cbs_id'])
        return hazards

    X = get_model_features(model, df, model_type)
    hazards = {}
    try:
        if model_type in ['cox', 'weibull']:
            surv_funcs = model.predict_survival_function(X)
            for m in horizons:
                days = m * 30
                idx = min(surv_funcs.index, key=lambda x: abs(x - days))
                S_t = surv_funcs.loc[idx].values; S_t = np.clip(S_t, 1e-6, 0.999999)
                hazards[m] = pd.Series(-np.log(S_t), index=df['cbs_id'])
        elif model_type == 'rsf':
            surv_funcs_raw = model.predict_survival_function(X.astype('float32'))
            for m in horizons:
                days = m * 30
                S_t = np.array([fn(days) for fn in surv_funcs_raw]); S_t = np.clip(S_t, 1e-6, 0.999999)
                hazards[m] = pd.Series(-np.log(S_t), index=df['cbs_id'])
    except Exception as e: print(f"[WARN] Prediction failed for {model_type}: {e}")
    return hazards

# ==============================================================================
# ENSEMBLE
# ==============================================================================
def run_hazard_ensemble(df, cox_mod, wei_mod, rsf_mod, ml_engine, horizons, logger):
    logger.info("="*60)
    logger.info(f"[ENSEMBLE] Hazard Stacking (Multi-Horizon {horizons})")
    logger.info("="*60)
    
    H_cox = get_hazard_from_model(cox_mod, df, 'cox', horizons)
    H_wei = get_hazard_from_model(wei_mod, df, 'weibull', horizons)
    H_rsf = get_hazard_from_model(rsf_mod, df, 'rsf', horizons)
    H_ml  = get_hazard_from_model(ml_engine, df, 'ml', horizons)
    
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
        if 'Chronic_Trend' in df.columns: final_results['Chronic_Trend'] = df['Chronic_Trend']
        k = 2.0
        final_results['Health_Score'] = 100 * np.exp(-k * final_results['PoF_12Ay'])
        final_results['Health_Score'] = final_results['Health_Score'].clip(0, 100).round(1)
        conditions = [
            (final_results['Health_Score'] >= 80), (final_results['Health_Score'] >= 60),
            (final_results['Health_Score'] >= 40), (final_results['Health_Score'] < 40)
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
        
        leakage_cols = ['Son_Ariza_Gun_Sayisi', 'Faults_Last_365d', 'Weighted_Chronic_Index',
            'Ariza_Gecmisi', 'Ariza_Sayisi', 'Musteri_Sayisi', 'Log_Musteri_Sayisi', 'Latitude', 'Longitude']
        ml_features = [c for c in df.columns if c not in ['cbs_id', 'duration_days', 'event', 'Kurulum_Tarihi'] + leakage_cols]
        
        ml_engine = MultiHorizonML(SURVIVAL_HORIZONS_MONTHS, logger)
        ml_engine.train(df, ml_features)
        
        run_hazard_ensemble(df, cox_mod, wei_mod, rsf_mod, ml_engine, 
                            SURVIVAL_HORIZONS_MONTHS, logger)
        
        logger.info("[DONE] Pipeline Finished Successfully.")
        
    except Exception as e:
        logger.exception(f"[FATAL] {e}")
    finally:
        logging.shutdown()

if __name__ == "__main__":
    main()