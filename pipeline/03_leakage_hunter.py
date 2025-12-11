"""
03_leakage_hunter.py (PoF3)

Purpose: Detect 'Data Leakage' features that give the model the answer key.
Method:
1. Correlation Analysis with Target ('event')
2. Single-Feature AUC (How well does this feature predict the target ALONE?)
3. XGBoost Feature Importance
"""

import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Setup Paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from config.config import FEATURE_OUTPUT_PATH, INTERMEDIATE_PATHS

def main():
    print("🕵️‍♂️ LEAKAGE HUNTER: Starting Investigation...")
    
    # 1. Load Data
    print("[1] Loading Data...")
    surv_df = pd.read_csv(INTERMEDIATE_PATHS["survival_base"])
    feat_df = pd.read_csv(FEATURE_OUTPUT_PATH)
    
    # Merge Features with Target
    df = surv_df[["cbs_id", "event"]].merge(feat_df, on="cbs_id", how="inner")
    
    # Drop identifiers and dates (not features)
    drop_cols = ["cbs_id", "Kurulum_Tarihi", "Ilk_Ariza_Tarihi", "Son_Ariza_Tarihi"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    # Fill NaNs for quick analysis
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0)
    
    # One-Hot Encode Categoricals
    cat_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    target = "event"
    features = [c for c in df.columns if c != target]
    
    print(f"[1] Dataset: {df.shape[0]} rows, {len(features)} features")

    # 2. Correlation Check
    print("\n[2] Running Correlation Analysis...")
    corrs = df[features].corrwith(df[target]).abs().sort_values(ascending=False)
    
    suspicious_corr = corrs[corrs > 0.5] # Threshold: 0.5 correlation is already very high for real-world failure
    print("⚠️ HIGH CORRELATION SUSPECTS:")
    if not suspicious_corr.empty:
        print(suspicious_corr)
    else:
        print("   None found > 0.5")

    # 3. Single-Feature Predictive Power (The "Magic Bullet" Test)
    # If a single feature gives AUC > 0.9, it is almost certainly leakage.
    print("\n[3] Running 'Magic Bullet' Test (Single Feature AUC)...")
    
    suspicious_auc = []
    
    for feat in features:
        # Simple Logistic Regression logic check: can this feature split the classes perfectly?
        try:
            score = roc_auc_score(df[target], df[feat])
            # Check both directions (high value = fail OR low value = fail)
            power = max(score, 1-score) 
            
            if power > 0.85:
                suspicious_auc.append((feat, power))
        except:
            pass # Skip columns that crash (constants etc)
            
    suspicious_auc.sort(key=lambda x: x[1], reverse=True)
    
    print("⚠️ 'MAGIC BULLET' SUSPECTS (AUC > 0.85):")
    if suspicious_auc:
        for name, score in suspicious_auc:
            print(f"   ❌ {name}: AUC = {score:.4f}")
    else:
        print("   None found.")

    # 4. XGBoost Importance (The Final Judge)
    print("\n[4] Training Quick XGBoost to find dominant features...")
    X = df[features]
    y = df[target]
    
    clf = xgb.XGBClassifier(n_estimators=50, max_depth=3, eval_metric='logloss')
    clf.fit(X, y)
    
    importance = pd.DataFrame({
        'feature': features,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("⚠️ TOP 5 MODEL DRIVERS:")
    print(importance.head(5))

    # 5. Recommendation
    print("\n" + "="*40)
    print("🧪 DIAGNOSIS")
    print("="*40)
    
    candidates = set(suspicious_corr.index.tolist() + [x[0] for x in suspicious_auc] + importance.head(3)['feature'].tolist())
    
    print(f"Investigate these columns immediately:\n{list(candidates)}")

if __name__ == "__main__":
    main()