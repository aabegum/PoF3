"""
05_raporlama_ve_gorsellestirme.py (PoF3 - Unified Reporting Engine)

PURPOSE:
  Combines Risk Segmentation, Visualization, and Reporting into a single master script.
  Ensures all outputs are synchronized and stored in structured folders.

PHASES:
  1. ACTION PLANNING: Generates work orders (e.g., 'Urgent Chronic Fixes').
  2. VISUALIZATION: Generates Risk Matrix, Health Scores, and Map plots.
  3. REPORTING: Packages everything into a professional Excel Dashboard.

OUTPUT STRUCTURE:
  data/sonuclar/
    ├── aksiyon_listeleri/       (CSV lists for maintenance teams)
    ├── gorseller/               (High-res PNG charts for PPT/PDF)
    └── PoF3_Analiz_Raporu.xlsx  (Final Executive Deliverable)
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Setup Paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from config.config import OUTPUT_DIR, LOG_DIR

STEP_NAME = "05_raporlama_ve_gorsellestirme"

# Folder Setup
ACTION_DIR = os.path.join(OUTPUT_DIR, "aksiyon_listeleri")
VISUAL_DIR = os.path.join(OUTPUT_DIR, "gorseller")
REPORT_DIR = OUTPUT_DIR # Root output folder for the Excel file

for d in [ACTION_DIR, VISUAL_DIR]:
    os.makedirs(d, exist_ok=True)

# Styling
plt.style.use('ggplot')
sns.set_palette("husl")

# ------------------------------------------------------------------------------
# LOGGING
# ------------------------------------------------------------------------------
def setup_logger():
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = logging.getLogger(STEP_NAME)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    fh = logging.FileHandler(os.path.join(LOG_DIR, f"{STEP_NAME}_{ts}.log"), encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger

# ------------------------------------------------------------------------------
# PHASE 1: ACTION PLANNING (Risk Segmentation)
# ------------------------------------------------------------------------------
def generate_action_lists(df, logger):
    logger.info("="*60)
    logger.info("[PHASE 1] Generating Action Lists...")
    logger.info("="*60)

    # 1. IMMEDIATE ATTENTION (Critical Risk + Chronic)
    # Assets failing often AND high consequence.
    crit_chronic = df[
        (df['Risk_Class'] == 'Critical') & 
        (df['Kronik_Seviye_Max'].isin(['KRITIK', 'YUKSEK']))
    ].copy()
    
    if not crit_chronic.empty:
        path = os.path.join(ACTION_DIR, "01_acil_mudahale_listesi.csv")
        crit_chronic.sort_values('Risk_Score', ascending=False).to_csv(path, index=False, encoding='utf-8-sig')
        logger.info(f"  > [URGENT] Chronic & Critical: {len(crit_chronic)} assets -> {os.path.basename(path)}")

    # 2. CAPEX PRIORITIES (High Risk Transformers)
    # Transformers are expensive; prioritize replacement.
    trafos = df[
        (df['Ekipman_Tipi'] == 'Trafo') & 
        (df['Risk_Class'].isin(['Critical', 'High']))
    ].copy()
    
    if not trafos.empty:
        path = os.path.join(ACTION_DIR, "02_yuksek_riskli_trafolar_capex.csv")
        trafos.sort_values('Risk_Score', ascending=False).to_csv(path, index=False, encoding='utf-8-sig')
        logger.info(f"  > [CAPEX] High Risk Transformers: {len(trafos)} assets -> {os.path.basename(path)}")

    # 3. INSPECTION ROUTE (High PoF / Low Impact)
    # Likely to fail but low impact. Good for route-based inspection.
    inspection = df[
        (df.get('PoF_Ensemble_12Ay', 0) > 0.10) & 
        (df['Risk_Class'].isin(['Low', 'Medium']))
    ].copy()
    
    if not inspection.empty:
        path = os.path.join(ACTION_DIR, "03_bakim_rotasi_kontrol.csv")
        inspection.sort_values('PoF_Ensemble_12Ay', ascending=False).to_csv(path, index=False, encoding='utf-8-sig')
        logger.info(f"  > [OPEX] High Prob/Low Impact: {len(inspection)} assets -> {os.path.basename(path)}")

    # 4. DATA QUALITY AUDIT
    # Critical risk calculated on missing data needs verification.
    if 'Musteri_Sayisi' in df.columns:
        audit = df[
            (df['Risk_Class'] == 'Critical') & 
            ((df['Musteri_Sayisi'] <= 1) | (df['Musteri_Sayisi'].isna()))
        ].copy()
        
        if not audit.empty:
            path = os.path.join(ACTION_DIR, "04_veri_kalitesi_kontrol.csv")
            audit.to_csv(path, index=False, encoding='utf-8-sig')
            logger.info(f"  > [DATA] Critical Risk/Missing Data: {len(audit)} assets")

    return crit_chronic # Return for Excel summary

# ------------------------------------------------------------------------------
# PHASE 2: VISUALIZATION ENGINE
# ------------------------------------------------------------------------------
def generate_visuals(df, logger):
    logger.info("="*60)
    logger.info("[PHASE 2] Generating Visual Dashboards...")
    logger.info("="*60)
    
    # 1. RISK MATRIX
    if 'CoF_Total_Score' in df.columns and 'PoF_Ensemble_12Ay' in df.columns:
        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            data=df, x='CoF_Total_Score', y='PoF_Ensemble_12Ay',
            hue='Risk_Class', hue_order=['Critical', 'High', 'Medium', 'Low'],
            palette={'Critical': 'red', 'High': 'orange', 'Medium': 'gold', 'Low': 'green'},
            alpha=0.6, s=60
        )
        plt.axhline(0.10, color='gray', linestyle='--', label='High Prob (10%)')
        plt.axvline(50, color='gray', linestyle='--', label='High Impact (50)')
        plt.title('Risk Matrisi (Risk Matrix)', fontsize=14)
        plt.xlabel('Etki Skoru (CoF)', fontsize=12)
        plt.ylabel('Arıza Olasılığı (PoF)', fontsize=12)
        plt.legend(title='Risk Sınıfı')
        
        path = os.path.join(VISUAL_DIR, "01_risk_matrisi.png")
        plt.savefig(path, dpi=300)
        plt.close()
        logger.info(f"  > Saved Chart: {os.path.basename(path)}")

    # 2. HEALTH SCORE DISTRIBUTION
    if 'Health_Score' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df['Health_Score'], bins=30, kde=True, color='teal', edgecolor='black')
        plt.axvline(40, color='red', linestyle='--', linewidth=2, label='Kritik Sınır (40)')
        plt.axvline(80, color='green', linestyle='--', linewidth=2, label='İyi Durum (80)')
        plt.title('Varlık Sağlık Skoru Dağılımı', fontsize=14)
        plt.xlabel('Sağlık Skoru (0=Kötü, 100=Mükemmel)')
        plt.legend()
        
        path = os.path.join(VISUAL_DIR, "02_saglik_skoru_dagilimi.png")
        plt.savefig(path, dpi=300)
        plt.close()
        logger.info(f"  > Saved Chart: {os.path.basename(path)}")

    # 3. CHRONIC BREAKDOWN
    if 'Kronik_Seviye_Max' in df.columns:
        plt.figure(figsize=(8, 6))
        counts = df['Kronik_Seviye_Max'].value_counts()
        order = ['KRITIK', 'YUKSEK', 'ORTA', 'IZLEME', 'NORMAL']
        counts = counts.reindex([x for x in order if x in counts.index])
        colors = ['red', 'orange', 'gold', 'lightblue', 'green']
        
        counts.plot(kind='bar', color=colors, edgecolor='black')
        plt.title('Kronik Arıza Seviyeleri', fontsize=14)
        plt.ylabel('Ekipman Sayısı')
        plt.xticks(rotation=0)
        
        path = os.path.join(VISUAL_DIR, "03_kronik_dagilimi.png")
        plt.savefig(path, dpi=300)
        plt.close()
        logger.info(f"  > Saved Chart: {os.path.basename(path)}")

    # 4. GEOSPATIAL MAP (If coordinates exist)
    if 'Latitude' in df.columns and 'Longitude' in df.columns:
        # Filter valid coords (non-zero)
        gdf = df[(df['Latitude'] != 0) & (df['Longitude'] != 0)].copy()
        if not gdf.empty:
            plt.figure(figsize=(10, 10))
            sns.scatterplot(
                data=gdf, x='Longitude', y='Latitude',
                hue='Health_Class', hue_order=['Critical', 'Poor', 'Good', 'Excellent'],
                palette={'Critical': 'red', 'Poor': 'orange', 'Good': 'yellow', 'Excellent': 'green'},
                s=30, alpha=0.8
            )
            plt.title('Coğrafi Risk Haritası', fontsize=14)
            plt.axis('equal')
            
            path = os.path.join(VISUAL_DIR, "04_cografi_risk_haritasi.png")
            plt.savefig(path, dpi=300)
            plt.close()
            logger.info(f"  > Saved Chart: {os.path.basename(path)}")

# ------------------------------------------------------------------------------
# PHASE 3: FINAL REPORTING (Excel)
# ------------------------------------------------------------------------------
def create_final_report(df, crit_chronic, logger):
    logger.info("="*60)
    logger.info("[PHASE 3] Creating Executive Excel Report...")
    logger.info("="*60)
    
    timestamp = datetime.now().strftime("%Y-%m-%d")
    out_path = os.path.join(REPORT_DIR, f"PoF3_Analiz_Raporu_Final.xlsx")
    
    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        # 1. Executive Summary
        total = len(df)
        crit_count = (df['Risk_Class'] == 'Critical').sum()
        avg_health = df['Health_Score'].mean() if 'Health_Score' in df.columns else 0
        
        summary_data = {
            'KPI': ['Toplam Varlık', 'Kritik Riskli Varlık', 'Kronik ve Kritik (Acil)', 'Ortalama Sağlık Skoru', 'Rapor Tarihi'],
            'Değer': [total, crit_count, len(crit_chronic), f"{avg_health:.1f}", timestamp],
            'Açıklama': [
                'Analiz edilen toplam ekipman sayısı',
                'Risk Skoru > 40 olan varlıklar',
                'Hem sık arızalanan hem de yüksek riskli varlıklar (Hemen Müdahale)',
                '0 (Kötü) - 100 (İyi) arası filo ortalaması',
                'Raporun oluşturulduğu tarih'
            ]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Yonetici_Ozeti', index=False)
        
        # 2. Action List (Urgent)
        if not crit_chronic.empty:
            crit_chronic.to_excel(writer, sheet_name='Acil_Mudahale_Listesi', index=False)
            
        # 3. Top 500 Riskiest Assets
        if 'Risk_Score' in df.columns:
            top_risk = df.sort_values('Risk_Score', ascending=False).head(500)
            top_risk.to_excel(writer, sheet_name='Top_500_Riskli_Varlik', index=False)
        else:
            df.head(500).to_excel(writer, sheet_name='Risk_Master_Ornek', index=False)
        
    logger.info(f"  > [DONE] Report saved: {out_path}")

# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------
def main():
    logger = setup_logger()
    
    # 1. Load Risk Master (Result of Step 04/04b)
    risk_path = os.path.join(OUTPUT_DIR, "risk_equipment_master.csv")
    if not os.path.exists(risk_path):
        logger.error(f"[FATAL] Risk Master not found at {risk_path}. Please run Step 04 first.")
        return
        
    # Load Master
    df = pd.read_csv(risk_path)
    
    # Context Merge (Latitude/Longitude) if missing in risk master
    # Sometimes risk_master comes from Step 04 statistics and might miss geo data
    # We fetch it from equipment_master just in case
    master_path = os.path.join(os.path.dirname(OUTPUT_DIR), "ara_ciktilar", "equipment_master.csv")
    if os.path.exists(master_path):
        meta = pd.read_csv(master_path)
        meta['cbs_id'] = meta['cbs_id'].astype(str).str.lower().str.strip()
        
        # Add Geo Data if missing
        cols_to_add = [c for c in ['Latitude', 'Longitude', 'Musteri_Sayisi'] if c in meta.columns and c not in df.columns]
        if cols_to_add:
            df['cbs_id'] = df['cbs_id'].astype(str).str.lower().str.strip()
            df = df.merge(meta[['cbs_id'] + cols_to_add], on='cbs_id', how='left')
    
    logger.info(f"[LOAD] Loaded {len(df):,} assets for reporting.")
    
    # 2. Run Phases
    crit_chronic_df = generate_action_lists(df, logger)
    generate_visuals(df, logger)
    create_final_report(df, crit_chronic_df, logger)
    
    logger.info("")
    logger.info("[SUCCESS] Pipeline Reporting Complete.")
    logger.info(f"  > Outputs located in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()