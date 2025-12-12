"""
05_raporlama_ve_gorsellestirme.py (PoF3 - Unified Reporting Engine v2)

PURPOSE:
  Combines Risk Segmentation, Visualization, Excel Reporting, AND PowerPoint generation.
  Ensures all deliverables are synchronized.

PHASES:
  1. ACTION PLANNING: Generates work orders (CSVs).
  2. VISUALIZATION: Generates charts (PNGs).
  3. EXCEL REPORT: Executive summary & lists.
  4. POWERPOINT: Presentation deck (Restored).

OUTPUTS:
  data/sonuclar/aksiyon_listeleri/*.csv
  data/sonuclar/gorseller/*.png
  data/sonuclar/PoF3_Analiz_Raporu_Final.xlsx
  data/sonuclar/PoF3_Yonetici_Sunumu_Final.pptx
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# PPTX Library Check
try:
    from pptx import Presentation
    from pptx.util import Inches, Pt
    HAS_PPTX = True
except ImportError:
    HAS_PPTX = False
    print("Warning: 'python-pptx' not installed. Skipping PPTX generation.")

# Setup Paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from config.config import OUTPUT_DIR, LOG_DIR

STEP_NAME = "05_raporlama_ve_gorsellestirme"

# Folder Setup
ACTION_DIR = os.path.join(OUTPUT_DIR, "aksiyon_listeleri")
VISUAL_DIR = os.path.join(OUTPUT_DIR, "gorseller")
REPORT_DIR = OUTPUT_DIR

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
# PHASE 1: ACTION PLANNING
# ------------------------------------------------------------------------------
def generate_action_lists(df, logger):
    logger.info("="*60)
    logger.info("[PHASE 1] Generating Action Lists...")
    
    crit_chronic = df[
        (df['Risk_Class'] == 'Critical') & 
        (df['Kronik_Seviye_Max'].isin(['KRITIK', 'YUKSEK']))
    ].copy()
    
    if not crit_chronic.empty:
        path = os.path.join(ACTION_DIR, "01_acil_mudahale_listesi.csv")
        crit_chronic.sort_values('Risk_Score', ascending=False).to_csv(path, index=False, encoding='utf-8-sig')
        logger.info(f"  > [URGENT] Chronic & Critical: {len(crit_chronic)} assets")

    trafos = df[
        (df['Ekipman_Tipi'] == 'Trafo') & 
        (df['Risk_Class'].isin(['Critical', 'High']))
    ].copy()
    
    if not trafos.empty:
        path = os.path.join(ACTION_DIR, "02_yuksek_riskli_trafolar_capex.csv")
        trafos.sort_values('Risk_Score', ascending=False).to_csv(path, index=False, encoding='utf-8-sig')
        logger.info(f"  > [CAPEX] High Risk Transformers: {len(trafos)} assets")

    inspection = df[
        (df.get('PoF_Ensemble_12Ay', 0) > 0.10) & 
        (df['Risk_Class'].isin(['Low', 'Medium']))
    ].copy()
    
    if not inspection.empty:
        path = os.path.join(ACTION_DIR, "03_bakim_rotasi_kontrol.csv")
        inspection.sort_values('PoF_Ensemble_12Ay', ascending=False).to_csv(path, index=False, encoding='utf-8-sig')
        logger.info(f"  > [OPEX] High Prob/Low Impact: {len(inspection)} assets")

    if 'Musteri_Sayisi' in df.columns:
        audit = df[
            (df['Risk_Class'] == 'Critical') & 
            ((df['Musteri_Sayisi'] <= 1) | (df['Musteri_Sayisi'].isna()))
        ].copy()
        if not audit.empty:
            path = os.path.join(ACTION_DIR, "04_veri_kalitesi_kontrol.csv")
            audit.to_csv(path, index=False, encoding='utf-8-sig')
            logger.info(f"  > [DATA] Critical Risk/Missing Data: {len(audit)} assets")

    return crit_chronic

# ------------------------------------------------------------------------------
# PHASE 2: VISUALIZATION
# ------------------------------------------------------------------------------
def generate_visuals(df, logger):
    logger.info("="*60)
    logger.info("[PHASE 2] Generating Visual Dashboards...")
    
    charts = {}

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
        plt.title('Varlık Risk Matrisi (Risk Matrix)', fontsize=14)
        plt.xlabel('Etki Skoru (CoF)', fontsize=12)
        plt.ylabel('Arıza Olasılığı (PoF)', fontsize=12)
        plt.legend(title='Risk Sınıfı')
        
        path = os.path.join(VISUAL_DIR, "01_risk_matrisi.png")
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        charts['risk_matrix'] = path
        logger.info(f"  > Saved: 01_risk_matrisi.png")

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
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        charts['health_dist'] = path
        logger.info(f"  > Saved: 02_saglik_skoru_dagilimi.png")

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
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        charts['chronic_dist'] = path
        logger.info(f"  > Saved: 03_kronik_dagilimi.png")

    # 4. GEOSPATIAL MAP
    if 'Latitude' in df.columns and 'Longitude' in df.columns:
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
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
            charts['geo_map'] = path
            logger.info(f"  > Saved: 04_cografi_risk_haritasi.png")

    return charts

# ------------------------------------------------------------------------------
# PHASE 3: EXCEL REPORTING
# ------------------------------------------------------------------------------
def create_excel_report(df, crit_chronic, logger):
    logger.info("="*60)
    logger.info("[PHASE 3] Creating Excel Report...")
    
    timestamp = datetime.now().strftime("%Y-%m-%d")
    out_path = os.path.join(REPORT_DIR, f"PoF3_Analiz_Raporu_Final.xlsx")
    
    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        # 1. Summary
        total = len(df)
        crit_count = (df['Risk_Class'] == 'Critical').sum() if 'Risk_Class' in df.columns else 0
        avg_health = df['Health_Score'].mean() if 'Health_Score' in df.columns else 0
        
        summary = pd.DataFrame({
            'KPI': ['Toplam Varlık', 'Kritik Riskli', 'Kronik ve Kritik', 'Ortalama Sağlık', 'Rapor Tarihi'],
            'Değer': [total, crit_count, len(crit_chronic), f"{avg_health:.1f}", timestamp]
        })
        summary.to_excel(writer, sheet_name='Yonetici_Ozeti', index=False)
        
        # 2. Action List
        if not crit_chronic.empty:
            crit_chronic.to_excel(writer, sheet_name='Acil_Mudahale', index=False)
            
        # 3. Top 1000 Master
        if 'Risk_Score' in df.columns:
            df.sort_values('Risk_Score', ascending=False).head(1000).to_excel(writer, sheet_name='Risk_Master_Top1000', index=False)
        else:
            df.head(1000).to_excel(writer, sheet_name='Risk_Master_Top1000', index=False)
            
    logger.info(f"  > Saved: {os.path.basename(out_path)}")

# ------------------------------------------------------------------------------
# PHASE 4: POWERPOINT PRESENTATION (RESTORED)
# ------------------------------------------------------------------------------
def create_pptx_presentation(df, charts, logger):
    if not HAS_PPTX:
        logger.warning("[PHASE 4] Skipping PPTX (Library missing).")
        return

    logger.info("="*60)
    logger.info("[PHASE 4] Creating PowerPoint Dashboard...")
    
    prs = Presentation()
    timestamp = datetime.now().strftime("%d %B %Y")
    
    # 1. Title Slide
    slide_layout = prs.slide_layouts[0] # Title Slide
    slide = prs.slides.add_slide(slide_layout)
    title = slide.shapes.title
    subtitle = slide.placeholders[1]
    title.text = "PoF3 Risk ve Sağlık Analizi"
    subtitle.text = f"Yönetici Özeti Raporu\n{timestamp}"
    
    # 2. Executive Summary Slide
    slide_layout = prs.slide_layouts[1] # Title and Content
    slide = prs.slides.add_slide(slide_layout)
    title = slide.shapes.title
    title.text = "Genel Durum Özeti"
    
    total_assets = len(df)
    critical_assets = (df['Risk_Class'] == 'Critical').sum() if 'Risk_Class' in df.columns else 0
    chronic_assets = (df['Kronik_Seviye_Max'] != 'NORMAL').sum() if 'Kronik_Seviye_Max' in df.columns else 0
    avg_health = df['Health_Score'].mean() if 'Health_Score' in df.columns else 0
    
    content = slide.placeholders[1]
    content.text = (
        f"Toplam Varlık Sayısı: {total_assets:,}\n"
        f"Kritik Riskli Varlıklar: {critical_assets:,} ({(critical_assets/total_assets):.1%})\n"
        f"Kronik Sorunlu Varlıklar: {chronic_assets:,}\n"
        f"Filo Ortalama Sağlık Skoru: {avg_health:.1f} / 100\n\n"
        "Öneri: 'Kritik' ve 'Kronik' kesişimindeki varlıklara öncelik verilmelidir."
    )

    # 3. Add Charts
    # Map chart names to Slide Titles
    chart_slides = {
        'risk_matrix': "Risk Matrisi (Etki vs Olasılık)",
        'health_dist': "Filo Sağlık Dağılımı",
        'chronic_dist': "Kronik Arıza Analizi",
        'geo_map': "Coğrafi Risk Haritası"
    }
    
    for key, slide_title in chart_slides.items():
        if key in charts and os.path.exists(charts[key]):
            slide = prs.slides.add_slide(prs.slide_layouts[5]) # Title Only
            title = slide.shapes.title
            title.text = slide_title
            
            # Add Image (Centered)
            left = Inches(1)
            top = Inches(1.5)
            height = Inches(5.5)
            slide.shapes.add_picture(charts[key], left, top, height=height)
            
    out_path = os.path.join(OUTPUT_DIR, "PoF3_Yonetici_Sunumu_Final.pptx")
    prs.save(out_path)
    logger.info(f"  > Saved: {os.path.basename(out_path)}")

# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------
def main():
    logger = setup_logger()
    
    # Load Data
    risk_path = os.path.join(OUTPUT_DIR, "risk_equipment_master.csv")
    if not os.path.exists(risk_path):
        logger.error(f"[FATAL] Risk Master not found at {risk_path}. Run Step 04 first.")
        return
        
    df = pd.read_csv(risk_path)
    
    # Ensure Context (Geo)
    master_path = os.path.join(os.path.dirname(OUTPUT_DIR), "ara_ciktilar", "equipment_master.csv")
    if os.path.exists(master_path):
        meta = pd.read_csv(master_path)
        meta['cbs_id'] = meta['cbs_id'].astype(str).str.lower().str.strip()
        cols_to_add = [c for c in ['Latitude', 'Longitude', 'Musteri_Sayisi'] if c in meta.columns and c not in df.columns]
        if cols_to_add:
            df['cbs_id'] = df['cbs_id'].astype(str).str.lower().str.strip()
            df = df.merge(meta[['cbs_id'] + cols_to_add], on='cbs_id', how='left')
    
    logger.info(f"[LOAD] Loaded {len(df):,} assets.")
    
    # Run All Phases
    crit_chronic = generate_action_lists(df, logger)
    charts = generate_visuals(df, logger)
    create_excel_report(df, crit_chronic, logger)
    create_pptx_presentation(df, charts, logger)
    
    logger.info("")
    logger.info("[SUCCESS] Reporting & Visualization Complete.")

if __name__ == "__main__":
    main()