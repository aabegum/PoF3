
"""
05_raporlama_ve_gorsellestirme.py (PoF3 - Ultimate Reporting Engine v3.1)

FIXES:
1. NameError: Ensures the 'charts' dictionary is explicitly defined and updated.
2. Full Visual Suite: Restores all 7 key visuals, including Aggregate Risk and Historical Trends.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

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
from config.config import OUTPUT_DIR, LOG_DIR, INTERMEDIATE_PATHS, FEATURE_OUTPUT_PATH # Added FEATURE_OUTPUT_PATH

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
# PHASE 2: VISUALIZATION CORE PLOTS
# ------------------------------------------------------------------------------

def plot_single_chart(df, col_x, col_y, plot_type, title, filename, logger, **kwargs):
    plt.figure(figsize=(kwargs.get('width', 10), kwargs.get('height', 6)))
    
    if plot_type == 'scatter':
        sns.scatterplot(data=df, x=col_x, y=col_y, **kwargs)
    elif plot_type == 'hist':
        sns.histplot(df[col_x], kde=True, **kwargs)
    elif plot_type == 'bar':
        sns.barplot(x=col_x, y=col_y, data=df, **kwargs)

    plt.title(title, fontsize=14)
    plt.tight_layout()
    path = os.path.join(VISUAL_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  > Saved: {filename}")
    return path

def generate_visuals(df, logger):
    logger.info("="*60)
    logger.info("[PHASE 2] Generating Visual Dashboards...")
    charts = {}

    # 1. RISK MATRIX
    if 'CoF_Total_Score' in df.columns and 'PoF_Ensemble_12Ay' in df.columns:
        path = plot_single_chart(df, 'CoF_Total_Score', 'PoF_Ensemble_12Ay', 'scatter', 
                                 'Varlık Risk Matrisi (Risk Matrix)', "01_risk_matrisi.png", logger,
                                 hue='Risk_Class', palette={'Critical': 'red', 'Moderate': 'orange', 'Good': 'yellow', 'Excellent': 'green', 'Unknown': 'gray'}, s=60, alpha=0.6)
        charts['risk_matrix'] = path

    # 2. HEALTH SCORE DISTRIBUTION
    if 'Health_Score' in df.columns:
        path = plot_single_chart(df, 'Health_Score', None, 'hist', 
                                 'Varlık Sağlık Skoru Dağılımı', "02_saglik_skoru_dagilimi.png", logger,
                                 bins=30, color='teal', edgecolor='black')
        charts['health_dist'] = path

    # 3. CHRONIC BREAKDOWN
    if 'Kronik_Seviye_Max' in df.columns:
        counts = df['Kronik_Seviye_Max'].value_counts()
        order = ['KRITIK', 'YUKSEK', 'ORTA', 'IZLEME', 'NORMAL']
        counts = counts.reindex([x for x in order if x in counts.index])
        
        plt.figure(figsize=(8, 6))
        colors = ['red', 'orange', 'gold', 'lightblue', 'green']
        counts.plot(kind='bar', color=colors, edgecolor='black')
        plt.title('Kronik Arıza Seviyeleri', fontsize=14)
        plt.ylabel('Ekipman Sayısı')
        plt.xticks(rotation=0)
        path = os.path.join(VISUAL_DIR, "03_kronik_dagilimi.png")
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        charts['chronic_dist'] = path

    # 4. GEOSPATIAL MAP
    if 'Latitude' in df.columns and 'Longitude' in df.columns:
        gdf = df[(df['Latitude'] != 0) & (df['Longitude'] != 0)].copy()
        if not gdf.empty:
                # Define the palette robustly
                palette_map = {'Critical': 'red', 'Moderate': 'orange', 'Good': 'yellow', 'Excellent': 'green'}
                
                # Add fallback for missing keys if needed
                for label in gdf['Health_Class'].unique():
                    if label not in palette_map:
                        palette_map[label] = 'gray'

                path = plot_single_chart(gdf, 'Longitude', 'Latitude', 'scatter', 
                                        'Coğrafi Risk Haritası', "04_cografi_risk_haritasi.png", logger,
                                        hue='Health_Class', height=10, width=10,
                                        palette=palette_map, s=30, alpha=0.8) # Use palette_map here
                charts['geo_map'] = path

    # 5. FEATURE IMPORTANCE (Correlation Proxy)
    corr_path = os.path.join(OUTPUT_DIR, "feature_correlations.csv")
    if os.path.exists(corr_path):
        try:
            corr_df = pd.read_csv(corr_path, index_col=0)
            if 'event' in corr_df.columns:
                top_features = corr_df['event'].abs().sort_values(ascending=False).head(10).drop('event', errors='ignore')
                plt.figure(figsize=(10, 6))
                top_features.plot(kind='barh', color='purple', edgecolor='black')
                plt.title('En Önemli Risk Faktörleri (Korelasyon)', fontsize=14)
                plt.xlabel('Korelasyon Gücü')
                plt.gca().invert_yaxis()
                path = os.path.join(VISUAL_DIR, "05_ozellik_onemi.png")
                plt.savefig(path, dpi=300, bbox_inches='tight')
                plt.close()
                charts['feature_imp'] = path
        except: pass

    # 6. HISTORICAL TREND
    events_path = INTERMEDIATE_PATHS["fault_events_clean"]
    if os.path.exists(events_path):
        try:
            ev = pd.read_csv(events_path, parse_dates=['Ariza_Baslangic_Zamani'])
            ev['Year'] = ev['Ariza_Baslangic_Zamani'].dt.year
            trend = ev.groupby('Year').size()
            plt.figure(figsize=(10, 6))
            trend.plot(kind='line', marker='o', linewidth=2, color='blue')
            plt.title('Yıllık Arıza Trendi', fontsize=14)
            plt.ylabel('Toplam Arıza Sayısı')
            plt.grid(True)
            path = os.path.join(VISUAL_DIR, "06_ariza_trendi.png")
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
            charts['fault_trend'] = path
        except: pass

    # 7. AGE DISTRIBUTION
    if 'Ekipman_Yasi_Gun' in df.columns:
        ages = df['Ekipman_Yasi_Gun'] / 365.25
        path = plot_single_chart(pd.DataFrame({'Ages': ages}), 'Ages', None, 'hist', 
                                 'Varlık Yaş Dağılımı (Yıl)', "07_yas_dagilimi.png", logger,
                                 bins=20, color='brown', edgecolor='black', width=10, height=6)
        charts['age_dist'] = path

    return charts

def plot_aggregate_risk_by_type(df, logger):
    """
    Plots mean PoF and Chronic Scores by Equipment Type (STOLEN FEATURE).
    """
    if 'PoF_Ensemble_12Ay' not in df.columns or 'Kronik_Kritik' not in df.columns:
        return None # Return None if data is insufficient

    agg_df = df.groupby('Ekipman_Tipi').agg(
        Mean_PoF_1Y=('PoF_Ensemble_12Ay', 'mean'),
        Mean_Chronic_Flag=('Kronik_Kritik', 'mean'),
        Count=('cbs_id', 'count')
    ).reset_index()
    
    agg_df = agg_df[agg_df['Count'] >= 100].sort_values('Mean_PoF_1Y', ascending=False).head(10)

    if agg_df.empty: return None

    fig, ax1 = plt.subplots(figsize=(12, 6))
    sns.barplot(x='Ekipman_Tipi', y='Mean_PoF_1Y', data=agg_df, ax=ax1, color='darkred', alpha=0.7)
    ax1.set_ylabel('Ortalama PoF (1 Yıl)', color='darkred', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='darkred')
    ax1.set_xlabel('Ekipman Tipi', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    
    ax2 = ax1.twinx()
    sns.lineplot(x='Ekipman_Tipi', y='Mean_Chronic_Flag', data=agg_df, ax=ax2, color='darkgreen', marker='o', linewidth=3)
    ax2.set_ylabel('Ortalama Kronik Skor (0-1)', color='darkgreen', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='darkgreen')
    ax2.set_ylim(0, agg_df['Mean_Chronic_Flag'].max() * 1.2)

    plt.title('Ekipman Tipine Göre Risk Yoğunluğu (Top 10)', fontsize=14)
    
    path = os.path.join(VISUAL_DIR, "08_aggregate_risk_by_type.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  > Saved: 08_aggregate_risk_by_type.png")
    return path # Return the path

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
    slide.shapes.title.text = "PoF3 Risk ve Sağlık Analizi"
    slide.placeholders[1].text = f"Yönetici Özeti Raporu\n{timestamp}"
    
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
        'aggregate_risk': "Ekipman Tipine Göre Risk Yoğunluğu", # NEW AGGREGATE PLOT
        'feature_imp': "Risk Faktörleri (Önem Derecesi)",
        'fault_trend': "Tarihsel Arıza Trendi",
        'age_dist': "Varlık Yaş Profili",
        'geo_map': "Coğrafi Risk Haritası"
    }
    
    for key, slide_title in chart_slides.items():
        if key in charts and os.path.exists(charts[key]):
            slide = prs.slides.add_slide(prs.slide_layouts[5]) # Title Only
            slide.shapes.title.text = slide_title
            
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
    
    # 1. Run Action Planning
    crit_chronic = generate_action_lists(df, logger)
    
    # 2. Generate Core Visuals
    charts = generate_visuals(df, logger)
    
    # 3. Generate Specialized Aggregate Plot (STOLEN FEATURE)
    aggregate_plot_path = plot_aggregate_risk_by_type(df, logger)
    if aggregate_plot_path:
        charts['aggregate_risk'] = aggregate_plot_path # Update charts dictionary
    
    # 4. Generate Reports
    create_excel_report(df, crit_chronic, logger)
    create_pptx_presentation(df, charts, logger)
    
    logger.info("")
    logger.info("[SUCCESS] Reporting & Visualization Complete.")

if __name__ == "__main__":
    main()
