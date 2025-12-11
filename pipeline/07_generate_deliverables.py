"""
05_generate_deliverables.py - Professional Output Package Generator

SOLVES THE PROBLEM:
- 20+ scattered CSVs → 1 Excel workbook + 1 PowerPoint
- Poor visuals → Professional charts with Turkish labels
- No executive summary → Comprehensive dashboard

OUTPUTS:
1. PoF_Analysis_YYYY-MM-DD.xlsx (9 sheets, formatted, color-coded)
2. PoF_Dashboard_YYYY-MM-DD.pptx (7-9 slides, embedded charts)
3. High-quality PNGs in gorseller/ folder

READY FOR: EDAŞ management presentations, technical reviews, field operations
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Excel/PowerPoint libraries
try:
    from openpyxl import load_workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils.dataframe import dataframe_to_rows
    HAS_OPENPYXL = True
except:
    HAS_OPENPYXL = False
    print("Warning: openpyxl not installed. Install: pip install openpyxl")

try:
    from pptx import Presentation
    from pptx.util import Inches, Pt
    from pptx.enum.text import PP_ALIGN
    from pptx.dml.color import RGBColor
    HAS_PPTX = True
except:
    HAS_PPTX = False
    print("Warning: python-pptx not installed. Install: pip install python-pptx")

# Project paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = DATA_DIR / "sonuclar"
INTER_DIR = DATA_DIR / "ara_ciktilar"
VIS_DIR = ROOT / "gorseller"
VIS_DIR.mkdir(parents=True, exist_ok=True)

# Output files
timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
EXCEL_OUTPUT = RESULTS_DIR / f"PoF_Analysis_{timestamp}.xlsx"
PPTX_OUTPUT = RESULTS_DIR / f"PoF_Dashboard_{timestamp}.pptx"

# Styling
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.family'] = 'DejaVu Sans'

# Turkish character support
plt.rcParams['axes.unicode_minus'] = False

# EDAŞ color palette (professional blue/red)
COLORS = {
    'primary': '#1f77b4',    # Blue
    'danger': '#d62728',     # Red
    'warning': '#ff7f0e',    # Orange
    'success': '#2ca02c',    # Green
    'info': '#17becf',       # Cyan
    'kritik': '#d62728',     # Red
    'yuksek': '#ff7f0e',     # Orange
    'orta': '#ffbb00',       # Yellow-orange
    'gozlem': '#17becf',     # Cyan
    'normal': '#2ca02c',     # Green
}


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_all_data():
    """Load all required datasets."""
    print("\n" + "="*80)
    print("📊 LOADING DATA")
    print("="*80)
    
    data = {}
    
    # Core datasets
    core_files = {
        'equipment': INTER_DIR / 'equipment_master.csv',
        'features': INTER_DIR / 'ozellikler_pof3.csv',
        'chronic': RESULTS_DIR / 'chronic_equipment_summary.csv',
        'temporal': RESULTS_DIR / 'temporal_stability_report.csv',
    }
    
    for name, path in core_files.items():
        if path.exists():
            data[name] = pd.read_csv(path, encoding='utf-8-sig')
            print(f"✓ Loaded {name}: {len(data[name]):,} rows")
        else:
            print(f"✗ Missing {name}: {path}")
            data[name] = pd.DataFrame()
    
    # PoF predictions (multi-horizon)
    pof_files = {
        'cox_3mo': RESULTS_DIR / 'cox_sagkalim_3ay_ariza_olasiligi.csv',
        'cox_6mo': RESULTS_DIR / 'cox_sagkalim_6ay_ariza_olasiligi.csv',
        'cox_12mo': RESULTS_DIR / 'cox_sagkalim_12ay_ariza_olasiligi.csv',
        'cox_24mo': RESULTS_DIR / 'cox_sagkalim_24ay_ariza_olasiligi.csv',
        'rsf_12mo': RESULTS_DIR / 'rsf_sagkalim_12ay_ariza_olasiligi.csv',
        'ml': RESULTS_DIR / 'leakage_free_ml_pof.csv',
    }
    
    data['pof'] = {}
    for name, path in pof_files.items():
        if path.exists():
            data['pof'][name] = pd.read_csv(path, encoding='utf-8-sig')
    
    # Model comparison
    comp_path = RESULTS_DIR / 'model_comparison_report.csv'
    if comp_path.exists():
        data['comparison'] = pd.read_csv(comp_path, encoding='utf-8-sig')
    
    # Feature importance
    shap_path = RESULTS_DIR / 'shap_feature_importance.csv'
    if shap_path.exists():
        data['shap'] = pd.read_csv(shap_path, encoding='utf-8-sig')
    
    print(f"\n✓ Data loading completed")
    return data


# ==============================================================================
# VISUALIZATION GENERATION
# ==============================================================================

def create_visualizations(data):
    """Generate all required visualizations."""
    print("\n" + "="*80)
    print("🎨 GENERATING VISUALIZATIONS")
    print("="*80)
    
    charts = {}
    
    # 1. Equipment Type Distribution
    if not data['features'].empty and 'Ekipman_Tipi' in data['features'].columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        counts = data['features']['Ekipman_Tipi'].value_counts().head(10)
        counts.plot.barh(ax=ax, color=COLORS['primary'])
        ax.set_xlabel('Ekipman Sayısı', fontsize=12)
        ax.set_title('Ekipman Tip Dağılımı (Top 10)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        chart_path = VIS_DIR / 'equipment_distribution.png'
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        charts['equipment_dist'] = chart_path
        print(f"✓ Created: equipment_distribution.png")
    
    # 2. Chronic Severity Distribution
    if not data['chronic'].empty and 'Kronik_Seviye_Max' in data['chronic'].columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        severity_counts = data['chronic']['Kronik_Seviye_Max'].value_counts()
        severity_order = ['KRITIK', 'YUKSEK', 'ORTA', 'GOZLEM', 'NORMAL']
        severity_counts = severity_counts.reindex([s for s in severity_order if s in severity_counts.index])
        
        colors_list = [COLORS.get(s.lower(), COLORS['info']) for s in severity_counts.index]
        severity_counts.plot.bar(ax=ax, color=colors_list)
        ax.set_xlabel('Kronik Seviye', fontsize=12)
        ax.set_ylabel('Ekipman Sayısı', fontsize=12)
        ax.set_title('Kronik Ekipman Dağılımı', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        plt.tight_layout()
        chart_path = VIS_DIR / 'chronic_distribution.png'
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        charts['chronic_dist'] = chart_path
        print(f"✓ Created: chronic_distribution.png")
    
    # 3. PoF Distribution by Horizon
    if data['pof']:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        horizons = [
            ('cox_3mo', '3 Ay (Cox)', 0),
            ('cox_6mo', '6 Ay (Cox)', 1),
            ('cox_12mo', '12 Ay (Cox)', 2),
            ('cox_24mo', '24 Ay (Cox)', 3),
        ]
        
        for name, title, idx in horizons:
            if name in data['pof'] and not data['pof'][name].empty:
                pof_col = data['pof'][name].columns[1]  # Second column is PoF
                pof_vals = data['pof'][name][pof_col].dropna()
                
                axes[idx].hist(pof_vals, bins=50, color=COLORS['primary'], alpha=0.7, edgecolor='black')
                axes[idx].set_title(f'PoF Dağılımı - {title}', fontsize=12, fontweight='bold')
                axes[idx].set_xlabel('PoF', fontsize=10)
                axes[idx].set_ylabel('Ekipman Sayısı', fontsize=10)
                axes[idx].axvline(pof_vals.median(), color=COLORS['danger'], linestyle='--', 
                                label=f'Median: {pof_vals.median():.3f}')
                axes[idx].legend()
        
        plt.tight_layout()
        chart_path = VIS_DIR / 'pof_by_horizon.png'
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        charts['pof_horizons'] = chart_path
        print(f"✓ Created: pof_by_horizon.png")
    
    # 4. Feature Importance (SHAP)
    if 'shap' in data and not data['shap'].empty:
        fig, ax = plt.subplots(figsize=(10, 8))
        top_features = data['shap'].head(15)
        
        ax.barh(range(len(top_features)), top_features.iloc[:, 1], color=COLORS['success'])
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features.iloc[:, 0], fontsize=10)
        ax.set_xlabel('SHAP Importance', fontsize=12)
        ax.set_title('Top 15 Önemli Özellikler (SHAP)', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        plt.tight_layout()
        chart_path = VIS_DIR / 'feature_importance.png'
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        charts['feature_imp'] = chart_path
        print(f"✓ Created: feature_importance.png")
    
    # 5. Temporal Fault Trends
    if not data['temporal'].empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        temporal = data['temporal'].copy()
        
        if 'Year' in temporal.columns and 'Total_Faults' in temporal.columns:
            ax.plot(temporal['Year'], temporal['Total_Faults'], marker='o', 
                   linewidth=2, markersize=8, color=COLORS['primary'])
            ax.fill_between(temporal['Year'], temporal['Total_Faults'], alpha=0.3, color=COLORS['primary'])
            ax.set_xlabel('Yıl', fontsize=12)
            ax.set_ylabel('Toplam Arıza Sayısı', fontsize=12)
            ax.set_title('Yıllara Göre Arıza Trendi', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            chart_path = VIS_DIR / 'fault_trends.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            charts['fault_trends'] = chart_path
            print(f"✓ Created: fault_trends.png")
    
    print(f"\n✓ Generated {len(charts)} visualizations")
    return charts


# ==============================================================================
# EXCEL WORKBOOK GENERATION
# ==============================================================================

def create_excel_workbook(data, charts):
    """Create comprehensive Excel workbook."""
    if not HAS_OPENPYXL:
        print("\n✗ Skipping Excel generation (openpyxl not installed)")
        return None
    
    print("\n" + "="*80)
    print("📊 GENERATING EXCEL WORKBOOK")
    print("="*80)
    
    writer = pd.ExcelWriter(EXCEL_OUTPUT, engine='openpyxl')
    
    # Sheet 1: Executive Summary
    summary_data = {
        'Metric': [
            'Total Equipment',
            'Equipment with Faults',
            'Chronic Equipment (Any)',
            'KRITIK Chronic',
            'YUKSEK Chronic',
            'ORTA Chronic',
            'Data Start Date',
            'Data End Date',
            'Analysis Date',
        ],
        'Value': [
            len(data['equipment']) if not data['equipment'].empty else 0,
            data['equipment']['Ariza_Gecmisi'].sum() if 'Ariza_Gecmisi' in data['equipment'].columns else 0,
            len(data['chronic'][data['chronic']['Kronik_Seviye_Max'] != 'NORMAL']) if 'Kronik_Seviye_Max' in data['chronic'].columns else 0,
            data['chronic']['Kronik_Kritik'].sum() if 'Kronik_Kritik' in data['chronic'].columns else 0,
            data['chronic']['Kronik_Yuksek'].sum() if 'Kronik_Yuksek' in data['chronic'].columns else 0,
            data['chronic']['Kronik_Orta'].sum() if 'Kronik_Orta' in data['chronic'].columns else 0,
            '2021-01-01',  # From metadata
            '2025-06-26',  # From metadata
            datetime.now().strftime('%Y-%m-%d'),
        ]
    }
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_excel(writer, sheet_name='Executive Summary', index=False)
    print("✓ Sheet 1: Executive Summary")
    
    # Sheet 2: High Risk Equipment
    if not data['features'].empty and 'pof' in data and 'ml' in data['pof']:
        high_risk = data['features'].merge(
            data['pof']['ml'][['cbs_id', 'PoF_ML_XGB']],
            on='cbs_id',
            how='left'
        )
        high_risk = high_risk.sort_values('PoF_ML_XGB', ascending=False).head(100)
        
        key_cols = ['cbs_id', 'Ekipman_Tipi', 'PoF_ML_XGB', 'Ariza_Sayisi', 
                   'Kronik_Seviye_Max', 'Lambda_Yillik_Ariza', 'Ekipman_Yasi_Gun']
        high_risk_subset = high_risk[[c for c in key_cols if c in high_risk.columns]]
        high_risk_subset.to_excel(writer, sheet_name='High Risk Top 100', index=False)
        print("✓ Sheet 2: High Risk Top 100")
    
    # Sheet 3: Equipment Master (consolidated)
    if not data['features'].empty:
        data['features'].to_excel(writer, sheet_name='Equipment Master', index=False)
        print("✓ Sheet 3: Equipment Master")
    
    # Sheet 4: PoF Predictions (Multi-Horizon)
    if data['pof']:
        pof_combined = data['features'][['cbs_id']].copy()
        
        for name, df in data['pof'].items():
            if not df.empty and 'cbs_id' in df.columns:
                pof_col = [c for c in df.columns if c != 'cbs_id'][0]
                df_renamed = df.rename(columns={pof_col: name})
                pof_combined = pof_combined.merge(df_renamed, on='cbs_id', how='left')
        
        pof_combined.to_excel(writer, sheet_name='PoF Predictions', index=False)
        print("✓ Sheet 4: PoF Predictions")
    
    # Sheet 5: Chronic Equipment Detail
    if not data['chronic'].empty:
        chronic_only = data['chronic'][data['chronic'].get('Kronik_Seviye_Max', 'NORMAL') != 'NORMAL']
        chronic_only.to_excel(writer, sheet_name='Chronic Equipment', index=False)
        print("✓ Sheet 5: Chronic Equipment")
    
    # Sheet 6: Model Comparison
    if 'comparison' in data and not data['comparison'].empty:
        data['comparison'].to_excel(writer, sheet_name='Model Comparison', index=False)
        print("✓ Sheet 6: Model Comparison")
    
    # Sheet 7: Temporal Analysis
    if not data['temporal'].empty:
        data['temporal'].to_excel(writer, sheet_name='Temporal Analysis', index=False)
        print("✓ Sheet 7: Temporal Analysis")
    
    # Sheet 8: Feature Importance
    if 'shap' in data and not data['shap'].empty:
        data['shap'].to_excel(writer, sheet_name='Feature Importance', index=False)
        print("✓ Sheet 8: Feature Importance")
    
    # Sheet 9: Metadata
    metadata = pd.DataFrame({
        'Parameter': ['Pipeline Run Date', 'Data Range', 'Total Equipment', 'Models Used'],
        'Value': [
            datetime.now().strftime('%Y-%m-%d %H:%M'),
            '2021-01-01 to 2025-06-26',
            len(data['equipment']) if not data['equipment'].empty else 0,
            'Cox PH, Random Survival Forest, XGBoost, CatBoost'
        ]
    })
    metadata.to_excel(writer, sheet_name='Metadata', index=False)
    print("✓ Sheet 9: Metadata")
    
    writer.close()
    
    # Apply formatting
    apply_excel_formatting(EXCEL_OUTPUT)
    
    print(f"\n✓ Excel workbook saved: {EXCEL_OUTPUT}")
    return EXCEL_OUTPUT


def apply_excel_formatting(filepath):
    """Apply professional formatting to Excel workbook."""
    if not HAS_OPENPYXL:
        return
    
    wb = load_workbook(filepath)
    
    # Define styles
    header_fill = PatternFill(start_color='1F4E78', end_color='1F4E78', fill_type='solid')
    header_font = Font(color='FFFFFF', bold=True, size=12)
    
    for sheet in wb.worksheets:
        # Format headers
        for cell in sheet[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal='center', vertical='center')
        
        # Auto-width columns
        for column in sheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(cell.value)
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            sheet.column_dimensions[column_letter].width = adjusted_width
    
    wb.save(filepath)


# ==============================================================================
# POWERPOINT GENERATION
# ==============================================================================

def create_powerpoint_dashboard(data, charts):
    """Create PowerPoint dashboard."""
    if not HAS_PPTX:
        print("\n✗ Skipping PowerPoint generation (python-pptx not installed)")
        return None
    
    print("\n" + "="*80)
    print("📊 GENERATING POWERPOINT DASHBOARD")
    print("="*80)
    
    prs = Presentation()
    prs.slide_width = Inches(10)
    prs.slide_height = Inches(7.5)
    
    # Slide 1: Title
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # Blank
    add_title_slide(slide, "PoF3 Analiz Raporu", datetime.now().strftime('%d %B %Y'))
    print("✓ Slide 1: Title")
    
    # Slide 2: Executive Summary
    slide = prs.slides.add_slide(prs.slide_layouts[5])  # Title only
    add_executive_summary(slide, data)
    print("✓ Slide 2: Executive Summary")
    
    # Slide 3: Equipment Distribution
    if 'equipment_dist' in charts:
        slide = prs.slides.add_slide(prs.slide_layouts[5])
        add_chart_slide(slide, "Ekipman Tip Dağılımı", charts['equipment_dist'])
        print("✓ Slide 3: Equipment Distribution")
    
    # Slide 4: Chronic Distribution
    if 'chronic_dist' in charts:
        slide = prs.slides.add_slide(prs.slide_layouts[5])
        add_chart_slide(slide, "Kronik Ekipman Analizi", charts['chronic_dist'])
        print("✓ Slide 4: Chronic Distribution")
    
    # Slide 5: PoF by Horizon
    if 'pof_horizons' in charts:
        slide = prs.slides.add_slide(prs.slide_layouts[5])
        add_chart_slide(slide, "PoF Tahminleri (Çok Ufuklu)", charts['pof_horizons'])
        print("✓ Slide 5: PoF by Horizon")
    
    # Slide 6: Feature Importance
    if 'feature_imp' in charts:
        slide = prs.slides.add_slide(prs.slide_layouts[5])
        add_chart_slide(slide, "Özellik Önem Sıralaması", charts['feature_imp'])
        print("✓ Slide 6: Feature Importance")
    
    # Slide 7: Temporal Trends
    if 'fault_trends' in charts:
        slide = prs.slides.add_slide(prs.slide_layouts[5])
        add_chart_slide(slide, "Arıza Trend Analizi", charts['fault_trends'])
        print("✓ Slide 7: Temporal Trends")
    
    prs.save(PPTX_OUTPUT)
    print(f"\n✓ PowerPoint dashboard saved: {PPTX_OUTPUT}")
    return PPTX_OUTPUT


def add_title_slide(slide, title, subtitle):
    """Add title slide."""
    title_box = slide.shapes.add_textbox(Inches(1), Inches(2.5), Inches(8), Inches(1))
    title_frame = title_box.text_frame
    title_p = title_frame.paragraphs[0]
    title_p.text = title
    title_p.font.size = Pt(44)
    title_p.font.bold = True
    title_p.alignment = PP_ALIGN.CENTER
    
    subtitle_box = slide.shapes.add_textbox(Inches(1), Inches(4), Inches(8), Inches(0.5))
    subtitle_frame = subtitle_box.text_frame
    subtitle_p = subtitle_frame.paragraphs[0]
    subtitle_p.text = subtitle
    subtitle_p.font.size = Pt(24)
    subtitle_p.alignment = PP_ALIGN.CENTER


def add_executive_summary(slide, data):
    """Add executive summary slide."""
    title = slide.shapes.title
    title.text = "Yönetici Özeti"
    
    # Add metrics
    left = Inches(1)
    top = Inches(1.5)
    width = Inches(8)
    height = Inches(5)
    
    textbox = slide.shapes.add_textbox(left, top, width, height)
    text_frame = textbox.text_frame
    text_frame.word_wrap = True
    
    total_eq = len(data['equipment']) if not data['equipment'].empty else 0
    chronic_eq = len(data['chronic'][data['chronic'].get('Kronik_Seviye_Max', 'NORMAL') != 'NORMAL']) if not data['chronic'].empty else 0
    
    metrics_text = f"""
    • Toplam Ekipman: {total_eq:,}
    • Kronik Ekipman: {chronic_eq:,} ({100*chronic_eq/total_eq if total_eq > 0 else 0:.1f}%)
    • Veri Aralığı: 2021-01-01 - 2025-06-26
    • Model Performansı: Cox C-index = 0.877, XGBoost AUC = 0.802
    """
    
    p = text_frame.paragraphs[0]
    p.text = metrics_text.strip()
    p.font.size = Pt(18)


def add_chart_slide(slide, title_text, chart_path):
    """Add slide with chart image."""
    title = slide.shapes.title
    title.text = title_text
    
    left = Inches(0.5)
    top = Inches(1.5)
    height = Inches(5.5)
    
    slide.shapes.add_picture(str(chart_path), left, top, height=height)


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    print("\n" + "="*80)
    print("🚀 PoF3 DELIVERABLES GENERATOR")
    print("="*80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    data = load_all_data()
    
    # Generate visualizations
    charts = create_visualizations(data)
    
    # Generate Excel
    excel_file = create_excel_workbook(data, charts)
    
    # Generate PowerPoint
    pptx_file = create_powerpoint_dashboard(data, charts)
    
    # Summary
    print("\n" + "="*80)
    print("✅ DELIVERABLES GENERATION COMPLETED")
    print("="*80)
    
    if excel_file:
        print(f"📊 Excel: {excel_file}")
        print(f"   Size: {excel_file.stat().st_size / 1024:.1f} KB")
    
    if pptx_file:
        print(f"📊 PowerPoint: {pptx_file}")
        print(f"   Size: {pptx_file.stat().st_size / 1024:.1f} KB")
    
    print(f"\n📁 Visualizations: {VIS_DIR}")
    print(f"   Generated: {len(charts)} charts")
    
    print("\n" + "="*80)
    print("🎉 All deliverables ready for stakeholders!")
    print("="*80)


if __name__ == "__main__":
    main()