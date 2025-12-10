"""
================================================================================
PoF3 HYBRID VISUAL SUITE
================================================================================
Combines:
- PoF3 scientific survival visuals (KM, hazard, distribution)
- Early-pipeline strong visuals (EDA dashboard, feature importance, load proxies)
- Risk quadrant (if CoF exists)
================================================================================
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Optional (geospatial)
try:
    import folium
    from folium.plugins import HeatMap
    GEO = True
except:
    GEO = False

sns.set(style="whitegrid")
plt.rcParams['figure.dpi'] = 150

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "data" / "sonuclar"
INTER_DIR = ROOT / "data" / "ara_ciktilar"
VIS_DIR = ROOT / "gorseller"
VIS_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load PoF predictions + features."""
    pof_file = RESULTS_DIR / "cox_sagkalim_12ay_ariza_olasiligi.csv"
    rsf_file = RESULTS_DIR / "rsf_sagkalim_12ay_ariza_olasiligi.csv"
    feat_file = INTER_DIR / "ozellikler_pof3.csv"

    pof_df = pd.read_csv(pof_file)
    rsf_df = pd.read_csv(rsf_file) if rsf_file.exists() else None
    feat_df = pd.read_csv(feat_file)

    # Merge PoF
    pof_df.rename(columns={"cbs_id": "cbs_id", pof_df.columns[1]: "PoF_12M_Cox"}, inplace=True)
    if rsf_df is not None:
        rsf_df.rename(columns={rsf_df.columns[0]: "cbs_id", rsf_df.columns[1]: "PoF_12M_RSF"}, inplace=True)
        pof_df = pof_df.merge(rsf_df, on="cbs_id", how="left")

    # Merge features
    full = feat_df.merge(pof_df, on="cbs_id", how="left")
    return full


# --------------------------------------------------------------------------
# 1) EDA DASHBOARD  (Early Pipeline Style)
# --------------------------------------------------------------------------

def plot_eda_dashboard(df):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Equipment Type Distribution
    df['Şebeke Unsuru'].value_counts().head(10).plot.bar(ax=axes[0,0], color='steelblue')
    axes[0,0].set_title("Ekipman Tip Dağılımı")

    # Age Distribution
    sns.histplot(df['Ekipman_Yasi_Gun'] / 365, bins=30, ax=axes[0,1], color='salmon')
    axes[0,1].set_title("Ekipman Yaşı (Yıl)")

    # MTBF Distribution
    sns.histplot(df['MTBF_Gun'], bins=30, ax=axes[0,2], color='seagreen')
    axes[0,2].set_title("MTBF Dağılımı")

    # Failure count by class
    sns.boxplot(data=df, x='Şebeke Unsuru', y='Ariza_Sayisi', ax=axes[1,0])
    axes[1,0].set_title("Arıza Sayısı – Ekipman Tipi")

    # Region-based (if exists)
    if 'ilce' in df.columns:
        df['ilce'].value_counts().head(15).plot.bar(ax=axes[1,1], color='purple')
        axes[1,1].set_title("İlçe Dağılımı")

    # Loading proxy (if exists)
    if 'Yuklenme_Proxy' in df.columns:
        sns.histplot(df['Yuklenme_Proxy'], ax=axes[1,2], color='darkorange')
        axes[1,2].set_title("Yüklenme Proxy Skoru")

    plt.tight_layout()
    plt.savefig(VIS_DIR / "01_eda_dashboard.png", dpi=300)
    plt.close()


# --------------------------------------------------------------------------
# 2) FEATURE IMPORTANCE (Early Pipeline Style)
# --------------------------------------------------------------------------

def plot_feature_importance(df):
    numeric_cols = [
        'Ekipman_Yasi_Gun', 'Ariza_Sayisi', 'MTBF_Gun', 'Bakim_Sayisi',
        'Son_Bakimdan_Gecen_Gun', 'Son_Ariza_Gun_Sayisi'
    ]

    imp = df[numeric_cols].corrwith(df['PoF_12M_Cox']).abs().sort_values(ascending=False)

    plt.figure(figsize=(8,6))
    sns.barplot(x=imp.values, y=imp.index, palette="viridis")
    plt.title("Feature Importance (Correlation with PoF - Cox)")
    plt.tight_layout()
    plt.savefig(VIS_DIR / "02_feature_importance.png", dpi=300)
    plt.close()


# --------------------------------------------------------------------------
# 3) SURVIVAL VISUALS (PoF3 Scientific Layer)
# --------------------------------------------------------------------------

def plot_pof_distribution(df):
    plt.figure(figsize=(7,5))
    sns.histplot(df['PoF_12M_Cox'], bins=30, color='royalblue')
    plt.title("PoF (12 Ay - Cox) Dağılımı")
    plt.tight_layout()
    plt.savefig(VIS_DIR / "03_pof_distribution_cox.png", dpi=300)
    plt.close()

    if 'PoF_12M_RSF' in df.columns:
        plt.figure(figsize=(7,5))
        sns.histplot(df['PoF_12M_RSF'], bins=30, color='darkred')
        plt.title("PoF (12 Ay - RSF) Dağılımı")
        plt.tight_layout()
        plt.savefig(VIS_DIR / "04_pof_distribution_rsf.png", dpi=300)
        plt.close()


# --------------------------------------------------------------------------
# 4) CHRONIC vs NONCHRONIC VISUALIZATION
# --------------------------------------------------------------------------

def plot_chronic_comparison(df):
    if 'Kronik_Flag' not in df.columns:
        return

    plt.figure(figsize=(7,5))
    sns.boxplot(data=df, x='Kronik_Flag', y='PoF_12M_Cox')
    plt.title("Kronik vs Non-Kronik – PoF (12 Ay Cox)")
    plt.tight_layout()
    plt.savefig(VIS_DIR / "05_chronic_vs_nonchronic.png", dpi=300)
    plt.close()


# --------------------------------------------------------------------------
# 5) RISK QUADRANT (if CoF exists)
# --------------------------------------------------------------------------

def plot_risk_quadrant(df):
    if "CoF" not in df.columns:
        return

    df = df.dropna(subset=['PoF_12M_Cox', 'CoF'])
    plt.figure(figsize=(7,6))
    plt.scatter(df['PoF_12M_Cox'], df['CoF'], s=12, alpha=0.5, color='slateblue')
    plt.xlabel("PoF (12 Ay – Cox)")
    plt.ylabel("CoF (TL)")
    plt.title("Risk Matrisi (PoF × CoF)")
    plt.axvline(df['PoF_12M_Cox'].median(), linestyle='--', color='gray')
    plt.axhline(df['CoF'].median(), linestyle='--', color='gray')
    plt.tight_layout()
    plt.savefig(VIS_DIR / "06_risk_quadrant.png", dpi=300)
    plt.close()


# --------------------------------------------------------------------------
# 6) OPTIONAL GEOSPATIAL HEATMAP (if lat/lon exists)
# --------------------------------------------------------------------------

def plot_geospatial(df):
    if not GEO:
        return
    if not all(col in df.columns for col in ["lat", "lon"]):
        return

    m = folium.Map(location=[df['lat'].mean(), df['lon'].mean()], zoom_start=11)
    heat_data = df[['lat','lon','PoF_12M_Cox']].dropna().values.tolist()
    HeatMap(heat_data, radius=12).add_to(m)
    m.save(str(VIS_DIR / "07_geospatial_pof_map.html"))


# --------------------------------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------------------------------

def main():
    df = load_data()

    print("📊 Creating hybrid visual suite...")

    plot_eda_dashboard(df)
    plot_feature_importance(df)
    plot_pof_distribution(df)
    plot_chronic_comparison(df)
    plot_risk_quadrant(df)
    plot_geospatial(df)

    print("\n🎉 Hybrid visual suite completed!")
    print(f"📁 Outputs saved to: {VIS_DIR}")


if __name__ == "__main__":
    main()
