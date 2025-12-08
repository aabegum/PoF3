"""
Turkish translation mappings for PoF3 output tables

This module provides comprehensive translation dictionaries to convert
technical/internal column names to customer-facing Turkish labels.

Usage:
    from utils.translations import translate_columns, COLUMN_TRANSLATIONS

    # Translate a DataFrame's columns
    df_turkish = translate_columns(df, 'risk_assessment')
"""

import pandas as pd


# Complete translation dictionary for all output tables
COLUMN_TRANSLATIONS = {
    # Technical feature names → Turkish customer-facing names
    "technical_to_turkish": {
        # Equipment identifiers
        "cbs_id": "CBS_ID",
        "CBS_ID": "CBS_ID",  # Already Turkish

        # Equipment characteristics
        "Ekipman_Tipi": "Ekipman_Sınıfı",
        "Equipment_Class": "Ekipman_Sınıfı",
        "Kurulum_Tarihi": "Kurulum_Tarihi",
        "Ekipman_Yasi_Gun": "Ekipman_Yaşı_Gün",

        # Fault history
        "Has_Ariza_Gecmisi": "Arıza_Geçmişi_Var",
        "Ariza_Gecmisine_Sahip": "Arıza_Geçmişi_Var",
        "Has_Failed": "Arıza_Yaptı",
        "Fault_Count": "Toplam_Arıza_Sayısı",
        "Toplam_Ariza_Sayisi": "Toplam_Arıza_Sayısı",

        # Fault dates and timing
        "Ilk_Ariza_Tarihi": "İlk_Arıza_Tarihi",
        "Son_Ariza_Tarihi": "Son_Arıza_Tarihi",
        "Ariza_Baslangic_Zamani": "Arıza_Başlangıç_Zamanı",
        "Ariza_Bitis_Zamani": "Arıza_Bitiş_Zamanı",
        "Son_Ariza_Gun_Sayisi": "Son_Arızadan_Geçen_Gün",

        # Duration metrics
        "MTBF_Gun": "Ortalama_Arıza_Arası_Süre_Gün",
        "Kesinti_Suresi_Dakika": "Kesinti_Süresi_Dakika",
        "duration_days": "Süre_Gün",
        "Sure_Gun": "Süre_Gün",

        # Survival analysis fields
        "event": "Arıza_Durumu",
        "Olay": "Arıza_Durumu",

        # Chronic risk flags
        "Tekrarlayan_Ariza_90g_Flag": "Tekrarlayan_Arıza_90Gün_Bayrak",
        "Chronic_90d_Flag": "Tekrarlayan_Arıza_90Gün_Bayrak",

        # Risk assessment outputs (for Step 05)
        "PoF_Score": "Arıza_Olasılığı_Skoru",
        "PoF_Category": "Arıza_Olasılığı_Kategorisi",
        "Risk_Score": "Risk_Skoru",
        "Risk_Category": "Risk_Kategorisi",
        "Hazard_Score": "Tehlike_Skoru",
        "Survival_Probability": "Hayatta_Kalma_Olasılığı",

        # Fault causes
        "Ariza_Nedeni": "Arıza_Nedeni",
        "cause_code": "Arıza_Nedeni_Kodu",
    },

    # Value translations (e.g., for categorical fields)
    "value_translations": {
        "Risk_Category": {
            "Low": "Düşük",
            "Medium": "Orta",
            "High": "Yüksek",
            "Critical": "Kritik",
        },
        "PoF_Category": {
            "Low": "Düşük",
            "Medium": "Orta",
            "High": "Yüksek",
            "Very High": "Çok Yüksek",
        },
    }
}


def translate_columns(df: pd.DataFrame, table_type: str = "general") -> pd.DataFrame:
    """
    Translate DataFrame column names to Turkish customer-facing labels.

    Args:
        df: DataFrame to translate
        table_type: Type of table (currently uses 'technical_to_turkish' for all)

    Returns:
        DataFrame with translated column names

    Example:
        >>> df_risk = pd.DataFrame({'cbs_id': [1, 2], 'PoF_Score': [0.5, 0.8]})
        >>> df_turkish = translate_columns(df_risk, 'risk_assessment')
        >>> df_turkish.columns
        Index(['CBS_ID', 'Arıza_Olasılığı_Skoru'], dtype='object')
    """
    translation_map = COLUMN_TRANSLATIONS["technical_to_turkish"]

    # Only translate columns that exist in the translation map
    rename_dict = {
        col: translation_map[col]
        for col in df.columns
        if col in translation_map
    }

    df_translated = df.rename(columns=rename_dict)
    return df_translated


def translate_values(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Translate categorical values in a specific column to Turkish.

    Args:
        df: DataFrame containing the column
        column: Name of column to translate values for

    Returns:
        DataFrame with translated values in the specified column

    Example:
        >>> df = pd.DataFrame({'Risk_Category': ['Low', 'High', 'Medium']})
        >>> df = translate_values(df, 'Risk_Category')
        >>> df['Risk_Category'].tolist()
        ['Düşük', 'Yüksek', 'Orta']
    """
    if column not in df.columns:
        return df

    value_map = COLUMN_TRANSLATIONS["value_translations"].get(column, {})
    if value_map:
        df = df.copy()
        df[column] = df[column].map(value_map).fillna(df[column])

    return df


def get_translation_report() -> str:
    """
    Generate a human-readable report of all available translations.

    Returns:
        Formatted string containing all translation mappings
    """
    report = []
    report.append("=" * 80)
    report.append("PoF3 Translation Dictionary")
    report.append("=" * 80)
    report.append("\nCOLUMN TRANSLATIONS (Technical → Turkish):\n")

    for eng, tur in sorted(COLUMN_TRANSLATIONS["technical_to_turkish"].items()):
        if eng != tur:  # Skip identity mappings
            report.append(f"  {eng:<40} → {tur}")

    report.append("\n" + "=" * 80)
    report.append("VALUE TRANSLATIONS:\n")

    for column, value_map in COLUMN_TRANSLATIONS["value_translations"].items():
        report.append(f"\n  {column}:")
        for eng_val, tur_val in value_map.items():
            report.append(f"    {eng_val:<20} → {tur_val}")

    report.append("\n" + "=" * 80)
    return "\n".join(report)


if __name__ == "__main__":
    # Print translation report when run directly
    print(get_translation_report())
