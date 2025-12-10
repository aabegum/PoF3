"""
Common Data Processing Utilities for PoF3 Pipeline

Bu modül, pipeline genelinde tekrar eden veri işleme fonksiyonlarını içerir.
"""

import logging
from typing import Optional, Tuple
import pandas as pd
import numpy as np


def normalize_column_names(*dataframes: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
    """
    DataFrame kolonlarını normalize eder (boşluk, özel karakter temizleme).

    Args:
        *dataframes: Normalize edilecek DataFrame'ler

    Returns:
        Tuple of normalized DataFrames
    """
    normalized = []

    for df in dataframes:
        df_copy = df.copy()
        df_copy.columns = df_copy.columns.str.strip().str.replace(r'\s+', '_', regex=True)
        normalized.append(df_copy)

    if len(normalized) == 1:
        return normalized[0]
    return tuple(normalized)


def standardize_id_column(df: pd.DataFrame, id_col: str = "cbs_id") -> pd.DataFrame:
    """
    ID kolonunu standardize eder (küçük harf, boşluk temizleme).

    Args:
        df: DataFrame
        id_col: ID kolonu ismi

    Returns:
        Standardized DataFrame
    """
    if id_col in df.columns:
        df[id_col] = df[id_col].astype(str).str.lower().str.strip()

    return df


def fill_missing_numeric(
    df: pd.DataFrame,
    columns: list,
    strategy: str = "median",
    fill_value: float = 0.0,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Sayısal kolonlardaki eksik değerleri doldurur.

    Args:
        df: DataFrame
        columns: Doldurulacak kolon listesi
        strategy: "median", "mean", "zero", "value"
        fill_value: strategy="value" için kullanılacak değer
        logger: Logger instance (opsiyonel)

    Returns:
        Filled DataFrame
    """
    df_copy = df.copy()

    for col in columns:
        if col not in df_copy.columns:
            continue

        if not df_copy[col].isna().any():
            continue

        if strategy == "median":
            fill_val = df_copy[col].median()
            if pd.isna(fill_val):
                fill_val = 0.0
                if logger:
                    logger.info(f"[INFO] {col}: Median bulunamadı, 0 kullanılıyor")
        elif strategy == "mean":
            fill_val = df_copy[col].mean()
            if pd.isna(fill_val):
                fill_val = 0.0
                if logger:
                    logger.info(f"[INFO] {col}: Mean bulunamadı, 0 kullanılıyor")
        elif strategy == "zero":
            fill_val = 0.0
        elif strategy == "value":
            fill_val = fill_value
        else:
            raise ValueError(f"Geçersiz strategy: {strategy}")

        df_copy[col] = df_copy[col].fillna(fill_val)

        if logger:
            logger.info(f"[INFO] {col}: NaN değerler {fill_val:.2f} ile dolduruldu")

    return df_copy


def remove_constant_columns(
    df: pd.DataFrame,
    exclude_cols: list = None,
    logger: Optional[logging.Logger] = None
) -> Tuple[pd.DataFrame, list]:
    """
    Sabit değerli (varyans = 0) kolonları kaldırır.

    Args:
        df: DataFrame
        exclude_cols: Kontrol edilmeyecek kolonlar
        logger: Logger instance (opsiyonel)

    Returns:
        (cleaned_df, removed_columns)
    """
    if exclude_cols is None:
        exclude_cols = []

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    check_cols = [c for c in numeric_cols if c not in exclude_cols]

    constant_cols = []
    for col in check_cols:
        if df[col].std() == 0:
            constant_cols.append(col)

    if constant_cols:
        if logger:
            logger.warning(f"[WARN] Sabit değerli kolonlar kaldırılıyor: {constant_cols}")
        df = df.drop(columns=constant_cols)

    return df, constant_cols


def group_rare_categories(
    df: pd.DataFrame,
    col: str,
    min_count: int = 30,
    other_label: str = "Other",
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Nadir kategorileri "Other" altında toplar.

    Args:
        df: DataFrame
        col: Kategorik kolon
        min_count: Minimum kategori sayısı
        other_label: Nadir kategoriler için etiket
        logger: Logger instance (opsiyonel)

    Returns:
        Grouped DataFrame
    """
    if col not in df.columns:
        return df

    df_copy = df.copy()
    value_counts = df_copy[col].value_counts()
    rare_categories = value_counts[value_counts < min_count].index.tolist()

    if rare_categories:
        df_copy[col] = df_copy[col].replace(rare_categories, other_label)

        if logger:
            logger.info(f"[INFO] {col}: {len(rare_categories)} nadir kategori '{other_label}' altında toplandı")
            logger.info(f"  Nadir kategoriler: {rare_categories}")

    return df_copy


def convert_voltage_to_numeric(
    df: pd.DataFrame,
    voltage_col: str = "Gerilim_Seviyesi",
    output_col: str = "Gerilim_Seviyesi_Sayisal",
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Gerilim seviyesini sayısal değere çevirir.

    Mapping:
        "0.4kV" -> 0.4
        "34.5kV" -> 34.5
        vb.

    Args:
        df: DataFrame
        voltage_col: Gerilim kolonu
        output_col: Çıktı kolonu
        logger: Logger instance (opsiyonel)

    Returns:
        DataFrame with numeric voltage column
    """
    if voltage_col not in df.columns:
        return df

    df_copy = df.copy()

    # "34.5kV" -> 34.5
    df_copy[output_col] = (
        df_copy[voltage_col]
        .astype(str)
        .str.replace("kV", "", regex=False)
        .str.strip()
    )

    # Convert to numeric
    df_copy[output_col] = pd.to_numeric(df_copy[output_col], errors="coerce")

    if logger:
        logger.info(f"[INFO] {output_col} türetildi (from {voltage_col})")

    return df_copy


def calculate_days_since(
    df: pd.DataFrame,
    date_col: str,
    reference_date: pd.Timestamp,
    output_col: str,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Referans tarihten itibaren geçen gün sayısını hesaplar.

    Args:
        df: DataFrame
        date_col: Tarih kolonu
        reference_date: Referans tarih
        output_col: Çıktı kolonu
        logger: Logger instance (opsiyonel)

    Returns:
        DataFrame with days_since column
    """
    if date_col not in df.columns:
        if logger:
            logger.warning(f"[WARN] {date_col} kolonu bulunamadı, {output_col} oluşturulamadı")
        return df

    df_copy = df.copy()
    df_copy[output_col] = (reference_date - pd.to_datetime(df_copy[date_col])).dt.days

    if logger:
        logger.info(f"[INFO] {output_col} hesaplandı (ref: {reference_date.date()})")

    return df_copy


def aggregate_string_list(
    df: pd.DataFrame,
    group_col: str,
    agg_col: str,
    output_col: str,
    separator: str = ", ",
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Gruplandırılmış string değerleri birleştirir.

    Örnek:
        cbs_id: [A, A, B]
        type: ["Preventive", "Corrective", "Preventive"]
        ->
        cbs_id: [A, B]
        type_list: ["Preventive, Corrective", "Preventive"]

    Args:
        df: DataFrame
        group_col: Gruplama kolonu
        agg_col: Birleştirilecek kolon
        output_col: Çıktı kolonu
        separator: Ayırıcı karakter
        logger: Logger instance (opsiyonel)

    Returns:
        Aggregated DataFrame
    """
    if group_col not in df.columns or agg_col not in df.columns:
        if logger:
            logger.warning(f"[WARN] {group_col} veya {agg_col} bulunamadı")
        return df

    agg_df = (
        df.groupby(group_col)[agg_col]
        .apply(lambda x: separator.join(x.dropna().unique()))
        .reset_index()
        .rename(columns={agg_col: output_col})
    )

    if logger:
        logger.info(f"[INFO] {output_col} oluşturuldu ({group_col} bazında)")

    return agg_df


def safe_merge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: str,
    how: str = "left",
    validate: Optional[str] = None,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Güvenli merge işlemi (duplicate column isimleri için kontrol).

    Args:
        left: Sol DataFrame
        right: Sağ DataFrame
        on: Merge kolonu
        how: Merge tipi
        validate: Merge validation ("1:1", "1:m", "m:1", "m:m")
        logger: Logger instance (opsiyonel)

    Returns:
        Merged DataFrame
    """
    # Check for overlapping columns (except merge key)
    overlap = set(left.columns).intersection(set(right.columns)) - {on}

    if overlap and logger:
        logger.warning(f"[WARN] Merge overlap detected: {overlap}")
        logger.warning("  Pandas will add _x and _y suffixes")

    result = left.merge(right, on=on, how=how, validate=validate)

    if logger:
        logger.info(f"[INFO] Merge completed: {len(left):,} + {len(right):,} -> {len(result):,} rows")

    return result
