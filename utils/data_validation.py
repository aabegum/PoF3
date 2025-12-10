"""
Data Quality Validation Utilities for PoF3 Pipeline

Bu modül, pipeline genelinde kullanılan veri kalitesi kontrol fonksiyonlarını içerir.
"""

import logging
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np


def validate_required_columns(
    df: pd.DataFrame,
    required_cols: List[str],
    df_name: str = "DataFrame",
    logger: Optional[logging.Logger] = None
) -> Tuple[bool, List[str]]:
    """
    DataFrame'de zorunlu kolonların varlığını kontrol eder.

    Args:
        df: Kontrol edilecek DataFrame
        required_cols: Zorunlu kolon listesi
        df_name: DataFrame ismi (log mesajları için)
        logger: Logger instance (opsiyonel)

    Returns:
        (is_valid, missing_cols): Tuple[bool, List[str]]
    """
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        msg = f"[VALIDATION ERROR] {df_name} içinde eksik kolonlar: {missing}"
        if logger:
            logger.error(msg)
        else:
            print(msg)
        return False, missing

    return True, []


def validate_no_nulls(
    df: pd.DataFrame,
    columns: List[str],
    df_name: str = "DataFrame",
    logger: Optional[logging.Logger] = None
) -> Tuple[bool, Dict[str, int]]:
    """
    Belirtilen kolonlarda NULL değer olmadığını kontrol eder.

    Args:
        df: Kontrol edilecek DataFrame
        columns: Kontrol edilecek kolonlar
        df_name: DataFrame ismi
        logger: Logger instance (opsiyonel)

    Returns:
        (is_valid, null_counts): Tuple[bool, Dict[str, int]]
    """
    null_counts = {}
    for col in columns:
        if col in df.columns:
            null_count = df[col].isna().sum()
            if null_count > 0:
                null_counts[col] = null_count

    if null_counts:
        msg = f"[VALIDATION WARNING] {df_name} içinde NULL değerler:\n"
        for col, count in null_counts.items():
            msg += f"  {col}: {count:,} NULL ({count/len(df)*100:.1f}%)\n"

        if logger:
            logger.warning(msg)
        else:
            print(msg)
        return False, null_counts

    return True, {}


def validate_date_range(
    df: pd.DataFrame,
    date_col: str,
    min_date: Optional[pd.Timestamp] = None,
    max_date: Optional[pd.Timestamp] = None,
    df_name: str = "DataFrame",
    logger: Optional[logging.Logger] = None
) -> Tuple[bool, int]:
    """
    Tarih kolonunun geçerli aralıkta olduğunu kontrol eder.

    Args:
        df: Kontrol edilecek DataFrame
        date_col: Tarih kolonu
        min_date: Minimum kabul edilebilir tarih
        max_date: Maximum kabul edilebilir tarih
        df_name: DataFrame ismi
        logger: Logger instance (opsiyonel)

    Returns:
        (is_valid, out_of_range_count): Tuple[bool, int]
    """
    if date_col not in df.columns:
        msg = f"[VALIDATION ERROR] {df_name} içinde '{date_col}' kolonu bulunamadı"
        if logger:
            logger.error(msg)
        else:
            print(msg)
        return False, 0

    dates = pd.to_datetime(df[date_col], errors='coerce')
    invalid_count = dates.isna().sum()

    if invalid_count > 0:
        msg = f"[VALIDATION WARNING] {df_name}.{date_col}: {invalid_count:,} geçersiz tarih"
        if logger:
            logger.warning(msg)
        else:
            print(msg)

    out_of_range = 0

    if min_date is not None:
        below_min = (dates < min_date).sum()
        if below_min > 0:
            out_of_range += below_min
            msg = f"[VALIDATION WARNING] {df_name}.{date_col}: {below_min:,} kayıt minimum tarihten ({min_date.date()}) önce"
            if logger:
                logger.warning(msg)
            else:
                print(msg)

    if max_date is not None:
        above_max = (dates > max_date).sum()
        if above_max > 0:
            out_of_range += above_max
            msg = f"[VALIDATION WARNING] {df_name}.{date_col}: {above_max:,} kayıt maximum tarihten ({max_date.date()}) sonra"
            if logger:
                logger.warning(msg)
            else:
                print(msg)

    return out_of_range == 0, out_of_range


def validate_numeric_range(
    df: pd.DataFrame,
    col: str,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    df_name: str = "DataFrame",
    logger: Optional[logging.Logger] = None
) -> Tuple[bool, int]:
    """
    Sayısal kolonun geçerli aralıkta olduğunu kontrol eder.

    Args:
        df: Kontrol edilecek DataFrame
        col: Sayısal kolon
        min_val: Minimum kabul edilebilir değer
        max_val: Maximum kabul edilebilir değer
        df_name: DataFrame ismi
        logger: Logger instance (opsiyonel)

    Returns:
        (is_valid, out_of_range_count): Tuple[bool, int]
    """
    if col not in df.columns:
        msg = f"[VALIDATION ERROR] {df_name} içinde '{col}' kolonu bulunamadı"
        if logger:
            logger.error(msg)
        else:
            print(msg)
        return False, 0

    out_of_range = 0

    if min_val is not None:
        below_min = (df[col] < min_val).sum()
        if below_min > 0:
            out_of_range += below_min
            msg = f"[VALIDATION WARNING] {df_name}.{col}: {below_min:,} kayıt minimum değerden ({min_val}) küçük"
            if logger:
                logger.warning(msg)
            else:
                print(msg)

    if max_val is not None:
        above_max = (df[col] > max_val).sum()
        if above_max > 0:
            out_of_range += above_max
            msg = f"[VALIDATION WARNING] {df_name}.{col}: {above_max:,} kayıt maximum değerden ({max_val}) büyük"
            if logger:
                logger.warning(msg)
            else:
                print(msg)

    return out_of_range == 0, out_of_range


def validate_no_duplicates(
    df: pd.DataFrame,
    id_col: str,
    df_name: str = "DataFrame",
    logger: Optional[logging.Logger] = None
) -> Tuple[bool, int]:
    """
    ID kolonunda duplicate olmadığını kontrol eder.

    Args:
        df: Kontrol edilecek DataFrame
        id_col: ID kolonu
        df_name: DataFrame ismi
        logger: Logger instance (opsiyonel)

    Returns:
        (is_valid, duplicate_count): Tuple[bool, int]
    """
    if id_col not in df.columns:
        msg = f"[VALIDATION ERROR] {df_name} içinde '{id_col}' kolonu bulunamadı"
        if logger:
            logger.error(msg)
        else:
            print(msg)
        return False, 0

    duplicates = df[id_col].duplicated().sum()

    if duplicates > 0:
        msg = f"[VALIDATION WARNING] {df_name}.{id_col}: {duplicates:,} duplicate kayıt"
        if logger:
            logger.warning(msg)
        else:
            print(msg)
        return False, duplicates

    return True, 0


def validate_categorical_values(
    df: pd.DataFrame,
    col: str,
    valid_values: List[str],
    df_name: str = "DataFrame",
    logger: Optional[logging.Logger] = None
) -> Tuple[bool, int]:
    """
    Kategorik kolonun sadece geçerli değerler içerdiğini kontrol eder.

    Args:
        df: Kontrol edilecek DataFrame
        col: Kategorik kolon
        valid_values: Geçerli değerler listesi
        df_name: DataFrame ismi
        logger: Logger instance (opsiyonel)

    Returns:
        (is_valid, invalid_count): Tuple[bool, int]
    """
    if col not in df.columns:
        msg = f"[VALIDATION ERROR] {df_name} içinde '{col}' kolonu bulunamadı"
        if logger:
            logger.error(msg)
        else:
            print(msg)
        return False, 0

    invalid = ~df[col].isin(valid_values + [np.nan, None, pd.NA])
    invalid_count = invalid.sum()

    if invalid_count > 0:
        invalid_vals = df.loc[invalid, col].unique()
        msg = f"[VALIDATION WARNING] {df_name}.{col}: {invalid_count:,} geçersiz değer\n"
        msg += f"  Geçersiz değerler: {invalid_vals[:10]}"  # İlk 10 göster
        if logger:
            logger.warning(msg)
        else:
            print(msg)
        return False, invalid_count

    return True, 0


def validate_constant_columns(
    df: pd.DataFrame,
    df_name: str = "DataFrame",
    logger: Optional[logging.Logger] = None
) -> Tuple[List[str], List[str]]:
    """
    Sabit değerli (varyans = 0) kolonları tespit eder.

    Args:
        df: Kontrol edilecek DataFrame
        df_name: DataFrame ismi
        logger: Logger instance (opsiyonel)

    Returns:
        (constant_cols, all_nan_cols): Tuple[List[str], List[str]]
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    constant_cols = []
    all_nan_cols = []

    for col in numeric_cols:
        if df[col].isna().all():
            all_nan_cols.append(col)
        elif df[col].std() == 0:
            constant_cols.append(col)

    if constant_cols:
        msg = f"[VALIDATION WARNING] {df_name}: Sabit değerli kolonlar: {constant_cols}"
        if logger:
            logger.warning(msg)
        else:
            print(msg)

    if all_nan_cols:
        msg = f"[VALIDATION WARNING] {df_name}: Tamamen NaN kolonlar: {all_nan_cols}"
        if logger:
            logger.warning(msg)
        else:
            print(msg)

    return constant_cols, all_nan_cols


def validate_dataframe_summary(
    df: pd.DataFrame,
    df_name: str = "DataFrame",
    logger: Optional[logging.Logger] = None
) -> Dict:
    """
    DataFrame'in genel kalite özetini oluşturur.

    Args:
        df: Kontrol edilecek DataFrame
        df_name: DataFrame ismi
        logger: Logger instance (opsiyonel)

    Returns:
        summary_dict: Özet istatistikler dictionary
    """
    summary = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "total_nulls": df.isna().sum().sum(),
        "null_percentage": (df.isna().sum().sum() / (len(df) * len(df.columns)) * 100),
        "duplicate_rows": df.duplicated().sum(),
        "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
    }

    msg = f"\n[VALIDATION SUMMARY] {df_name}:\n"
    msg += f"  Satır sayısı: {summary['row_count']:,}\n"
    msg += f"  Kolon sayısı: {summary['column_count']:,}\n"
    msg += f"  Toplam NULL: {summary['total_nulls']:,} ({summary['null_percentage']:.2f}%)\n"
    msg += f"  Duplicate satır: {summary['duplicate_rows']:,}\n"
    msg += f"  Bellek kullanımı: {summary['memory_usage_mb']:.2f} MB\n"

    if logger:
        logger.info(msg)
    else:
        print(msg)

    return summary
