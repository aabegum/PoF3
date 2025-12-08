"""
Safe date parser for PoF3 - handles mixed Turkish date formats

This module provides a deterministic multi-format date parser to avoid
ambiguous date parsing issues that can arise with pd.to_datetime().

Supported formats:
    - 1.2.2021 16:59
    - 07-01-2024 21:17:45
    - 2021-02-01 14:30:00
    - 01/02/2021 09:30
    - And variations without time components
"""

from datetime import datetime
import pandas as pd


def parse_date_safely(x):
    """
    Parses mixed date formats with deterministic format priority.

    This parser tries multiple common Turkish date formats in order,
    avoiding the ambiguity of pd.to_datetime() automatic format detection.

    Args:
        x: Date string or value to parse

    Returns:
        datetime object or pd.NaT if parsing fails

    Examples:
        >>> parse_date_safely("1.2.2021 16:59")
        datetime.datetime(2021, 2, 1, 16, 59)

        >>> parse_date_safely("07-01-2024 21:17:45")
        datetime.datetime(2024, 1, 7, 21, 17, 45)

        >>> parse_date_safely("2021-02-01 14:30:00")
        datetime.datetime(2021, 2, 1, 14, 30)
    """
    if pd.isna(x):
        return pd.NaT

    x = str(x).strip()

    # List of supported formats (day-first formats prioritized for Turkish data)
    date_formats = [
        "%d.%m.%Y %H:%M",
        "%d.%m.%Y %H:%M:%S",
        "%d-%m-%Y %H:%M",
        "%d-%m-%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%d/%m/%Y %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%d.%m.%Y",
        "%d-%m-%Y",
        "%d/%m/%Y",
    ]

    for fmt in date_formats:
        try:
            return datetime.strptime(x, fmt)
        except (ValueError, TypeError):
            pass

    # Final fallback with explicit dayfirst=True
    try:
        return pd.to_datetime(x, errors="coerce", dayfirst=True)
    except Exception:
        return pd.NaT


def parse_date_column(series: pd.Series, column_name: str = "date") -> pd.Series:
    """
    Apply safe date parsing to an entire pandas Series.

    Args:
        series: Pandas Series containing date strings
        column_name: Name of column for logging purposes

    Returns:
        Pandas Series with parsed datetime values

    Example:
        >>> df['Kurulum_Tarihi'] = parse_date_column(df['Kurulum_Tarihi'], 'Kurulum_Tarihi')
    """
    return series.apply(parse_date_safely)
