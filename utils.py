import re
import pandas as pd


def clean_text(text: str) -> str:
    """Lowercase and remove special characters."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def assign_grade(score: float) -> str:
    """Convert numeric seller trust score into A-F grade."""
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"


def safe_datetime(series: pd.Series) -> pd.Series:
    """Convert a pandas series to datetime safely."""
    return pd.to_datetime(series, errors="coerce")