import pandas as pd

from utils import assign_grade


def calculate_seller_trust_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate seller trust score using authenticity, verified purchases,
    and suspicious cluster penalties.
    """
    seller_scores = df.groupby("seller_id").agg({
        "authenticity_score": "mean",
        "verified_purchase": "mean",
        "suspicious_cluster_flag": "mean",
        "fake_probability": "mean"
    }).reset_index()

    seller_scores["seller_trust_score"] = (
        0.50 * seller_scores["authenticity_score"] +
        0.25 * (seller_scores["verified_purchase"] * 100) +
        0.15 * ((1 - seller_scores["suspicious_cluster_flag"]) * 100) +
        0.10 * ((1 - seller_scores["fake_probability"]) * 100)
    )

    seller_scores["seller_trust_score"] = seller_scores["seller_trust_score"].clip(0, 100)
    seller_scores["seller_grade"] = seller_scores["seller_trust_score"].apply(assign_grade)

    return seller_scores


def attach_seller_scores(df: pd.DataFrame, seller_scores: pd.DataFrame) -> pd.DataFrame:
    """Merge seller trust scores back into review-level dataframe."""
    return df.merge(seller_scores, on="seller_id", how="left")