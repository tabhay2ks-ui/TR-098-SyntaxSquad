import pandas as pd

from utils import assign_grade


def seller_explanation(row):
    """Generate a human-readable explanation for seller trust score."""
    reasons = []

    if row["authenticity_score"] < 50:
        reasons.append("low average review authenticity")

    if row["verified_purchase"] < 0.5:
        reasons.append("low verified purchase rate")

    if row["suspicious_cluster_flag"] > 0.3:
        reasons.append("elevated suspicious cluster activity")

    if row["fake_probability"] > 0.5:
        reasons.append("many reviews predicted as fake")

    if not reasons:
        return "Seller appears trustworthy based on review authenticity, verification behavior, and low suspicious activity."

    return "Low trust because of " + ", ".join(reasons) + "."


def calculate_seller_trust_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate seller trust score using review authenticity, verified purchase rate,
    suspicious cluster behavior, and fake review probability.
    """

    seller_scores = df.groupby("seller_id").agg({
        "authenticity_score": "mean",
        "verified_purchase": "mean",
        "suspicious_cluster_flag": "mean",
        "fake_probability": "mean"
    }).reset_index()

    # Positive components
    seller_scores["authenticity_component"] = seller_scores["authenticity_score"]
    seller_scores["verification_component"] = seller_scores["verified_purchase"] * 100

    # Safer trust-oriented components
    seller_scores["cluster_trust_component"] = (1 - seller_scores["suspicious_cluster_flag"]) * 100
    seller_scores["fake_trust_component"] = (1 - seller_scores["fake_probability"]) * 100

    # Final trust score
    seller_scores["seller_trust_score"] = (
        0.40 * seller_scores["authenticity_component"] +
        0.25 * seller_scores["verification_component"] +
        0.20 * seller_scores["cluster_trust_component"] +
        0.15 * seller_scores["fake_trust_component"]
    )

    seller_scores["seller_trust_score"] = seller_scores["seller_trust_score"].clip(0, 100)
    seller_scores["seller_grade"] = seller_scores["seller_trust_score"].apply(assign_grade)

    # Human-readable explanation
    seller_scores["trust_explanation"] = seller_scores.apply(seller_explanation, axis=1)

    return seller_scores


def attach_seller_scores(df: pd.DataFrame, seller_scores: pd.DataFrame) -> pd.DataFrame:
    """Merge seller trust scores back into the review-level dataframe."""
    return df.merge(seller_scores, on="seller_id", how="left")