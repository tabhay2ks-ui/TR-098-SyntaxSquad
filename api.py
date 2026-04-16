from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import re

from model import train_fake_review_model, predict_fake_review_scores
from clustering import detect_suspicious_clusters
from scoring import calculate_seller_trust_scores, attach_seller_scores
from explanation import generate_explanation


app = FastAPI(
    title="AI Fake Review Detection API",
    description="Real-time fake review detection, clustering, and seller trust scoring",
    version="2.1.0"
)


class ReviewInput(BaseModel):
    review_text: str = Field(..., min_length=1)
    rating: float = Field(..., ge=1, le=5)
    verified_purchase: int = Field(..., ge=0, le=1)
    product_id: str = "demo_product"
    reviewer_id: int = 999999
    seller_id: str = "demo_seller"


def is_invalid_review_text(text: str) -> bool:
    """Detect obviously invalid or gibberish-like review input."""
    if not text or not text.strip():
        return True

    cleaned = text.strip()
    words = cleaned.split()

    # Single long token with no meaningful structure
    if len(words) == 1 and len(words[0]) >= 10:
        return True

    # Extremely short non-review text
    if len(words) < 2 and len(cleaned) < 12:
        return True

    # Very low vowel ratio can indicate gibberish
    letters_only = re.sub(r"[^a-zA-Z]", "", cleaned)
    if letters_only:
        vowels = sum(1 for ch in letters_only.lower() if ch in "aeiou")
        vowel_ratio = vowels / len(letters_only)
        if len(letters_only) >= 8 and vowel_ratio < 0.2:
            return True

    return False


def normalize_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rename_map = {}

    if "verified_purchase_x" in df.columns and "verified_purchase" not in df.columns:
        rename_map["verified_purchase_x"] = "verified_purchase"

    if "fake_probability_x" in df.columns and "fake_probability" not in df.columns:
        rename_map["fake_probability_x"] = "fake_probability"

    if "authenticity_score_x" in df.columns and "authenticity_score" not in df.columns:
        rename_map["authenticity_score_x"] = "authenticity_score"

    if "suspicious_cluster_flag_x" in df.columns and "suspicious_cluster_flag" not in df.columns:
        rename_map["suspicious_cluster_flag_x"] = "suspicious_cluster_flag"

    if rename_map:
        df = df.rename(columns=rename_map)

    return df


def safe_get(row, col, default=None):
    return row[col] if col in row.index else default


def build_input_dataframe(review: ReviewInput) -> pd.DataFrame:
    return pd.DataFrame([{
        "review_text": review.review_text,
        "rating": review.rating,
        "verified_purchase": review.verified_purchase,
        "product_id": review.product_id,
        "reviewer_id": review.reviewer_id,
        "seller_id": review.seller_id,
        "timestamp": pd.Timestamp.now(),
        "label": 0
    }])


print("Loading dataset and training model...")
df = pd.read_csv("data/processed/reviews.csv")
model_artifacts = train_fake_review_model(df)
print("Model ready.")


@app.get("/")
def home():
    return {
        "message": "Fake Review Detection API is running",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict_review(review: ReviewInput):
    # Step 1: Reject obviously invalid/gibberish input
    if is_invalid_review_text(review.review_text):
        return {
            "review_text": review.review_text,
            "rating": review.rating,
            "verified_purchase": review.verified_purchase,
            "product_id": review.product_id,
            "reviewer_id": review.reviewer_id,
            "seller_id": review.seller_id,
            "fake_probability": 0.99,
            "predicted_label": "Fake",
            "authenticity_score": 1.0,
            "cluster_id": -1,
            "suspicious_cluster_flag": 0,
            "seller_trust_score": 0.0,
            "seller_grade": "F",
            "invalid_input": True,
            "trust_explanation": "Seller trust cannot be evaluated because the review text is invalid.",
            "review_explanation": "Suspicious because the submitted text does not appear to be a meaningful review."
        }

    # Step 2: Build input row
    input_df = build_input_dataframe(review)

    # Step 3: Review scoring
    scored_df = predict_fake_review_scores(
        input_df,
        model_artifacts,
        threshold=0.5
    )

    # Step 4: Cluster detection
    clustered_df = detect_suspicious_clusters(scored_df)

    if "cluster_id" not in clustered_df.columns:
        clustered_df["cluster_id"] = -1

    if "suspicious_cluster_flag" not in clustered_df.columns:
        clustered_df["suspicious_cluster_flag"] = 0

    # Step 5: Seller trust scoring
    seller_scores = calculate_seller_trust_scores(clustered_df)
    final_df = attach_seller_scores(clustered_df, seller_scores)

    # Step 6: Normalize merge columns
    final_df = normalize_output_columns(final_df)

    # Step 7: Explanation
    final_df["explanation"] = final_df.apply(generate_explanation, axis=1)

    row = final_df.iloc[0]

    return {
        "review_text": safe_get(row, "review_text", ""),
        "rating": float(safe_get(row, "rating", 0)),
        "verified_purchase": int(safe_get(row, "verified_purchase", 0)),
        "product_id": safe_get(row, "product_id", ""),
        "reviewer_id": int(safe_get(row, "reviewer_id", 0)),
        "seller_id": safe_get(row, "seller_id", ""),
        "fake_probability": round(float(safe_get(row, "fake_probability", 0)), 4),
        "predicted_label": "Fake" if safe_get(row, "predicted_label", 0) == 1 else "Real",
        "authenticity_score": round(float(safe_get(row, "authenticity_score", 0)), 2),
        "cluster_id": int(safe_get(row, "cluster_id", -1)),
        "suspicious_cluster_flag": int(safe_get(row, "suspicious_cluster_flag", 0)),
        "seller_trust_score": round(float(safe_get(row, "seller_trust_score", 0)), 2),
        "seller_grade": safe_get(row, "seller_grade", "N/A"),
        "invalid_input": False,
        "trust_explanation": safe_get(row, "trust_explanation", ""),
        "review_explanation": safe_get(row, "explanation", "")
    }