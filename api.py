from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List
import pandas as pd
import re

from model import train_fake_review_model, predict_fake_review_scores
from clustering import detect_suspicious_clusters
from scoring import calculate_seller_trust_scores, attach_seller_scores
from explanation import generate_explanation


app = FastAPI(
    title="AI Fake Review Detection API",
    description="Real-time fake review detection, batch review analysis, clustering, and seller trust scoring",
    version="3.0.0"
)


# =========================
# INPUT MODELS
# =========================
class ReviewInput(BaseModel):
    review_text: str = Field(..., min_length=1)
    rating: float = Field(..., ge=1, le=5)
    verified_purchase: int = Field(..., ge=0, le=1)
    product_id: str = "demo_product"
    reviewer_id: int = 999999
    seller_id: str = "demo_seller"


class BatchReviewInput(BaseModel):
    reviews: List[ReviewInput]


# =========================
# INPUT VALIDATION
# =========================
def is_invalid_review_text(text: str) -> bool:
    """Detect obviously invalid or gibberish-like review input."""
    if not text or not text.strip():
        return True

    cleaned = text.strip()
    words = cleaned.split()

    # Single long meaningless token
    if len(words) == 1 and len(words[0]) >= 10:
        return True

    # Extremely short non-review text
    if len(words) < 2 and len(cleaned) < 12:
        return True

    # Low vowel ratio can indicate gibberish
    letters_only = re.sub(r"[^a-zA-Z]", "", cleaned)
    if letters_only:
        vowels = sum(1 for ch in letters_only.lower() if ch in "aeiou")
        vowel_ratio = vowels / len(letters_only)
        if len(letters_only) >= 8 and vowel_ratio < 0.2:
            return True

    return False


# =========================
# HELPERS
# =========================
def normalize_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize duplicate columns created after merges."""
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


def build_invalid_result(review: ReviewInput) -> dict:
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


def process_reviews_dataframe(input_df: pd.DataFrame) -> pd.DataFrame:
    """Run the full review pipeline on a DataFrame of reviews."""
    scored_df = predict_fake_review_scores(
        input_df,
        model_artifacts,
        threshold=0.5
    )

    clustered_df = detect_suspicious_clusters(scored_df)

    if "cluster_id" not in clustered_df.columns:
        clustered_df["cluster_id"] = -1

    if "suspicious_cluster_flag" not in clustered_df.columns:
        clustered_df["suspicious_cluster_flag"] = 0

    seller_scores = calculate_seller_trust_scores(clustered_df)
    final_df = attach_seller_scores(clustered_df, seller_scores)

    final_df = normalize_output_columns(final_df)
    final_df["invalid_input"] = False
    final_df["explanation"] = final_df.apply(generate_explanation, axis=1)

    return final_df


def dataframe_row_to_response(row) -> dict:
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
        "invalid_input": bool(safe_get(row, "invalid_input", False)),
        "trust_explanation": safe_get(row, "trust_explanation", ""),
        "review_explanation": safe_get(row, "explanation", "")
    }


# =========================
# LOAD MODEL
# =========================
print("Loading dataset and training model...")
df = pd.read_csv("data/processed/reviews.csv")
model_artifacts = train_fake_review_model(df)
print("Model ready.")


# =========================
# ROUTES
# =========================
@app.get("/")
def home():
    return {
        "message": "Fake Review Detection API is running",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model_artifacts is not None
    }


@app.post("/predict")
def predict_review(review: ReviewInput):
    # Invalid input handling
    if is_invalid_review_text(review.review_text):
        return build_invalid_result(review)

    # Single review pipeline
    input_df = build_input_dataframe(review)
    final_df = process_reviews_dataframe(input_df)
    row = final_df.iloc[0]

    return dataframe_row_to_response(row)


@app.post("/predict-batch")
def predict_batch(batch: BatchReviewInput):
    if not batch.reviews:
        return {
            "total_reviews": 0,
            "fake_reviews": 0,
            "fake_percentage": 0.0,
            "results": []
        }

    valid_rows = []
    invalid_results = []

    for review in batch.reviews:
        if is_invalid_review_text(review.review_text):
            invalid_results.append(build_invalid_result(review))
        else:
            valid_rows.append({
                "review_text": review.review_text,
                "rating": review.rating,
                "verified_purchase": review.verified_purchase,
                "product_id": review.product_id,
                "reviewer_id": review.reviewer_id,
                "seller_id": review.seller_id,
                "timestamp": pd.Timestamp.now(),
                "label": 0
            })

    batch_results = []

    if valid_rows:
        input_df = pd.DataFrame(valid_rows)
        final_df = process_reviews_dataframe(input_df)

        for _, row in final_df.iterrows():
            batch_results.append(dataframe_row_to_response(row))

    # Combine results
    all_results = batch_results + invalid_results

    total = len(all_results)
    fake_count = sum(1 for r in all_results if str(r.get("predicted_label", "")).lower() == "fake")
    fake_percentage = round((fake_count / total) * 100, 2) if total > 0 else 0.0

    # Risky sellers summary
    risky_sellers = []
    if all_results:
        results_df = pd.DataFrame(all_results)
        if not results_df.empty and "seller_id" in results_df.columns:
            seller_summary = (
                results_df.groupby("seller_id", as_index=False)
                .agg(
                    avg_fake_probability=("fake_probability", "mean"),
                    avg_seller_trust_score=("seller_trust_score", "mean"),
                    review_count=("seller_id", "count")
                )
                .sort_values(by="avg_fake_probability", ascending=False)
            )

            risky_sellers = seller_summary.head(10).round(4).to_dict(orient="records")

    return {
        "total_reviews": total,
        "fake_reviews": fake_count,
        "fake_percentage": fake_percentage,
        "risky_sellers": risky_sellers,
        "results": all_results
    }