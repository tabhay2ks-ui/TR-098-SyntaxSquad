import pandas as pd

from model import train_fake_review_model, predict_fake_review_scores
from clustering import detect_suspicious_clusters
from scoring import calculate_seller_trust_scores, attach_seller_scores
from explanation import generate_explanation


def main():
    # Load cleaned dataset
    df = pd.read_csv("data/processed/reviews.csv")

    # Train model
    model_artifacts = train_fake_review_model(df)

    print("Model Metrics:")
    print(model_artifacts["metrics"]["classification_report"])
    print("Accuracy:", model_artifacts["metrics"]["accuracy"])
    print("F1 Score:", model_artifacts["metrics"]["f1_score"])

    if "confusion_matrix" in model_artifacts["metrics"]:
        print("Confusion Matrix:", model_artifacts["metrics"]["confusion_matrix"])

    # Review-level predictions
    scored_df = predict_fake_review_scores(df, model_artifacts)

    # Cluster suspicious reviewers
    clustered_df = detect_suspicious_clusters(scored_df)

    # Seller-level trust scores
    seller_scores = calculate_seller_trust_scores(clustered_df)

    # Merge seller scores back into review-level dataframe
    final_df = attach_seller_scores(clustered_df, seller_scores)

    # Keep the most useful output columns
    columns_to_keep = [
        "product_id",
        "rating",
        "label",
        "review_text",
        "reviewer_id",
        "seller_id",
        "timestamp",
        "verified_purchase",
        "review_length",
        "word_count",
        "exclamation_count",
        "uppercase_ratio",
        "sentiment_polarity",
        "reviewer_review_count",
        "rating_deviation",
        "is_short_review",
        "is_extreme_rating",
        "reviewer_daily_review_count",
        "product_daily_review_count",
        "fake_probability",
        "predicted_label",
        "authenticity_score",
        "cluster_id",
        "suspicious_cluster_flag",
        "seller_trust_score",
        "seller_grade",
        "trust_explanation",
    ]

    existing_columns = [col for col in columns_to_keep if col in final_df.columns]
    final_df = final_df[existing_columns].copy()

    # Add final human-readable explanation
    final_df["explanation"] = final_df.apply(generate_explanation, axis=1)

    # Save outputs
    final_df.to_csv("outputs/predictions.csv", index=False)
    seller_scores.to_csv("outputs/seller_scores.csv", index=False)

    print("\nSaved files successfully:")
    print("- outputs/predictions.csv")
    print("- outputs/seller_scores.csv")


if __name__ == "__main__":
    main()