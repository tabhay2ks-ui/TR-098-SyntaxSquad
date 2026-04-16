import pandas as pd

from model import train_fake_review_model, predict_fake_review_scores
from clustering import detect_suspicious_clusters
from scoring import calculate_seller_trust_scores, attach_seller_scores


def main():
    df = pd.read_csv("data/processed/reviews.csv")

    model_artifacts = train_fake_review_model(df)
    print("Model Metrics:")
    print(model_artifacts["metrics"]["classification_report"])
    print("Accuracy:", model_artifacts["metrics"]["accuracy"])
    print("F1 Score:", model_artifacts["metrics"]["f1_score"])

    scored_df = predict_fake_review_scores(df, model_artifacts)
    clustered_df = detect_suspicious_clusters(scored_df)

    seller_scores = calculate_seller_trust_scores(clustered_df)
    final_df = attach_seller_scores(clustered_df, seller_scores)

    final_df.to_csv("outputs/predictions.csv", index=False)
    seller_scores.to_csv("outputs/seller_scores.csv", index=False)

    print("\nSaved:")
    print("- outputs/predictions.csv")
    print("- outputs/seller_scores.csv")


if __name__ == "__main__":
    main()