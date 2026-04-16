import pandas as pd

from model import train_fake_review_model, predict_fake_review_scores
from clustering import detect_suspicious_clusters
from scoring import calculate_seller_trust_scores, attach_seller_scores
from explanation import generate_explanation


def build_single_review_df(
    review_text: str,
    rating: float,
    product_id: str = "demo_product",
    reviewer_id: int = 999999,
    seller_id: str = "demo_seller",
    verified_purchase: int = 0,
    label: int = 0
) -> pd.DataFrame:
    return pd.DataFrame([{
        "review_text": review_text,
        "rating": rating,
        "label": label,
        "product_id": product_id,
        "reviewer_id": reviewer_id,
        "seller_id": seller_id,
        "timestamp": pd.Timestamp.now(),
        "verified_purchase": verified_purchase
    }])


def predict_one_review(
    review_text: str,
    rating: float,
    model_artifacts: dict,
    product_id: str = "demo_product",
    reviewer_id: int = 999999,
    seller_id: str = "demo_seller",
    verified_purchase: int = 0
) -> pd.DataFrame:
    single_df = build_single_review_df(
        review_text=review_text,
        rating=rating,
        product_id=product_id,
        reviewer_id=reviewer_id,
        seller_id=seller_id,
        verified_purchase=verified_purchase,
        label=0
    )

    scored_df = predict_fake_review_scores(single_df, model_artifacts)
    clustered_df = detect_suspicious_clusters(scored_df)
    seller_scores = calculate_seller_trust_scores(clustered_df)
    final_df = attach_seller_scores(clustered_df, seller_scores)

    rename_map = {}
    if "verified_purchase_x" in final_df.columns:
        rename_map["verified_purchase_x"] = "verified_purchase"
    if "fake_probability_x" in final_df.columns:
        rename_map["fake_probability_x"] = "fake_probability"
    if "authenticity_score_x" in final_df.columns:
        rename_map["authenticity_score_x"] = "authenticity_score"
    if "suspicious_cluster_flag_x" in final_df.columns:
        rename_map["suspicious_cluster_flag_x"] = "suspicious_cluster_flag"

    if rename_map:
        final_df = final_df.rename(columns=rename_map)

    if "cluster_id" not in final_df.columns:
        final_df["cluster_id"] = -1
    if "suspicious_cluster_flag" not in final_df.columns:
        final_df["suspicious_cluster_flag"] = 0

    final_df["explanation"] = final_df.apply(generate_explanation, axis=1)

    important_cols = [
        "review_text",
        "rating",
        "product_id",
        "seller_id",
        "verified_purchase",
        "fake_probability",
        "predicted_label",
        "authenticity_score",
        "cluster_id",
        "suspicious_cluster_flag",
        "seller_trust_score",
        "seller_grade",
        "trust_explanation",
        "explanation"
    ]

    existing_cols = [col for col in important_cols if col in final_df.columns]
    return final_df[existing_cols]


def main():
    print("Training model...")
    df = pd.read_csv("data/processed/reviews.csv")
    model_artifacts = train_fake_review_model(df)
    print("Model ready.\n")

    while True:
        print("\n--- Fake Review Detector ---")
        review_text = input("Enter review text: ").strip()

        if review_text.lower() == "exit":
            print("Exiting program.")
            break

        try:
            rating = float(input("Enter rating (1-5): ").strip())
        except ValueError:
            print("Invalid rating. Please enter a number.")
            continue

        try:
            verified_purchase = int(input("Verified purchase? (1 = Yes, 0 = No): ").strip())
            if verified_purchase not in [0, 1]:
                print("Please enter only 1 or 0.")
                continue
        except ValueError:
            print("Invalid input. Please enter 1 or 0.")
            continue

        result = predict_one_review(
            review_text=review_text,
            rating=rating,
            model_artifacts=model_artifacts,
            product_id="demo_product_1",
            reviewer_id=12345,
            seller_id="demo_seller_1",
            verified_purchase=verified_purchase
        )

        row = result.iloc[0]

        print("\n--- Prediction Result ---")
        print("Review Text:", row["review_text"])
        print("Rating:", row["rating"])
        print("Verified Purchase:", row["verified_purchase"])
        print("Fake Probability:", round(row["fake_probability"], 4))
        print("Predicted Label:", "Fake" if row["predicted_label"] == 1 else "Real")
        print("Authenticity Score:", round(row["authenticity_score"], 2))
        print("Seller Trust Score:", round(row["seller_trust_score"], 2))
        print("Seller Grade:", row["seller_grade"])
        print("Trust Explanation:", row["trust_explanation"])
        print("Review Explanation:", row["explanation"])

        again = input("\nType another review? (yes/no): ").strip().lower()
        if again != "yes":
            print("Exiting program.")
            break


if __name__ == "__main__":
    main()