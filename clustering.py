import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


def detect_suspicious_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cluster reviewers based on simple behavioral features.
    Reviews in non-noise clusters can be treated as suspicious campaign candidates.
    """
    df = df.copy()

    reviewer_features = df.groupby("reviewer_id").agg({
        "rating": "mean",
        "verified_purchase": "mean",
        "fake_probability": "mean",
        "review_text": "count"
    }).rename(columns={"review_text": "review_count"}).reset_index()

    feature_cols = ["rating", "verified_purchase", "fake_probability", "review_count"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(reviewer_features[feature_cols])

    clustering_model = DBSCAN(eps=1.2, min_samples=3)
    reviewer_features["cluster_id"] = clustering_model.fit_predict(X_scaled)

    reviewer_features["suspicious_cluster_flag"] = reviewer_features["cluster_id"].apply(
        lambda x: 0 if x == -1 else 1
    )

    df = df.merge(
        reviewer_features[["reviewer_id", "cluster_id", "suspicious_cluster_flag"]],
        on="reviewer_id",
        how="left"
    )

    return df