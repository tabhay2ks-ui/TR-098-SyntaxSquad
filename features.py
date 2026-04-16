import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob

from utils import clean_text, safe_datetime


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create basic linguistic and behavioral features."""
    df = df.copy()

    df["review_text"] = df["review_text"].fillna("").apply(clean_text)
    df["timestamp"] = safe_datetime(df["timestamp"])

    df["review_length"] = df["review_text"].apply(len)
    df["word_count"] = df["review_text"].apply(lambda x: len(x.split()))
    df["exclamation_count"] = df["review_text"].apply(lambda x: x.count("!"))
    df["uppercase_ratio"] = df["review_text"].apply(
        lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0
    )

    df["sentiment_polarity"] = df["review_text"].apply(
        lambda x: TextBlob(x).sentiment.polarity if x else 0
    )

    df["verified_purchase"] = df["verified_purchase"].fillna(0).astype(int)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0)

    reviewer_counts = df.groupby("reviewer_id")["review_text"].transform("count")
    df["reviewer_review_count"] = reviewer_counts

    product_avg_rating = df.groupby("product_id")["rating"].transform("mean")
    df["rating_deviation"] = (df["rating"] - product_avg_rating).abs()

    if df["timestamp"].notna().sum() > 0:
        daily_counts = (
            df.groupby(df["timestamp"].dt.date)["review_text"]
            .transform("count")
            .fillna(1)
        )
        df["daily_review_volume"] = daily_counts
    else:
        df["daily_review_volume"] = 1

    return df


def build_tfidf_features(train_texts, test_texts, max_features: int = 500):
    """Fit TF-IDF on train texts and transform both train and test."""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        ngram_range=(1, 2)
    )

    X_train_tfidf = vectorizer.fit_transform(train_texts)
    X_test_tfidf = vectorizer.transform(test_texts)

    return X_train_tfidf, X_test_tfidf, vectorizer


def get_numeric_feature_columns():
    """List numeric feature columns used alongside TF-IDF."""
    return [
        "rating",
        "verified_purchase",
        "review_length",
        "word_count",
        "exclamation_count",
        "uppercase_ratio",
        "sentiment_polarity",
        "reviewer_review_count",
        "rating_deviation",
        "daily_review_volume",
    ]