import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob

from utils import clean_text, safe_datetime


SPAM_PHRASES = [
    "buy now",
    "best product",
    "amazing product",
    "highly recommended",
    "must buy",
    "excellent product",
    "great product",
    "perfect product",
    "love it",
    "worth every penny"
]


def count_spam_phrases(text: str) -> int:
    """Count promotional/spam-like phrases in cleaned text."""
    text = text or ""
    return sum(1 for phrase in SPAM_PHRASES if phrase in text)


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create improved linguistic and behavioral features."""
    df = df.copy()

    df["raw_review_text"] = df["review_text"].fillna("")
    df["clean_review_text"] = df["raw_review_text"].apply(clean_text)
    df["timestamp"] = safe_datetime(df["timestamp"])

    # Text features
    df["review_length"] = df["clean_review_text"].apply(len)
    df["word_count"] = df["clean_review_text"].apply(lambda x: len(x.split()))
    df["exclamation_count"] = df["raw_review_text"].apply(lambda x: x.count("!"))
    df["uppercase_ratio"] = df["raw_review_text"].apply(
        lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0
    )

    df["sentiment_polarity"] = df["clean_review_text"].apply(
        lambda x: TextBlob(x).sentiment.polarity if x else 0
    )

    # Stronger spam-style features
    df["spam_word_count"] = df["clean_review_text"].apply(count_spam_phrases)
    df["has_buy_now"] = df["clean_review_text"].apply(lambda x: int("buy now" in x))
    df["has_best_product"] = df["clean_review_text"].apply(lambda x: int("best product" in x))
    df["repeated_word_ratio"] = df["clean_review_text"].apply(
        lambda x: (
            1 - (len(set(x.split())) / len(x.split()))
            if len(x.split()) > 0 else 0
        )
    )

    # Numeric/base features
    df["verified_purchase"] = df["verified_purchase"].fillna(0).astype(int)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0)

    # Reviewer behavior
    df["reviewer_review_count"] = (
        df.groupby("reviewer_id")["clean_review_text"].transform("count")
    )

    # Product behavior
    product_avg_rating = df.groupby("product_id")["rating"].transform("mean")
    df["rating_deviation"] = (df["rating"] - product_avg_rating).abs()

    # Fraud-style flags
    df["is_short_review"] = (df["word_count"] <= 4).astype(int)
    df["is_extreme_rating"] = df["rating"].isin([1, 5]).astype(int)
    df["excessive_exclamations"] = (df["exclamation_count"] >= 3).astype(int)
    df["high_uppercase_flag"] = (df["uppercase_ratio"] > 0.3).astype(int)
    df["spam_phrase_flag"] = (df["spam_word_count"] >= 1).astype(int)

    # Time features
    if df["timestamp"].notna().sum() > 0:
        review_dates = df["timestamp"].dt.date

        df["reviewer_daily_review_count"] = (
            df.groupby(["reviewer_id", review_dates])["clean_review_text"]
            .transform("count")
            .fillna(1)
        )

        df["product_daily_review_count"] = (
            df.groupby(["product_id", review_dates])["clean_review_text"]
            .transform("count")
            .fillna(1)
        )
    else:
        df["reviewer_daily_review_count"] = 1
        df["product_daily_review_count"] = 1

    return df


def build_tfidf_features(train_texts, test_texts, max_features: int = 3000):
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
        "is_short_review",
        "is_extreme_rating",
        "reviewer_daily_review_count",
        "product_daily_review_count",
        "spam_word_count",
        "has_buy_now",
        "has_best_product",
        "repeated_word_ratio",
        "excessive_exclamations",
        "high_uppercase_flag",
        "spam_phrase_flag",
    ]