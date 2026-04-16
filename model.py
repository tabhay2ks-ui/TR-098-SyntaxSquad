import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from features import add_basic_features, build_tfidf_features, get_numeric_feature_columns


def train_fake_review_model(df: pd.DataFrame):
    """Train a fake review classifier and return trained artifacts."""
    df = add_basic_features(df.copy())

    X_text = df["clean_review_text"].fillna("")
    y = df["label"].astype(int)

    numeric_cols = get_numeric_feature_columns()
    X_numeric = df[numeric_cols].fillna(0)

    X_train_text, X_test_text, X_train_num, X_test_num, y_train, y_test = train_test_split(
        X_text, X_numeric, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train_tfidf, X_test_tfidf, vectorizer = build_tfidf_features(X_train_text, X_test_text)

    scaler = StandardScaler()
    X_train_num_scaled = scaler.fit_transform(X_train_num)
    X_test_num_scaled = scaler.transform(X_test_num)

    X_train_final = hstack([X_train_tfidf, csr_matrix(X_train_num_scaled)])
    X_test_final = hstack([X_test_tfidf, csr_matrix(X_test_num_scaled)])

    model = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        C=0.7
    )
    model.fit(X_train_final, y_train)

    y_pred = model.predict(X_test_final)
    y_prob = model.predict_proba(X_test_final)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }

    return {
        "model": model,
        "vectorizer": vectorizer,
        "scaler": scaler,
        "numeric_cols": numeric_cols,
        "metrics": metrics
    }


def predict_fake_review_scores(df: pd.DataFrame, model_artifacts: dict, threshold: float = 0.5) -> pd.DataFrame:
    """Predict fake probability/authenticity score for each review."""
    df = add_basic_features(df.copy())

    vectorizer = model_artifacts["vectorizer"]
    model = model_artifacts["model"]
    scaler = model_artifacts["scaler"]
    numeric_cols = model_artifacts["numeric_cols"]

    X_text = vectorizer.transform(df["clean_review_text"].fillna(""))
    X_numeric = df[numeric_cols].fillna(0)
    X_numeric_scaled = scaler.transform(X_numeric)

    X_final = hstack([X_text, csr_matrix(X_numeric_scaled)])

    fake_probability = model.predict_proba(X_final)[:, 1]

    # Hybrid rule layer to catch obvious spam-like patterns
    rule_based_suspicion = (
        (df["spam_word_count"] >= 1) |
        (df["excessive_exclamations"] == 1) |
        (df["high_uppercase_flag"] == 1) |
        ((df["verified_purchase"] == 0) & (df["is_extreme_rating"] == 1) & (df["spam_phrase_flag"] == 1))
    )

    adjusted_probability = fake_probability.copy()
    adjusted_probability = adjusted_probability + (rule_based_suspicion.astype(float) * 0.12)
    adjusted_probability = adjusted_probability.clip(0, 1)

    predicted_label = (adjusted_probability >= threshold).astype(int)

    df["fake_probability"] = adjusted_probability
    df["predicted_label"] = predicted_label
    df["authenticity_score"] = (1 - df["fake_probability"]) * 100

    return df