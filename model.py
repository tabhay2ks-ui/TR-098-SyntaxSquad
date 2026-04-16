import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from features import add_basic_features, build_tfidf_features, get_numeric_feature_columns


def train_fake_review_model(df: pd.DataFrame):
    """Train a fake review classifier and return trained artifacts."""
    df = add_basic_features(df)

    X_text = df["review_text"]
    y = df["label"].astype(int)

    numeric_cols = get_numeric_feature_columns()
    X_numeric = df[numeric_cols].fillna(0)

    X_train_text, X_test_text, X_train_num, X_test_num, y_train, y_test = train_test_split(
        X_text, X_numeric, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train_tfidf, X_test_tfidf, vectorizer = build_tfidf_features(X_train_text, X_test_text)

    X_train_final = hstack([X_train_tfidf, csr_matrix(X_train_num.values)])
    X_test_final = hstack([X_test_tfidf, csr_matrix(X_test_num.values)])

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_final, y_train)

    y_pred = model.predict(X_test_final)
    y_prob = model.predict_proba(X_test_final)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred)
    }

    return {
        "model": model,
        "vectorizer": vectorizer,
        "numeric_cols": numeric_cols,
        "metrics": metrics
    }


def predict_fake_review_scores(df: pd.DataFrame, model_artifacts: dict) -> pd.DataFrame:
    """Predict fake probability/authenticity score for each review."""
    df = add_basic_features(df)

    vectorizer = model_artifacts["vectorizer"]
    model = model_artifacts["model"]
    numeric_cols = model_artifacts["numeric_cols"]

    X_text = vectorizer.transform(df["review_text"])
    X_numeric = csr_matrix(df[numeric_cols].fillna(0).values)

    X_final = hstack([X_text, X_numeric])

    fake_probability = model.predict_proba(X_final)[:, 1]
    predicted_label = model.predict(X_final)

    df = df.copy()
    df["fake_probability"] = fake_probability
    df["predicted_label"] = predicted_label
    df["authenticity_score"] = (1 - df["fake_probability"]) * 100

    return df