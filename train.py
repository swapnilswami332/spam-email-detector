"""
Train a spam vs ham classifier on labeled email text.

Uses the same preprocessing as inference, TF-IDF features, and Multinomial Naive Bayes.
"""

import csv
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from app.preprocess import ensure_nltk_data, normalize_text

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "emails.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "spam_classifier.joblib"


def load_dataset(path: Path) -> tuple[list[str], list[int]]:
    texts: list[str] = []
    labels: list[int] = []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = (row.get("label") or "").strip().lower()
            text = (row.get("text") or "").strip()
            if label not in ("ham", "spam") or not text:
                continue
            texts.append(text)
            labels.append(1 if label == "spam" else 0)
    if len(texts) < 10:
        raise ValueError("Need at least 10 labeled rows to train.")
    return texts, labels


def main() -> None:
    ensure_nltk_data()
    print(f"Loading data from {DATA_PATH}")
    raw_texts, raw_labels = load_dataset(DATA_PATH)

    print("Preprocessing text (tokenize, stopwords, lemmatize)...")
    cleaned: list[str] = []
    labels: list[int] = []
    for text, label in zip(raw_texts, raw_labels):
        norm = normalize_text(text)
        if norm:
            cleaned.append(norm)
            labels.append(label)
    if len(cleaned) < 10:
        raise SystemExit("Not enough usable text after preprocessing.")

    X_train, X_test, y_train, y_test = train_test_split(
        cleaned, labels, test_size=0.2, random_state=42, stratify=labels
    )

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
    )
    classifier = MultinomialNB(alpha=0.1)
    labels_map = {"ham": 0, "spam": 1}

    print("Training TF-IDF + Multinomial Naive Bayes...")
    X_train_vec = vectorizer.fit_transform(X_train)
    classifier.fit(X_train_vec, y_train)

    X_test_vec = vectorizer.transform(X_test)
    y_pred = classifier.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {acc:.2%}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_test, y_pred))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "vectorizer": vectorizer,
        "classifier": classifier,
        "labels": labels_map,
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"\nSaved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
