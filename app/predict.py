"""Load trained model and run inference."""

from pathlib import Path

import joblib

from app.preprocess import normalize_text

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "spam_classifier.joblib"


class SpamDetector:
    def __init__(self, model_path: Path | None = None) -> None:
        path = model_path or MODEL_PATH
        if not path.is_file():
            raise FileNotFoundError(
                f"No trained model at {path}. Run: python train.py"
            )
        bundle = joblib.load(path)
        self.vectorizer = bundle["vectorizer"]
        self.classifier = bundle["classifier"]
        self.labels = bundle.get("labels", {"ham": 0, "spam": 1})

    def predict(self, email_text: str) -> dict:
        cleaned = normalize_text(email_text)
        if not cleaned:
            return {
                "label": "ham",
                "is_spam": False,
                "confidence": 0.0,
                "message": "Empty or unprocessable text; treated as not spam.",
            }

        features = self.vectorizer.transform([cleaned])
        prediction = int(self.classifier.predict(features)[0])
        proba = self.classifier.predict_proba(features)[0]
        confidence = float(max(proba))
        is_spam = prediction == self.labels.get("spam", 1)
        label = "spam" if is_spam else "ham"

        return {
            "label": label,
            "is_spam": is_spam,
            "confidence": round(confidence, 4),
            "spam_probability": round(float(proba[self.labels.get("spam", 1)]), 4),
        }


_detector: SpamDetector | None = None


def get_detector() -> SpamDetector:
    global _detector
    if _detector is None:
        _detector = SpamDetector()
    return _detector
