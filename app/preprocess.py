"""Text preprocessing for email bodies (NLP pipeline)."""

import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

_nltk_ready = False


def ensure_nltk_data() -> None:
    """Download NLTK resources used by the pipeline (safe to call multiple times)."""
    global _nltk_ready
    if _nltk_ready:
        return
    for package in ("punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"):
        nltk.download(package, quiet=True)
    _nltk_ready = True


_lemmatizer: WordNetLemmatizer | None = None
_stop_words: set[str] | None = None


def _get_lemmatizer() -> WordNetLemmatizer:
    global _lemmatizer
    ensure_nltk_data()
    if _lemmatizer is None:
        _lemmatizer = WordNetLemmatizer()
    return _lemmatizer


def _get_stop_words() -> set[str]:
    global _stop_words
    ensure_nltk_data()
    if _stop_words is None:
        _stop_words = set(stopwords.words("english"))
    return _stop_words


def normalize_text(text: str) -> str:
    """
    Clean raw email text: lowercase, strip URLs/emails, remove punctuation,
    tokenize, drop stopwords, lemmatize, join back to a single string.
    """
    if not text or not text.strip():
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = nltk.word_tokenize(text)

    stop = _get_stop_words()
    lemmatizer = _get_lemmatizer()
    tokens = [
        lemmatizer.lemmatize(t)
        for t in tokens
        if t.isalpha() and len(t) > 1 and t not in stop
    ]
    return " ".join(tokens)
