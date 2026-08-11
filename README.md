# Spam Email Detector (Python + NLP)

A **backend-only** project that classifies email text as **spam** or **ham** (not spam) using classic NLP and machine learning. There is no web UI—only a **FastAPI** service you call with HTTP (curl, Postman, or your own scripts).
 
This README is written for **learning **: it explains *what* each piece does and *why* it is used at a beginner level.

---

## What problem are we solving?

Spam emails waste time, carry scams, and sometimes deliver malware. Automated filters use patterns in words and phrases to block unwanted mail.

This project builds a **small text classifier**:

1. You give it the **text** of an email (subject + body is fine as one string).
2. It returns **spam** or **ham**, plus a **confidence** score.

It is **not** production-grade (small dataset, simple model). It is meant to teach the **NLP + ML pipeline** you would see in larger systems.

---

## High-level architecture

```text
  emails.csv (labeled examples)
        |
        v
   train.py  --------->  models/spam_classifier.joblib
        |                        |
   NLTK preprocessing             |
   TF-IDF vectorizer              |
   Naive Bayes classifier         v
                           app/main.py (FastAPI)
                           POST /predict { "email_text": "..." }
```

**Training** (offline): read labeled emails → clean text → convert to numbers → learn weights → save model.

**Inference** (online): new email → same cleaning → same vectorizer → model predicts class.

---

## Tech stack (and why)

| Component | Role |
|-----------|------|
| **Python 3.10+** | Language for ML and API |
| **NLTK** | Tokenization, English stopwords, lemmatization |
| **scikit-learn** | TF-IDF features, train/test split, Naive Bayes, metrics |
| **joblib** | Save/load vectorizer + classifier together |
| **FastAPI** | REST API (backend only; auto-generated docs at `/docs`) |
| **uvicorn** | ASGI server that runs FastAPI |

**Not used:** Streamlit or a separate frontend (per project goal). **FastAPI** is enough to expose predictions over HTTP.

---

## NLP pipeline (step by step)

All cleaning lives in `app/preprocess.py` in `normalize_text()`.

### 1. Lowercasing

`"WIN"` and `"win"` should be the same feature. We convert everything to lowercase.

### 2. Remove URLs and email addresses

Spam often contains phishing links. Replacing URLs with space stops the model from memorizing exact malicious URLs and focuses on surrounding language.

### 3. Remove digits

Phone numbers and account IDs vary a lot; for a basic model, digits often add noise.

### 4. Remove punctuation

Punctuation is stripped so `"FREE!!!"` becomes tokens `free`.

### 5. Tokenization

**Tokenization** splits text into words using NLTK `word_tokenize`.

Example: `"Click here now"` → `["click", "here", "now"]`

### 6. Stopword removal

**Stopwords** are very common words (`the`, `is`, `at`) that usually do not help tell spam from normal mail. NLTK provides an English list.

### 7. Lemmatization

**Lemmatization** reduces words to a base form: `running` → `run`, `accounts` → `account`. That merges related words into one feature.

Output of preprocessing is one **space-separated string** of cleaned tokens, ready for scikit-learn.

---

## Machine learning (basic theory)

### Features: TF-IDF

Models need **numbers**, not raw text. We use **TF-IDF** (`TfidfVectorizer`):

- **TF (term frequency):** how often a word appears in *this* email.
- **IDF (inverse document frequency):** down-weights words that appear in *every* email.

So words like `"congratulations"` or `"verify"` can get high weight if they are common in spam but rare in ham.

We also use **bigrams** (`ngram_range=(1, 2)`): pairs of adjacent words like `"click here"` or `"bank account"` capture short phrases.

### Classifier: Multinomial Naive Bayes

**Naive Bayes** is a classic choice for text:

- Fast to train on sparse word counts.
- Works well with TF-IDF or counts for spam/ham style problems.
- Assumes features are independent (not strictly true, but often good enough for learning).

`MultinomialNB` with a small `alpha` smooths rare words so the model does not overfit on typos in the training set.

### Labels

- `ham` → legitimate mail (class 0)
- `spam` → unwanted mail (class 1)

---

## Project layout

```text
spam email detector/
├── README.md                 # This file
├── requirements.txt
├── train.py                  # Train and save model
├── data/
│   └── emails.csv            # label,text columns (ham/spam)
├── models/
│   └── spam_classifier.joblib   # Created after training (gitignored)
└── app/
    ├── main.py               # FastAPI routes
    ├── preprocess.py         # NLTK text pipeline
    └── predict.py            # Load model and predict
```

---

## Setup

### 1. Create a virtual environment (recommended)

```powershell
cd "c:\projects\spam email detector"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Train the model

First run downloads NLTK data (punkt, stopwords, wordnet) automatically.

```powershell
python train.py
```

You should see **test accuracy**, a **classification report**, and a saved file under `models/spam_classifier.joblib`.

### 3. Start the API

```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

- Health check: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
- Interactive API docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 4. Questions?

Post in [Discussions → Q&A](https://github.com/swapnilswami332/spam-email-detector/discussions/categories/q-a). See [docs/discussions-guide.md](docs/discussions-guide.md) for sample questions and answers.

---

## Using the API

### POST `/predict`

**Request body (JSON):**

```json
{
  "email_text": "URGENT: Verify your bank account now or it will be closed!!!"
}
```

**Example with curl (PowerShell):**

```powershell
curl -X POST "http://127.0.0.1:8000/predict" `
  -H "Content-Type: application/json" `
  -d "{\"email_text\": \"Hi, please find the meeting notes attached.\"}"
```

**Example response:**

```json
{
  "label": "ham",
  "is_spam": false,
  "confidence": 0.92,
  "spam_probability": 0.08,
  "message": null
}
```

| Field | Meaning |
|-------|---------|
| `label` | `"spam"` or `"ham"` |
| `is_spam` | Boolean for automation |
| `confidence` | Max class probability from the model |
| `spam_probability` | Estimated P(spam) |

If the model file is missing, `/predict` returns **503** with instructions to run `train.py`.

---

## Dataset: `data/emails.csv`

Format:

```csv
label,text
ham,"Normal email text..."
spam,"Promotional or scam text..."
```

The included file has **40 ham** and **40 spam** examples for demo training. For better real-world accuracy:

1. Download a public set (e.g. **SMS Spam Collection** or **Enron spam** datasets).
2. Map labels to `ham` / `spam` and a single `text` column.
3. Replace or append rows in `emails.csv` and re-run `train.py`.

**Study tip:** Always keep a **held-out test set** you do not train on, so accuracy numbers mean something. `train.py` already splits 80% train / 20% test with stratification.

---

## How training code works (`train.py`)

1. Load CSV and drop invalid rows.
2. Apply `normalize_text()` to every row (same as production).
3. `train_test_split` with `stratify=y` so spam/ham ratio is similar in train and test.
4. `TfidfVectorizer.fit_transform` on training text only (no data leakage from test set).
5. `MultinomialNB.fit` on TF-IDF matrix.
6. Evaluate on test set: accuracy, precision/recall, confusion matrix.
7. Save `vectorizer`, `classifier`, and label map with **joblib**.

**Important:** The vectorizer must be the **same object** saved at training time and loaded at prediction time. Otherwise vocabulary and IDF weights would not match.

---

## How inference works (`app/predict.py`)

1. Load the joblib bundle once (cached in `get_detector()`).
2. `normalize_text(email_text)` — identical pipeline as training.
3. `vectorizer.transform([cleaned])` — turn text into TF-IDF row.
4. `classifier.predict` / `predict_proba` — class and probabilities.

Empty or all-stopword input returns **ham** with a short message (safe default for this demo).

---

## Study exercises (try these yourself)

1. **Add subject line:** Concatenate `"Subject: ... Body: ..."` in your API client and see if accuracy on your own examples improves mentally.
2. **Change `max_features` or `ngram_range`** in `train.py` and compare test accuracy.
3. **Swap Naive Bayes** for `sklearn.linear_model.LogisticRegression` — same TF-IDF pipeline.
4. **Inspect top features:** After training, look at `classifier.feature_log_prob_` and vectorizer vocabulary for words most associated with spam.
5. **Error analysis:** Print test emails where `y_pred != y_test` and decide if preprocessing or more data would help.

---

## Limitations (honest notes for learners)

- Small handcrafted dataset → metrics are illustrative, not benchmark quality.
- English-only stopwords and tokenization.
- No attachment analysis, headers, or sender reputation (real filters use those too).
- Naive Bayes + TF-IDF is a **baseline**; modern systems may use transformers (BERT, etc.) but the **preprocess → features → classify** idea remains.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `No trained model at ...` | Run `python train.py` |
| NLTK download errors | Run Python once: `import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')` |
| Low accuracy on your emails | Add more labeled examples to `emails.csv` and retrain |
| Port 8000 in use | `uvicorn app.main:app --port 8001` |

---

## Sample inputs for testing

Try these with `POST /predict` after starting the API:

| Type | Example text |
|------|----------------|
| Ham | `Team meeting moved to Thursday at 3 PM. Agenda is in the shared doc.` |
| Spam | `CONGRATULATIONS! You won $1,000,000. Click here to claim your prize now!!!` |

---

## License and use

Educational starter project. Use public datasets and respect their licenses if you expand the training data.

MIT License — free to use for learning and personal projects.

See `requirements.txt` for third-party library licenses.
