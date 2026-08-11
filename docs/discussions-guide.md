# Discussions & Q&A Guide

Use [GitHub Discussions](https://github.com/swapnilswami332/spam-email-detector/discussions) for questions about training, the API, or NLP concepts in this project.

## Where to post

| Category | Use for |
|----------|---------|
| **Q&A** | How-to questions (training, API errors, dataset format) |
| **General** | Feedback and ideas |
| **Ideas** | Feature suggestions |

## Sample questions (for collaborators)

If you are helping test or learn with this repo, post one of these in **Q&A**:

1. **How do I train the model after editing `emails.csv`?**
2. **Why does `/predict` return 503 Model not trained?**
3. **What NLP steps does `normalize_text()` perform?**
4. **How do I call the API from PowerShell or curl?**

## Sample answers (maintainer)

### Q: How do I train the model after editing `emails.csv`?

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train.py
```

This saves `models/spam_classifier.joblib`. Restart the API with `uvicorn app.main:app --reload`.

### Q: Why does `/predict` return 503?

The API loads `models/spam_classifier.joblib` at startup. Run `python train.py` first. Check `/health` — `model_loaded` should be `true`.

### Q: What NLP steps does preprocessing use?

`app/preprocess.py` lowercases text, removes URLs/emails/digits, strips punctuation, tokenizes with NLTK, removes English stopwords, lemmatizes words, and returns a cleaned string for TF-IDF.

### Q: How do I call the API?

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8001

Invoke-RestMethod -Uri "http://127.0.0.1:8001/predict" -Method POST `
  -ContentType "application/json" `
  -Body '{"email_text":"Meeting notes attached for tomorrow."}'
```

Or open http://127.0.0.1:8001/docs for the interactive Swagger UI.

## Marking an answer (question author)

1. Open the Q&A thread.
2. Find the reply that solves your question.
3. Click **Mark as answer** under that comment.

Only the **person who asked the question** can mark an answer.
