"""FastAPI backend for spam email classification."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.predict import get_detector


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        get_detector()
    except FileNotFoundError:
        pass
    yield


app = FastAPI(
    title="Spam Email Detector",
    description="Classify email text as spam or ham using TF-IDF + Naive Bayes.",
    version="1.0.0",
    lifespan=lifespan,
)


class EmailRequest(BaseModel):
    email_text: str = Field(..., min_length=1, description="Full email body or subject+body")


class PredictionResponse(BaseModel):
    label: str
    is_spam: bool
    confidence: float
    spam_probability: float | None = None
    message: str | None = None


@app.get("/")
def root():
    return {
        "service": "Spam Email Detector",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict",
    }


@app.get("/health")
def health():
    ready = False
    try:
        get_detector()
        ready = True
    except FileNotFoundError:
        pass
    return {"status": "ok", "model_loaded": ready}


@app.post("/predict", response_model=PredictionResponse)
def predict_email(body: EmailRequest):
    try:
        detector = get_detector()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail="Model not trained. Run `python train.py` from the project root.",
        ) from exc

    result = detector.predict(body.email_text)
    return PredictionResponse(**result)
