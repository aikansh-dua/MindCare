"""Emotion detection — GoEmotions DistilBERT (28 emotions)."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import torch
from transformers import pipeline

EMOTION_MODEL = "joeddav/distilbert-base-uncased-go-emotions-student"

# GoEmotions label set (28 classes used by this student model)
GO_EMOTIONS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization", "relief",
    "remorse", "sadness", "surprise", "neutral",
]


def _device() -> int | str:
    if torch.cuda.is_available():
        return 0
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return -1


@lru_cache(maxsize=1)
def get_emotion_pipeline():
    device = _device()
    return pipeline(
        "text-classification",
        model=EMOTION_MODEL,
        tokenizer=EMOTION_MODEL,
        top_k=None,
        device=device if device != "mps" else -1,  # MPS can be flaky with pipelines; CPU is safe
    )


def detect_emotion(text: str, top_k: int = 3) -> dict[str, Any]:
    """Return primary emotion and top-k scores."""
    clf = get_emotion_pipeline()
    raw = clf(text[:512])
    # pipeline with top_k=None returns [[{label, score}, ...]]
    scores = raw[0] if isinstance(raw, list) and raw and isinstance(raw[0], list) else raw
    ranked = sorted(scores, key=lambda x: float(x["score"]), reverse=True)
    primary = ranked[0]["label"] if ranked else "neutral"
    return {
        "emotion": primary,
        "confidence": float(ranked[0]["score"]) if ranked else 0.0,
        "top_emotions": [
            {"label": r["label"], "score": round(float(r["score"]), 4)}
            for r in ranked[:top_k]
        ],
    }
