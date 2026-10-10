"""Crisis detection — keyword gate + BART MNLI confirmation.

BART alone over-fires on sad/breakup messages, so we require clear
crisis language (or very high suicide/self-harm scores) before escalating.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

import torch
from transformers import pipeline

CRISIS_MODEL = "facebook/bart-large-mnli"

CRISIS_LABELS = [
    "suicide risk",
    "self-harm",
    "severe hopelessness",
    "immediate danger",
    "emotional distress",
    "seeking support",
]

# Only these can escalate — and only at high confidence
CRITICAL_LABELS = {"suicide risk", "self-harm"}
CRITICAL_SCORE = 0.92
CONFIRM_SCORE = 0.80

# Explicit phrases that indicate real crisis intent
CRISIS_PATTERNS = [
    r"\bkill myself\b",
    r"\bend my life\b",
    r"\btake my (own )?life\b",
    r"\bsuicid(e|al)\b",
    r"\bself[-\s]?harm\b",
    r"\bcut myself\b",
    r"\bwant to die\b",
    r"\bwanna die\b",
    r"\bdon't want to (live|be alive)\b",
    r"\bdo not want to (live|be alive)\b",
    r"\bno reason to live\b",
    r"\bbetter off dead\b",
    r"\bhurt myself\b",
]

_CRISIS_RE = re.compile("|".join(CRISIS_PATTERNS), re.IGNORECASE)


@lru_cache(maxsize=1)
def get_crisis_pipeline():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "zero-shot-classification",
        model=CRISIS_MODEL,
        device=device,
    )


def _keyword_hit(text: str) -> bool:
    return bool(_CRISIS_RE.search(text))


def detect_crisis(text: str) -> dict[str, Any]:
    """
    Return crisis=True only when:
      1) explicit crisis keywords are present, AND BART confirms
         suicide risk / self-harm >= CONFIRM_SCORE, OR
      2) no keywords, but suicide risk OR self-harm >= CRITICAL_SCORE
         (catches paraphrases that still score extremely high)
    """
    clf = get_crisis_pipeline()
    result = clf(text[:512], candidate_labels=CRISIS_LABELS, multi_label=True)

    ranked = [
        {"label": lab, "score": round(float(sc), 4)}
        for lab, sc in zip(result["labels"], result["scores"])
    ]
    score_map = {item["label"]: item["score"] for item in ranked}
    keywords = _keyword_hit(text)

    critical_scores = {lab: score_map.get(lab, 0.0) for lab in CRITICAL_LABELS}
    max_critical = max(critical_scores.values()) if critical_scores else 0.0

    crisis = False
    triggered = []
    reason = "none"

    if keywords:
        for lab, sc in critical_scores.items():
            if sc >= CONFIRM_SCORE:
                crisis = True
                triggered.append(lab)
        # Keyword alone with strong "immediate danger" also escalates
        if not crisis and score_map.get("immediate danger", 0) >= CONFIRM_SCORE:
            crisis = True
            triggered.append("immediate danger")
        reason = "keyword+model" if crisis else "keyword_without_confirm"
        # Explicit "kill myself" / "suicide" style phrases: escalate even if MNLI is unsure
        if not crisis and re.search(
            r"\b(kill myself|end my life|suicid|want to die|wanna die|better off dead)\b",
            text,
            re.I,
        ):
            crisis = True
            triggered.append("explicit_keyword")
            reason = "explicit_keyword"
    else:
        # No crisis language → only escalate on extremely high critical scores
        for lab, sc in critical_scores.items():
            if sc >= CRITICAL_SCORE:
                crisis = True
                triggered.append(lab)
        reason = "high_confidence_model" if crisis else "no_crisis"

    return {
        "crisis": crisis,
        "triggered_labels": triggered,
        "scores": ranked,
        "keyword_hit": keywords,
        "reason": reason,
        "max_critical_score": round(max_critical, 4),
    }
