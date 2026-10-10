#!/usr/bin/env python3
"""Expand MindCare knowledge base from public counseling data + curated intents.

Sources:
  - Existing mindcare_dataset.json (80 intents)
  - Amod/mental_health_counseling_conversations (HF)
  - Extra curated intents (career/internship stress, academics, etc.)

Outputs:
  - mindcare_dataset.json  (merged intents)
  - counseling_rag.json    (Q&A pairs for retrieval)
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INTENTS_PATH = ROOT / "mindcare_dataset.json"
RAG_PATH = ROOT / "counseling_rag.json"

EXTRA_INTENTS = [
    {
        "tag": "internship_stress",
        "patterns": [
            "will I get an internship",
            "I didn't get an internship",
            "internship rejection",
            "worried about internship",
            "no internship offers",
            "scared I won't get hired",
            "career anxiety",
            "job search stress",
        ],
        "responses": [
            "Internship uncertainty is really stressful — it can feel like your whole future is on the line. "
            "I can't predict outcomes, but I can help you sit with the worry. What's weighing on you most about it?",
            "Job and internship pressure is exhausting. Your worth isn't defined by one application result. "
            "Want to talk about the fear behind it, or brainstorm one small next step that feels manageable?",
        ],
    },
    {
        "tag": "academic_pressure",
        "patterns": [
            "exam stress",
            "I will fail my exams",
            "too much studying",
            "academic burnout",
            "college pressure",
            "assignment overload",
            "grades anxiety",
        ],
        "responses": [
            "Academic pressure can feel crushing. It's okay to feel overwhelmed — you're not alone in that. "
            "What's the most urgent piece right now, and what would 'good enough' look like for today?",
            "Exams and grades can make everything feel high-stakes. Let's slow down for a moment — "
            "how is your body feeling with this stress?",
        ],
    },
    {
        "tag": "relationship_conflict",
        "patterns": [
            "fighting with my partner",
            "argument with friend",
            "family conflict",
            "they don't understand me",
            "toxic relationship",
        ],
        "responses": [
            "Relationship conflict is painful. It makes sense that you're hurt or frustrated. "
            "Do you want to unpack what happened, or focus on how you're feeling afterward?",
        ],
    },
    {
        "tag": "loneliness",
        "patterns": [
            "I feel so alone",
            "nobody understands me",
            "I have no friends",
            "isolated",
            "lonely all the time",
        ],
        "responses": [
            "Loneliness can feel heavy and invisible to others. Thank you for saying it out loud. "
            "I'm here with you — when does the loneliness feel strongest?",
        ],
    },
    {
        "tag": "burnout",
        "patterns": [
            "I am burned out",
            "burnout",
            "can't keep going at work",
            "exhausted all the time",
            "no motivation left",
        ],
        "responses": [
            "Burnout is your system asking for rest, not proof that you're failing. "
            "What has been draining you the most lately?",
        ],
    },
]


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def load_counseling_pairs(limit: int = 2500) -> list[dict]:
    print("Downloading Amod/mental_health_counseling_conversations ...")
    from datasets import load_dataset

    ds = load_dataset("Amod/mental_health_counseling_conversations", split="train")
    pairs = []
    for row in ds:
        ctx = (row.get("Context") or "").strip()
        resp = (row.get("Response") or "").strip()
        if len(ctx) < 20 or len(resp) < 40:
            continue
        if len(ctx) > 400:
            ctx = ctx[:400]
        if len(resp) > 600:
            resp = resp[:600] + "..."
        pairs.append({"context": ctx, "response": resp, "tokens": list(_tokenize(ctx))})
        if len(pairs) >= limit:
            break
    print(f"  kept {len(pairs)} counseling Q&A pairs")
    return pairs


def merge_intents(existing: list[dict], extra: list[dict]) -> list[dict]:
    by_tag = {i.get("tag"): dict(i) for i in existing if i.get("tag")}
    for intent in extra:
        tag = intent["tag"]
        if tag in by_tag:
            pats = list(dict.fromkeys(by_tag[tag].get("patterns", []) + intent.get("patterns", [])))
            resps = list(dict.fromkeys(by_tag[tag].get("responses", []) + intent.get("responses", [])))
            by_tag[tag]["patterns"] = pats
            by_tag[tag]["responses"] = resps
        else:
            by_tag[tag] = intent
    return list(by_tag.values())


def main():
    existing = {"intents": []}
    if INTENTS_PATH.exists():
        existing = json.loads(INTENTS_PATH.read_text(encoding="utf-8"))

    intents = merge_intents(existing.get("intents", []), EXTRA_INTENTS)
    INTENTS_PATH.write_text(
        json.dumps({"intents": intents}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {len(intents)} intents → {INTENTS_PATH.name}")

    try:
        pairs = load_counseling_pairs(limit=2500)
    except Exception as e:
        print(f"Counseling dataset download failed: {e}")
        pairs = []

    RAG_PATH.write_text(json.dumps({"pairs": pairs}, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(pairs)} RAG pairs → {RAG_PATH.name}")

    if pairs:
        try:
            from rag import build_embedding_cache

            info = build_embedding_cache(force=True)
            print(f"Built embedding cache → {info}")
        except Exception as e:
            print(f"Embedding cache build skipped: {e}")


if __name__ == "__main__":
    main()
