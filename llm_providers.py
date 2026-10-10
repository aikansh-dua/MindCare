"""MindCare Gemini LLM + RAG intent retrieval."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

DATASET_PATH = Path(__file__).resolve().parent / "mindcare_dataset.json"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()

SYSTEM_PROMPT = """You are MindCare, an empathetic mental health support companion.
Your role:
- Listen with warmth, validate feelings, and respond with clear, practical support.
- Use the provided knowledge-base context when it is relevant.
- Keep replies concise (2–5 short paragraphs) unless the user asks for more detail.
- Never claim to be a doctor or replace professional care.
- If the user appears to be in crisis, urge them to contact local emergency services
  or India helplines 1860 266 2345 / 9152987821 immediately.
- Do not invent medical diagnoses. Prefer supportive guidance and coping strategies.
"""

_LOW_PRIORITY_TAGS = {"default", "no-response", "fallback", "unknown", "out-of-scope"}
_STOPWORDS = {
    "i", "me", "my", "am", "im", "i'm", "a", "an", "the", "to", "and", "or", "of",
    "in", "on", "is", "it", "for", "you", "your", "so", "do", "how", "can", "what",
}


class IntentRetriever:
    """Retrieve relevant guidance from the MindCare intents dataset (RAG)."""

    def __init__(self, intents: list[dict[str, Any]]):
        self._compiled = []
        for intent in intents:
            patterns = [p.lower().strip() for p in intent.get("patterns", []) if p]
            responses = intent.get("responses") or []
            tag = (intent.get("tag") or "unknown").lower()
            if patterns and responses:
                self._compiled.append((tag, patterns, responses))

    @staticmethod
    def _score_intent(query_l: str, query_tokens: set[str], tag: str, patterns: list[str]) -> float:
        score = 0.0
        tag_key = tag.replace("-", " ").replace("_", " ")
        if tag_key and re.search(rf"\b{re.escape(tag_key)}\b", query_l):
            score = max(score, 70.0)

        aliases = {
            "anxious": ["anxiety", "anxious", "panic"],
            "stressed": ["stress", "stressed", "overwhelmed", "pressure"],
            "sad": ["sad", "lonely", "down"],
            "depressed": ["depression", "depressed", "empty"],
            "sleep": ["sleep", "insomnia", "sleepless", "awake"],
            "greeting": ["hello", "hi", "hey", "namaste", "hola"],
        }
        for alias in aliases.get(tag, []):
            if re.search(rf"\b{re.escape(alias)}\b", query_l):
                score = max(score, 70.0)

        for pattern in patterns:
            if not pattern:
                continue
            if pattern == query_l:
                score = max(score, 100.0)
                continue

            pattern_tokens = set(re.findall(r"[a-z0-9']+", pattern))
            content_tokens = pattern_tokens - _STOPWORDS

            if len(pattern) >= 8 and (pattern in query_l or query_l in pattern):
                score = max(score, 55.0 + min(len(pattern), 25))
                continue

            if len(content_tokens) <= 1:
                for tok in content_tokens:
                    if len(tok) >= 4 and re.search(rf"\b{re.escape(tok)}\b", query_l):
                        score = max(score, 45.0)
                continue

            if not pattern_tokens:
                continue
            if content_tokens:
                content_overlap = len(query_tokens & content_tokens) / len(content_tokens)
            else:
                content_overlap = 0.0
            overlap = len(query_tokens & pattern_tokens) / len(pattern_tokens)
            if content_overlap >= 0.5:
                score = max(score, 30.0 + content_overlap * 40)
            elif overlap >= 0.7:
                score = max(score, overlap * 35)

        if tag in _LOW_PRIORITY_TAGS:
            score *= 0.15
        return score

    def best_match(self, user_query: str, min_score: float = 25.0) -> dict[str, Any] | None:
        query_l = user_query.lower()
        query_tokens = set(re.findall(r"[a-z0-9']+", query_l))
        best_score, best = 0.0, None
        for tag, patterns, responses in self._compiled:
            score = self._score_intent(query_l, query_tokens, tag, patterns)
            if score > best_score:
                best_score = score
                best = {"tag": tag, "score": score, "responses": responses}
        if best and best["score"] >= min_score:
            return best
        return None


class GeminiMindCareLLM:
    """Gemini chat model with MindCare system prompt."""

    def __init__(self, api_key: str, model_name: str):
        from langchain_google_genai import ChatGoogleGenerativeAI

        self._chat = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.6,
            max_output_tokens=1024,
        )

    def invoke(self, prompt: str, max_tokens: int = 1024) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage

        result = self._chat.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
        )
        content = result.content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and "text" in item:
                    parts.append(item["text"])
            content = "".join(parts)
        return str(content).strip()


def load_intents() -> list[dict[str, Any]]:
    try:
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "intents" in data:
            return data["intents"]
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


def dataset_advice_for_query(user_query: str, intents: list[dict[str, Any]] | None = None) -> str | None:
    intents = intents if intents is not None else load_intents()
    match = IntentRetriever(intents).best_match(user_query)
    if not match:
        return None
    return f"Topic={match['tag']}; Example guidance: {match['responses'][0]}"


def build_models():
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Get a free key at https://aistudio.google.com/apikey "
            "and add it to your .env file."
        )

    llm = GeminiMindCareLLM(GEMINI_API_KEY, GEMINI_MODEL)
    # Keep routing keys used by chat logic (emotion / casual routing)
    return {
        "default": llm,
        "deep_emotion": llm,
        "casual": llm,
    }


def provider_status() -> dict[str, Any]:
    return {
        "provider": "gemini",
        "model": GEMINI_MODEL,
        "api_key_configured": bool(GEMINI_API_KEY),
        "dataset_intents": len(load_intents()),
    }
