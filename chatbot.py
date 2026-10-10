"""MindCare orchestration — emotion + crisis + RAG + generative reply."""

from __future__ import annotations

import json
import random
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from crisis import detect_crisis
from emotion import detect_emotion
from generation import generate_reply, generation_status
from memory import get_store
from rag import rag_status, retrieve_counseling

DATASET_PATH = Path(__file__).resolve().parent / "mindcare_dataset.json"

CRISIS_REPLY = (
    "I'm really concerned about your safety. Please seek help right away. "
    "In India you can call 1860 266 2345 or 9152987821. "
    "If you are in immediate danger, contact local emergency services. You are not alone."
)


@lru_cache(maxsize=1)
def _load_intents() -> list[dict]:
    try:
        with open(DATASET_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("intents", []) if isinstance(data, dict) else data
    except Exception:
        return []


def _tokenize(text: str) -> set[str]:
    stop = {
        "i", "me", "my", "am", "im", "a", "an", "the", "to", "and", "or", "of",
        "in", "on", "is", "it", "for", "you", "your", "so", "do", "how", "can",
        "will", "get", "an", "be", "with",
    }
    return set(re.findall(r"[a-z0-9']+", text.lower())) - stop


def _best_intent(user_message: str) -> dict[str, Any] | None:
    query = user_message.lower()
    query_tokens = _tokenize(user_message)
    best = None
    best_score = 0.0

    aliases = {
        "sad": ["sad", "lonely", "down", "upset", "cry", "crying"],
        "depressed": ["depression", "depressed", "empty"],
        "anxious": ["anxiety", "anxious", "panic", "worried"],
        "stressed": ["stress", "stressed", "overwhelmed"],
        "internship_stress": ["internship", "intern", "job", "career", "hire", "hiring"],
        "academic_pressure": ["exam", "exams", "grades", "college", "assignment", "study"],
        "burnout": ["burnout", "burned", "exhausted"],
        "loneliness": ["alone", "lonely", "isolated"],
        "greeting": ["hi", "hello", "hey"],
    }

    if re.search(r"\b(broke\s*up|break\s*up|breakup|gf|bf|girlfriend|boyfriend)\b", query):
        for intent in _load_intents():
            if (intent.get("tag") or "").lower() in {"sad", "depressed", "relationship_conflict"}:
                responses = intent.get("responses") or []
                if responses:
                    return {
                        "tag": intent["tag"],
                        "score": 80.0,
                        "response": random.choice(responses),
                    }

    for intent in _load_intents():
        tag = (intent.get("tag") or "").lower()
        if tag in {"default", "no-response"}:
            continue
        responses = intent.get("responses") or []
        patterns = [p.lower() for p in intent.get("patterns", [])]
        if not responses:
            continue

        score = 0.0
        if tag and re.search(rf"\b{re.escape(tag.replace('_', ' '))}\b", query):
            score = max(score, 70.0)
        for alias in aliases.get(tag, []):
            if re.search(rf"\b{re.escape(alias)}\b", query):
                score = max(score, 65.0)
        for pattern in patterns:
            if pattern == query:
                score = max(score, 100.0)
            elif len(pattern) >= 8 and pattern in query:
                score = max(score, 55.0)
            else:
                ptoks = _tokenize(pattern)
                if ptoks:
                    overlap = len(query_tokens & ptoks) / len(ptoks)
                    if overlap >= 0.5:
                        score = max(score, 30.0 + overlap * 50)

        if score > best_score:
            best_score = score
            best = {"tag": tag, "score": score, "response": random.choice(responses)}

    return best if best and best_score >= 35 else None


def _supportive_template(user_message: str, emotion: str) -> str | None:
    """Greetings only — everything else goes through generation (+ failover)."""
    q = user_message.lower().strip()
    if re.match(r"^(hi|hello|hey|hola|namaste)\b[!?.]*$", q) or (
        re.match(r"^(hi|hello|hey)\b", q)
        and len(q.split()) <= 3
        and not re.search(r"\b(sad|anxious|stress|lonely|hurt|broke|depress|panic)\b", q)
    ):
        return random.choice(
            [
                "Hi there. I'm MindCare — how are you feeling today?",
                "Hello. I'm here with you. What's on your mind?",
                "Hey. Glad you reached out. How are you doing right now?",
            ]
        )
    return None


def _safe_fallback_reply(message: str, emotion: str, intent: dict | None) -> tuple[str, str]:
    """Last resort if all generators fail — never paste raw RAG essays."""
    if intent and intent.get("score", 0) >= 55:
        return intent["response"], "dataset_fallback"
    return (
        "Thanks for sharing that. It sounds like something's weighing on you. "
        "I'm here with you — what about this feels hardest right now?",
        "safe_fallback",
    )


def _build_generation_prompt(
    user_message: str,
    memory_ctx: dict[str, Any],
    emotion: str,
    intent: dict | None,
    rag_hits: list[dict],
) -> str:
    history = memory_ctx.get("recent") or []
    summary = (memory_ctx.get("summary") or "").strip()
    profile = memory_ctx.get("profile") or {}

    turns = []
    for m in history:
        if m["role"] != "user" and "concerned about your safety" in m.get("text", "").lower():
            continue
        who = "User" if m["role"] == "user" else "MindCare"
        turns.append(f"{who}: {m['text']}")
    history_block = "\n".join(turns) if turns else "(none)"

    profile_bits = []
    if profile.get("concerns"):
        profile_bits.append("concerns: " + ", ".join(profile["concerns"][:8]))
    if profile.get("triggers"):
        profile_bits.append("triggers: " + "; ".join(profile["triggers"][:5]))
    if profile.get("what_helped"):
        profile_bits.append("what helped before: " + "; ".join(profile["what_helped"][:5]))
    if profile.get("last_emotion"):
        profile_bits.append(f"last emotion: {profile['last_emotion']}")
    profile_block = "; ".join(profile_bits) if profile_bits else "(new user)"

    rag_bits = []
    if intent and intent.get("score", 0) >= 50:
        rag_bits.append(f"- Intent hint ({intent['tag']}): {intent['response']}")
    for hit in rag_hits:
        rag_bits.append(
            f"- Similar counseling note: Q: {hit['context'][:160]} → A: {hit['response'][:180]}"
        )
    rag_block = "\n".join(rag_bits) if rag_bits else "- (no close knowledge match — reply from empathy only)"

    summary_block = summary if summary else "(none)"

    return (
        "Stay on-topic for the user's latest message. "
        "Do not invent unrelated therapy advice (e.g. anger worksheets) if they talked about something else. "
        "Use memory/profile only when it clearly helps; do not over-reference old topics.\n"
        f"Detected emotion: {emotion}\n"
        f"User profile: {profile_block}\n"
        f"Earlier conversation summary: {summary_block}\n"
        f"Knowledge context (use only if clearly relevant):\n{rag_block}\n\n"
        f"Recent conversation:\n{history_block}\n\n"
        f"User: {user_message}\n"
        f"Write MindCare's next reply:"
    )


def chat(
    user_id: str,
    message: str,
    *,
    owner_id: str | None = None,
) -> dict[str, Any]:
    """
    user_id: conversation id (history / summary key)
    owner_id: browser owner for profile + session index (defaults to user_id)
    """
    store = get_store()
    profile_id = owner_id or user_id
    emotion_info = detect_emotion(message)
    crisis_info = detect_crisis(message)

    if crisis_info["crisis"]:
        store.append(user_id, "user", message)
        store.append(user_id, "assistant", CRISIS_REPLY)
        store.update_profile_from_turn(
            profile_id, message, emotion=emotion_info["emotion"], reply=CRISIS_REPLY
        )
        if owner_id:
            from memory import resolve_conversation_title

            hist = store.get_history(user_id)
            meta = store.get_conversation(owner_id, user_id) or {}
            title = resolve_conversation_title(
                hist,
                meta.get("title"),
                emotion_info["emotion"],
                first_message_only=True,
            )
            store.touch_conversation(
                owner_id,
                user_id,
                title=title,
                message_count=len(hist),
                force_title=bool(title),
            )
        return {
            "response": CRISIS_REPLY,
            "emotion": emotion_info["emotion"],
            "emotion_details": emotion_info,
            "crisis": True,
            "crisis_details": crisis_info,
            "memory_backend": store.backend,
            "conversation_id": user_id,
            "generation": generation_status(),
        }

    memory_ctx = store.context_for_prompt(user_id)
    # Use owner profile in prompt when available
    if owner_id:
        memory_ctx["profile"] = store.get_profile(owner_id)

    intent = _best_intent(message)
    rag_hits = retrieve_counseling(message)

    reply = _supportive_template(message, emotion_info["emotion"])
    source = "template" if reply else None
    gen_error = None
    failover_errors: list[str] = []

    if not reply:
        try:
            prompt = _build_generation_prompt(
                message, memory_ctx, emotion_info["emotion"], intent, rag_hits
            )
            reply, source, failover_errors = generate_reply(prompt)
        except Exception as e:
            gen_error = f"{type(e).__name__}: {e}"
            reply, source = _safe_fallback_reply(message, emotion_info["emotion"], intent)

    if not reply:
        reply, source = _safe_fallback_reply(message, emotion_info["emotion"], intent)

    # Title from conversation substance (not greetings / first-line copy)
    prior = store.get_history(user_id)

    store.append(user_id, "user", message)
    store.append(user_id, "assistant", reply)
    profile = store.update_profile_from_turn(
        profile_id, message, emotion=emotion_info["emotion"], reply=reply
    )

    session_meta = None
    if owner_id:
        from memory import resolve_conversation_title

        hist = store.get_history(user_id)
        meta = store.get_conversation(owner_id, user_id) or {}
        # Fast title from first substantial message only (Flash polish happens on leave).
        title = resolve_conversation_title(
            hist,
            meta.get("title"),
            emotion_info["emotion"],
            first_message_only=True,
        )
        session_meta = store.touch_conversation(
            owner_id,
            user_id,
            title=title,
            message_count=len(hist),
            force_title=bool(title),
        )

    result = {
        "response": reply,
        "emotion": emotion_info["emotion"],
        "emotion_details": emotion_info,
        "crisis": False,
        "crisis_details": crisis_info,
        "reply_source": source,
        "rag_hits": len(rag_hits),
        "rag": rag_status(),
        "intent": intent["tag"] if intent else None,
        "memory_backend": store.backend,
        "conversation_id": user_id,
        "conversation_title": (session_meta or {}).get("title"),
        "profile": {
            "concerns": profile.get("concerns", []),
            "triggers": profile.get("triggers", []),
            "what_helped": profile.get("what_helped", []),
            "turn_count": profile.get("turn_count", 0),
        },
        "has_summary": bool(store.get_summary(user_id)),
        "generation": generation_status(),
    }
    if gen_error:
        result["generation_error"] = gen_error[:240]
    if failover_errors:
        result["failover_errors"] = [e[:160] for e in failover_errors[:3]]
    return result


def model_status() -> dict[str, Any]:
    return {**generation_status(), "rag": rag_status()}
