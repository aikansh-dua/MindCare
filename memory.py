"""Conversation memory — Redis with in-memory fallback.

Supports multiple titled conversations per browser owner (ChatGPT-style recents),
plus rolling summaries and a structured profile per owner.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from typing import Any

REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
MEMORY_KEY_PREFIX = "mindcare:memory:"
PROFILE_KEY_PREFIX = "mindcare:profile:"
SUMMARY_KEY_PREFIX = "mindcare:summary:"
SESSIONS_KEY_PREFIX = "mindcare:sessions:"
MAX_TURNS = int(os.getenv("MEMORY_MAX_TURNS", "40"))
SUMMARY_AFTER_TURNS = int(os.getenv("MEMORY_SUMMARY_AFTER", "12"))
KEEP_RECENT_TURNS = int(os.getenv("MEMORY_KEEP_RECENT", "8"))
MEMORY_TTL_SECONDS = int(os.getenv("MEMORY_TTL_SECONDS", "0"))
MAX_SESSIONS = int(os.getenv("MEMORY_MAX_SESSIONS", "40"))

CONCERN_PATTERNS: list[tuple[str, str]] = [
    (r"\b(anxi\w*|panic|worried|worry)\b", "anxiety"),
    (r"\b(stress\w*|overwhelm\w*)\b", "stress"),
    (r"\b(depress\w*|hopeless|empty)\b", "depression"),
    (r"\b(sleep|insomnia|can't sleep|cannot sleep)\b", "sleep"),
    (r"\b(lonely|alone|isolat\w*)\b", "loneliness"),
    (r"\b(intern(ship)?|job search|career)\b", "internship"),
    (r"\b(exam\w*|grades?|college|assignment)\b", "academic"),
    (r"\b(burnout|burned out|exhausted)\b", "burnout"),
    (r"\b(broke\s*up|break\s*up|breakup|dumped)\b", "breakup"),
    (r"\b(gf|bf|girlfriend|boyfriend|partner)\b.*\b(left|gone|ended)\b", "breakup"),
]


def empty_profile(user_id: str) -> dict[str, Any]:
    return {
        "user_id": user_id,
        "concerns": [],
        "triggers": [],
        "what_helped": [],
        "last_emotion": None,
        "turn_count": 0,
    }


GREETING_RE = re.compile(
    r"^(hi+|hello|hey|hola|namaste|yo|sup|hii+|hello\s*ji|hey\s*ji|"
    r"good\s*(morning|afternoon|evening|night)|kaise\s*ho|how\s*are\s*you)"
    r"[\s!.?]*$",
    re.IGNORECASE,
)


def is_greeting_message(text: str) -> bool:
    t = " ".join((text or "").strip().split())
    if not t:
        return True
    if GREETING_RE.match(t):
        return True
    # Very short social openers
    return len(t.split()) <= 3 and bool(
        re.search(r"\b(hi|hello|hey|hola|namaste)\b", t, re.IGNORECASE)
    )


def is_weak_title(title: str | None) -> bool:
    if not title or title in {"New conversation", "This moment", "Untitled moment"}:
        return True
    return is_greeting_message(title)


def title_from_message(text: str) -> str:
    """Short title from a single substantial message."""
    cleaned = " ".join((text or "").strip().split())
    cleaned = re.sub(r"^[\W_]+", "", cleaned)
    if not cleaned or is_greeting_message(cleaned):
        return "New conversation"
    for sep in [".", "?", "!", "\n"]:
        if sep in cleaned:
            cleaned = cleaned.split(sep)[0].strip()
            break
    if len(cleaned) > 52:
        cut = cleaned[:49]
        if " " in cut:
            cut = cut.rsplit(" ", 1)[0]
        cleaned = cut + "…"
    return cleaned[:1].upper() + cleaned[1:] if cleaned else "New conversation"


def heuristic_conversation_title(history: list[dict[str, Any]], emotion: str | None = None) -> str:
    """Build a title from user substance, not greetings."""
    user_msgs = [
        (m.get("text") or "").strip()
        for m in history
        if m.get("role") == "user" and (m.get("text") or "").strip()
    ]
    substantial = [m for m in user_msgs if not is_greeting_message(m)]
    if not substantial:
        return "New conversation"

    primary = substantial[-1] if len(substantial) == 1 else substantial[0]
    # Prefer the most contentful line
    primary = max(substantial, key=lambda s: len(s))

    theme_bits = []
    blob = " ".join(substantial).lower()
    themes = [
        (r"\b(sad|dukhi|upset|cry|crying|hurt)\b", "Feeling low"),
        (r"\b(anxi\w*|panic|worried|tension)\b", "Anxiety"),
        (r"\b(stress\w*|overwhelm\w*)\b", "Feeling overwhelmed"),
        (r"\b(sleep|insomnia)\b", "Sleep struggles"),
        (r"\b(lonely|alone|isolat)\b", "Feeling alone"),
        (r"\b(break\s*up|breakup|dumped|gf|bf|girlfriend|boyfriend)\b", "Relationship pain"),
        (r"\b(intern|job|career)\b", "Career pressure"),
        (r"\b(exam|college|grades|study)\b", "Academic stress"),
        (r"\b(burnout|exhausted)\b", "Burnout"),
    ]
    for pattern, label in themes:
        if re.search(pattern, blob):
            theme_bits.append(label)
            break

    if theme_bits:
        return theme_bits[0]

    if emotion and emotion not in {"neutral", "curiosity", "approval", "caring"}:
        soft = {
            "sadness": "Feeling sad",
            "grief": "Grief",
            "remorse": "Heavy feelings",
            "nervousness": "Feeling anxious",
            "fear": "Feeling afraid",
            "anger": "Feeling angry",
            "disappointment": "Feeling disappointed",
        }
        if emotion in soft:
            return soft[emotion]

    return title_from_message(primary)


def resolve_conversation_title(
    history: list[dict[str, Any]],
    current_title: str | None,
    emotion: str | None = None,
    *,
    prefer_llm: bool = False,
    first_message_only: bool = False,
) -> str | None:
    """
    Return a new title when we should update, else None.

    - first_message_only: set a fast heuristic title once from the first
      substantial user message (chat path — never calls Gemini).
    - prefer_llm: Flash polish when the conversation ends.
    """
    user_msgs = [
        (m.get("text") or "").strip()
        for m in history
        if m.get("role") == "user" and (m.get("text") or "").strip()
    ]
    substantial = [m for m in user_msgs if not is_greeting_message(m)]
    if not substantial:
        return None

    weak = is_weak_title(current_title)

    if first_message_only:
        # Title once from the first real message; do not keep rewriting mid-chat.
        if not weak:
            return None
        return heuristic_conversation_title(history, emotion)

    if prefer_llm:
        try:
            from generation import generate_conversation_title

            llm_title = generate_conversation_title(history)
            if llm_title and not is_weak_title(llm_title):
                return llm_title
        except Exception:
            pass
        # Fall back to heuristic if Flash fails and title is still weak
        if weak:
            return heuristic_conversation_title(history, emotion)
        return None

    if not (weak or len(substantial) >= 2):
        return None
    return heuristic_conversation_title(history, emotion)

def _set_with_ttl(client, key: str, value: str) -> None:
    if MEMORY_TTL_SECONDS > 0:
        client.setex(key, MEMORY_TTL_SECONDS, value)
    else:
        client.set(key, value)


class MemoryStore:
    def __init__(self):
        self._redis = None
        self._local_memory: dict[str, list] = {}
        self._local_profiles: dict[str, dict] = {}
        self._local_summaries: dict[str, str] = {}
        self._local_sessions: dict[str, list] = {}
        self.backend = "memory"

        try:
            import redis

            client = redis.Redis.from_url(
                REDIS_URL, decode_responses=True, socket_connect_timeout=1
            )
            client.ping()
            self._redis = client
            self.backend = "redis"
        except Exception:
            self._redis = None
            self.backend = "memory"

    # ---- history / summary (keyed by conversation id) ----

    def get_history(self, user_id: str) -> list[dict[str, Any]]:
        if self._redis is not None:
            raw = self._redis.get(MEMORY_KEY_PREFIX + user_id)
            return json.loads(raw) if raw else []
        return list(self._local_memory.get(user_id, []))

    def get_summary(self, user_id: str) -> str:
        if self._redis is not None:
            return self._redis.get(SUMMARY_KEY_PREFIX + user_id) or ""
        return self._local_summaries.get(user_id, "")

    def set_summary(self, user_id: str, summary: str) -> None:
        summary = (summary or "").strip()
        if self._redis is not None:
            if summary:
                _set_with_ttl(self._redis, SUMMARY_KEY_PREFIX + user_id, summary)
            else:
                self._redis.delete(SUMMARY_KEY_PREFIX + user_id)
        else:
            if summary:
                self._local_summaries[user_id] = summary
            else:
                self._local_summaries.pop(user_id, None)

    def append(self, user_id: str, role: str, text: str) -> list[dict[str, Any]]:
        history = self.get_history(user_id)
        history.append({"role": role, "text": text})
        history = self._maybe_summarize(user_id, history)
        history = history[-MAX_TURNS:]
        self._save_history(user_id, history)
        return history

    def set_history(self, user_id: str, history: list[dict[str, Any]]) -> None:
        if not history:
            self.set_summary(user_id, "")
        self._save_history(user_id, history[-MAX_TURNS:])

    def clear_user(self, user_id: str) -> None:
        self.set_history(user_id, [])
        self.set_summary(user_id, "")
        self.save_profile(user_id, empty_profile(user_id))

    def _maybe_summarize(
        self, user_id: str, history: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if len(history) <= SUMMARY_AFTER_TURNS:
            return history

        keep = max(KEEP_RECENT_TURNS, 4)
        older = history[:-keep]
        recent = history[-keep:]
        existing = self.get_summary(user_id)
        chunk = _extractive_summary(older)
        merged = _merge_summaries(existing, chunk)
        self.set_summary(user_id, merged)
        return recent

    def _save_history(self, user_id: str, history: list[dict[str, Any]]) -> None:
        if self._redis is not None:
            _set_with_ttl(
                self._redis, MEMORY_KEY_PREFIX + user_id, json.dumps(history)
            )
        else:
            self._local_memory[user_id] = history

    # ---- profile (keyed by owner id) ----

    def get_profile(self, user_id: str) -> dict[str, Any]:
        default = empty_profile(user_id)
        if self._redis is not None:
            raw = self._redis.get(PROFILE_KEY_PREFIX + user_id)
            if not raw:
                return default
            profile = json.loads(raw)
        else:
            profile = dict(self._local_profiles.get(user_id, default))

        for key, value in default.items():
            profile.setdefault(key, value if not isinstance(value, list) else list(value))
        return profile

    def save_profile(self, user_id: str, profile: dict[str, Any]) -> None:
        profile = {**empty_profile(user_id), **profile, "user_id": user_id}
        if self._redis is not None:
            _set_with_ttl(
                self._redis, PROFILE_KEY_PREFIX + user_id, json.dumps(profile)
            )
        else:
            self._local_profiles[user_id] = profile

    def update_profile_from_turn(
        self,
        user_id: str,
        message: str,
        emotion: str | None = None,
        reply: str | None = None,
    ) -> dict[str, Any]:
        profile = self.get_profile(user_id)
        text = message.lower()

        for pattern, concern in CONCERN_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                if concern not in profile["concerns"]:
                    profile["concerns"].append(concern)

        trig = re.search(
            r"\b(?:when|after|whenever)\s+(.{5,60}?)(?:\.|,|$)",
            message,
            re.IGNORECASE,
        )
        if trig:
            phrase = trig.group(1).strip()
            if phrase and phrase not in profile["triggers"] and len(profile["triggers"]) < 8:
                profile["triggers"].append(phrase[:80])

        if reply and re.search(
            r"\b(that helped|that helps|feel(?:s|ing)? better|worked|useful)\b",
            message,
            re.IGNORECASE,
        ):
            tip = reply.strip().split(".")[0][:100]
            if tip and tip not in profile["what_helped"] and len(profile["what_helped"]) < 8:
                profile["what_helped"].append(tip)

        if emotion:
            profile["last_emotion"] = emotion
        profile["turn_count"] = int(profile.get("turn_count") or 0) + 1
        profile["concerns"] = profile["concerns"][-12:]
        profile["triggers"] = profile["triggers"][-8:]
        profile["what_helped"] = profile["what_helped"][-8:]
        self.save_profile(user_id, profile)
        return profile

    def context_for_prompt(self, user_id: str, recent_n: int = 6) -> dict[str, Any]:
        history = self.get_history(user_id)
        profile = self.get_profile(user_id)
        return {
            "summary": self.get_summary(user_id),
            "recent": history[-recent_n:],
            "profile": profile,
        }

    # ---- multi-conversation index (per owner) ----

    def list_conversations(self, owner_id: str) -> list[dict[str, Any]]:
        sessions = self._load_sessions(owner_id)
        sessions.sort(key=lambda s: s.get("updated_at", 0), reverse=True)
        return sessions

    def create_conversation(self, owner_id: str, title: str | None = None) -> dict[str, Any]:
        now = time.time()
        session = {
            "id": uuid.uuid4().hex[:12],
            "title": title or "New conversation",
            "updated_at": now,
            "created_at": now,
            "message_count": 0,
        }
        sessions = self._load_sessions(owner_id)
        sessions.insert(0, session)
        self._save_sessions(owner_id, sessions[:MAX_SESSIONS])
        self.set_history(session["id"], [])
        return session

    def get_conversation(self, owner_id: str, conversation_id: str) -> dict[str, Any] | None:
        for s in self._load_sessions(owner_id):
            if s.get("id") == conversation_id:
                return {
                    **s,
                    "memory": self.get_history(conversation_id),
                    "summary": self.get_summary(conversation_id),
                }
        return None

    def delete_conversation(self, owner_id: str, conversation_id: str) -> bool:
        sessions = self._load_sessions(owner_id)
        new_sessions = [s for s in sessions if s.get("id") != conversation_id]
        if len(new_sessions) == len(sessions):
            return False
        self._save_sessions(owner_id, new_sessions)
        self.set_history(conversation_id, [])
        self.set_summary(conversation_id, "")
        return True

    def touch_conversation(
        self,
        owner_id: str,
        conversation_id: str,
        *,
        title: str | None = None,
        message_count: int | None = None,
        force_title: bool = False,
    ) -> dict[str, Any] | None:
        sessions = self._load_sessions(owner_id)
        found = None
        for s in sessions:
            if s.get("id") == conversation_id:
                s["updated_at"] = time.time()
                if title:
                    current = s.get("title") or "New conversation"
                    if force_title or is_weak_title(current) or current == "New conversation":
                        s["title"] = title
                if message_count is not None:
                    s["message_count"] = message_count
                found = s
                break
        if not found:
            return None
        sessions.sort(key=lambda x: x.get("updated_at", 0), reverse=True)
        self._save_sessions(owner_id, sessions[:MAX_SESSIONS])
        return found

    def ensure_conversation(self, owner_id: str, conversation_id: str | None) -> dict[str, Any]:
        if conversation_id:
            existing = self.get_conversation(owner_id, conversation_id)
            if existing:
                return {"id": conversation_id, "title": existing.get("title", "New conversation")}
        return self.create_conversation(owner_id)

    def _load_sessions(self, owner_id: str) -> list[dict[str, Any]]:
        if self._redis is not None:
            raw = self._redis.get(SESSIONS_KEY_PREFIX + owner_id)
            return json.loads(raw) if raw else []
        return list(self._local_sessions.get(owner_id, []))

    def _save_sessions(self, owner_id: str, sessions: list[dict[str, Any]]) -> None:
        if self._redis is not None:
            _set_with_ttl(
                self._redis, SESSIONS_KEY_PREFIX + owner_id, json.dumps(sessions)
            )
        else:
            self._local_sessions[owner_id] = sessions


def _extractive_summary(turns: list[dict[str, Any]]) -> str:
    bits = []
    for m in turns:
        text = (m.get("text") or "").strip()
        if not text:
            continue
        if m.get("role") == "assistant" and "concerned about your safety" in text.lower():
            continue
        if m.get("role") == "user":
            bits.append(f"User shared: {text[:140]}")
        else:
            bits.append(f"MindCare responded about: {text[:80]}")
    seen = set()
    out = []
    for b in bits:
        key = b.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(b)
    return " | ".join(out[-10:])


def _merge_summaries(existing: str, new_chunk: str) -> str:
    if not existing:
        return new_chunk[:1200]
    if not new_chunk:
        return existing[:1200]
    merged = f"{existing} || {new_chunk}"
    return merged[-1200:]


_store: MemoryStore | None = None


def get_store() -> MemoryStore:
    global _store
    if _store is None:
        _store = MemoryStore()
    return _store
