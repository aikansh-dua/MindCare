"""Reply generation backends: Gemini (recommended), Hugging Face Inference API, or local HF.

Failover chain (when GENERATION_FAILOVER=true, default):
  primary provider → remaining configured providers → raises if all fail
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv

load_dotenv()

GENERATION_PROVIDER = os.getenv("GENERATION_PROVIDER", "gemini").strip().lower()
# Flash-only mode: keep failover off unless explicitly enabled.
GENERATION_FAILOVER = os.getenv("GENERATION_FAILOVER", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
# Local 7B download/load can hang the chat path for minutes — off unless opted in.
FAILOVER_INCLUDE_LOCAL = os.getenv("FAILOVER_INCLUDE_LOCAL", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
# Flash-only — never pin a Pro/thinking-heavy model for chat.
_DEFAULT_FLASH = "gemini-flash-lite-latest"
_raw_model = os.getenv("GEMINI_MODEL", _DEFAULT_FLASH).strip() or _DEFAULT_FLASH
GEMINI_MODEL = _raw_model if "flash" in _raw_model.lower() else _DEFAULT_FLASH
# Gemini 3+: minimal|low|medium|high — only applied to gemini-3* flash models.
GEMINI_THINKING_LEVEL = os.getenv("GEMINI_THINKING_LEVEL", "minimal").strip().lower()
GEMINI_TIMEOUT_SECONDS = float(os.getenv("GEMINI_TIMEOUT_SECONDS", "18"))
HF_TOKEN = os.getenv("HF_TOKEN", "").strip() or os.getenv("HUGGINGFACEHUB_API_TOKEN", "").strip()
HF_CHAT_MODEL = os.getenv("HF_CHAT_MODEL", "tanusrich/Mental_Health_Chatbot").strip()

SYSTEM_PROMPT = (
    "You are MindCare, a warm, empathetic mental health support companion. "
    "Listen, validate feelings, and offer practical coping ideas. "
    "Do not invent a medical history or diagnose. "
    "Do not predict careers, exams, or internships as facts — if asked, "
    "acknowledge uncertainty and gently connect back to how the user is feeling. "
    "Keep replies to 2–4 short sentences unless the user asks for more. "
    "Ask at most one gentle clarifying question when it helps. "
    "Only mention crisis helplines if the user expresses clear self-harm or suicide risk."
)


class GeminiGenerator:
    def __init__(self, api_key: str, model_name: str):
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import HumanMessage, SystemMessage

        self._HumanMessage = HumanMessage
        self._SystemMessage = SystemMessage
        self._timeout = GEMINI_TIMEOUT_SECONDS
        kwargs: dict[str, Any] = {
            "model": model_name,
            "google_api_key": api_key,
            "temperature": 0.65,
            "max_output_tokens": 512,
        }
        # Thinking only helps (and slows) Gemini 3.x reasoning models.
        if model_name.startswith("gemini-3") and GEMINI_THINKING_LEVEL in {
            "minimal",
            "low",
            "medium",
            "high",
        }:
            kwargs["thinking_level"] = GEMINI_THINKING_LEVEL
        self._chat = ChatGoogleGenerativeAI(**kwargs)
        self.model_id = model_name

    def generate(self, user_prompt: str) -> str:
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

        def _invoke():
            result = self._chat.invoke(
                [
                    self._SystemMessage(content=SYSTEM_PROMPT),
                    self._HumanMessage(content=user_prompt),
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

        with ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(_invoke)
            try:
                text = fut.result(timeout=self._timeout)
            except FuturesTimeout as e:
                raise RuntimeError(
                    f"gemini timed out after {self._timeout:.0f}s ({self.model_id})"
                ) from e
        if not text:
            raise RuntimeError(f"gemini returned empty reply ({self.model_id})")
        return text


class HuggingFaceAPIGenerator:
    """Uses Hugging Face Inference API for the 7B mental-health chatbot (no local GPU)."""

    def __init__(self, model_id: str, token: str | None):
        from huggingface_hub import InferenceClient

        self.model_id = model_id
        self._client = InferenceClient(model=model_id, token=token or None)

    def generate(self, user_prompt: str) -> str:
        prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}\nMindCare:"
        try:
            completion = self._client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=256,
                temperature=0.7,
            )
            return (completion.choices[0].message.content or "").strip()
        except Exception:
            text = self._client.text_generation(
                prompt,
                max_new_tokens=200,
                temperature=0.7,
                return_full_text=False,
            )
            return str(text).strip()


def _build_provider(name: str):
    if name == "gemini":
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY missing")
        return GeminiGenerator(GEMINI_API_KEY, GEMINI_MODEL), "gemini"
    if name == "hf_api":
        return HuggingFaceAPIGenerator(HF_CHAT_MODEL, HF_TOKEN or None), "hf_api"
    if name == "local":
        from chatbot_local import LocalHFGenerator

        return LocalHFGenerator(), "local"
    raise RuntimeError(f"Unknown provider {name!r}")


def _failover_order(primary: str) -> list[str]:
    """Primary first, then other *fast* backends. Never pull local HF unless opted in."""
    preferred = ["gemini", "hf_api"]
    if primary == "local" or FAILOVER_INCLUDE_LOCAL:
        preferred.append("local")
    order = [primary] + [p for p in preferred if p != primary]
    usable = []
    for name in order:
        if name == "gemini" and not GEMINI_API_KEY:
            continue
        if name == "hf_api":
            # Unauthenticated HF hangs / rate-limits; only fail over with a token.
            if not HF_TOKEN:
                continue
        if name == "local" and primary != "local" and not FAILOVER_INCLUDE_LOCAL:
            continue
        usable.append(name)
    if primary not in usable and primary in {"gemini", "hf_api", "local"}:
        usable.insert(0, primary)
    return usable


@lru_cache(maxsize=1)
def get_generator():
    """Build the configured primary generator."""
    if GENERATION_PROVIDER not in {"gemini", "hf_api", "local"}:
        raise RuntimeError(
            f"Unknown GENERATION_PROVIDER={GENERATION_PROVIDER!r}. Use gemini | hf_api | local."
        )
    return _build_provider(GENERATION_PROVIDER)


def generate_reply(user_prompt: str) -> tuple[str, str, list[str]]:
    """
    Generate a reply with optional provider failover.

    Returns: (text, provider_used, errors_from_failed_providers)
    """
    errors: list[str] = []
    providers = (
        _failover_order(GENERATION_PROVIDER)
        if GENERATION_FAILOVER
        else [GENERATION_PROVIDER]
    )

    for name in providers:
        try:
            if name == GENERATION_PROVIDER:
                gen, provider = get_generator()
            else:
                gen, provider = _build_provider(name)
            text = (gen.generate(user_prompt) or "").strip()
            if not text:
                raise RuntimeError(f"{provider} returned empty reply")
            return text, provider, errors
        except Exception as e:
            errors.append(f"{name}: {type(e).__name__}: {e}")
            continue

    raise RuntimeError("All generation providers failed: " + " | ".join(errors[:3]))


def generate_conversation_title(history: list[dict]) -> str | None:
    """Gemini-only short title. Never falls back to local HF (too slow for chat path)."""
    if not (GENERATION_PROVIDER == "gemini" and GEMINI_API_KEY):
        return None

    turns = []
    for m in history[-10:]:
        text = (m.get("text") or "").strip()
        if not text:
            continue
        who = "User" if m.get("role") == "user" else "MindCare"
        turns.append(f"{who}: {text[:160]}")
    if not turns:
        return None

    prompt = (
        "Create a short conversation title (3 to 6 words) that captures what the user "
        "is going through emotionally or situationally. "
        "Do NOT copy a greeting. Do NOT use quotes. Title Case. "
        "Reply with ONLY the title.\n\n"
        + "\n".join(turns)
        + "\n\nTitle:"
    )

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import HumanMessage, SystemMessage

        chat = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=0.3,
            max_output_tokens=48,
        )
        result = chat.invoke(
            [
                SystemMessage(
                    content="You write brief conversation titles for a mental health companion app."
                ),
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
        title = str(content).strip().strip('"').strip("'")
        title = title.split("\n")[0].strip()
        title = re.sub(r"^(title:\s*)", "", title, flags=re.I)
        if 2 <= len(title.split()) <= 10 and len(title) <= 60:
            return title[:1].upper() + title[1:]
    except Exception:
        return None
    return None


def generation_status() -> dict[str, Any]:
    return {
        "provider": GENERATION_PROVIDER,
        "failover": GENERATION_FAILOVER,
        "failover_order": _failover_order(GENERATION_PROVIDER) if GENERATION_FAILOVER else [GENERATION_PROVIDER],
        "gemini_model": GEMINI_MODEL if GEMINI_API_KEY else None,
        "hf_chat_model": HF_CHAT_MODEL,
        "gemini_key_configured": bool(GEMINI_API_KEY),
        "hf_token_configured": bool(HF_TOKEN),
    }
