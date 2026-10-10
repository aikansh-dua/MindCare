#!/usr/bin/env python3
"""Extensive MindCare pipeline evaluator.

Suites:
  project  — files, imports, corpus sizes, env/config
  emotion  — DistilBERT GoEmotions labels
  crisis   — keyword + BART gate (true positives & false positives)
  rag      — embedding retrieval quality / anti-false-matches
  chat     — full orchestration (template / crisis / generation routing)
  memory   — Redis/in-memory history + profile tags
  live     — real Gemini/HF calls (only with --live)

Usage:
  python evaluate.py                  # all suites except live API spend
  python evaluate.py --live           # + real generation provider
  python evaluate.py --suite rag
  python evaluate.py --suite chat,crisis
  python evaluate.py --rag-only       # alias for --suite rag
  python evaluate.py --quick          # project + rag + crisis only (no HF emotion/chat load)
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
CASES_PATH = ROOT / "eval_cases.json"

REQUIRED_FILES = [
    "app.py",
    "bot.py",
    "chatbot.py",
    "emotion.py",
    "crisis.py",
    "rag.py",
    "generation.py",
    "memory.py",
    "expand_dataset.py",
    "mindcare_dataset.json",
    "counseling_rag.json",
    "requirements.txt",
    "README.md",
    ".env.example",
    "run.sh",
    "eval_cases.json",
    "frontend/index.html",
    "frontend/styles.css",
    "frontend/app.js",
]

REQUIRED_MODULES = [
    "emotion",
    "crisis",
    "rag",
    "generation",
    "memory",
    "chatbot",
    "app",
]


class _StubGenerator:
    model_id = "eval-stub"

    def generate(self, user_prompt: str) -> str:
        return (
            "It makes sense that this feels heavy. "
            "I'm here with you — what part feels hardest right now?"
        )


def _stub_generate_reply(user_prompt: str):
    return _StubGenerator().generate(user_prompt), "gemini", []


class EvalReport:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.failures: list[str] = []

    def ok(self, suite: str, case_id: str, detail: str = "") -> None:
        self.passed += 1
        extra = f" — {detail}" if detail else ""
        print(f"PASS  [{suite}] {case_id}{extra}")

    def fail(self, suite: str, case_id: str, errors: list[str], detail: str = "") -> None:
        self.failed += 1
        msg = "; ".join(errors)
        line = f"FAIL  [{suite}] {case_id}: {msg}"
        if detail:
            line += f" | {detail}"
        print(line)
        self.failures.append(line)

    def skip(self, suite: str, case_id: str, reason: str) -> None:
        self.skipped += 1
        print(f"SKIP  [{suite}] {case_id}: {reason}")


def _load_cases() -> dict[str, Any]:
    with open(CASES_PATH, encoding="utf-8") as f:
        data = json.load(f)
    # Back-compat: old flat list format
    if isinstance(data, list):
        return {"chat": data, "rag": [], "emotion": [], "crisis": [], "memory": []}
    return data


def _contains_any(text: str, needles: list[str]) -> bool:
    lower = text.lower()
    return any(n.lower() in lower for n in needles)


def _rag_blob(hits: list[dict]) -> str:
    return " ".join(f"{h.get('context', '')} {h.get('response', '')}" for h in hits).lower()


def _check_rag_expect(message: str, expect: dict[str, Any]) -> tuple[list[str], list[dict]]:
    from rag import retrieve_counseling

    errors: list[str] = []
    hits = retrieve_counseling(message)
    blob = _rag_blob(hits)

    if "rag_min_hits" in expect and len(hits) < int(expect["rag_min_hits"]):
        errors.append(f"rag_hits={len(hits)} < min {expect['rag_min_hits']}")
    if "rag_max_hits" in expect and len(hits) > int(expect["rag_max_hits"]):
        errors.append(f"rag_hits={len(hits)} > max {expect['rag_max_hits']}")

    min_score = expect.get("rag_min_score")
    if min_score is not None and hits:
        top = float(hits[0].get("score") or 0)
        if top < float(min_score):
            errors.append(f"top_rag_score={top:.3f} < {min_score}")

    for bad in expect.get("rag_context_must_not_contain") or []:
        if bad and bad.lower() in blob:
            errors.append(f"rag contained forbidden: {bad!r}")

    must_any = expect.get("rag_context_must_contain_any") or []
    if must_any and hits and not _contains_any(blob, must_any):
        errors.append(f"rag missing any of {must_any}")

    for h in hits:
        if h.get("retrieval") not in {None, "embeddings", "token"}:
            errors.append(f"unknown retrieval mode: {h.get('retrieval')!r}")

    return errors, hits


def _check_chat_expect(result: dict[str, Any], expect: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    reply = result.get("response") or ""

    if "crisis" in expect and bool(result.get("crisis")) != bool(expect["crisis"]):
        errors.append(f"crisis={result.get('crisis')} expected {expect['crisis']}")

    if not expect.get("crisis") and "reply_source_in" in expect:
        source = result.get("reply_source")
        if source not in expect["reply_source_in"]:
            errors.append(f"reply_source={source!r} not in {expect['reply_source_in']}")

    if "reply_min_chars" in expect and len(reply.strip()) < int(expect["reply_min_chars"]):
        errors.append(f"reply too short ({len(reply)} chars)")

    for bad in expect.get("reply_must_not_contain") or []:
        if bad.lower() in reply.lower():
            errors.append(f"reply contained forbidden: {bad!r}")

    must_any = expect.get("reply_must_contain_any") or []
    if must_any and not _contains_any(reply, must_any):
        errors.append(f"reply missing any of {must_any}")

    if "intent_in" in expect:
        allowed = expect["intent_in"]
        intent = result.get("intent")
        if intent not in allowed:
            errors.append(f"intent={intent!r} not in {allowed}")

    if "emotion_in" in expect:
        emotion = result.get("emotion")
        if emotion not in expect["emotion_in"]:
            errors.append(f"emotion={emotion!r} not in {expect['emotion_in']}")

    return errors


# ---------------------------------------------------------------------------
# Suites
# ---------------------------------------------------------------------------

def suite_project(report: EvalReport) -> None:
    suite = "project"

    missing = [f for f in REQUIRED_FILES if not (ROOT / f).exists()]
    if missing:
        report.fail(suite, "required_files", [f"missing: {missing}"])
    else:
        report.ok(suite, "required_files", f"{len(REQUIRED_FILES)} present")

    import_errors = []
    for name in REQUIRED_MODULES:
        try:
            importlib.import_module(name)
        except Exception as e:
            import_errors.append(f"{name}: {type(e).__name__}: {e}")
    if import_errors:
        report.fail(suite, "imports", import_errors)
    else:
        report.ok(suite, "imports", f"{len(REQUIRED_MODULES)} modules")

    try:
        intents = json.loads((ROOT / "mindcare_dataset.json").read_text(encoding="utf-8"))
        n_intents = len(intents.get("intents", []))
        if n_intents < 40:
            report.fail(suite, "intents_corpus", [f"only {n_intents} intents (want >= 40)"])
        else:
            report.ok(suite, "intents_corpus", f"{n_intents} intents")
    except Exception as e:
        report.fail(suite, "intents_corpus", [str(e)])

    try:
        rag = json.loads((ROOT / "counseling_rag.json").read_text(encoding="utf-8"))
        n_pairs = len(rag.get("pairs", []))
        if n_pairs < 500:
            report.fail(suite, "rag_corpus", [f"only {n_pairs} pairs (want >= 500)"])
        else:
            sample = rag["pairs"][0]
            need = {"context", "response"}
            if not need.issubset(sample.keys()):
                report.fail(suite, "rag_corpus", [f"pair missing keys {need - set(sample)}"])
            else:
                report.ok(suite, "rag_corpus", f"{n_pairs} pairs")
    except Exception as e:
        report.fail(suite, "rag_corpus", [str(e)])

    from generation import generation_status
    from rag import rag_status

    g = generation_status()
    r = rag_status()
    if g.get("provider") not in {"gemini", "hf_api", "local"}:
        report.fail(suite, "generation_config", [f"bad provider {g.get('provider')!r}"])
    else:
        detail = f"provider={g.get('provider')} model={g.get('gemini_model') or g.get('hf_chat_model')}"
        if g.get("provider") == "gemini" and not g.get("gemini_key_configured"):
            report.fail(suite, "generation_config", ["GEMINI_API_KEY missing"], detail)
        else:
            report.ok(suite, "generation_config", detail)

    if r.get("backend") not in {"embeddings", "token"}:
        report.fail(suite, "rag_config", [f"bad backend {r.get('backend')!r}"])
    else:
        report.ok(
            suite,
            "rag_config",
            f"backend={r.get('backend')} model={r.get('embed_model')} pairs={r.get('corpus_pairs')}",
        )

    # Flask app factory-ish sanity
    try:
        import app as flask_app

        client_routes = sorted({rule.rule for rule in flask_app.app.url_map.iter_rules()})
        needed = {
            "/",
            "/health",
            "/chat",
            "/chat_memory",
            "/user_profile",
            "/clear",
            "/api",
            "/conversations",
        }
        missing_routes = sorted(needed - set(client_routes))
        # Also allow parameterized conversation routes
        has_conv_id = any("/conversations/" in r for r in client_routes)
        if missing_routes:
            report.fail(suite, "api_routes", [f"missing routes: {missing_routes}"])
        elif not has_conv_id:
            report.fail(suite, "api_routes", ["missing /conversations/<id> routes"])
        else:
            report.ok(suite, "api_routes", f"{len(client_routes)} routes")
    except Exception as e:
        report.fail(suite, "api_routes", [str(e)])


def suite_emotion(report: EvalReport, cases: list[dict]) -> None:
    from emotion import detect_emotion

    for case in cases:
        expect = case.get("expect") or {}
        try:
            info = detect_emotion(case["message"])
            emotion = info.get("emotion")
            errors = []
            if "emotion_in" in expect and emotion not in expect["emotion_in"]:
                errors.append(f"emotion={emotion!r} not in {expect['emotion_in']}")
            if not emotion:
                errors.append("empty emotion")
            if errors:
                report.fail("emotion", case["id"], errors, f"scores={info.get('top', info)[:3] if isinstance(info.get('top'), list) else info}")
            else:
                report.ok("emotion", case["id"], f"emotion={emotion}")
        except Exception as e:
            report.fail("emotion", case["id"], [f"{type(e).__name__}: {e}"])


def suite_crisis(report: EvalReport, cases: list[dict]) -> None:
    from crisis import detect_crisis

    for case in cases:
        expect = case.get("expect") or {}
        try:
            info = detect_crisis(case["message"])
            errors = []
            if "crisis" in expect and bool(info.get("crisis")) != bool(expect["crisis"]):
                errors.append(
                    f"crisis={info.get('crisis')} expected {expect['crisis']} "
                    f"(reason={info.get('reason')})"
                )
            if errors:
                report.fail("crisis", case["id"], errors)
            else:
                report.ok(
                    "crisis",
                    case["id"],
                    f"crisis={info.get('crisis')} reason={info.get('reason')}",
                )
        except Exception as e:
            report.fail("crisis", case["id"], [f"{type(e).__name__}: {e}"])


def suite_rag(report: EvalReport, cases: list[dict]) -> None:
    from rag import build_embedding_cache, rag_status

    try:
        info = build_embedding_cache()
        if info.get("pairs", 0) < 500:
            report.fail("rag", "embed_cache", [f"pairs={info.get('pairs')}"])
        elif info.get("dim", 0) < 64:
            report.fail("rag", "embed_cache", [f"dim={info.get('dim')}"])
        else:
            report.ok("rag", "embed_cache", str(info))
    except Exception as e:
        report.fail("rag", "embed_cache", [f"{type(e).__name__}: {e}"])
        return

    status = rag_status()
    if status.get("backend") == "embeddings" and not status.get("cache_exists"):
        report.fail("rag", "cache_file", ["cache_exists=false after build"])
    else:
        report.ok("rag", "cache_file", f"cache_exists={status.get('cache_exists')}")

    # Ranking sanity: anxiety query should outrank weather for an anxiety doc family
    try:
        from rag import retrieve_counseling

        anx = retrieve_counseling("I have severe anxiety and panic")
        weather = retrieve_counseling("the weather is cloudy today")
        anx_score = float(anx[0]["score"]) if anx else 0.0
        weather_score = float(weather[0]["score"]) if weather else 0.0
        if anx and anx_score < 0.35:
            report.fail("rag", "ranking_anxiety", [f"top anxiety score too low: {anx_score:.3f}"])
        elif weather and weather_score > anx_score and weather_score > 0.55:
            report.fail(
                "rag",
                "ranking_anxiety",
                [f"weather score {weather_score:.3f} > anxiety {anx_score:.3f}"],
            )
        else:
            report.ok(
                "rag",
                "ranking_anxiety",
                f"anxiety={anx_score:.3f} hits={len(anx)} weather_hits={len(weather)}",
            )
    except Exception as e:
        report.fail("rag", "ranking_anxiety", [f"{type(e).__name__}: {e}"])

    for case in cases:
        expect = case.get("expect") or {}
        try:
            errors, hits = _check_rag_expect(case["message"], expect)
            if errors:
                top = hits[0] if hits else None
                detail = (
                    f"hits={len(hits)} top={top.get('score') if top else None} "
                    f"ctx={(top.get('context')[:80] if top else '')!r}"
                )
                report.fail("rag", case["id"], errors, detail)
            else:
                top = hits[0] if hits else None
                report.ok(
                    "rag",
                    case["id"],
                    f"hits={len(hits)} top_score={top.get('score') if top else None} "
                    f"mode={top.get('retrieval') if top else None}",
                )
        except Exception as e:
            report.fail("rag", case["id"], [f"{type(e).__name__}: {e}"])


def suite_chat(report: EvalReport, cases: list[dict], *, live: bool) -> None:
    from chatbot import chat
    from memory import get_store

    for case in cases:
        expect = case.get("expect") or {}
        message = case["message"]
        errors: list[str] = []

        # RAG expectations attached to chat cases
        if any(k.startswith("rag_") for k in expect):
            rag_errors, _ = _check_rag_expect(message, expect)
            errors.extend(rag_errors)

        needs_gen = bool(expect.get("needs_generation"))
        user_id = f"eval_{case['id']}_{uuid.uuid4().hex[:8]}"
        store = get_store()
        store.set_history(user_id, [])

        try:
            if needs_gen and not live:
                with patch("chatbot.generate_reply", side_effect=_stub_generate_reply):
                    result = chat(user_id, message)
                mode = "stub-gen"
            else:
                result = chat(user_id, message)
                mode = "live" if needs_gen else "route"

            errors.extend(_check_chat_expect(result, expect))

            # Memory should contain the turn (except we always append)
            hist = store.get_history(user_id)
            if len(hist) < 2:
                errors.append(f"history_len={len(hist)} expected >= 2")

            if errors:
                detail = (
                    f"source={result.get('reply_source')} crisis={result.get('crisis')} "
                    f"emotion={result.get('emotion')} rag_hits={result.get('rag_hits')} "
                    f"reply={str(result.get('response', ''))[:100]!r}"
                )
                if result.get("generation_error"):
                    detail += f" gen_err={result['generation_error'][:120]}"
                report.fail("chat", case["id"], errors, detail)
            else:
                report.ok(
                    "chat",
                    case["id"],
                    f"{mode} source={result.get('reply_source')} "
                    f"crisis={result.get('crisis')} emotion={result.get('emotion')} "
                    f"rag_hits={result.get('rag_hits')}",
                )
        except Exception as e:
            report.fail("chat", case["id"], [f"{type(e).__name__}: {e}"])


def suite_memory(report: EvalReport, cases: list[dict]) -> None:
    from chatbot import chat
    from memory import KEEP_RECENT_TURNS, SUMMARY_AFTER_TURNS, get_store

    store = get_store()
    report.ok("memory", "backend", f"backend={store.backend}")

    for case in cases:
        expect = case.get("expect") or {}
        user_id = f"eval_mem_{case['id']}_{uuid.uuid4().hex[:8]}"
        errors: list[str] = []

        try:
            store.clear_user(user_id)

            if case.get("kind") == "summary":
                # Exceed SUMMARY_AFTER_TURNS to force rolling summary
                for i in range(SUMMARY_AFTER_TURNS + 2):
                    store.append(user_id, "user", f"user note about stress number {i}")
                    store.append(
                        user_id, "assistant", f"supportive reply number {i} about coping"
                    )
                summary = store.get_summary(user_id)
                hist = store.get_history(user_id)
                if expect.get("has_summary") and not summary:
                    errors.append("expected rolling summary to be set")
                if "recent_max" in expect and len(hist) > int(expect["recent_max"]) + 2:
                    # allow small slack; summarizer keeps KEEP_RECENT_TURNS
                    if len(hist) > KEEP_RECENT_TURNS + 2:
                        errors.append(
                            f"history_len={len(hist)} expected around {KEEP_RECENT_TURNS}"
                        )
                if summary and "stress" not in summary.lower():
                    errors.append("summary missing expected user content")
            elif "steps" in case:
                for step in case["steps"]:
                    store.append(user_id, step["role"], step["text"])
                hist = store.get_history(user_id)
                if "history_len" in expect and len(hist) != int(expect["history_len"]):
                    errors.append(f"history_len={len(hist)} expected {expect['history_len']}")
                if "last_role" in expect and hist and hist[-1].get("role") != expect["last_role"]:
                    errors.append(f"last_role={hist[-1].get('role')}")
            elif "message" in case:
                with patch("chatbot.generate_reply", side_effect=_stub_generate_reply):
                    chat(user_id, case["message"])
                profile = store.get_profile(user_id)
                concerns = profile.get("concerns") or []
                must_any = expect.get("profile_concerns_contain_any") or []
                if must_any and not any(c in concerns for c in must_any):
                    errors.append(f"concerns={concerns} missing any of {must_any}")
                for bad in expect.get("profile_concerns_must_not_contain") or []:
                    if bad in concerns:
                        errors.append(f"concerns unexpectedly contains {bad!r}: {concerns}")

            store.clear_user(user_id)
            if store.get_history(user_id) or store.get_summary(user_id):
                errors.append("clear_user did not wipe history/summary")

            if errors:
                report.fail("memory", case["id"], errors)
            else:
                report.ok("memory", case["id"])
        except Exception as e:
            report.fail("memory", case["id"], [f"{type(e).__name__}: {e}"])


def suite_live_generation(report: EvalReport) -> None:
    """One real end-to-end generation smoke test."""
    from chatbot import chat
    from generation import generation_status, get_generator
    from memory import get_store

    status = generation_status()
    report.ok("live", "config", str(status))

    try:
        get_generator.cache_clear()
        gen, provider = get_generator()
        text = gen.generate(
            "User feels overwhelmed at work. Write a short MindCare reply (2 sentences)."
        )
        if not text or len(text.strip()) < 20:
            report.fail("live", "direct_generate", [f"empty/short reply: {text!r}"])
        else:
            report.ok("live", "direct_generate", f"provider={provider} chars={len(text)}")
    except Exception as e:
        report.fail("live", "direct_generate", [f"{type(e).__name__}: {e}"])
        return

    user_id = f"eval_live_{uuid.uuid4().hex[:8]}"
    get_store().set_history(user_id, [])
    try:
        result = chat(
            user_id,
            "I feel overwhelmed at work and do not know where to start",
        )
        errors = _check_chat_expect(
            result,
            {
                "crisis": False,
                "reply_source_in": ["gemini", "hf_api", "local"],
                "reply_min_chars": 40,
                "reply_must_not_contain": [
                    "anger is like a blanket",
                    "concerned about your safety",
                ],
            },
        )
        if errors:
            report.fail(
                "live",
                "chat_generate",
                errors,
                f"source={result.get('reply_source')} reply={str(result.get('response'))[:120]!r}",
            )
        else:
            report.ok(
                "live",
                "chat_generate",
                f"source={result.get('reply_source')} chars={len(result.get('response') or '')}",
            )
    except Exception as e:
        report.fail("live", "chat_generate", [f"{type(e).__name__}: {e}"])


def run_evaluation(
    *,
    live: bool = False,
    suites: list[str] | None = None,
    quick: bool = False,
) -> int:
    data = _load_cases()
    report = EvalReport()
    t0 = time.time()

    available = ["project", "emotion", "crisis", "rag", "chat", "memory"]
    if live:
        available.append("live")

    if quick:
        selected = ["project", "rag", "crisis"]
    elif suites:
        selected = suites
    else:
        selected = ["project", "emotion", "crisis", "rag", "chat", "memory"]
        if live:
            selected.append("live")

    unknown = [s for s in selected if s not in available and s != "live"]
    if unknown:
        print(f"Unknown suites: {unknown}. Available: {available}")
        return 2

    print(
        f"MindCare full eval — suites={selected} live={live} "
        f"cases_file={CASES_PATH.name}"
    )
    print("=" * 72)

    if "project" in selected:
        print("\n## project")
        suite_project(report)

    if "rag" in selected:
        print("\n## rag")
        suite_rag(report, data.get("rag") or [])

    if "emotion" in selected:
        print("\n## emotion")
        suite_emotion(report, data.get("emotion") or [])

    if "crisis" in selected:
        print("\n## crisis")
        suite_crisis(report, data.get("crisis") or [])

    if "chat" in selected:
        print("\n## chat")
        suite_chat(report, data.get("chat") or [], live=live)

    if "memory" in selected:
        print("\n## memory")
        suite_memory(report, data.get("memory") or [])

    if "live" in selected:
        print("\n## live")
        suite_live_generation(report)

    elapsed = time.time() - t0
    print("\n" + "=" * 72)
    print(
        f"Result: {report.passed} passed, {report.failed} failed, "
        f"{report.skipped} skipped  ({elapsed:.1f}s)"
    )
    if report.failures:
        print("\nFailures:")
        for line in report.failures:
            print(f"  - {line}")
    return 0 if report.failed == 0 else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="MindCare full pipeline evaluator")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Call the real generation provider (also enables live suite)",
    )
    parser.add_argument(
        "--suite",
        type=str,
        default="",
        help="Comma-separated suites: project,emotion,crisis,rag,chat,memory,live",
    )
    parser.add_argument(
        "--rag-only",
        action="store_true",
        help="Alias for --suite rag",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="project + rag + crisis only (faster smoke)",
    )
    args = parser.parse_args()

    suites = None
    if args.rag_only:
        suites = ["rag"]
    elif args.suite.strip():
        suites = [s.strip() for s in args.suite.split(",") if s.strip()]

    sys.exit(
        run_evaluation(
            live=args.live,
            suites=suites,
            quick=args.quick,
        )
    )


if __name__ == "__main__":
    main()
