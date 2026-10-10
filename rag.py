"""Embedding-based counseling retrieval (with token-overlap fallback)."""

from __future__ import annotations

import hashlib
import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
RAG_PATH = ROOT / "counseling_rag.json"
EMBED_CACHE_PATH = ROOT / "counseling_rag_embeddings.npz"

EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2").strip()
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "2"))
RAG_MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.42"))
# Prefer embeddings; set RAG_BACKEND=token to force old overlap retrieval
RAG_BACKEND = os.getenv("RAG_BACKEND", "embeddings").strip().lower()


def _tokenize(text: str) -> set[str]:
    stop = {
        "i", "me", "my", "am", "im", "a", "an", "the", "to", "and", "or", "of",
        "in", "on", "is", "it", "for", "you", "your", "so", "do", "how", "can",
        "will", "get", "be", "with",
    }
    return set(re.findall(r"[a-z0-9']+", text.lower())) - stop


@lru_cache(maxsize=1)
def _load_rag_pairs() -> list[dict]:
    try:
        with open(RAG_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("pairs", [])
    except Exception:
        return []


@lru_cache(maxsize=1)
def _get_embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBED_MODEL)


def _encode(texts: list[str]) -> np.ndarray:
    model = _get_embedder()
    vectors = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=len(texts) > 64,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return np.asarray(vectors, dtype=np.float32)


@lru_cache(maxsize=1)
def _load_corpus_embeddings() -> tuple[list[dict], np.ndarray]:
    """Load pairs + embedding matrix (build/cache on first use)."""
    pairs = _load_rag_pairs()
    if not pairs:
        return [], np.zeros((0, 384), dtype=np.float32)

    contexts = [p.get("context", "") for p in pairs]
    digest = hashlib.sha256()
    digest.update(EMBED_MODEL.encode())
    digest.update(str(len(pairs)).encode())
    for ctx in contexts:
        digest.update(ctx.encode("utf-8", errors="ignore"))
        digest.update(b"\0")
    fingerprint = digest.hexdigest()

    if EMBED_CACHE_PATH.exists():
        try:
            cached = np.load(EMBED_CACHE_PATH, allow_pickle=False)
            if (
                str(cached["fingerprint"]) == fingerprint
                and cached["embeddings"].shape[0] == len(pairs)
            ):
                return pairs, cached["embeddings"].astype(np.float32)
        except Exception:
            pass

    embeddings = _encode(contexts)
    np.savez_compressed(
        EMBED_CACHE_PATH,
        embeddings=embeddings,
        fingerprint=np.asarray(fingerprint),
    )
    return pairs, embeddings


def _retrieve_embeddings(user_message: str, top_k: int, min_score: float) -> list[dict]:
    pairs, matrix = _load_corpus_embeddings()
    if matrix.shape[0] == 0:
        return []

    query = _encode([user_message])[0]
    scores = matrix @ query  # cosine because both are L2-normalized
    top_idx = np.argsort(-scores)[: max(top_k * 3, top_k)]

    hits: list[dict] = []
    seen: set[str] = set()
    for i in top_idx:
        score = float(scores[i])
        if score < min_score:
            continue
        pair = dict(pairs[int(i)])
        key = (pair.get("context") or "")[:200]
        if key in seen:
            continue
        seen.add(key)
        pair["score"] = round(score, 4)
        pair["retrieval"] = "embeddings"
        hits.append(pair)
        if len(hits) >= top_k:
            break
    return hits


def _retrieve_token_overlap(user_message: str, top_k: int) -> list[dict]:
    """Legacy fallback — strict token overlap."""
    q = _tokenize(user_message)
    if len(q) < 2:
        return []

    scored = []
    for pair in _load_rag_pairs():
        tokens = set(pair.get("tokens") or []) or _tokenize(pair.get("context", ""))
        overlap_tokens = q & tokens
        overlap = len(overlap_tokens)
        if overlap < 2:
            continue
        score = overlap / (len(q) ** 0.5 * len(tokens) ** 0.5 + 1e-6)
        coverage = overlap / max(len(q), 1)
        if coverage < 0.35 and overlap < 3:
            continue
        scored.append((score, coverage, pair))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    hits = []
    for s, c, p in scored[:top_k]:
        if s < 0.18 or c < 0.35:
            continue
        item = dict(p)
        item["score"] = round(float(s), 4)
        item["retrieval"] = "token"
        hits.append(item)
    return hits


def retrieve_counseling(
    user_message: str,
    top_k: int | None = None,
    min_score: float | None = None,
) -> list[dict]:
    """
    Retrieve counseling Q&A context for the generator prompt.
    Default: sentence-transformer embeddings + cosine similarity.
    """
    k = top_k if top_k is not None else RAG_TOP_K
    threshold = min_score if min_score is not None else RAG_MIN_SCORE

    if RAG_BACKEND == "token":
        return _retrieve_token_overlap(user_message, k)

    try:
        return _retrieve_embeddings(user_message, k, threshold)
    except Exception:
        # Model download / OOM / etc. — degrade gracefully
        return _retrieve_token_overlap(user_message, k)


def rag_status() -> dict[str, Any]:
    return {
        "backend": RAG_BACKEND,
        "embed_model": EMBED_MODEL if RAG_BACKEND != "token" else None,
        "top_k": RAG_TOP_K,
        "min_score": RAG_MIN_SCORE,
        "corpus_pairs": len(_load_rag_pairs()),
        "cache_exists": EMBED_CACHE_PATH.exists(),
    }


def build_embedding_cache(force: bool = False) -> dict[str, Any]:
    """Precompute corpus embeddings (used by expand_dataset / eval setup)."""
    if force and EMBED_CACHE_PATH.exists():
        EMBED_CACHE_PATH.unlink()
    _load_corpus_embeddings.cache_clear()
    pairs, matrix = _load_corpus_embeddings()
    return {
        "pairs": len(pairs),
        "dim": int(matrix.shape[1]) if matrix.size else 0,
        "cache": str(EMBED_CACHE_PATH),
        "model": EMBED_MODEL,
    }
