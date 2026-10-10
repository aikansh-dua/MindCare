"""MindCare Flask app — serves the web UI and chat API."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from chatbot import chat, model_status
from generation import generation_status
from memory import get_store

ROOT = Path(__file__).resolve().parent
FRONTEND = ROOT / "frontend"

app = Flask(
    __name__,
    static_folder=str(FRONTEND),
    static_url_path="/static",
)
CORS(app)


@app.get("/")
def index():
    return send_from_directory(FRONTEND, "index.html")


@app.get("/api")
def api_info():
    store = get_store()
    return jsonify(
        {
            "name": "MindCare",
            "description": "Mental health support chatbot (emotion + crisis + RAG + generative reply)",
            "memory_backend": store.backend,
            "generation": generation_status(),
        }
    )


@app.get("/health")
def health():
    store = get_store()
    return jsonify(
        {
            "status": "ok",
            "memory_backend": store.backend,
            "generation": generation_status(),
            "generator": model_status(),
        }
    )


@app.get("/conversations")
def list_conversations():
    owner_id = request.args.get("owner_id") or request.args.get("user_id") or "default"
    store = get_store()
    return jsonify({"owner_id": owner_id, "conversations": store.list_conversations(owner_id)})


@app.post("/conversations")
def create_conversation():
    data = request.get_json(silent=True) or {}
    owner_id = data.get("owner_id") or data.get("user_id") or "default"
    store = get_store()
    session = store.create_conversation(owner_id)
    return jsonify(session)


@app.get("/conversations/<conversation_id>")
def get_conversation(conversation_id: str):
    owner_id = request.args.get("owner_id") or request.args.get("user_id") or "default"
    store = get_store()
    session = store.get_conversation(owner_id, conversation_id)
    if not session:
        return jsonify({"error": "conversation not found"}), 404
    return jsonify(session)


@app.delete("/conversations/<conversation_id>")
def delete_conversation(conversation_id: str):
    data = request.get_json(silent=True) or {}
    owner_id = (
        data.get("owner_id")
        or request.args.get("owner_id")
        or request.args.get("user_id")
        or "default"
    )
    store = get_store()
    ok = store.delete_conversation(owner_id, conversation_id)
    if not ok:
        return jsonify({"error": "conversation not found"}), 404
    return jsonify({"ok": True, "id": conversation_id})


@app.post("/conversations/<conversation_id>/title")
def finalize_conversation_title(conversation_id: str):
    """Flash-polish title when the user leaves a conversation (not on the chat hot path)."""
    data = request.get_json(silent=True) or {}
    owner_id = (
        data.get("owner_id")
        or request.args.get("owner_id")
        or request.args.get("user_id")
        or "default"
    )
    store = get_store()
    meta = store.get_conversation(owner_id, conversation_id)
    if not meta:
        return jsonify({"error": "conversation not found"}), 404

    from memory import resolve_conversation_title

    hist = store.get_history(conversation_id)
    emotion = None
    profile = store.get_profile(owner_id)
    if isinstance(profile, dict):
        emotion = profile.get("last_emotion")

    title = resolve_conversation_title(
        hist,
        meta.get("title"),
        emotion,
        prefer_llm=True,
    )
    if title:
        meta = store.touch_conversation(
            owner_id,
            conversation_id,
            title=title,
            message_count=len(hist),
            force_title=True,
        )
    return jsonify(
        {
            "id": conversation_id,
            "title": (meta or {}).get("title") or title or "New conversation",
        }
    )


@app.post("/chat")
def chat_endpoint():
    """
    JSON body:
      {
        "owner_id": "browser_id",
        "conversation_id": "optional existing id",
        "message": "..."
      }
    Legacy: { "user_id": "...", "message": "..." }
    """
    data = request.get_json(silent=True) or {}
    message = data.get("message") or request.args.get("message") or ""
    if not message.strip():
        return jsonify({"error": "message is required"}), 400

    store = get_store()
    owner_id = data.get("owner_id") or data.get("user_id") or request.args.get("user_id")
    conversation_id = data.get("conversation_id")

    if owner_id:
        session = store.ensure_conversation(owner_id, conversation_id)
        conversation_id = session["id"]
        result = chat(
            conversation_id,
            message.strip(),
            owner_id=owner_id,
        )
        result["owner_id"] = owner_id
        result["conversation_id"] = conversation_id
        # Refresh title from index
        meta = store.get_conversation(owner_id, conversation_id)
        if meta:
            result["conversation_title"] = meta.get("title")
        return jsonify(result)

    # Legacy single-thread path
    user_id = data.get("user_id") or request.args.get("user_id") or "default"
    result = chat(user_id=user_id, message=message.strip())
    return jsonify(result)


@app.get("/chat_memory")
def chat_memory():
    owner_id = request.args.get("owner_id")
    conversation_id = request.args.get("conversation_id") or request.args.get("user_id", "default")
    store = get_store()
    payload = {
        "conversation_id": conversation_id,
        "memory": store.get_history(conversation_id),
        "summary": store.get_summary(conversation_id),
    }
    if owner_id:
        payload["owner_id"] = owner_id
        payload["profile"] = store.get_profile(owner_id)
        payload["conversations"] = store.list_conversations(owner_id)
    else:
        payload["user_id"] = conversation_id
        payload["profile"] = store.get_profile(conversation_id)
    return jsonify(payload)


@app.post("/clear")
def clear_conversation():
    data = request.get_json(silent=True) or {}
    owner_id = data.get("owner_id")
    conversation_id = data.get("conversation_id") or data.get("user_id") or "default"
    store = get_store()
    if owner_id and conversation_id:
        store.delete_conversation(owner_id, conversation_id)
        return jsonify({"ok": True, "owner_id": owner_id, "conversation_id": conversation_id})
    store.clear_user(conversation_id)
    return jsonify({"ok": True, "user_id": conversation_id})


@app.get("/user_profile")
def user_profile():
    user_id = request.args.get("owner_id") or request.args.get("user_id", "default")
    store = get_store()
    return jsonify(store.get_profile(user_id))


if __name__ == "__main__":
    import threading

    def _warmup():
        try:
            from emotion import detect_emotion
            from crisis import detect_crisis
            from rag import retrieve_counseling

            detect_emotion("warmup")
            detect_crisis("warmup")
            retrieve_counseling("feeling stressed about school")
            print("Models warmed up.")
        except Exception as e:
            print(f"Warmup skipped: {type(e).__name__}: {e}")

    threading.Thread(target=_warmup, daemon=True).start()
    port = int(os.getenv("PORT", "5050"))
    print(f"MindCare UI + API → http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
