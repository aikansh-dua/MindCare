"""MindCare Streamlit UI — emotion + crisis + template/RAG + generative reply."""

import streamlit as st

from chatbot import chat
from memory import get_store

st.set_page_config(page_title="MindCare", layout="wide")
st.title("MindCare")
store = get_store()
st.caption(
    f"Emotion + crisis (local HF) · RAG counseling corpus · generator via .env · memory: **{store.backend}**"
)

if "user_id" not in st.session_state:
    st.session_state.user_id = "streamlit_user"

if st.button("Clear conversation"):
    store.clear_user(st.session_state.user_id)
    st.session_state.pop("last_result", None)
    st.success("Conversation cleared.")

user_input = st.text_input("Type your message here...")

if user_input:
    with st.spinner("Thinking..."):
        result = chat(st.session_state.user_id, user_input)
    st.session_state["last_result"] = {"input": user_input, "result": result}

if "last_result" in st.session_state:
    item = st.session_state["last_result"]
    result = item["result"]
    st.markdown(f"**You:** {item['input']}")
    st.markdown(f"**MindCare:** {result['response']}")
    st.write(
        {
            "emotion": result.get("emotion"),
            "crisis": result.get("crisis"),
            "reply_source": result.get("reply_source"),
            "memory_backend": result.get("memory_backend"),
            "profile": result.get("profile"),
            "has_summary": result.get("has_summary"),
        }
    )

history = store.get_history(st.session_state.user_id)
profile = store.get_profile(st.session_state.user_id)
summary = store.get_summary(st.session_state.user_id)
with st.expander("Conversation memory"):
    if summary:
        st.caption("Summary of earlier turns")
        st.write(summary)
    if profile.get("concerns") or profile.get("triggers"):
        st.caption("Profile")
        st.write(
            {
                "concerns": profile.get("concerns"),
                "triggers": profile.get("triggers"),
                "what_helped": profile.get("what_helped"),
                "last_emotion": profile.get("last_emotion"),
                "turn_count": profile.get("turn_count"),
            }
        )
    for m in history[-10:]:
        st.write(f"{m['role']}: {m['text']}")
