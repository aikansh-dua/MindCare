from fastapi import FastAPI, HTTPException, Form
import json
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

from llm_providers import build_models, dataset_advice_for_query, load_intents, provider_status

load_dotenv()

app = FastAPI(
    title="MindCare",
    description="Empathetic mental health support chatbot powered by Gemini + RAG",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    models = build_models()
except Exception as e:
    models = None
    _model_error = str(e)
else:
    _model_error = None

memory_storage = []
user_profiles_storage = {}
dataset_intents = []


@app.on_event("startup")
async def startup_event():
    global dataset_intents
    dataset_intents = load_intents()


@app.get("/health")
def health():
    status = provider_status()
    status["models_ready"] = models is not None
    status["model_error"] = _model_error
    return status


@app.get("/chat_memory")
def get_chat_memory(user_id: str = "default"):
    return {"memory": memory_storage}


@app.post("/save_chat_memory")
async def update_chat_memory(user_id: str = Form(...), memory: str = Form(...)):
    try:
        memory_data = json.loads(memory)
        global memory_storage
        memory_storage = memory_data
        return {"message": "Chat memory updated successfully."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid memory format: {str(e)}")


@app.get("/user_profile")
def get_user_profile(user_id: str = "default"):
    if user_id not in user_profiles_storage:
        user_profiles_storage[user_id] = {"concerns": [], "coping_strategies": []}
    return user_profiles_storage[user_id]


@app.post("/save_user_profile")
async def update_user_profile(profile: dict):
    user_id = profile.get("user_id", "default")
    user_profiles_storage[user_id] = profile
    return {"message": "User profile updated successfully."}


@app.post("/chat")
async def chat(user_id: str, user_query: str):
    global memory_storage

    if models is None:
        raise HTTPException(
            status_code=503,
            detail=_model_error
            or "GEMINI_API_KEY not configured. Add it to .env — see README.",
        )

    if user_id not in user_profiles_storage:
        user_profiles_storage[user_id] = {"concerns": [], "coping_strategies": []}
    user_profile = user_profiles_storage[user_id]

    crisis_keywords = ["suicide", "end my life", "self-harm", "hopeless", "kill myself"]
    deep_emotion_keywords = ["anxious", "overwhelmed", "panic attack", "lonely", "heartbroken"]
    casual_keywords = ["motivate me", "how to be happy", "mental health tips", "relaxation"]

    selected_model = models["default"]

    if any(word in user_query.lower() for word in crisis_keywords):
        return {
            "response": (
                "URGENT: If you are in crisis, seek help immediately. "
                "India helpline: 1860 266 2345 / 9152987821. You are not alone."
            ),
            "provider": "crisis_protocol",
        }

    if any(word in user_query.lower() for word in deep_emotion_keywords):
        selected_model = models["deep_emotion"]
    elif any(word in user_query.lower() for word in casual_keywords):
        selected_model = models["casual"]

    advice = dataset_advice_for_query(user_query, dataset_intents)
    enriched_query = user_query
    if advice:
        enriched_query = f"{user_query}\n[Knowledge base context: {advice}]"

    history = "\n".join(
        [
            f"User: {m['text']}" if m["role"] == "user" else f"AI: {m['text']}"
            for m in memory_storage[-5:]
            if memory_storage
        ]
    )

    user_context = f"User concerns: {', '.join(user_profile.get('concerns', []))}\n"
    user_context += f"Past coping strategies: {', '.join(user_profile.get('coping_strategies', []))}\n"

    modified_query = (
        f"{user_context}\n"
        f"Recent conversation:\n{history}\n\n"
        f"User: {enriched_query}\n"
        f"MindCare:"
    )

    try:
        response = selected_model.invoke(modified_query, max_tokens=1024)
        if isinstance(response, str):
            response = response.strip()
        else:
            response = str(response).strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini error: {str(e)}")

    if not isinstance(memory_storage, list):
        memory_storage = []

    memory_storage.append({"role": "user", "text": user_query})
    memory_storage.append({"role": "ai", "text": response})

    return {
        "user_query": user_query,
        "response": response,
        "provider": "gemini",
        "rag_context_used": bool(advice),
    }
