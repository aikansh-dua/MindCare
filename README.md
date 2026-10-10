# MindCare

Empathetic mental health companion. Supportive conversation only — **not** professional care.

India helplines: **1860 266 2345** / **9152987821**. If someone is in immediate danger, contact local emergency services.

---

## What it does

Each user message goes through a fixed pipeline. Emotion and crisis models run **locally** (Hugging Face). Replies are generated with **Gemini** by default (or HF Inference API / local weights). Conversation history lives in **Redis** (falls back to in-memory if Redis is down).

| Stage | Technology |
|-------|------------|
| Emotion (28 labels) | [`joeddav/distilbert-base-uncased-go-emotions-student`](https://huggingface.co/joeddav/distilbert-base-uncased-go-emotions-student) |
| Crisis detection | Keyword gate + [`facebook/bart-large-mnli`](https://huggingface.co/facebook/bart-large-mnli) |
| Knowledge | ~85 intents + ~2,500 counseling pairs via **embedding RAG** (`sentence-transformers/all-MiniLM-L6-v2`) |
| Reply generation | **Gemini** (default) · HF Inference API · optional local HF |
| Memory | Redis (`REDIS_URL`), with in-process fallback |

---

## Pipeline (how a reply is chosen)

```
user message
    │
    ├─► 1. Emotion  (DistilBERT GoEmotions)
    ├─► 2. Crisis   (keywords → optional BART confirm)
    │         │
    │         └─ if crisis ──► fixed safety reply  (reply_source omitted / crisis=true)
    │
    ├─► 3. Retrieve knowledge
    │         • best intent from mindcare_dataset.json
    │         • embedding RAG (MiniLM cosine sim) over counseling_rag.json
    │           (falls back to token-overlap if embeddings unavailable)
    │
    └─► 4. Pick reply (first match wins)
              │
              ├─ greeting template?  → canned hello                    (reply_source: template)
              ├─ else generate       → Gemini → HF API → local (failover)
              │     (RAG + intent + memory summary/profile in prompt)
              └─ all generators fail → safe / dataset fallback         (reply_source: safe_fallback | dataset_fallback)
                    │
                    └─► 5. Save turn, rolling summary, structured profile in Redis
```

### Reply priority (important)

| Priority | When | `reply_source` |
|----------|------|----------------|
| 1. Crisis | Clear self-harm / suicide risk language | crisis path (`crisis: true`) |
| 2. Template | Short greetings only (`hi` / `hello` / `hey`) | `template` |
| 3. Generation | Everything else (with provider failover) | `gemini` / `hf_api` / `local` |
| 4. Fallback | All generators fail | `safe_fallback` or `dataset_fallback` |

**Templates are greetings-only** so sadness, breakup, device frustration, etc. go through Gemini (or failover) for more natural replies.

**RAG is never shown raw as the user-facing answer.** Weak token matches used to dump unrelated counseling text; retrieval is stricter now and only feeds the generator prompt when relevant.

### Debug fields in `/chat` response

```json
{
  "response": "...",
  "emotion": "sadness",
  "crisis": false,
  "reply_source": "gemini",
  "rag_hits": 0,
  "intent": "stressed",
  "memory_backend": "redis",
  "generation": { "provider": "gemini", "gemini_model": "gemini-3.5-flash", ... }
}
```

If generation fails, `generation_error` is included for debugging.

---

## Setup

```bash
git clone https://github.com/aikansh-dua/MindCare.git
cd MindCare
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Optional: refresh intents + counseling RAG corpus
python expand_dataset.py

cp .env.example .env
# edit .env — set GEMINI_API_KEY (or HF_TOKEN)
```

### Redis

```bash
brew services start redis
# or: ./run.sh redis
```

Without Redis, memory still works in-process (lost on restart).

---

## Generation providers

Set `GENERATION_PROVIDER` in `.env`.

### A — Gemini (recommended)

1. Free key: https://aistudio.google.com/apikey  
2. `.env`:

```env
GENERATION_PROVIDER=gemini
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-3.5-flash
```

Use a current model such as `gemini-3.5-flash` or `gemini-flash-lite-latest`. Older IDs like `gemini-2.5-flash` may return **404 / not available to new users**. Some 2.0 free-tier quotas can show **limit: 0**.

### B — Hugging Face Inference API

```env
GENERATION_PROVIDER=hf_api
HF_TOKEN=hf_xxx
HF_CHAT_MODEL=tanusrich/Mental_Health_Chatbot
```

### C — Local weights

Needs substantial RAM for the 7B chatbot. FLAN-T5 is only a last-resort local fallback and is **not** recommended for quality replies.

```env
GENERATION_PROVIDER=local
CHATBOT_MODEL=tanusrich/Mental_Health_Chatbot
CHATBOT_FALLBACK_MODEL=google/flan-t5-base
```

---

## Run

```bash
./run.sh ui     # Web UI + API → http://127.0.0.1:5050
```

Open that URL in your browser. First chat may load HF models (slow once).

Optional legacy Streamlit: `./run.sh streamlit` → http://127.0.0.1:8501

### API smoke test

```bash
curl -s http://127.0.0.1:5050/health

curl -s -X POST http://127.0.0.1:5050/chat \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"u1","message":"I feel overwhelmed at work and do not know where to start"}'
```

Expect `"reply_source":"gemini"` when Gemini is configured. Short greetings like `"hi"` return `"template"`.

### Useful endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/` | Web UI |
| `GET` | `/health` | Status + generation config |
| `GET` | `/api` | API metadata JSON |
| `POST` | `/chat` | Main chat turn |
| `POST` | `/clear` | Clear history + profile for a user |
| `GET` | `/chat_memory?user_id=` | Conversation history + summary |
| `GET` | `/user_profile?user_id=` | Structured profile |

In the UI, use **New conversation** to clear Redis memory for your browser session.

---

## Evaluation

Golden cases live in `eval_cases.json` (greeting, laptop, breakup, crisis, work overwhelm, RAG relevance).

```bash
./run.sh eval                 # full suite (generation stubbed)
./run.sh eval --live          # + real Gemini/HF smoke
./run.sh eval --quick         # project + rag + crisis only
./run.sh eval --suite rag,chat
./run.sh eval --rag-only
```

Suites: **project** (files/imports/config/API routes) · **emotion** · **crisis** · **rag** (embeddings) · **chat** (full routing) · **memory** · **live** (optional).
Fixtures: `eval_cases.json`.

---

## Project structure

```
MindCare/
├── app.py                 # Flask: web UI + API
├── frontend/              # Custom calm chat UI
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── bot.py                 # Legacy Streamlit UI (optional)
├── chatbot.py             # Orchestration (emotion → crisis → template/RAG → generate)
├── emotion.py             # DistilBERT GoEmotions
├── crisis.py              # Keyword gate + BART MNLI
├── rag.py                 # Embedding retrieval (+ token fallback)
├── generation.py          # Gemini / HF API / local + failover
├── chatbot_local.py       # Local HF dialogue loader (optional)
├── memory.py              # Redis history, summary, profile
├── expand_dataset.py      # Build / refresh intents + counseling RAG + embed cache
├── evaluate.py            # Full pipeline evaluation
├── eval_cases.json        # Eval fixtures
├── mindcare_dataset.json  # Intent knowledge base
├── counseling_rag.json    # Counseling Q&A retrieval corpus
├── counseling_rag_embeddings.npz  # Cached MiniLM vectors (auto-built)
├── run.sh                 # ui | api | streamlit | eval | redis
├── .env.example
└── README.md
```

---

## Dataset notes

- Base intents from the original mental-health chatbot dataset  
- Expanded with career/internship, academic, burnout, loneliness, and related intents  
- RAG corpus from public counseling conversations (Amod dataset), retrieved with **sentence embeddings**  
- Re-run `python expand_dataset.py` to refresh intents, pairs, and embedding cache  

Respect upstream dataset licenses if you redistribute the JSON files.

---

## More docs

See [CONTEXT.md](CONTEXT.md) for an architecture and design-decision reference.

## Disclaimer

MindCare is a supportive companion, not a substitute for professional mental-health care.

## License

No license has been chosen yet; all rights reserved by the author until one is added.
