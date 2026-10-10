#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

if [ -x "${ROOT}/.venv/bin/python" ]; then
  PATH="${ROOT}/.venv/bin:${PATH}"
fi

if [ -f "${ROOT}/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/.env"
  set +a
fi

MODE="${1:-api}"

case "${MODE}" in
  api|ui)
    echo "Starting MindCare UI + API on http://127.0.0.1:${PORT:-5050}"
    echo "Open the URL in your browser. First chat may load HF models (slow once)."
    export PORT="${PORT:-5050}"
    exec python app.py
    ;;
  streamlit)
    echo "Legacy Streamlit UI on http://127.0.0.1:8501"
    export STREAMLIT_SERVER_FILE_WATCHER_TYPE="${STREAMLIT_SERVER_FILE_WATCHER_TYPE:-none}"
    exec python -m streamlit run bot.py \
      --server.headless true \
      --server.address 127.0.0.1 \
      --server.port 8501 \
      --browser.gatherUsageStats false \
      --server.fileWatcherType none
    ;;
  test)
    exec python evaluate.py
    ;;
  eval)
    shift || true
    exec python evaluate.py "$@"
    ;;
  redis)
    echo "Starting Redis on port 6379..."
    exec redis-server
    ;;
  *)
    echo "Usage: ./run.sh [ui|api|streamlit|test|eval|redis]"
    echo "  ./run.sh ui                   # web UI + API (http://127.0.0.1:5050)"
    echo "  ./run.sh streamlit            # legacy Streamlit UI"
    echo "  ./run.sh eval                 # full pipeline suites"
    echo "  ./run.sh eval --live          # include real Gemini/HF"
    echo "  ./run.sh eval --quick         # project + rag + crisis"
    echo "  ./run.sh eval --suite rag,chat"
    echo "  ./run.sh eval --rag-only"
    exit 1
    ;;
esac
