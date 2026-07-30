"""Central configuration: environment variables, paths and tunable constants.

Everything that a user might reasonably want to change lives here, so it only
has to be changed in one place.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]

load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
DOCS_DIR = BASE_DIR / "docs"
STATIC_DIR = BASE_DIR / "static"

DB_PATH = DATA_DIR / "app.db"
CHROMA_DIR = DATA_DIR / "chroma"

# ---------------------------------------------------------------------------
# LLM  (NVIDIA NIM -- OpenAI-compatible endpoint, free tier)
# ---------------------------------------------------------------------------
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "").strip()
NVIDIA_BASE_URL = os.getenv(
    "NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"
).strip()

# Writing the SQL is the accuracy-critical step, so it gets the bigger model.
# Measured on the 11 example questions:
#   70B : 11/11, but free-tier latency is erratic (0.9s to 25s, occasional
#         60s timeout - the endpoint is queue-contended, not our code)
#   8B  : 6/6 in isolation, but only 8-10/11 in a real conversation. It flakes
#         on "lowest attendance" (LEFT JOIN NULL ordering) and on normalising
#         "average score". Too unreliable for this step.
# Set this to ROUTER_MODEL if you would rather have speed than accuracy.
#   "meta/llama-3.3-70b-instruct" is listed by NIM but times out on every call.
# Do NOT use a "thinking" / "reasoning" / R1 style model: they emit
# chain-of-thought that breaks strict JSON and SQL parsing.
LLM_MODEL = "meta/llama-3.1-70b-instruct"

# Routing is a two-way classification and composition is a short rewrite of
# results already computed - an 8B model does both as well as a 70B one and, on
# the free tier, far more predictably. Measured over 8 identical calls:
#   8B  : 0.41-0.50s  (consistent)
#   70B : 0.94-25.19s (queue-contended, one 90s timeout)
# Set either to LLM_MODEL to use one model everywhere.
ROUTER_MODEL = "meta/llama-3.1-8b-instruct"
COMPOSE_MODEL = "meta/llama-3.1-8b-instruct"

# Run the router, the SQL generator and document retrieval concurrently instead
# of waiting for the router before starting the tool. Cuts a voice turn roughly
# in half. Costs one wasted SQL-generation call whenever the router picks
# documents, so set False if you are bumping the 40 req/min free-tier cap.
SPECULATIVE_TOOLS = True

LLM_TIMEOUT_SECONDS = 60
LLM_MAX_RETRIES = 3          # retries on 429 / transient network errors
LLM_BACKOFF_SECONDS = 2.0    # base for exponential backoff (2s, 4s, 8s)

# ---------------------------------------------------------------------------
# Speech to text  (faster-whisper, local, CPU)
# ---------------------------------------------------------------------------
WHISPER_MODEL_SIZE = "base"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"

# None lets Whisper detect the language per clip, which is what allows a
# question in Hindi and the next one in English. Force a code such as "en" or
# "hi" if detection picks the wrong one on your accent or microphone.
WHISPER_LANGUAGE: str | None = None

# ---------------------------------------------------------------------------
# Embeddings + vector store
# ---------------------------------------------------------------------------
EMBED_MODEL = "all-MiniLM-L6-v2"
CHROMA_COLLECTION = "college_docs"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
RAG_TOP_K = 4

# Chunks below this similarity are dropped rather than fed to the composer.
# Retrieval always returns its top K no matter how poor the match, and an
# off-topic question was measured retrieving chunks at 0.19 similarity - which
# the composer would then dutifully answer from. Below this floor the assistant
# says the documents do not cover it instead of inventing an answer.
RAG_MIN_SCORE = 0.25

# ---------------------------------------------------------------------------
# SQL tool
# ---------------------------------------------------------------------------
MAX_SQL_ROWS = 100

# ---------------------------------------------------------------------------
# Conversation memory
# ---------------------------------------------------------------------------
MAX_HISTORY_TURNS = 6

# How many of those turns the text-to-SQL prompt sees. Keep this small: a long
# history measurably degrades SQL accuracy on smaller models (they answer the
# earlier question instead of the current one), and a follow-up such as
# "and what about semester 4?" only needs the turn immediately before it.
SQL_HISTORY_TURNS = 2

# Idle sessions are dropped after this long, with a hard cap as a backstop.
# Without eviction every browser that ever loads the page leaks a history
# buffer for the lifetime of the process.
SESSION_TTL_SECONDS = 60 * 60 * 2
MAX_SESSIONS = 500

# ---------------------------------------------------------------------------
# Text to speech
# ---------------------------------------------------------------------------
TTS_DEFAULT_RATE = 175
