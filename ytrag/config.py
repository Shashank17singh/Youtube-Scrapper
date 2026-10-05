"""All tunables live here, every one of them env-driven.

Nothing else in the package reads os.getenv directly.
"""

import logging
import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
load_dotenv(find_dotenv(usecwd=True))

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ROOT_ENV = _REPO_ROOT / ".env"
if _ROOT_ENV.exists():
    load_dotenv(_ROOT_ENV, override=False)
ROOT = Path(os.getenv("YTRAG_ROOT", Path.home() / ".ytrag"))
AUDIO_DIR = ROOT / "audio"
TRANSCRIPT_DIR = ROOT / "transcripts"

for _d in (AUDIO_DIR, TRANSCRIPT_DIR):
    _d.mkdir(parents=True, exist_ok=True)
WHISPER_MODEL = os.getenv("YTRAG_WHISPER_MODEL", "large-v3")
WHISPER_DEVICE = os.getenv("YTRAG_WHISPER_DEVICE", "auto")  # auto | cuda | cpu
WHISPER_COMPUTE = os.getenv("YTRAG_WHISPER_COMPUTE", "")  # blank = pick per device
WHISPER_LANG = os.getenv(
    "YTRAG_WHISPER_LANG", "en"
)  # see README §6.0 - TEST THIS FIRST
WHISPER_BEAM = int(os.getenv("YTRAG_WHISPER_BEAM", 5))  # noqa: PLW1508
WHISPER_BATCH = int(os.getenv("YTRAG_WHISPER_BATCH", 8))  # noqa: PLW1508
CHUNK_SECONDS = int(os.getenv("YTRAG_CHUNK_SECONDS", 75))  # noqa: PLW1508
CHUNK_OVERLAP_SECONDS = int(os.getenv("YTRAG_CHUNK_OVERLAP", 15))  # noqa: PLW1508
MIN_CHUNK_WORDS = int(os.getenv("YTRAG_MIN_CHUNK_WORDS", 15))  # noqa: PLW1508
LINK_REWIND_SECONDS = int(os.getenv("YTRAG_LINK_REWIND", 5))  # noqa: PLW1508
EMBED_MODEL = os.getenv("YTRAG_EMBED_MODEL", "all-MiniLM-L6-v2")
EMBED_BATCH = int(os.getenv("YTRAG_EMBED_BATCH", 16))  # noqa: PLW1508
EMBED_QUERY_PREFIX = os.getenv("YTRAG_EMBED_QUERY_PREFIX", "")
DOWNLOAD_SLEEP_MIN = float(os.getenv("YTRAG_DOWNLOAD_SLEEP_MIN", 2))  # noqa: PLW1508
DOWNLOAD_SLEEP_MAX = float(os.getenv("YTRAG_DOWNLOAD_SLEEP_MAX", 6))  # noqa: PLW1508
COOKIES_FROM_BROWSER = os.getenv("YTRAG_COOKIES_FROM_BROWSER", "")
COOKIES_FILE = os.getenv("YTRAG_COOKIES_FILE", "")
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_PATH = Path(os.getenv("YTRAG_QDRANT_PATH", ROOT / "qdrant"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION = os.getenv("YTRAG_COLLECTION", "dsa_lectures")
UPSERT_BATCH = int(os.getenv("YTRAG_UPSERT_BATCH", 128))  # noqa: PLW1508
TOP_K = int(os.getenv("YTRAG_TOP_K", 6))  # noqa: PLW1508
MAX_DISTANCE = float(os.getenv("YTRAG_MAX_DISTANCE", 0.6))  # noqa: PLW1508
CONFIDENT_DISTANCE = float(os.getenv("YTRAG_CONFIDENT_DISTANCE", 0.45))  # noqa: PLW1508
TITLE_BOOST = float(os.getenv("YTRAG_TITLE_BOOST", 0.06))  # noqa: PLW1508
_DEFAULT_BACKEND = (
    "gemini"
    if (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    else ("groq" if os.getenv("GROQ_API_KEY") else "none")
)
LLM_BACKEND = os.getenv("YTRAG_LLM_BACKEND", _DEFAULT_BACKEND)  # gemini | groq | none
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
LLM_MODEL = os.getenv("YTRAG_LLM_MODEL", "")  # blank = per-backend default
GROQ_MODEL = os.getenv("YTRAG_GROQ_MODEL", "openai/gpt-oss-120b")
GEMINI_MODEL = os.getenv("YTRAG_GEMINI_MODEL", "gemini-3.8-flash")
REFUSAL = "Ye topic in lectures me cover nahi hua."
MAX_QUESTION_CHARS = int(os.getenv("YTRAG_MAX_QUESTION_CHARS", 500))  # noqa: PLW1508
RATE_LIMIT_REQUESTS = int(os.getenv("YTRAG_RATE_LIMIT_REQUESTS", 20))  # noqa: PLW1508
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("YTRAG_RATE_LIMIT_WINDOW", 60))  # noqa: PLW1508
