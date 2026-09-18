import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
DATABASE_DIR = BASE_DIR / "database"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

DATABASE_PATH = DATABASE_DIR / "case_study.db"

INTERVIEW_GUIDE_PATH = DATA_DIR / "Interview_Guide.txt"

TRANSCRIPT_FILES = [
    DATA_DIR / "Transcript_1_France.txt",
    DATA_DIR / "Transcript_2_Germany.txt",
    DATA_DIR / "Transcript_3_UK.txt",
]


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")


GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
GEMINI_EMBEDDING_MODEL = os.getenv(
    "GEMINI_EMBEDDING_MODEL",
    "models/gemini-embedding-001"
)


GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.0"))


RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "6"))
CROSS_CALL_TOP_K_PER_EXPERT = int(os.getenv("CROSS_CALL_TOP_K_PER_EXPERT", "8"))


def ensure_directories() -> None:
    """Create the runtime directories (database/, vectorstore/) if missing."""
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)


def require_api_key() -> None:
    """Raise a clear, actionable error if GOOGLE_API_KEY is missing."""
    if not GOOGLE_API_KEY:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Copy .env.example to .env and add your "
            "Google AI Studio API key before running the app."
        )
