"""Settings read from the environment (.env is loaded when present)."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

ACRUXCORE_API_KEY = os.environ.get("ACRUXCORE_API_KEY", "")
ACRUXCORE_BASE_URL = os.environ.get("ACRUXCORE_BASE_URL", "https://api.acruxcore.com/api/v1")
ACRUXCORE_DASHBOARD_URL = os.environ.get("ACRUXCORE_DASHBOARD_URL", "https://acruxcore.com")

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Stage 1 reads the image. Stage 2 (in AcruxCore) turns the text into fields.
READER_MODEL = os.environ.get("READER_MODEL", "google/gemma-4-31b-it")
EXTRACT_MODEL = os.environ.get("EXTRACT_MODEL", "rs-gemma-4-31b")  # a name registered in AcruxCore
PROMPT_NAME = "receipt-extract"
PROMPT_ALIAS = os.environ.get("PROMPT_ALIAS", "production")

DATA = ROOT / "data"
RECEIPTS = DATA / "receipts"
LABELS = DATA / "labels.json"
TRANSCRIPTS = DATA / "transcripts"
