"""
Central config for the entire project.
Reads from .env so we can swap LLM providers, tweak retrieval params,
etc. without touching any code.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# -- paths --
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CHROMA_DIR = PROCESSED_DATA_DIR / "chroma_store"
DUCKDB_PATH = PROCESSED_DATA_DIR / "financials.duckdb"
BM25_INDEX_PATH = PROCESSED_DATA_DIR / "bm25_index.pkl"

# -- LLM setup --
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# default models per provider — override with LLM_MODEL in .env
_DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-4-20250514",
    "groq": "llama-3.3-70b-versatile",
}
LLM_MODEL = os.getenv("LLM_MODEL") or _DEFAULT_MODELS.get(LLM_PROVIDER, "gpt-4o-mini")

# embedding model — runs locally, no API key needed
# use "or" instead of default param so empty string in .env still falls back
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL") or "all-MiniLM-L6-v2"

# -- retrieval tuning --
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.35"))
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "10"))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "5"))
CHUNK_SIZE = 1500       # chars per chunk — bigger chunks = fewer total = faster indexing
CHUNK_OVERLAP = 100     # overlap between consecutive chunks

# -- logging --
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# -- dataset info (shown to the planner agent + UI) --
DATASET_DESCRIPTION = """
Available structured data (DuckDB — SQL queries):
  Table: companies
    Columns: company_name, ticker, sector, revenue_mn (in millions USD),
             net_income_mn, total_assets_mn, market_cap_mn, year, quarter
    Coverage: ~5,000 publicly traded US companies, 2020–2024 quarterly data

Available unstructured data (ChromaDB + BM25 — semantic & keyword search):
  Source: Financial news articles and analyst reports
  Fields per chunk: text, source, date, companies_mentioned, sector
  Coverage: ~100,000 articles from 2020–2024
  Topics: earnings, market analysis, sector trends, M&A, analyst opinions
""".strip()
