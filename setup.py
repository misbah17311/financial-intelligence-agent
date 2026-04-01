"""
One-shot setup script — run this once to download data, build all indexes,
and verify everything works.

Usage: python setup.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.logger import logger


def main():
    logger.info("=" * 60)
    logger.info("FINANCIAL INTELLIGENCE AGENT — SETUP")
    logger.info("=" * 60)

    # step 1: check .env
    from src.config import LLM_PROVIDER, OPENAI_API_KEY, ANTHROPIC_API_KEY, GROQ_API_KEY
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.exists(env_path):
        logger.error(
            "No .env file found. Copy .env.example to .env and fill in your API key:\n"
            "  cp .env.example .env"
        )
        sys.exit(1)

    key_map = {"openai": OPENAI_API_KEY, "anthropic": ANTHROPIC_API_KEY, "groq": GROQ_API_KEY}
    active_key = key_map.get(LLM_PROVIDER, "")
    if not active_key or active_key.startswith("sk-your") or active_key.startswith("gsk_your"):
        logger.error(
            f"LLM_PROVIDER is set to '{LLM_PROVIDER}' but the API key looks unset. "
            f"Edit your .env file."
        )
        sys.exit(1)

    logger.info(f"LLM provider: {LLM_PROVIDER}")

    # step 2: ingest data
    logger.info("\n--- Step 1: Data Ingestion ---")
    from src.data_platform.ingest import run_ingestion
    run_ingestion()

    # step 3: load into DuckDB
    logger.info("\n--- Step 2: DuckDB Setup ---")
    from src.data_platform.duckdb_store import init_tables, get_schema_info
    init_tables()
    print(get_schema_info())

    # step 4: build ChromaDB vector index
    logger.info("\n--- Step 3: ChromaDB Vector Index ---")
    from src.data_platform.chroma_store import build_index as build_chroma
    build_chroma()

    # step 5: build BM25 keyword index
    logger.info("\n--- Step 4: BM25 Keyword Index ---")
    from src.data_platform.bm25_store import build_index as build_bm25
    build_bm25()

    # step 6: quick sanity test
    logger.info("\n--- Step 5: Sanity Test ---")
    from src.agents.graph import run_query
    test_q = "What is the average revenue of Technology sector companies in 2024?"
    logger.info(f"Test query: {test_q}")
    result = run_query(test_q)
    print(f"\nAnswer:\n{result['answer'][:500]}")
    print(f"\nConfidence: {result['confidence']}")

    logger.info("\n" + "=" * 60)
    logger.info("SETUP COMPLETE")
    logger.info("Run the UI with: streamlit run src/ui/app.py")
    logger.info("Run evaluation with: python evaluation/evaluate.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
