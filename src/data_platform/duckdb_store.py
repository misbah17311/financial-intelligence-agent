"""
DuckDB wrapper — loads the company fundamentals parquet into a SQL table
and provides a safe query interface for the agent.
"""

import duckdb
import pandas as pd
from src.config import DUCKDB_PATH, PROCESSED_DATA_DIR
from src.logger import logger


_conn = None
_read_only = True  # default to read-only so multiple processes can share


def get_connection(write=False):
    """Lazy singleton connection to the DuckDB database."""
    global _conn, _read_only
    if _conn is not None and write and _read_only:
        # need to upgrade to writable — close and reconnect
        _conn.close()
        _conn = None
    if _conn is None:
        DUCKDB_PATH.parent.mkdir(parents=True, exist_ok=True)
        ro = not write
        _conn = duckdb.connect(str(DUCKDB_PATH), read_only=ro)
        _read_only = ro
    return _conn


def init_tables():
    """
    Create the companies table from the preprocessed parquet file.
    Idempotent — safe to call multiple times.
    """
    conn = get_connection(write=True)
    parquet_path = PROCESSED_DATA_DIR / "company_fundamentals.parquet"

    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Company fundamentals not found at {parquet_path}. "
            f"Run the ingestion pipeline first."
        )

    conn.execute("DROP TABLE IF EXISTS companies")
    conn.execute(f"""
        CREATE TABLE companies AS
        SELECT * FROM read_parquet('{parquet_path}')
    """)

    count = conn.execute("SELECT COUNT(*) FROM companies").fetchone()[0]
    logger.info(f"DuckDB: loaded {count} rows into 'companies' table")
    return count


def run_query(sql: str) -> pd.DataFrame:
    """
    Execute a read-only SQL query and return results as a DataFrame.
    We only allow SELECT statements to prevent any writes from the agent.
    """
    cleaned = sql.strip().rstrip(";").strip()

    # basic safety check — only allow read queries
    first_word = cleaned.split()[0].upper() if cleaned else ""
    if first_word not in ("SELECT", "WITH", "EXPLAIN"):
        raise ValueError(
            f"Only SELECT queries are allowed. Got: {first_word}..."
        )

    conn = get_connection()
    try:
        result = conn.execute(cleaned).fetchdf()
        logger.debug(f"DuckDB query returned {len(result)} rows: {cleaned[:100]}...")
        return result
    except Exception as e:
        logger.error(f"DuckDB query failed: {e}\nQuery: {cleaned}")
        raise


def get_schema_info() -> str:
    """
    Returns a human-readable schema description for the planner agent.
    This helps the LLM write correct SQL.
    """
    conn = get_connection()
    try:
        cols = conn.execute("DESCRIBE companies").fetchdf()
        schema = "Table: companies\nColumns:\n"
        for _, row in cols.iterrows():
            schema += f"  - {row['column_name']} ({row['column_type']})\n"

        # also show some sample values so the LLM knows what data looks like
        sample = conn.execute(
            "SELECT DISTINCT sector FROM companies ORDER BY sector"
        ).fetchdf()
        schema += f"\nSectors: {', '.join(sample['sector'].tolist())}\n"

        years = conn.execute(
            "SELECT DISTINCT year FROM companies ORDER BY year"
        ).fetchdf()
        schema += f"Years: {', '.join(str(y) for y in years['year'].tolist())}\n"

        count = conn.execute("SELECT COUNT(DISTINCT company_name) FROM companies").fetchone()[0]
        schema += f"Total companies: {count}\n"

        return schema
    except Exception:
        return "Table 'companies' not yet initialized. Run ingestion first."


if __name__ == "__main__":
    init_tables()
    print(get_schema_info())
    print(run_query("SELECT sector, year, ROUND(AVG(revenue_mn), 2) as avg_rev FROM companies GROUP BY sector, year ORDER BY sector, year LIMIT 10"))
