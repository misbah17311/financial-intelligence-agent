# LangChain-compatible tools the agents can call
# three tools: sql_query, semantic_search, get_dataset_info

from langchain_core.tools import tool
from src.data_platform.duckdb_store import run_query, get_schema_info
from src.retrieval.hybrid import hybrid_search
from src.config import DATASET_DESCRIPTION
from src.logger import logger


@tool
def sql_query(query: str) -> str:
    """
    Run a SQL query against the company financials database.
    Use this for questions about specific numbers: revenue, profit,
    market cap, comparisons between companies, aggregations by sector/year, etc.

    The table is called 'companies' with columns:
    company_name, ticker, sector, year, quarter, revenue_mn, net_income_mn,
    total_assets_mn, market_cap_mn

    Args:
        query: A valid SQL SELECT statement.

    Returns:
        Query results as a formatted string table, or an error message.
    """
    logger.info(f"Tool call: sql_query({query[:100]}...)")
    try:
        df = run_query(query)
        if df.empty:
            return "Query returned no results. Check your filters — maybe the company name or date range doesn't match the data."

        # format as a readable table, cap at 50 rows to not blow up the context
        if len(df) > 50:
            result = df.head(50).to_string(index=False)
            result += f"\n\n... showing 50 of {len(df)} total rows"
        else:
            result = df.to_string(index=False)

        return result
    except ValueError as e:
        return f"Query blocked: {e}"
    except Exception as e:
        return f"SQL error: {e}. Double-check the table/column names."


@tool
def semantic_search(query: str) -> str:
    """
    Search financial news articles and analyst reports by meaning.
    Use this for questions about opinions, trends, sentiment, reasons,
    explanations — anything that isn't a simple number lookup.

    Args:
        query: A natural language search query describing what you're looking for.

    Returns:
        Top matching article passages with relevance scores and confidence level.
    """
    logger.info(f"Tool call: semantic_search({query[:100]}...)")
    try:
        results = hybrid_search(query)
        confidence = results["confidence"]

        if not results["results"]:
            return "No relevant articles found for this query."

        # format results for the LLM to consume
        output_parts = [f"Search confidence: {confidence}"]
        output_parts.append(f"Strategy: {results['strategy']}")
        output_parts.append(f"---")

        for i, doc in enumerate(results["results"], 1):
            score_info = ""
            if "rerank_score" in doc:
                score_info = f" (relevance: {doc['rerank_score']:.3f})"
            elif "rrf_score" in doc:
                score_info = f" (RRF: {doc['rrf_score']:.4f})"

            source = doc.get("metadata", {}).get("source", "unknown")
            output_parts.append(f"[{i}]{score_info} ({source}): {doc['text']}")

        # add a warning if confidence is low
        if confidence in ("LOW", "NONE"):
            output_parts.append(f"\n⚠ Low confidence — the retrieved passages may not be directly relevant. "
                              f"Consider rephrasing or note that the dataset may not cover this topic.")

        return "\n\n".join(output_parts)
    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        return f"Search error: {e}"


@tool
def get_dataset_info() -> str:
    """
    Returns information about what data is available in the system.
    Call this when you need to understand what questions the system can answer,
    or to check the database schema before writing SQL.

    Returns:
        Description of available datasets and the SQL table schema.
    """
    schema = get_schema_info()
    return f"{DATASET_DESCRIPTION}\n\nSQL Schema:\n{schema}"


# convenience list for registering all tools with agents
ALL_TOOLS = [sql_query, semantic_search, get_dataset_info]
