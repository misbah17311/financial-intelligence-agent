# Architecture

## System Overview

```
User Question
     │
     ▼
┌──────────────┐
│  FastAPI +    │  (production UI with dark theme)
│  HTML/CSS/JS  │
└──────┬───────┘
       │
       ▼
┌─ Input Guardrails ──────────────────────────┐
│  Length → PII → SQL Injection → Prompt      │
│  Injection → Topic Relevance                │
│  (short-circuits on first failure)          │
└──────┬──────────────────────────────────────┘
       │ all passed
       ▼
┌──────────────────────────────────────────────┐
│              LangGraph Pipeline               │
│                                               │
│  ┌─────────┐   ┌──────────┐   ┌──────────┐  │
│  │ Planner │──▶│ Retriever│──▶│ Analyst  │  │
│  └─────────┘   └──────────┘   └──────────┘  │
│                                     │         │
│                               ┌──────────┐   │
│                               │  Critic  │   │
│                               └────┬─────┘   │
│                                    │          │
│                          APPROVED? ──▶ Done   │
│                          REVISE?   ──▶ Retry  │
└──────────────────────────────────────────────┘
       │                    │
       ▼                    ▼
┌──────────┐       ┌──────────────┐
│  DuckDB  │       │ Hybrid Search│
│  (SQL)   │       │              │
└──────────┘       │ ChromaDB     │
                   │ + BM25       │
                   │ + RRF Fusion │
                   │ + Reranker   │
                   └──────────────┘
```

## Guardrails System

Every query passes through input guardrails before reaching the agent pipeline, and answers pass through output validation before reaching the user.

### Input Guardrails (pre-pipeline)

Run sequentially, cheapest checks first. Short-circuits on first failure — no LLM tokens spent on blocked queries.

1. **Input Length** (`check_input_length`): Rejects queries > 2,000 chars (prompt stuffing defense) and empty inputs.
2. **PII Detection** (`check_pii`): Regex patterns for SSNs, credit cards, emails, phone numbers. Blocks the query and tells the user no personal data is needed.
3. **SQL Injection** (`check_sql_injection`): Catches DROP/DELETE/UPDATE injection, UNION SELECT, comment injection (`--`), DuckDB-specific exploits (LOAD_EXTENSION, COPY, ATTACH DATABASE). Additionally, the DuckDB layer itself only allows SELECT/WITH/EXPLAIN.
4. **Prompt Injection** (`check_prompt_injection`): Pattern matching for jailbreak phrases ("ignore previous instructions", "act as", "DAN", system prompt extraction attempts, chat template markers).
5. **Topic Relevance** (`check_topic_relevance`): Blocks queries about cooking, weather, creative writing, hacking, medical/legal advice, dating — anything clearly outside financial domain.

### Output Guardrails (post-pipeline)

1. **Response Validation** (`validate_response`): Checks the LLM's output for signs of prompt leaking, generic AI refusal patterns, or instruction-following behavior that shouldn't appear in financial answers.

### API Integration

The `POST /api/query` endpoint runs both layers:
```python
# input check — blocks before any LLM call
validation = validate_input(question)
if not validation["passed"]:
    return blocked_response(validation)

# run agent pipeline
result = run_query(question)

# output check
resp_ok, issue = validate_response(result["answer"])
```

The response includes guardrail details so the UI can show which checks passed/failed as colored pills.

## Agent Pipeline

The system uses four specialized agents wired together as a LangGraph state machine. Each agent has a single job and passes its output to the next through a shared state object.

### 1. Planner

Receives the user's question along with the database schema and dataset description. Outputs a JSON execution plan — a list of tool calls with their inputs.

Example output for "Compare Apple and Microsoft revenue in 2024":
```json
[
  {"tool": "sql_query", "input": "SELECT company_name, quarter, revenue_mn FROM companies WHERE company_name IN ('Apple', 'Microsoft') AND year = 2024 ORDER BY quarter"},
  {"tool": "semantic_search", "input": "Apple Microsoft revenue comparison 2024 analysis"}
]
```

The planner decides which tools to call and in what order. For pure data lookups it'll use SQL only. For opinion/sentiment questions it'll use semantic search. For complex questions it combines both.

### 2. Retriever

Executes each step from the planner's output. Calls the underlying functions directly (bypassing LangChain's tool serialization layer to avoid edge cases with input formatting).

Two tools are available:
- **sql_query**: Runs a SQL SELECT against DuckDB. Has safety checks — only SELECT statements are allowed, no DDL/DML.
- **semantic_search**: Runs the hybrid retrieval pipeline (described below). Returns the top passages with relevance scores.

### 3. Analyst

Takes the user's original question plus all retrieved data and writes a comprehensive answer. The prompt instructs it to cite sources, format currency values properly (millions with $ prefix), and distinguish between what the data says vs. what it doesn't cover.

### 4. Critic

Reviews the draft answer against the retrieved data. Checks for factual errors, hallucinated numbers, and whether the answer actually addresses the question. If the answer is good, it responds with APPROVED and the answer goes to the user. If there's a real factual problem, it responds with REVISE and specific feedback — the analyst gets one retry attempt to fix it.

The critic is deliberately lenient about style issues — it only rejects for wrong numbers or fabricated claims. This keeps response times reasonable (most answers are approved on the first pass).

## Retrieval Pipeline

The hybrid retrieval system combines three search strategies and merges them for maximum recall and precision.

### Step 1: Parallel Search

Two searches run on the same query:
- **Vector search** (ChromaDB): Encodes the query with `all-MiniLM-L6-v2` and finds the nearest document embeddings by cosine similarity. Good at catching semantically related content even when exact words don't match.
- **BM25 keyword search**: Classic TF-IDF-style scoring. Good at finding documents with specific company names, ticker symbols, or financial terms that semantic search might miss.

### Step 2: Reciprocal Rank Fusion (RRF)

Results from both searches are merged using RRF with k=60:

```
RRF_score(doc) = Σ 1 / (k + rank_in_list)
```

Documents appearing in both lists get boosted. This produces a single ranked list that balances semantic relevance with keyword precision.

### Step 3: Cross-Encoder Reranking

The top candidates from RRF are re-scored by a cross-encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`). Unlike the embedding model which encodes query and document separately, the cross-encoder processes them together, giving much more accurate relevance scores at the cost of being slower (which is why we only apply it to the top candidates, not the full corpus).

### Step 4: Confidence Scoring

Based on the reranker scores, the system assigns a confidence level:
- **HIGH**: Top result scores above 2.0
- **MEDIUM**: Top result scores between 0.0 and 2.0
- **LOW/NONE**: Top result scores below 0.0

This confidence level is shown to the user alongside the answer.

## Data Layer

### DuckDB (Structured Data)

Embedded analytical database holding company fundamentals. Single table schema:

| Column | Type | Example |
|--------|------|---------|
| company_name | VARCHAR | Apple |
| ticker | VARCHAR | AAPL |
| sector | VARCHAR | Technology |
| year | INTEGER | 2024 |
| quarter | VARCHAR | Q4 |
| revenue_mn | DOUBLE | 59619.85 |
| net_income_mn | DOUBLE | 14773.21 |
| total_assets_mn | DOUBLE | 338932.45 |
| market_cap_mn | DOUBLE | 2891043.12 |

Coverage: 105 companies, 7 sectors, 20 quarters (Q1 2020 – Q4 2024) = 2,100 rows.

The connection opens in read-only mode at runtime so multiple processes (e.g. Streamlit + evaluation) can access it concurrently.

### ChromaDB (Vector Store)

Stores document embeddings for ~110K news article chunks. Each chunk is embedded using `all-MiniLM-L6-v2` (384-dimensional vectors, runs locally on CPU). Chunks are ~1500 characters with 100 character overlap to preserve context across boundaries.

### BM25 Index

A serialized BM25 index (pickle file) built from the same chunks. Tokenized with simple whitespace + lowercasing. Loaded into memory at startup for fast keyword search.

## LLM Abstraction

The `llm.py` module provides a `get_llm()` factory that returns the appropriate LangChain chat model based on `.env` configuration:

```python
# .env controls which provider is used
LLM_PROVIDER=openai    # or "anthropic" or "groq"
LLM_MODEL=             # blank = use default for the provider
```

No code changes needed to swap providers. The factory handles API key validation, model selection, and temperature defaults.

## Evaluation

The evaluation suite (`evaluation/evaluate.py`) runs 20 curated queries and measures:

- **Response rate**: Did every query get an answer?
- **Latency**: Wall-clock time per query (p50 and p95)
- **Confidence distribution**: % of HIGH vs MEDIUM vs LOW confidence answers
- **LLM-as-judge scoring**: The LLM grades its own answers on accuracy, relevance, and completeness (1-5 scale)
- **Tool selection accuracy**: Did the planner pick SQL for data queries and semantic search for opinion queries?

Queries span six categories: simple SQL lookups, company comparisons, sector aggregations, trend analysis, sentiment/opinion search, and multi-hop questions requiring multiple tool calls.
