"""
Multi-agent system built on LangGraph.

Four agents collaborate in a pipeline:
  Planner  → figures out what data to fetch and how
  Retriever → calls the right tools (SQL, semantic search)
  Analyst  → synthesizes everything into a coherent answer
  Critic   → checks the answer for accuracy and completeness

The flow is: Planner → Retriever → Analyst → Critic → (done or retry)
"""

import json
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from src.llm import get_llm
from src.tools.agent_tools import ALL_TOOLS, sql_query, semantic_search, get_dataset_info
from src.data_platform.duckdb_store import run_query as duckdb_run_query
from src.retrieval.hybrid import hybrid_search
from src.config import DATASET_DESCRIPTION
from src.logger import logger


# ---------- direct function wrappers (bypass LangChain .invoke) ----------
# LangChain's tool.invoke() has serialization edge cases that can blow up
# when the LLM produces unexpected input shapes. These thin wrappers call
# the same code the @tool-decorated functions call, minus the wrapper.

def _run_sql(query: str) -> str:
    """Run a SQL query, same logic as the sql_query tool."""
    logger.info(f"Direct call: sql_query({query[:100]}...)")
    try:
        df = duckdb_run_query(query)
        if df.empty:
            return "Query returned no results. Check your filters."
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


def _run_search(query: str) -> str:
    """Run hybrid search, same logic as the semantic_search tool."""
    logger.info(f"Direct call: semantic_search({query[:100]}...)")
    try:
        results = hybrid_search(query)
        confidence = results["confidence"]
        if not results["results"]:
            return "No relevant articles found for this query."

        parts = [f"Search confidence: {confidence}",
                 f"Strategy: {results['strategy']}", "---"]
        for i, doc in enumerate(results["results"], 1):
            score_info = ""
            if "rerank_score" in doc:
                score_info = f" (relevance: {doc['rerank_score']:.3f})"
            elif "rrf_score" in doc:
                score_info = f" (RRF: {doc['rrf_score']:.4f})"
            source = doc.get("metadata", {}).get("source", "unknown")
            parts.append(f"[{i}]{score_info} ({source}): {doc['text']}")

        if confidence in ("LOW", "NONE"):
            parts.append("\n⚠ Low confidence — passages may not be directly relevant.")
        return "\n\n".join(parts)
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return f"Search error: {e}"


# -- state that flows through the graph --
class AgentState(TypedDict):
    query: str                    # original user question
    plan: str                     # planner's execution plan
    retrieved_data: str           # raw data from retrieval step
    draft_answer: str             # analyst's synthesized answer
    critique: str                 # critic's feedback
    final_answer: str             # the answer we return to the user
    confidence: str               # HIGH / MEDIUM / LOW / NONE
    sources: list[str]            # source citations
    retry_count: int              # how many times we've looped back
    error: str                    # any error message


# -- system prompts for each agent --

PLANNER_PROMPT = f"""You are a financial data analyst planner. Your job is to look at a user's question
and create a step-by-step plan for answering it.

You have access to:
{DATASET_DESCRIPTION}

For each step in your plan, specify which tool to use:
- sql_query: for numerical lookups, aggregations, comparisons, filtering
- semantic_search: for opinions, trends, explanations, analyst sentiment
- get_dataset_info: if you need to check what data is available

Think about:
1. Does this question need structured data (numbers)? → SQL
2. Does it need unstructured insights (opinions, reasons)? → semantic search
3. Does it need both? → plan multiple steps
4. Is it a multi-hop question? → break it into sub-questions

Output your plan as a JSON array of steps, each with "tool" and "input" fields.
If the question is outside the dataset's coverage, say so in your plan.
Be specific with SQL — use the actual column names from the schema."""

ANALYST_PROMPT = """You are a financial analyst. You receive a user's question along with
data retrieved from a financial database and news articles.

Your job:
1. Combine the structured data (SQL results) and unstructured data (article passages)
   into a clear, well-organized answer
2. Cite specific numbers from the SQL data when available — always include currency symbols
   (e.g. $59.6 billion, not just 59,619 million). Revenue and financial figures are in USD.
3. Reference the article sources when using qualitative information
4. If the retrieved data doesn't fully answer the question, say what's missing
5. Don't make up numbers or facts that aren't in the provided data

Note: All monetary values in the database are in millions USD (column names ending in _mn).
So revenue_mn = 59619.85 means $59,619.85 million = $59.6 billion.

Keep your tone professional but readable. Use bullet points for comparisons.
Include a "Sources" section at the end listing where each piece of info came from."""

CRITIC_PROMPT = """You are a quick quality checker for financial analysis answers.

Only reject an answer if it has one of these SERIOUS problems:
1. WRONG NUMBERS: The answer states a number that directly contradicts the retrieved data.
2. HALLUCINATION: The answer invents facts not present in the retrieved data.
3. WRONG TOPIC: The answer doesn't address the user's question at all.

Minor style issues, missing disclaimers, or incomplete sourcing are NOT reasons to reject.
When the retrieved data is limited, the answer should work with what's available — don't reject just because the data is sparse.

Respond with exactly one of:
- APPROVED (if there are no serious problems listed above)
- REVISE: [one sentence describing the specific factual error to fix]

Default to APPROVED unless there is a clear factual error."""


# -- agent node functions --

def planner_node(state: AgentState) -> dict:
    """Analyze the query and produce an execution plan."""
    logger.info(f"Planner: analyzing query")
    llm = get_llm()

    # first check the schema so the planner knows exact column names
    from src.data_platform.duckdb_store import get_schema_info
    schema_info = DATASET_DESCRIPTION + "\n\n" + get_schema_info()

    messages = [
        SystemMessage(content=PLANNER_PROMPT),
        HumanMessage(content=f"Schema info:\n{schema_info}\n\nUser question: {state['query']}"),
    ]

    response = llm.invoke(messages)
    plan = response.content
    logger.info(f"Planner output: {plan[:200]}...")

    return {"plan": plan}


def retriever_node(state: AgentState) -> dict:
    """Execute the plan by calling the appropriate tools."""
    logger.info("Retriever: executing plan")
    llm = get_llm()

    plan_text = state["plan"]
    all_results = []

    # try to parse the plan as JSON steps
    try:
        # extract JSON array from the plan text (LLM might wrap it in markdown)
        import re
        json_match = re.search(r'\[.*\]', plan_text, re.DOTALL)
        if json_match:
            steps = json.loads(json_match.group())
        else:
            # fallback — ask the LLM to decide what to do
            steps = [
                {"tool": "semantic_search", "input": state["query"]},
                {"tool": "sql_query", "input": f"SELECT * FROM companies LIMIT 5"},
            ]
    except json.JSONDecodeError:
        logger.warning("Could not parse plan as JSON, falling back to direct search")
        steps = [{"tool": "semantic_search", "input": state["query"]}]

    # run each step
    for i, step in enumerate(steps):
        tool_name = step.get("tool", "semantic_search")
        tool_input = step.get("input", state["query"])

        # LLM sometimes returns input as a dict or list instead of a plain string
        # e.g. {"query": "NVIDIA"} or ["NVIDIA", "analyst"] — normalize to string
        if isinstance(tool_input, dict):
            tool_input = (
                tool_input.get("query")
                or tool_input.get("input")
                or tool_input.get("sql")
                or next(iter(tool_input.values()), state["query"])
            )
        if isinstance(tool_input, (list, tuple)):
            tool_input = " ".join(str(x) for x in tool_input)
        if not isinstance(tool_input, str):
            tool_input = str(tool_input)

        logger.info(f"  Step {i+1}: {tool_name}({tool_input[:80]}...)")

        try:
            # call the underlying functions directly instead of going through
            # LangChain's .invoke() — avoids serialization quirks
            if tool_name == "sql_query":
                result = _run_sql(tool_input)
            elif tool_name == "semantic_search":
                result = _run_search(tool_input)
            elif tool_name == "get_dataset_info":
                result = DATASET_DESCRIPTION
            else:
                result = f"Unknown tool: {tool_name}"

            all_results.append(f"--- Step {i+1} ({tool_name}) ---\n{result}")
        except Exception as e:
            logger.error(f"  Step {i+1} failed: {e}")
            all_results.append(f"--- Step {i+1} ({tool_name}) --- ERROR: {e}")

    retrieved = "\n\n".join(all_results)
    logger.info(f"Retriever: collected {len(all_results)} results")

    return {"retrieved_data": retrieved}


def analyst_node(state: AgentState) -> dict:
    """Synthesize the retrieved data into a coherent answer."""
    logger.info("Analyst: synthesizing answer")
    llm = get_llm()

    messages = [
        SystemMessage(content=ANALYST_PROMPT),
        HumanMessage(content=(
            f"User question: {state['query']}\n\n"
            f"Execution plan:\n{state['plan']}\n\n"
            f"Retrieved data:\n{state['retrieved_data']}"
        )),
    ]

    response = llm.invoke(messages)
    draft = response.content
    logger.info(f"Analyst: produced {len(draft)} char answer")

    return {"draft_answer": draft}


def critic_node(state: AgentState) -> dict:
    """Review the draft answer for accuracy and completeness."""
    logger.info("Critic: reviewing answer")
    llm = get_llm()

    messages = [
        SystemMessage(content=CRITIC_PROMPT),
        HumanMessage(content=(
            f"User question: {state['query']}\n\n"
            f"Retrieved data:\n{state['retrieved_data']}\n\n"
            f"Draft answer:\n{state['draft_answer']}"
        )),
    ]

    response = llm.invoke(messages)
    critique = response.content.strip()
    logger.info(f"Critic verdict: {critique[:100]}...")

    return {"critique": critique}


def decide_after_critic(state: AgentState) -> Literal["finalize", "retry"]:
    """Route based on critic's verdict — approve or send back for revision."""
    critique = state.get("critique", "")
    retry_count = state.get("retry_count", 0)

    if "APPROVED" in critique.upper():
        return "finalize"

    # allow max 1 retry — each retry adds ~10 sec of LLM calls
    if retry_count >= 1:
        logger.warning("Max retries reached, finalizing with current answer")
        return "finalize"

    return "retry"


def retry_node(state: AgentState) -> dict:
    """Take critic feedback and revise the answer."""
    logger.info(f"Retry: incorporating feedback (attempt {state.get('retry_count', 0) + 1})")
    llm = get_llm()

    messages = [
        SystemMessage(content=ANALYST_PROMPT),
        HumanMessage(content=(
            f"User question: {state['query']}\n\n"
            f"Retrieved data:\n{state['retrieved_data']}\n\n"
            f"Your previous answer:\n{state['draft_answer']}\n\n"
            f"Reviewer feedback:\n{state['critique']}\n\n"
            f"Please revise your answer based on the feedback above."
        )),
    ]

    response = llm.invoke(messages)
    return {
        "draft_answer": response.content,
        "retry_count": state.get("retry_count", 0) + 1,
    }


def finalize_node(state: AgentState) -> dict:
    """Package the final answer with metadata."""
    answer = state.get("draft_answer", "I wasn't able to generate an answer.")

    # attach a confidence note if retrieval confidence was low
    critique = state.get("critique", "")
    if "APPROVED" in critique.upper():
        confidence = "HIGH"
    elif state.get("retry_count", 0) > 0:
        confidence = "MEDIUM"
    else:
        confidence = "MEDIUM"

    return {
        "final_answer": answer,
        "confidence": confidence,
    }


def build_graph() -> StateGraph:
    """
    Wire up the multi-agent pipeline as a LangGraph state machine.
    """
    graph = StateGraph(AgentState)

    # add all the nodes
    graph.add_node("planner", planner_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("critic", critic_node)
    graph.add_node("retry", retry_node)
    graph.add_node("finalize", finalize_node)

    # wire the edges: planner → retriever → analyst → critic → decide
    graph.set_entry_point("planner")
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "analyst")
    graph.add_edge("analyst", "critic")

    # conditional branching after critic
    graph.add_conditional_edges(
        "critic",
        decide_after_critic,
        {"finalize": "finalize", "retry": "retry"},
    )

    # after retry, go back to critic for another review
    graph.add_edge("retry", "critic")
    graph.add_edge("finalize", END)

    return graph.compile()


# module-level compiled graph — import and use directly
agent_graph = None


def get_agent():
    """Get the compiled agent graph (lazy init)."""
    global agent_graph
    if agent_graph is None:
        agent_graph = build_graph()
    return agent_graph


def run_query(query: str) -> dict:
    """
    Main entry point — send a question, get an answer.
    Returns dict with 'final_answer', 'confidence', and trace info.
    """
    agent = get_agent()

    initial_state = {
        "query": query,
        "plan": "",
        "retrieved_data": "",
        "draft_answer": "",
        "critique": "",
        "final_answer": "",
        "confidence": "",
        "sources": [],
        "retry_count": 0,
        "error": "",
    }

    logger.info(f"Running agent pipeline for: {query[:100]}...")
    result = agent.invoke(initial_state)

    return {
        "answer": result.get("final_answer", "No answer generated."),
        "confidence": result.get("confidence", "UNKNOWN"),
        "plan": result.get("plan", ""),
        "critique": result.get("critique", ""),
        "retries": result.get("retry_count", 0),
    }
