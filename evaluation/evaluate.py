# Evaluation runner — tests the agent against curated queries
# and grades answers using LLM-as-judge (1-5 scale)
# tracks: response rate, confidence dist, latency, tool accuracy

import json
import time
import sys
import os
from pathlib import Path
from statistics import mean, median

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.graph import run_query
from src.llm import get_llm
from src.logger import logger
from langchain_core.messages import HumanMessage, SystemMessage

EVAL_DIR = Path(__file__).parent
RESULTS_DIR = EVAL_DIR / "results"


def load_test_queries() -> list[dict]:
    with open(EVAL_DIR / "test_queries.json") as f:
        return json.load(f)


def grade_answer(query: str, answer: str, query_type: str) -> dict:
    # use the LLM itself to grade answer quality on 1-5 scale
    llm = get_llm()
    grading_prompt = f"""Grade this answer on a scale of 1-5 for each criterion.
    
Question: {query}
Question type: {query_type}
Answer: {answer}

Criteria:
1. RELEVANCE (1-5): Does the answer address the actual question?
2. ACCURACY (1-5): Are the facts/numbers plausible and consistent?
3. COMPLETENESS (1-5): Does it cover all parts of the question?
4. CLARITY (1-5): Is it well-organized and easy to understand?

For out_of_scope questions, give 5/5 if the system correctly identifies it's outside the dataset.

Respond in JSON format only:
{{"relevance": X, "accuracy": X, "completeness": X, "clarity": X, "reasoning": "brief explanation"}}"""

    try:
        response = llm.invoke([HumanMessage(content=grading_prompt)])
        # parse the JSON from the response
        import re
        json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except Exception as e:
        logger.error(f"Grading failed: {e}")

    return {"relevance": 0, "accuracy": 0, "completeness": 0, "clarity": 0, "reasoning": "grading failed"}


def run_evaluation():
    # run the full evaluation suite across test queries
    queries = load_test_queries()
    results = []
    latencies = []

    print(f"\nRunning evaluation on {len(queries)} test queries...\n")
    print("-" * 80)

    for i, test in enumerate(queries):
        query = test["query"]
        query_type = test["type"]

        print(f"[{i+1}/{len(queries)}] {query[:70]}...")

        start = time.time()
        try:
            result = run_query(query)
            elapsed = time.time() - start
            latencies.append(elapsed)

            answer = result["answer"]
            confidence = result["confidence"]

            # check if the agent used the right tool by looking at the plan
            plan = result.get("plan", "")
            used_sql = "sql_query" in plan.lower()
            used_search = "semantic_search" in plan.lower()
            expected = test["expected_tool"]

            tool_correct = False
            if expected == "sql_query" and used_sql:
                tool_correct = True
            elif expected == "semantic_search" and used_search:
                tool_correct = True
            elif expected == "both" and (used_sql or used_search):
                tool_correct = True
            elif expected == "none":
                tool_correct = True  # any response is fine for out-of-scope

            # grade the answer
            grades = grade_answer(query, answer, query_type)

            result_entry = {
                "id": test["id"],
                "query": query,
                "type": query_type,
                "answer": answer[:500],  # truncate for storage
                "confidence": confidence,
                "tool_correct": tool_correct,
                "latency_seconds": round(elapsed, 2),
                "grades": grades,
                "retries": result.get("retries", 0),
            }
            results.append(result_entry)

            avg_grade = mean([
                grades.get("relevance", 0),
                grades.get("accuracy", 0),
                grades.get("completeness", 0),
                grades.get("clarity", 0),
            ])
            print(f"  → Confidence: {confidence} | Avg grade: {avg_grade:.1f}/5 | "
                  f"Tool correct: {tool_correct} | Time: {elapsed:.1f}s")

        except Exception as e:
            elapsed = time.time() - start
            print(f"  → ERROR: {e} ({elapsed:.1f}s)")
            results.append({
                "id": test["id"],
                "query": query,
                "type": query_type,
                "answer": f"ERROR: {e}",
                "confidence": "ERROR",
                "tool_correct": False,
                "latency_seconds": round(elapsed, 2),
                "grades": {"relevance": 0, "accuracy": 0, "completeness": 0, "clarity": 0},
                "retries": 0,
            })

    # compute aggregate metrics
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    total = len(results)
    errors = sum(1 for r in results if r["confidence"] == "ERROR")
    response_rate = (total - errors) / total * 100

    confidence_counts = {}
    for r in results:
        c = r["confidence"]
        confidence_counts[c] = confidence_counts.get(c, 0) + 1

    tool_accuracy = sum(1 for r in results if r["tool_correct"]) / total * 100

    all_grades = [r["grades"] for r in results if r["confidence"] != "ERROR"]
    avg_relevance = mean([g["relevance"] for g in all_grades]) if all_grades else 0
    avg_accuracy = mean([g["accuracy"] for g in all_grades]) if all_grades else 0
    avg_completeness = mean([g["completeness"] for g in all_grades]) if all_grades else 0
    avg_clarity = mean([g["clarity"] for g in all_grades]) if all_grades else 0

    sorted_latencies = sorted(latencies)
    p50 = sorted_latencies[len(sorted_latencies) // 2] if sorted_latencies else 0
    p95_idx = int(len(sorted_latencies) * 0.95)
    p95 = sorted_latencies[min(p95_idx, len(sorted_latencies) - 1)] if sorted_latencies else 0

    summary = {
        "total_queries": total,
        "response_rate": f"{response_rate:.1f}%",
        "confidence_distribution": confidence_counts,
        "tool_routing_accuracy": f"{tool_accuracy:.1f}%",
        "answer_quality": {
            "relevance": round(avg_relevance, 2),
            "accuracy": round(avg_accuracy, 2),
            "completeness": round(avg_completeness, 2),
            "clarity": round(avg_clarity, 2),
            "overall": round(mean([avg_relevance, avg_accuracy, avg_completeness, avg_clarity]), 2),
        },
        "latency": {
            "median_seconds": round(p50, 2),
            "p95_seconds": round(p95, 2),
        },
    }

    print(json.dumps(summary, indent=2))

    # save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    with open(RESULTS_DIR / f"eval_{timestamp}.json", "w") as f:
        json.dump({"summary": summary, "details": results}, f, indent=2)

    with open(RESULTS_DIR / f"summary_{timestamp}.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to evaluation/results/eval_{timestamp}.json")
    return summary


if __name__ == "__main__":
    run_evaluation()
