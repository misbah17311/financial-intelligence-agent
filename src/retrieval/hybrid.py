# Hybrid retrieval pipeline:
#   1. Vector search (ChromaDB) for meaning-based matches
#   2. Keyword search (BM25) for exact term matches
#   3. Reciprocal Rank Fusion to merge both lists
#   4. Cross-encoder reranking for final precision

from sentence_transformers import CrossEncoder
from src.data_platform import chroma_store, bm25_store
from src.config import TOP_K_RESULTS, RERANK_TOP_N, SIMILARITY_THRESHOLD
from src.logger import logger


_reranker = None


def _get_reranker():
    # lazy-load the cross-encoder reranker (~80MB model)
    global _reranker
    if _reranker is None:
        logger.info("Loading cross-encoder reranker...")
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


def reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    k: int = 60,
) -> list[dict]:
    # merge multiple ranked lists using RRF
    # docs appearing in multiple lists get boosted (k=60 from original paper)
    fused_scores = {}  # text -> cumulative score
    doc_map = {}       # text -> full doc dict (for returning later)

    for result_list in result_lists:
        for rank, doc in enumerate(result_list):
            text = doc["text"]
            rrf_score = 1.0 / (k + rank + 1)  # +1 because rank is 0-indexed

            if text in fused_scores:
                fused_scores[text] += rrf_score
            else:
                fused_scores[text] = rrf_score
                doc_map[text] = doc

    # sort by fused score, highest first
    sorted_texts = sorted(fused_scores.keys(), key=lambda t: fused_scores[t], reverse=True)

    results = []
    for text in sorted_texts:
        doc = doc_map[text].copy()
        doc["rrf_score"] = round(fused_scores[text], 6)
        results.append(doc)

    return results


def rerank(query: str, documents: list[dict], top_n: int = RERANK_TOP_N) -> list[dict]:
    # re-score using cross-encoder (looks at query+doc pairs together)
    # more accurate than independent embeddings but slower, so only top candidates
    if not documents:
        return []

    reranker = _get_reranker()

    pairs = [(query, doc["text"]) for doc in documents]
    scores = reranker.predict(pairs)

    for doc, score in zip(documents, scores):
        doc["rerank_score"] = round(float(score), 4)

    # sort by rerank score descending
    reranked = sorted(documents, key=lambda d: d["rerank_score"], reverse=True)
    return reranked[:top_n]


def hybrid_search(
    query: str,
    n_results: int = TOP_K_RESULTS,
    top_n: int = RERANK_TOP_N,
    use_reranker: bool = True,
) -> dict:
    # full hybrid retrieval: vector + BM25 → RRF → rerank
    # returns results list, confidence level, and strategy description
    logger.info(f"Hybrid search: '{query[:80]}...'")

    # step 1 — run both searches in parallel (they're independent)
    vector_results = chroma_store.search(query, n_results=n_results)
    bm25_results = bm25_store.search(query, n_results=n_results)

    logger.debug(f"  Vector search: {len(vector_results)} results, top score={vector_results[0]['score'] if vector_results else 'N/A'}")
    logger.debug(f"  BM25 search: {len(bm25_results)} results, top score={bm25_results[0]['score'] if bm25_results else 'N/A'}")

    # step 2 — fuse the two lists
    fused = reciprocal_rank_fusion([vector_results, bm25_results])
    logger.debug(f"  After RRF fusion: {len(fused)} unique chunks")

    # step 3 — rerank the top candidates
    if use_reranker and fused:
        candidates = fused[:n_results]  # rerank top N from fusion
        final = rerank(query, candidates, top_n=top_n)
        strategy = "hybrid (vector + BM25) → RRF fusion → cross-encoder rerank"
    else:
        final = fused[:top_n]
        strategy = "hybrid (vector + BM25) → RRF fusion"

    # step 4 — assess confidence based on the best score
    confidence = _assess_confidence(final, vector_results)

    logger.info(f"  Returning {len(final)} results, confidence={confidence}")

    return {
        "results": final,
        "confidence": confidence,
        "strategy": strategy,
        "num_vector_hits": len(vector_results),
        "num_bm25_hits": len(bm25_results),
    }


def _assess_confidence(final_results: list[dict], vector_results: list[dict]) -> str:
    # determine retrieval confidence from vector similarity scores (0–1 range)
    if not vector_results:
        return "NONE"

    best_vector_score = max(r["score"] for r in vector_results[:3])

    if best_vector_score >= 0.7:
        return "HIGH"
    elif best_vector_score >= SIMILARITY_THRESHOLD:
        return "MEDIUM"
    elif best_vector_score >= 0.2:
        return "LOW"
    else:
        return "NONE"
