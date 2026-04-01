# BM25 keyword search over news chunks
# complements ChromaDB's semantic search by catching exact keyword matches

import pickle
import pandas as pd
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from src.config import BM25_INDEX_PATH, PROCESSED_DATA_DIR
from src.data_platform.chroma_store import chunk_text
from src.logger import logger


_bm25 = None
_corpus_chunks = None  # keeps the original text for each chunk


def _tokenize(text: str) -> list[str]:
    # simple whitespace + lowercase tokenizer, good enough for BM25
    return text.lower().split()


def build_index(force_rebuild: bool = False):
    # build BM25 from the same news articles as ChromaDB, saves to disk
    global _bm25, _corpus_chunks

    if BM25_INDEX_PATH.exists() and not force_rebuild:
        logger.info("BM25 index already exists on disk. Loading...")
        load_index()
        return len(_corpus_chunks)

    parquet_path = PROCESSED_DATA_DIR / "financial_news.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"News data not found at {parquet_path}. Run ingestion first.")

    df = pd.read_parquet(parquet_path)

    # chunk the same way ChromaDB does, so results are comparable
    logger.info(f"Building BM25 index from {len(df)} articles...")
    all_chunks = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Chunking for BM25"):
        chunks = chunk_text(row["text"])
        all_chunks.extend(chunks)

    # tokenize each chunk
    tokenized = [_tokenize(c) for c in tqdm(all_chunks, desc="Tokenizing")]

    _bm25 = BM25Okapi(tokenized)
    _corpus_chunks = all_chunks

    # save to disk
    BM25_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"bm25": _bm25, "chunks": _corpus_chunks}, f)

    logger.info(f"BM25: indexed {len(all_chunks)} chunks, saved to {BM25_INDEX_PATH}")
    return len(all_chunks)


def load_index():
    # load previously built BM25 index from pickle
    global _bm25, _corpus_chunks

    if not BM25_INDEX_PATH.exists():
        raise FileNotFoundError("BM25 index not found. Run build_index() first.")

    with open(BM25_INDEX_PATH, "rb") as f:
        data = pickle.load(f)

    _bm25 = data["bm25"]
    _corpus_chunks = data["chunks"]
    logger.info(f"BM25: loaded index with {len(_corpus_chunks)} chunks")


def search(query: str, n_results: int = 10) -> list[dict]:
    # keyword search — ranks chunks by TF-IDF scoring
    global _bm25, _corpus_chunks

    if _bm25 is None:
        load_index()

    tokens = _tokenize(query)
    scores = _bm25.get_scores(tokens)

    # get top N indices
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_results]

    results = []
    for idx in top_indices:
        results.append({
            "text": _corpus_chunks[idx],
            "score": round(float(scores[idx]), 4),
            "metadata": {"source": "bm25"},
        })

    return results


if __name__ == "__main__":
    count = build_index()
    print(f"\nBM25 index: {count} chunks")
    print("\nTest search: 'Apple revenue growth'")
    for r in search("Apple revenue growth", n_results=3):
        print(f"  [{r['score']:.4f}] {r['text'][:100]}...")
