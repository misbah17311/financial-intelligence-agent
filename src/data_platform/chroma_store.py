# ChromaDB vector store — embeds financial news chunks
# and provides semantic (meaning-based) search
# uses sentence-transformers locally, no API cost for embeddings

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import pandas as pd
import hashlib
from tqdm import tqdm
from src.config import CHROMA_DIR, PROCESSED_DATA_DIR, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP
from src.logger import logger


_client = None
_collection = None
_embed_model = None

COLLECTION_NAME = "financial_articles"


def _get_embed_model():
    # lazy-load embedding model (~80MB download first time)
    global _embed_model
    if _embed_model is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        _embed_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embed_model


def get_client():
    # persistent ChromaDB client — data survives restarts
    global _client
    if _client is None:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _client


def get_collection():
    # get or create the main articles collection
    global _collection
    if _collection is None:
        client = get_client()
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},  # cosine similarity for search
        )
    return _collection


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    # split text into overlapping chunks, trying to break on sentence boundaries
    if len(text) <= chunk_size:
        return [text]

    # try splitting on sentence boundaries
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # keep overlap by starting the next chunk with the tail of the current one
            words = current_chunk.split()
            overlap_text = " ".join(words[-overlap:]) if len(words) > overlap else current_chunk
            current_chunk = overlap_text + " " + sentence
        else:
            current_chunk += " " + sentence if current_chunk else sentence

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks if chunks else [text]


def build_index(force_rebuild: bool = False):
    # read preprocessed news, chunk it, embed it, store in ChromaDB
    # skips if collection already has data (unless force_rebuild)
    collection = get_collection()

    if collection.count() > 0 and not force_rebuild:
        logger.info(f"ChromaDB already has {collection.count()} chunks. Skipping index build.")
        return collection.count()

    if force_rebuild:
        client = get_client()
        client.delete_collection(COLLECTION_NAME)
        global _collection
        _collection = None
        collection = get_collection()

    parquet_path = PROCESSED_DATA_DIR / "financial_news.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"News data not found at {parquet_path}. Run ingestion first.")

    df = pd.read_parquet(parquet_path)
    model = _get_embed_model()

    all_chunks = []
    all_ids = []
    all_metadatas = []

    logger.info(f"Chunking {len(df)} articles...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Chunking"):
        chunks = chunk_text(row["text"])
        for ci, chunk in enumerate(chunks):
            # deterministic ID so re-runs don't create duplicates
            chunk_id = hashlib.md5(f"{idx}_{ci}_{chunk[:50]}".encode()).hexdigest()
            all_chunks.append(chunk)
            all_ids.append(chunk_id)
            all_metadatas.append({
                "source": str(row.get("source", "unknown")),
                "date": str(row.get("date", "unknown")),
                "article_index": int(idx),
            })

    # embed and insert in batches (ChromaDB has a batch limit)
    batch_size = 2000
    total_inserted = 0

    logger.info(f"Embedding and indexing {len(all_chunks)} chunks...")
    for i in tqdm(range(0, len(all_chunks), batch_size), desc="Indexing"):
        batch_chunks = all_chunks[i:i + batch_size]
        batch_ids = all_ids[i:i + batch_size]
        batch_meta = all_metadatas[i:i + batch_size]

        embeddings = model.encode(batch_chunks, show_progress_bar=False).tolist()

        collection.add(
            documents=batch_chunks,
            embeddings=embeddings,
            ids=batch_ids,
            metadatas=batch_meta,
        )
        total_inserted += len(batch_chunks)

    logger.info(f"ChromaDB: indexed {total_inserted} chunks total")
    return total_inserted


def search(query: str, n_results: int = 10) -> list[dict]:
    # semantic search — finds chunks closest in meaning to the query
    collection = get_collection()
    model = _get_embed_model()

    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        include=["documents", "distances", "metadatas"],
    )

    output = []
    for i in range(len(results["documents"][0])):
        # ChromaDB returns cosine distance; convert to similarity (1 - distance)
        distance = results["distances"][0][i]
        similarity = 1 - distance

        output.append({
            "text": results["documents"][0][i],
            "score": round(similarity, 4),
            "metadata": results["metadatas"][0][i],
        })

    return output


if __name__ == "__main__":
    count = build_index()
    print(f"\nIndex built with {count} chunks")
    print("\nTest search: 'Apple revenue growth'")
    for r in search("Apple revenue growth", n_results=3):
        print(f"  [{r['score']:.3f}] {r['text'][:100]}...")
