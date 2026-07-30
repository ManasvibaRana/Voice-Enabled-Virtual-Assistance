"""RAG tool: ingest ./docs into Chroma, retrieve top-k chunks for a question.

Embeddings are computed locally with sentence-transformers and passed to Chroma
explicitly, so Chroma never has to download its own ONNX embedding model.
"""

import hashlib
import threading
from typing import Any, TypedDict

from app import config

_embedder: Any = None
_client: Any = None
_collection: Any = None
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Lazy resources
# ---------------------------------------------------------------------------


def get_embedder():
    """Load MiniLM once. First call downloads ~90MB."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        print(f"[rag] loading embedding model {config.EMBED_MODEL} ...")
        _embedder = SentenceTransformer(config.EMBED_MODEL)
        print("[rag] embedding model ready")
    return _embedder


def get_collection():
    global _client, _collection
    if _collection is None:
        import chromadb

        config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        _collection = _client.get_or_create_collection(
            name=config.CHROMA_COLLECTION,
            embedding_function=None,  # we always supply embeddings ourselves
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def embed(texts: list[str]) -> list[list[float]]:
    vectors = get_embedder().encode(
        texts, normalize_embeddings=True, show_progress_bar=False
    )
    return [v.tolist() for v in vectors]


# ---------------------------------------------------------------------------
# Chunking + ingestion
# ---------------------------------------------------------------------------


def chunk_text(
    text: str,
    size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> list[str]:
    """Fixed-width character chunks with overlap, snapped to whitespace."""
    text = " ".join(text.split())
    if not text:
        return []
    if len(text) <= size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            # Snap backwards to the last space so we do not cut words in half.
            space = text.rfind(" ", start + int(size * 0.6), end)
            if space != -1:
                end = space
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def ingest_docs(force: bool = False) -> dict[str, Any]:
    """Ingest every .txt/.md file in ./docs. Skipped if already populated."""
    with _lock:
        try:
            collection = get_collection()
        except Exception as exc:  # noqa: BLE001
            print(f"[rag] could not open Chroma: {exc!r}")
            return {"ingested": 0, "skipped": True, "error": str(exc)}

        try:
            existing = collection.count()
        except Exception:  # noqa: BLE001
            existing = 0

        if existing > 0 and not force:
            print(f"[rag] collection already has {existing} chunks, skipping ingest")
            return {"ingested": 0, "skipped": True, "chunks": existing, "error": ""}

        files = sorted(
            p
            for p in config.DOCS_DIR.glob("**/*")
            if p.is_file() and p.suffix.lower() in {".txt", ".md"}
        )
        if not files:
            print(f"[rag] no .txt/.md files found in {config.DOCS_DIR}")
            return {"ingested": 0, "skipped": False, "chunks": 0, "error": ""}

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for path in files:
            try:
                raw = path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                print(f"[rag] could not read {path.name}: {exc}")
                continue
            for index, chunk in enumerate(chunk_text(raw)):
                digest = hashlib.sha1(
                    f"{path.name}:{index}:{chunk}".encode("utf-8")
                ).hexdigest()[:16]
                ids.append(f"{path.stem}-{index}-{digest}")
                documents.append(chunk)
                metadatas.append({"source": path.name, "chunk": index})

        if not documents:
            return {"ingested": 0, "skipped": False, "chunks": 0, "error": ""}

        try:
            vectors = embed(documents)
            collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=vectors,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[rag] ingest failed: {exc!r}")
            return {"ingested": 0, "skipped": False, "error": str(exc)}

        print(f"[rag] ingested {len(documents)} chunks from {len(files)} files")
        return {
            "ingested": len(documents),
            "skipped": False,
            "chunks": len(documents),
            "files": [p.name for p in files],
            "error": "",
        }


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


class Chunk(TypedDict):
    text: str
    source: str
    score: float


class RagResult(TypedDict):
    ok: bool
    chunks: list[Chunk]
    error: str


def retrieve(question: str, top_k: int = config.RAG_TOP_K) -> RagResult:
    """Embed the question and pull the top-k nearest chunks."""
    try:
        collection = get_collection()
        vector = embed([question])[0]
        result = collection.query(
            query_embeddings=[vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[rag] retrieval failed: {exc!r}")
        return {"ok": False, "chunks": [], "error": str(exc)}

    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]

    chunks: list[Chunk] = []
    for i, text in enumerate(documents):
        meta = metadatas[i] if i < len(metadatas) and metadatas[i] else {}
        distance = distances[i] if i < len(distances) else 1.0
        chunks.append(
            {
                "text": text,
                "source": str(meta.get("source", "unknown")),
                # cosine distance -> rough 0..1 similarity, for display only
                "score": round(max(0.0, 1.0 - float(distance)), 3),
            }
        )

    # Drop weak matches. Chroma always returns its top K however poor the fit,
    # and answering from a 0.19-similarity chunk produces confident nonsense.
    kept = [c for c in chunks if c["score"] >= config.RAG_MIN_SCORE]
    if not kept and chunks:
        best = max(c["score"] for c in chunks)
        print(
            f"[rag] no chunk cleared the {config.RAG_MIN_SCORE} threshold "
            f"(best was {best}) - answering 'not covered'"
        )

    return {"ok": bool(kept), "chunks": kept, "error": ""}


def chunks_as_text(chunks: list[Chunk]) -> str:
    if not chunks:
        return "(no relevant passages were retrieved)"
    return "\n\n".join(
        f"[{i + 1}] (source: {c['source']})\n{c['text']}" for i, c in enumerate(chunks)
    )
