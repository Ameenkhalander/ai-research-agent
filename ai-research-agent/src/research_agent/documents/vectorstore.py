"""ChromaDB-backed semantic index for uploaded documents."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

EmbedFn = Callable[[list[str]], Sequence[Sequence[float]]]
COLLECTION = "documents"


class VectorStore:
    """Thin wrapper around a Chroma collection.

    Embeddings are computed here (Chroma's default ONNX all-MiniLM-L6-v2 model unless ``embed_fn`` is
    supplied) so the same code path works for local persistent storage and a remote Chroma server.
    """

    def __init__(self, persist_dir: str | Path | None = None, *, host: str | None = None, port: int = 8000,
                 embed_fn: EmbedFn | None = None, collection_name: str = COLLECTION):
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        if host:
            self.client = chromadb.HttpClient(host=host, port=port)
        else:
            if persist_dir is None:
                raise ValueError("persist_dir is required for a local Chroma store")
            Path(persist_dir).mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=str(persist_dir),
                                                    settings=ChromaSettings(anonymized_telemetry=False))
        if embed_fn is None:
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
            embed_fn = DefaultEmbeddingFunction()
        self._embed = embed_fn
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}, embedding_function=None)

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[float(x) for x in vec] for vec in self._embed(texts)]

    def add_chunks(self, doc_id: int, filename: str, chunks: list, batch_size: int = 64) -> int:
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [c.text for c in batch]
            self.collection.add(
                ids=[f"{doc_id}:{c.index}" for c in batch],
                documents=texts,
                embeddings=self._embed_texts(texts),
                metadatas=[{"document_id": doc_id, "filename": filename, "chunk_index": c.index, "page": c.page}
                           for c in batch],
            )
        return len(chunks)

    def delete_document(self, doc_id: int) -> None:
        self.collection.delete(where={"document_id": doc_id})

    def count(self) -> int:
        return int(self.collection.count())

    def query(self, text: str, top_k: int = 5, filename: str | None = None) -> list[dict]:
        total = self.count()
        if total == 0 or not text.strip():
            return []
        kwargs = {"query_embeddings": self._embed_texts([text]), "n_results": max(1, min(top_k, total)),
                  "include": ["documents", "metadatas", "distances"]}
        if filename:
            kwargs["where"] = {"filename": filename}
        res = self.collection.query(**kwargs)
        out = []
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            out.append({
                "document": meta.get("filename"), "chunk_index": meta.get("chunk_index"),
                "page": meta.get("page", 0), "score": round(1.0 - float(dist), 4), "text": doc,
            })
        return out


def build_store(settings) -> VectorStore:
    return VectorStore(settings.chroma_dir, host=settings.chroma_host, port=settings.chroma_port)
