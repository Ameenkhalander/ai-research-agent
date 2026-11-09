import pytest

from helpers import hash_embed
from research_agent.documents.processor import Chunk


def test_semantic_retrieval_and_delete(tmp_path):
    pytest.importorskip("chromadb")
    from research_agent.documents.vectorstore import VectorStore

    store = VectorStore(tmp_path / "chroma", embed_fn=hash_embed)
    store.add_chunks(1, "ml.txt", [Chunk(0, 0, "Gradient descent optimises neural network weights."),
                                   Chunk(1, 0, "Backpropagation computes gradients layer by layer.")])
    store.add_chunks(2, "cook.txt", [Chunk(0, 0, "Sourdough bread needs flour water salt and a starter.")])
    assert store.count() == 3
    hits = store.query("how are gradients computed in backpropagation", top_k=2)
    assert hits[0]["document"] == "ml.txt" and hits[0]["chunk_index"] == 1
    only_cook = store.query("flour", top_k=5, filename="cook.txt")
    assert {h["document"] for h in only_cook} == {"cook.txt"}
    store.delete_document(1)
    assert store.count() == 1
    assert store.query("", top_k=3) == []
