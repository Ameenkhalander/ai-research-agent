import pytest

from helpers import FakeStore
from research_agent.config import Settings
from research_agent.db import Database
from research_agent.documents.processor import (PageText, chunk_pages, extract_text, ingest_document, split_text)


def test_split_text_respects_size_and_covers_text():
    text = " ".join(f"Sentence number {i} talks about topic {i % 7}." for i in range(400))
    chunks = split_text(text, chunk_size=500, overlap=100)
    assert len(chunks) > 5
    assert all(0 < len(c) <= 500 for c in chunks)
    assert chunks[0].startswith("Sentence number 0")
    assert chunks[-1].endswith("topic 0.")  # 399 % 7 == 0: the final sentence is included


def test_split_text_overlaps_between_neighbours():
    text = " ".join(f"word{i}" for i in range(600))
    chunks = split_text(text, chunk_size=300, overlap=100)
    shared = set(chunks[0].split()[-10:]) & set(chunks[1].split())
    assert shared, "consecutive chunks should share some text"


def test_split_text_terminates_and_handles_edges():
    assert split_text("") == []
    assert split_text("   \n  ") == []
    assert split_text("short text", 1000, 100) == ["short text"]
    no_spaces = "x" * 5000
    chunks = split_text(no_spaces, chunk_size=1000, overlap=500)
    assert 4 <= len(chunks) <= 12


def test_split_text_validates_arguments():
    with pytest.raises(ValueError):
        split_text("abc", chunk_size=0)
    with pytest.raises(ValueError):
        split_text("abc", chunk_size=100, overlap=80)


def test_chunk_pages_keeps_page_numbers():
    chunks = chunk_pages([PageText(1, "First page text. " * 5), PageText(2, "Second page text. " * 5)], 1000, 100)
    assert [c.page for c in chunks] == [1, 2]
    assert [c.index for c in chunks] == [0, 1]


def test_extract_txt_and_rejects_unknown_type():
    pages = extract_text("notes.txt", "Héllo\r\n\r\n\r\nworld".encode("utf-8"))
    assert pages[0].text == "Héllo\n\nworld"
    with pytest.raises(ValueError):
        extract_text("data.docx", b"x")


def test_extract_pdf_roundtrip(tmp_path):
    pytest.importorskip("reportlab")
    pytest.importorskip("pypdf")
    from reportlab.pdfgen import canvas

    path = tmp_path / "t.pdf"
    c = canvas.Canvas(str(path))
    c.drawString(72, 750, "Page one mentions photosynthesis.")
    c.showPage()
    c.drawString(72, 750, "Page two mentions mitochondria.")
    c.save()
    pages = extract_text("t.pdf", path.read_bytes())
    assert len(pages) == 2
    assert "photosynthesis" in pages[0].text and "mitochondria" in pages[1].text


def test_corrupt_pdf_raises_value_error():
    pytest.importorskip("pypdf")
    with pytest.raises(ValueError):
        extract_text("bad.pdf", b"not a pdf at all")


def test_ingest_document_stores_and_dedupes(tmp_path):
    settings = Settings(data_dir=tmp_path, chunk_size=200, chunk_overlap=20)
    db, store = Database(settings.sqlite_path), FakeStore()
    data = ("Retrieval augmented generation grounds answers in documents. " * 20).encode()
    doc = ingest_document("rag.txt", data, db=db, store=store, settings=settings)
    assert doc["n_chunks"] == store.count() > 1 and doc["duplicate"] is False
    again = ingest_document("copy.txt", data, db=db, store=store, settings=settings)
    assert again["duplicate"] is True and again["id"] == doc["id"]
    assert len(db.list_documents()) == 1


def test_ingest_rejects_empty_and_oversized(tmp_path):
    settings = Settings(data_dir=tmp_path, max_upload_mb=1)
    db, store = Database(settings.sqlite_path), FakeStore()
    with pytest.raises(ValueError):
        ingest_document("a.txt", b"", db=db, store=store, settings=settings)
    with pytest.raises(ValueError):
        ingest_document("a.txt", b"x" * (1024 * 1024 + 1), db=db, store=store, settings=settings)
    with pytest.raises(ValueError):
        ingest_document("blank.txt", b"   \n  ", db=db, store=store, settings=settings)
