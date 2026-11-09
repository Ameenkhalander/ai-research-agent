"""Text extraction (PDF/TXT), chunking and ingestion into SQLite + the vector store."""
from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_EXTENSIONS = {".pdf": "pdf", ".txt": "txt"}


@dataclass
class PageText:
    page: int  # 1-based; 0 means "not paginated" (TXT)
    text: str


@dataclass
class Chunk:
    index: int
    page: int
    text: str


def normalize_whitespace(text: str) -> str:
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text(filename: str, data: bytes) -> list[PageText]:
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{ext}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}")
    if ext == ".txt":
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = data.decode("latin-1")
        return [PageText(0, normalize_whitespace(text))]

    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and reader.decrypt("") == 0:
            raise ValueError("PDF is password-protected")
        pages = [PageText(i, normalize_whitespace(p.extract_text() or "")) for i, p in enumerate(reader.pages, 1)]
    except ValueError:
        raise
    except Exception as exc:  # pypdf raises many exception types for corrupt files
        raise ValueError(f"Could not read PDF: {exc}") from exc
    return pages


_BREAKS = ("\n\n", ". ", "! ", "? ", "\n", "; ", ", ", " ")


def split_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    """Split text into overlapping chunks, preferring paragraph/sentence boundaries."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap > chunk_size // 2:
        raise ValueError("overlap must be between 0 and chunk_size // 2")
    text = normalize_whitespace(text)
    n = len(text)
    chunks: list[str] = []
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            window_start = start + int(chunk_size * 0.6)
            for sep in _BREAKS:
                idx = text.rfind(sep, window_start, end)
                if idx != -1:
                    end = idx + len(sep)
                    break
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + max(1, chunk_size // 10))
    return chunks


def chunk_pages(pages: list[PageText], chunk_size: int = 1000, overlap: int = 150) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in pages:
        for piece in split_text(page.text, chunk_size, overlap):
            chunks.append(Chunk(index=len(chunks), page=page.page, text=piece))
    return chunks


def ingest_document(filename: str, data: bytes, *, db, store, settings) -> dict:
    """Extract, chunk, embed and index a document. Returns the stored document row."""
    filename = Path(filename).name or "upload"
    if len(data) == 0:
        raise ValueError("The uploaded file is empty")
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise ValueError(f"File exceeds the {settings.max_upload_mb} MB upload limit")

    sha = hashlib.sha256(data).hexdigest()
    existing = db.get_document_by_hash(sha)
    if existing:
        return {**existing, "duplicate": True}

    pages = extract_text(filename, data)
    chunks = chunk_pages(pages, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        raise ValueError("No extractable text found (scanned/image-only PDFs need OCR, which is not supported)")

    is_pdf = Path(filename).suffix.lower() == ".pdf"
    doc_id = db.add_document(
        filename=filename, file_type=SUPPORTED_EXTENSIONS[Path(filename).suffix.lower()], sha256=sha,
        size_bytes=len(data), n_pages=len(pages) if is_pdf else 1,
        n_chunks=len(chunks), n_chars=sum(len(p.text) for p in pages),
    )
    try:
        store.add_chunks(doc_id, filename, chunks)
    except Exception:
        db.delete_document(doc_id)
        raise
    return {**db.get_document(doc_id), "duplicate": False}
