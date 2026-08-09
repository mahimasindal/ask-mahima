"""Ingestion pipeline: reads data/*.md, chunks it, embeds it, stores it in ChromaDB.

Chunking strategy: split each file on "## " (level-2) headings. Personal bio
docs are short and already organized into coherent sections (Who I Am, Work
History, one heading per project, ...) — heading-based chunking keeps each
chunk semantically self-contained, which suits this content better than
fixed-size+overlap chunking (which would cut mid-thought in short documents).
As a fallback, any section longer than ~800 characters is further split by
paragraph with a small overlap, so no single chunk becomes too large to embed
meaningfully — this rarely triggers on typical personal-bio content.

Run:
    python scripts/ingest.py            # full rebuild (default)
    python scripts/ingest.py --no-reset # upsert into the existing collection
"""

import argparse
import re
import sys
from pathlib import Path

# Allow running as `python scripts/ingest.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.config import get_settings
from app.services.retrieval import ChromaVectorStore

MAX_CHUNK_CHARS = 800
MIN_CHUNK_CHARS = 60  # sections shorter than this carry too little signal to
# stand alone — e.g. a "## GitHub\nhttps://..." section is almost all
# boilerplate to a sentence embedding model, so it can rank artificially
# close to unrelated queries and crowd out longer, genuinely relevant
# chunks. Merging short sections into a neighbor fixes this generically,
# without special-casing any particular file.
HEADING_PATTERN = re.compile(r"^##\s+(.+)$", re.MULTILINE)


def _merge_tiny_sections(sections: list[str], min_chars: int) -> list[str]:
    """Merges any section shorter than min_chars into the following section
    (or the previous one, if it's the last section in the file)."""
    merged: list[str] = []
    carry = ""
    for section in sections:
        combined = f"{carry}\n\n{section}" if carry else section
        if len(combined) < min_chars:
            carry = combined
            continue
        merged.append(combined)
        carry = ""
    if carry:
        if merged:
            merged[-1] = f"{merged[-1]}\n\n{carry}"
        else:
            merged.append(carry)
    return merged


def load_markdown_files(data_dir: str) -> list[tuple[str, str]]:
    """Returns [(filename, raw_text), ...] for every .md file in data_dir."""
    data_path = Path(data_dir)
    files = sorted(data_path.glob("*.md"))
    return [(f.name, f.read_text(encoding="utf-8")) for f in files]


def _split_large_section(text: str, max_chars: int) -> list[str]:
    """Fallback splitter for oversized sections: split by paragraph, with a
    one-paragraph overlap between consecutive pieces so context isn't lost
    at the boundary."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return [text.strip()] if text.strip() else []

    pieces: list[str] = []
    current: list[str] = []
    current_len = 0
    for para in paragraphs:
        if current_len + len(para) > max_chars and current:
            pieces.append("\n\n".join(current))
            # overlap: keep the last paragraph as the start of the next piece
            current = [current[-1], para]
            current_len = len(current[-2]) + len(para)
        else:
            current.append(para)
            current_len += len(para)
    if current:
        pieces.append("\n\n".join(current))
    return pieces


def chunk_document(filename: str, text: str) -> list[dict]:
    """Splits a markdown file into chunk dicts with id/text/metadata.

    Each chunk dict has: id, text, source, document_type, chunk_index.
    """
    document_type = Path(filename).stem

    # Split on level-2 headings, keeping the heading text with its section.
    matches = list(HEADING_PATTERN.finditer(text))
    sections: list[str] = []
    if not matches:
        # No "## " headings found — treat the whole file as one section.
        sections = [text.strip()]
    else:
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            section_text = text[start:end].strip()
            if section_text:
                sections.append(section_text)

    sections = _merge_tiny_sections(sections, MIN_CHUNK_CHARS)

    chunks: list[dict] = []
    chunk_index = 0
    for section in sections:
        pieces = (
            [section]
            if len(section) <= MAX_CHUNK_CHARS
            else _split_large_section(section, MAX_CHUNK_CHARS)
        )
        for piece in pieces:
            if not piece.strip():
                continue
            chunk_id = f"{document_type}_{chunk_index}"
            chunks.append(
                {
                    "id": chunk_id,
                    "text": piece,
                    "source": filename,
                    "document_type": document_type,
                    "chunk_index": chunk_index,
                }
            )
            chunk_index += 1

    return chunks


def main(reset: bool = True) -> None:
    settings = get_settings()
    store = ChromaVectorStore(settings)

    if reset:
        print(f"Resetting collection '{settings.collection_name}'...")
        store.reset()

    files = load_markdown_files(settings.data_dir)
    if not files:
        print(f"No .md files found in {settings.data_dir}. Nothing to ingest.")
        return

    all_ids: list[str] = []
    all_documents: list[str] = []
    all_metadatas: list[dict] = []

    print(f"{'File':<20} {'Chunks':>7}")
    print("-" * 28)
    for filename, text in files:
        chunks = chunk_document(filename, text)
        print(f"{filename:<20} {len(chunks):>7}")
        for chunk in chunks:
            all_ids.append(chunk["id"])
            all_documents.append(chunk["text"])
            all_metadatas.append(
                {
                    "source": chunk["source"],
                    "document_type": chunk["document_type"],
                    "chunk_id": chunk["id"],
                    "chunk_index": chunk["chunk_index"],
                }
            )

    if not all_ids:
        print("No chunks produced — check your markdown files.")
        return

    store.add(ids=all_ids, documents=all_documents, metadatas=all_metadatas)
    print("-" * 28)
    print(
        f"Ingested {len(all_ids)} chunks from {len(files)} files "
        f"into '{settings.chroma_persist_dir}'."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest data/*.md into the vector store.")
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Upsert into the existing collection instead of rebuilding it from scratch.",
    )
    args = parser.parse_args()
    main(reset=not args.no_reset)
