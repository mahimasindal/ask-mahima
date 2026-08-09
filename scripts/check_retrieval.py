"""Manual eyeball check for retrieval quality — run this after ingest.py and
before writing any LLM code, to confirm the vector search actually returns
sensible chunks for a given query.

Run:
    python scripts/check_retrieval.py "What does Mahima do?"
    python scripts/check_retrieval.py   # uses a default query
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.config import get_settings
from app.services.retrieval import ChromaVectorStore, Retriever

DEFAULT_QUERY = "What does Mahima do?"


def main() -> None:
    query = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUERY
    settings = get_settings()
    store = ChromaVectorStore(settings)
    retriever = Retriever(store, top_k=settings.top_k)

    print(f"Query: {query!r}\n")
    chunks = retriever.retrieve(query)

    if not chunks:
        print("No chunks retrieved. Did you run `python scripts/ingest.py` first?")
        return

    for i, chunk in enumerate(chunks, start=1):
        preview = chunk.text.replace("\n", " ")
        if len(preview) > 120:
            preview = preview[:120] + "..."
        print(f"[{i}] source={chunk.source}  document_type={chunk.document_type}  distance={chunk.distance:.4f}")
        print(f"    {preview}\n")


if __name__ == "__main__":
    main()
