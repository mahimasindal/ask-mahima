# Ask Mahima — V0

A small personal RAG (retrieval-augmented generation) chatbot that answers questions about Mahima using only her personal knowledge base (`data/*.md`), with an explicit "I don't know" fallback instead of fabricating answers.

## Stack

- **Backend:** FastAPI + Pydantic + Uvicorn
- **Embeddings:** local, via ChromaDB's built-in ONNX `all-MiniLM-L6-v2` (no `torch`, no API key needed — chosen over `sentence-transformers` to stay well under Render free tier's 512MB RAM limit)
- **Vector DB:** ChromaDB, local persistent storage (`data/chroma_db/`)
- **LLM:** OpenRouter (`openai` SDK pointed at OpenRouter's OpenAI-compatible API), default model `openai/gpt-4o-mini`
- **Frontend:** a single static HTML/JS page, no framework

## Setup

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
cp .env.example .env   # then fill in OPENROUTER_API_KEY
```

## Run

```bash
# 1. Ingest the knowledge base (re-run any time data/*.md changes)
./venv/bin/python scripts/ingest.py

# 2. (Optional) sanity-check retrieval before starting the server
./venv/bin/python scripts/check_retrieval.py "What does Mahima do?"

# 3. Start the server
./venv/bin/uvicorn app.main:app --reload
```

## Deploy (Render)

`render.yaml` defines a free-tier web service as a Blueprint:

1. Push this repo to GitHub (private — `data/*.md` has real personal content).
2. In Render: **New +** → **Blueprint** → connect the `ask-mahima` repo. Render reads `render.yaml` automatically.
3. When prompted, set `OPENROUTER_API_KEY` (it's marked `sync: false` in `render.yaml`, so Render asks for it in the dashboard rather than reading it from the repo — the key is never committed).
4. Deploy. The build command installs dependencies and re-runs `scripts/ingest.py` on every deploy, so the vector index is always rebuilt fresh from whatever's currently in `data/*.md` — no dependency on persistent disk.

To update the live knowledge base later: edit `data/*.md`, commit, push — the next Render deploy re-ingests automatically.

Then open `http://localhost:8000`, or call the API directly:

```bash
curl -X POST localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What AI projects has Mahima built?"}'
```

## Architecture

```
User → POST /chat → Retriever.retrieve(query)
                        └─ ChromaVectorStore.query()  (embeds query, searches locally)
                     → RagService.answer()
                        ├─ format retrieved chunks into context
                        ├─ LLMClient.generate(system_prompt, context + question)
                        └─ ChatResponse(answer, sources)
```

| Layer | File |
|---|---|
| Settings | `app/services/config.py` |
| Embedding function (shared by ingest + retrieval) | `app/services/embeddings.py` |
| Vector store abstraction | `app/services/retrieval.py` (`VectorStore` → `ChromaVectorStore`, `Retriever`) |
| Ingestion | `scripts/ingest.py` |
| LLM client | `app/services/llm.py` |
| RAG orchestration (prompt + composition) | `app/services/rag.py` |
| API | `app/main.py`, `app/api/*.py` |

`rag.py` is the only module that imports both `Retriever` and `LLMClient` — swapping the vector DB (e.g. to FAISS) or the LLM provider is a one-file change, not a rewrite.

## Known Limitations (V0)

These were found by testing the app against real questions after ingesting real content, not hypothetically — each is a concrete, reproduced failure and its fix.

### Issues reported in iteration 1 (exact questions, exact fixes)

| # | Question asked | What happened (reported) | Root cause | Fix applied |
|---|---|---|---|---|
| 1 | "Why should you hire mahima?" | Answered "I don't know" — wrong, since her skills/experience were retrieved | The `career.md` "Current Role" chunk **was** in the retrieved context, but the system prompt's "do not invent, infer, or fabricate" instruction was read too literally by the LLM — it treated *building an argument from stated facts* as fabrication rather than legitimate synthesis. | Rewrote `SYSTEM_PROMPT` in `app/services/rag.py` to explicitly say synthesis/reasoning *grounded in the retrieved context* is allowed and expected; inventing facts not present in the context is what's actually prohibited. |
| 2 | "Should I hire mahima for a backend role in netherlands?" | Answered "I don't know" | Same root cause as #1 — a reasoning/argument question, refused for the same over-literal reason. | Same fix as #1 (the prompt rewrite fixed both, since they're the same failure mode). |
| 3 | "Can you tell me your favourite?" | Said the info "isn't provided," but the KB clearly states things she loves (Ghibli movies, deep conversations, specific hobbies/topics) | The question is genuinely ambiguous (favourite *what*?) — the model treated that ambiguity itself as grounds to refuse, even though relevant "things she loves" content was sitting right there in the retrieved context. | Added an explicit clause to `SYSTEM_PROMPT`: on a broad/ambiguous question, answer with whatever the context most directly speaks to and note what the answer is based on, instead of refusing outright. |
| 4 | "What is Mahima good at?" | Said "I don't know" — should have surfaced hobbies and work experience | This one was a genuine **retrieval** failure, not a prompt failure: `career.md`'s "Skills"/"Current Role" chunks weren't even in the top-`k` results. Short, low-information `links.md` chunks (just a heading + a bare URL, e.g. `## GitHub\nhttps://github.com/...`) ranked *closer* to the query than the real skills content, because "GitHub"/"LinkedIn" are lexically career-adjacent to the embedding model even though the URLs carry no actual skill information. | Two-part fix: (a) added `_merge_tiny_sections` to `scripts/ingest.py` to merge very short sections into a neighboring chunk during ingestion, so single-line link chunks stop occupying so many top-`k` slots; (b) raised `top_k` from 4 → 8 in `app/services/config.py` so real content has more chances to make the cut even when noisy chunks still outrank it. |

Verified after the fixes: all four questions above now return grounded, sourced answers, the retrieval test suite still passes, and a sanity check ("What is Mahima's home address and phone number?") still correctly declines — confirming the fixes made the system *less falsely conservative* without reopening the hallucination risk they were designed to guard against.

### Underlying causes, generalized (for the "why," not just the "what")

1. **Small local embedding models struggle to distinguish "mentions a topic" from "actually describes it."** `all-MiniLM-L6-v2` ranked `links.md` chunks (just a heading + a bare URL, e.g. `## GitHub\nhttps://github.com/...`) as *closer* to career-related queries like "What is Mahima good at?" than the actual `career.md` "Skills"/"Current Role" content — because "GitHub" and "LinkedIn" are lexically career-adjacent even though the URLs themselves carry no information about skills. Mitigated in V0 by (a) merging very short, low-information sections into neighboring chunks during ingestion (`scripts/ingest.py::_merge_tiny_sections`) and (b) raising `top_k` so genuinely relevant chunks have more chances to make the cut even when noisy chunks outrank them. Neither fix eliminates the underlying issue — a better embedding model, a similarity-score cutoff, or (most effectively) a cross-encoder **reranking** step after initial retrieval would address this properly. Reranking was explicitly out of scope for V0.

2. **An anti-hallucination system prompt can be read too literally by the LLM.** The original prompt ("do not invent, infer, or fabricate") caused the model to refuse questions that required *reasoning across* retrieved facts rather than reciting them verbatim — e.g. "Why should you hire Mahima?" was answered "I don't know" even though the "Current Role" chunk (with her actual skills/experience) was present in the retrieved context. The model was treating synthesis as fabrication. Fixed by explicitly distinguishing the two in `SYSTEM_PROMPT` (`app/services/rag.py`): synthesis/summarization *grounded in retrieved context* is allowed and expected; inventing facts not present in the context is not.

3. **Ambiguous questions triggered over-cautious refusals even with directly relevant context retrieved.** "What is your favourite?" (no object specified) was answered "I don't know" even though the retrieved context clearly contained things Mahima has said she loves (Ghibli movies, deep conversations, specific hobbies). Fixed with an explicit prompt instruction: answer with the best-available signal from context on a broad/ambiguous question rather than refusing outright, and note what the answer is based on.

4. **No relevance/similarity threshold.** Retrieval always returns the `top_k` nearest chunks, even if none of them are a good match for the query — there's no cutoff distance below which the system says "nothing relevant found" instead of passing marginal matches to the LLM. This is why the model (not the retrieval short-circuit) is the layer currently doing most of the "is this actually relevant" judgment — a second line of defense worth adding in V1.

5. **No reranking, hybrid search, or evaluation harness.** Retrieval is a single-pass vector similarity search over an all-MiniLM embedding. All four items above are the practical consequences of that choice for a V0 — cheap, fast, no API key required, but with real precision limits that a reranker or a stronger embedding model would substantially reduce.
