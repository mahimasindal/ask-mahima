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

## Evaluation

`evaluation/` is a read-only, offline retrieval-quality harness — it never writes to the vector store and never touches the request-serving path (`app/api`, `app/main.py`, `app/api/deps.py`); it builds its own `Retriever` against the same persisted Chroma collection, exactly like `scripts/check_retrieval.py` does.

```bash
./venv/bin/python scripts/evaluate_retrieval.py
```

This runs the hand-labeled queries in `evaluation/data/eval_dataset.json` (each with the source file(s) that should be retrieved) through the live retriever and writes a Markdown report to `evaluation/reports/latest.md` (gitignored — regenerated per run) showing, per query: every retrieved chunk with its source, distance, and text preview, plus **Recall@K** — the fraction of labeled-relevant sources found somewhere in the top K results, averaged across the dataset. It's how the two issues in "Iteration 2" below were actually found, not guessed at.

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
| Retrieval evaluation (Recall@K, offline) | `evaluation/`, `scripts/evaluate_retrieval.py` |

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

### Issues reported in iteration 2 (found via the retrieval eval harness, `evaluation/`)

Unlike iteration 1, these weren't found by manually trying questions and noticing a bad answer — they were found by running `scripts/evaluate_retrieval.py` against a 10-query hand-labeled dataset and reading the Recall@K numbers and per-query retrieved-chunk report it produces.

| # | Found via | What it looked like | Root cause | Fix applied |
|---|---|---|---|---|
| 5 | Eval report: `about_me.md` chunk previews | Two indexed chunks, `about_me_0` and `about_me_2`, contained *only* a bare markdown heading (`"## Who I Am"`, 11 chars; `"## Personality & Values"`, 23 chars) — no body text, occupying real retrieval slots with zero information. | `scripts/ingest.py::_split_large_section`'s overflow-splitting loop flushed whatever it had accumulated so far the moment the *next* paragraph alone would exceed `MAX_CHUNK_CHARS`. When a section is short-heading-then-one-huge-paragraph (the actual shape of `about_me.md`'s sections), that flushes the heading alone as its own "piece" before the real content is ever considered. | Added a `min_chars` floor to the flush condition — an accumulated piece can no longer be flushed until it's at least `MIN_CHUNK_CHARS` long, so a short leading fragment keeps absorbing paragraphs instead of shipping alone. Traded off: occasionally produces a piece somewhat larger than `MAX_CHUNK_CHARS`, accepted as the better failure mode. |
| 6 | Eval report: same chunk (`projects_2`, "Ask Mahima (this project!)") ranked rank #1 for 8 of 10 unrelated queries | Recall@1 was 0.20 — the correct source was almost never the top result, because a single generic chunk kept winning regardless of topic. | `projects_2`'s text described the project itself as something that *"answers questions about Mahima"* — phrasing that's nearly a paraphrase of every eval query (`"What does Mahima..."`, `"Tell me about Mahima..."`). Confirmed via direct embedding check: cosine similarity to `"What does Mahima currently do for work?"` was 0.50 for `projects_2` vs. 0.22 for the actually-correct `career.md` chunk — a real content/phrasing collision, not a distance-metric artifact (all stored vectors are unit-normalized, so squared-L2 and cosine rank identically here). | Reworded the chunk in `data/projects.md` to describe the tech stack and behavior without query-mirroring language ("answers questions about") — re-ingested and re-evaluated. |

Verified after both fixes: `about_me.md`'s chunk count dropped from 5 → 3 with no near-empty chunks remaining; the full test suite (`tests/`) still passes; and re-running the eval harness shows Mean Recall@5 rose from **0.700 → 1.000** — the labeled-relevant source now shows up within the top 5 results for every query in the dataset (Recall@1/@3 are unchanged, since the next-highest-ranking distractor is `links.md`'s bare-URL chunks, which is the already-known, already-documented issue in cause #1 below — fixing #6 removed one distractor, it didn't eliminate the underlying embedding-model weakness).

### Iteration 3: chasing Recall@1/@3 — one real fix, one reverted experiment, one open question

| # | Found via | What it looked like | Root cause | Fix applied |
|---|---|---|---|---|
| 7 | Eval report: correlated chunk length against similarity to 4 unrelated test queries across the whole collection | Pearson r = **−0.616** between chunk length and mean similarity-to-arbitrary-query. The 3 shortest chunks in the entire 22-chunk collection were exactly `links.md`'s 3 chunks (63/97/122 chars) — and they were also the top-3 highest-similarity chunks to every unrelated query tested. Cause #1 below had attributed this to "GitHub/LinkedIn are lexically career-adjacent"; the real driver is more general — very short mean-pooled MiniLM embeddings get systematically inflated similarity to almost any query, not just career-adjacent ones. | Raised `MIN_CHUNK_CHARS` in `scripts/ingest.py` from 60 → 150 — the merging mechanism already existed, it was just calibrated below what this failure mode actually needed. Merges `links.md`'s 4 sections and `interests.md`'s 2 sections into one chunk each; no effect on `about_me.md`/`career.md`/`projects.md`/`writing.md`, already comfortably above threshold. | **Confirmed effect:** Mean Recall@3 rose from 0.400 → **1.000**. Fewer, larger low-signal chunks meant fewer of them could crowd multiple top-3 slots at once. |
| 8 | Same eval report, after fix #7 | Recall@1 stayed at 0.200 — the single merged `links.md` chunk still won rank #1 for 8 of 10 queries outright, unchanged by the merge. Tried rewording it with descriptive prose ("Where to find Mahima's *work* and profiles online...") — the same kind of rewording that fixed `projects_2` in iteration 2. | **This made it worse (Recall@1 dropped to 0.100).** The added natural-language framing itself introduced new query-mirroring phrasing ("Mahima's work") — the exact failure mode being fixed, recreated by the fix. Reverted to plain `Label: URL` lines (no narrative prose) — back to 0.200, no regression, but no gain either. | Kept the plain-labeled-lines version (strictly no worse than the original, better structured) and **stopped further data-content experiments** — with two different wordings both landing at "links.md still wins rank #1 regardless of phrasing," this is evidence the ceiling for a content-only fix has been reached, not a wording problem to keep iterating on. |

**Where this leaves Recall@1:** genuinely open. Closing it further needs a retrieval-*algorithm* change, not more markdown edits — e.g. a similarity-score cutoff, a fixed per-document-type down-weighting, or (most robust) a reranking pass after initial retrieval. All three were already flagged as explicit V1 scope in cause #5 below, before this investigation — this investigation is the evidence for why that scope call was correct, not a reason to revisit it casually. In production terms this is lower-urgency than it sounds: `top_k=8` means the correct chunk is still in every response's context (Recall@8 = 1.000), so answer quality isn't visibly affected today — this is a ranking-precision gap, not a coverage gap.

### Underlying causes, generalized (for the "why," not just the "what")

1. **Small local embedding models struggle to distinguish "mentions a topic" from "actually describes it" — and, more precisely, systematically over-rank very short chunks regardless of topic.** `all-MiniLM-L6-v2` consistently ranked `links.md` content *closer* to nearly every query than the genuinely relevant chunk. Originally attributed (iteration 1) to "GitHub"/"LinkedIn" being lexically career-adjacent — plausible, but incomplete: measured directly in iteration 3, chunk length correlates with similarity-to-arbitrary-query at Pearson r = −0.616 across the whole collection, and the effect held up even after merging `links.md` into one longer, non-URL-dominated chunk. Mitigated in V0 by (a) merging very short, low-information sections into neighboring chunks during ingestion (`scripts/ingest.py::_merge_tiny_sections`, threshold raised 60 → 150 in iteration 3) and (b) raising `top_k` so genuinely relevant chunks have more chances to make the cut even when noisy chunks outrank them. These fixes brought Recall@3/5/8 to a perfect 1.000, but **Recall@1 is still stuck at 0.200** — content-level fixes (merging, rewording) have hit their ceiling; a similarity-score cutoff, a per-document-type prior, or (most effectively) a cross-encoder **reranking** step after initial retrieval is what closing it further actually needs. Reranking remains explicitly out of scope for V0.

2. **An anti-hallucination system prompt can be read too literally by the LLM.** The original prompt ("do not invent, infer, or fabricate") caused the model to refuse questions that required *reasoning across* retrieved facts rather than reciting them verbatim — e.g. "Why should you hire Mahima?" was answered "I don't know" even though the "Current Role" chunk (with her actual skills/experience) was present in the retrieved context. The model was treating synthesis as fabrication. Fixed by explicitly distinguishing the two in `SYSTEM_PROMPT` (`app/services/rag.py`): synthesis/summarization *grounded in retrieved context* is allowed and expected; inventing facts not present in the context is not.

3. **Ambiguous questions triggered over-cautious refusals even with directly relevant context retrieved.** "What is your favourite?" (no object specified) was answered "I don't know" even though the retrieved context clearly contained things Mahima has said she loves (Ghibli movies, deep conversations, specific hobbies). Fixed with an explicit prompt instruction: answer with the best-available signal from context on a broad/ambiguous question rather than refusing outright, and note what the answer is based on.

4. **No relevance/similarity threshold.** Retrieval always returns the `top_k` nearest chunks, even if none of them are a good match for the query — there's no cutoff distance below which the system says "nothing relevant found" instead of passing marginal matches to the LLM. This is why the model (not the retrieval short-circuit) is the layer currently doing most of the "is this actually relevant" judgment — a second line of defense worth adding in V1.

5. **No reranking or hybrid search.** Retrieval is a single-pass vector similarity search over an all-MiniLM embedding. Items 1 and 4 above are practical consequences of that choice for a V0 — cheap, fast, no API key required, but with real precision limits that a reranker or a stronger embedding model would substantially reduce. (There is now an evaluation harness — `evaluation/` — that at least makes this measurable instead of anecdotal; see the Evaluation section above.)

6. **A splitter that flushes on overflow, not on adequacy, can silently emit near-empty chunks.** `_split_large_section`'s fallback splitter used to flush whatever it had accumulated the instant the next paragraph would overflow `MAX_CHUNK_CHARS` — without checking whether what it had accumulated so far was actually substantial. A short heading followed by one huge paragraph (a common shape here) meant the heading got flushed alone. Fixed by requiring an accumulated piece to clear `MIN_CHUNK_CHARS` before it's allowed to flush (see iteration 2, #5).

7. **Self-referential/meta content can rank artificially high.** A chunk describing what this RAG system itself does used phrasing ("answers questions about Mahima") that lexically resembles the shape of real user queries, so it out-ranked genuinely relevant content for unrelated topics regardless of the embedding model's actual topic understanding. Worth remembering when writing any knowledge-base content: describing the system in query-like language creates an accidental "hub" chunk (see iteration 2, #6).
