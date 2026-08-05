# CLAUDE.md — tina4-groundwire

Guidance for Claude Code working in this repo. Read this first.

## What groundwire is

A retrieval + verification layer that sits **in front of** an LLM (never inside it — no
attention/KV surgery). The flow is `ingest → chunk → index → retrieve → prompt → read`.
The corpus lives off-GPU (RAM/disk); only the small reader model touches the GPU, and it
only ever sees the top-k retrieved chunks. Two origins fused in one package:

1. **Long-context NIAH harness** — measure retrieval recall + throughput at 64K–3M tokens
   (`harness.py`, `tasks.py`, `baselines.py`, `memory_model.py`).
2. **Retrieval-augmented code assistant** — the real product: index real source (the Tina4
   framework, 6 language targets) and ground a **weak reader (Qwen-7B)** so it writes
   idiomatic, import-correct code. This is the use case that matters most.

## Layout

```
groundwire/
  pipeline.py        # Groundwire: ingest_repo/ingest_code, chunk_text/chunk_code,
                     #   retrieve() (repo-scope, source-over-tests, guide slot),
                     #   _stitch() (neighbor merge + code neighbor-pull), ask()
  memory_systems.py  # backends + tokenizer: fold(), terms(), _light_stem();
                     #   InMemoryBM25, SqliteFTS (default), HashEmbedding,
                     #   DenseEncoderMemory, HybridRetriever, IterativeRetriever,
                     #   MultiQueryRetriever
  rerank.py          # DenseReranker: reorder a lexical pool by dense cosine,
                     #   query embedded once + candidates cached by id (no corpus
                     #   embedding, off-GPU). Wired via Groundwire(rerank="dense").
  encoders.py        # stdlib HTTP embedding clients (Ollama / OpenAI-compat); HashEncoder
  answer.py          # readers: RegexExtract (GPU-free), Qwen, APIGenerator, LLMRewriter
  harness.py         # NIAH generation/planting/scoring/throughput + CLI
  server.py          # OpenAI-compat proxy w/ import verify+correct loop  ── OUT OF SCOPE
  mcp_server.py      # standalone MCP server (agentic path)               ── OUT OF SCOPE
examples/  tina4_corpus.py, tina4_eval.py, tina4_eval.jsonl (30-Q multi-lang eval), batteries
tests/     test_core.py  — stdlib unittest, network-free, deterministic core
```

## Working constraints (current focus)

- **Library only.** Improve the `groundwire` retrieval engine (`pipeline.py`, `memory_systems.py`,
  `encoders.py`, `answer.py`, and NIAH scaffolding). **Do NOT touch `server.py` or
  `mcp_server.py`** — the user has explicitly de-scoped the server and agentic paths.
- **Optimize for the code-retrieval use case** (real source, weak reader), not just NIAH filler.
- **Validate on the weakest reader.** Retrieval-change effects flip sign with reader
  capability (a usage/example slot once helped Llama-8B +2 but hurt Qwen-7B −25). Judge every
  change on Qwen-7B, never on the strong reader. Live Qwen-7B runs happen on the GPU box, not
  in the dev sandbox — so land logic + offline tests here, run the reader benchmark there.
- **Keep it stdlib-first.** Only numpy is a hard dep (for dense). No torch/transformers/
  sentence-transformers in the retrieval path — encoders are thin HTTP clients we own.
- Run tests: `python3 -m unittest discover tests` (seconds, no network). Add a deterministic
  test for every fix.

## Invariants & hard-won lessons (don't regress these)

- **`fold()` symmetry.** `fold()` (lowercase + strip diacritics + de-comma numbers + split
  camelCase) MUST be applied to **both** the document side and the query side, or code
  identifiers become unreachable (`field` can't find `IntegerField`). Both `InMemoryBM25` and
  `SqliteFTS` now fold both sides (`SqliteFTS` indexes `fold(text)` in `body`, keeps the
  original in an `UNINDEXED raw` column). Don't reintroduce a raw-body index.
- **source-over-tests / guide slot / def-boost / dense reranker are RETAINED wins.** The
  usage/example slot was reverted (net-harmful to the weak reader). The dense reranker is now a
  first-class cached `DenseReranker` (`rerank.py`) — retrieval stays lexical (0 calls), the query
  is embedded once/query, each candidate embedded ≤ once ever (cached by id), reader GPU
  untouched; falls back to lexical order if no embed endpoint. Ordering is byte-identical to the
  old inline reorder. Also exposed as `harness.py --rerank dense [--rerank-encoder hash:DIM]`.
- **Neighbor-pull is code-only.** Prose must not stitch neighbors (`test_prose_not_pulled`).
- **`retrieve()` scoping is a no-op on a single-repo index** by design; it only shapes
  multi-repo pools. Don't "fix" that.
- **Persistence is data, not GPU state** — `save()`/`load()` snapshot the index; round-trips
  must reproduce identical retrieval (`test_save_load_roundtrip`).

## Game plan — library hardening (DONE 2026-07-06)

A background audit (5 module reviewers + adversarial per-finding verify) confirmed 11 defects;
all fixed, each with a `tests/test_core.py` regression that fails on the pre-fix code (verified
via `git stash`) and passes after. Suite: 42 tests, network-free, `python3 -m unittest discover
tests`. **These were the fixes — done, don't redo:**

Retrieval quality (biggest lift for code retrieval, all in `SqliteFTS`, the default backend):
1. **Fold the document side** — `body` now stores `fold(text)`, original kept in `raw`
   (`UNINDEXED`); `get()`/`query()` return `raw`, `_state()` snapshots `raw`, `_restore()`
   re-folds. `field` now reaches `IntegerField`; numeric ids (`24,601`) reachable.
2. **Dropped the blanket `t+"*"` wildcard** — matched junk short tokens (`i*`,`a*`,`to*`). Now
   ORs the exact token with its `_light_stem`, mirroring `InMemoryBM25`.
3. **Hybrid** benefits for free (its sqlite arm was a dead/noisy arm; now contributes correctly).

Correctness bugs:
4. `_stitch` merged neighbors across **different untitled documents** (`None == None` guard) —
   guarded with `src is not None` (`pipeline.py`).
5. `MultiQueryRetriever` safety-floor **negative slice at k=1** returned ~4 chunks w/ a dup —
   `max(0, k-len(keep))` + de-dupe + `[:k]`.
6. `SqliteFTS` used a **fixed shared temp path** → two instances clobbered each other. Now a
   unique per-instance file, cleaned up on `close()`.
7. `DenseEncoderMemory` **crashed on an empty first ingest** (`vstack (0,1)` vs `(N,d)`) — early
   return on empty.
8. `HashEncoder`/`HashEmbedding` used salted builtin `hash()` → **non-deterministic across
   processes** (broke dense save/load) — now `zlib.crc32`.
9. `harness.run_one` `approx_tokens` **double-counted overlap** — computed from `sentences`
   pre-chunk; `run_one` also guards empty `depths`.
10. `make_distractors` `continue`-on-collision returned **fewer than `count`** — retry loop.

Hardening (verifier rated lower-impact: only reachable via the out-of-scope server today, but
harmless and correct — kept): `SqliteFTS` `check_same_thread=False` + lock (thread-safe);
`OpenAIEmbeddingsEncoder` sorts `data` by `index` (spec-correct if a server reorders).

**Deliberately deferred:** the `MultiQueryRetriever.phrase_pat` single-token off-topic-probe
finding — the auditor flagged it as likely-intentional off-topic gating; changing retrieval
scoring needs a **Qwen-7B** measurement first (sign-flip risk), so not touched.
