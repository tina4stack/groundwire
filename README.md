# Groundwire

**A retrieval + verification layer that sits in front of an LLM.** Groundwire
keeps your corpus *off the GPU* and puts only the top-k retrieved chunks in front
of the model — so a small, weak reader can answer correctly over a body of source
far larger than its context window. Part of the [tina4stack](https://github.com/tina4stack).

```bash
pip install tina4-groundwire      # zero runtime dependencies — stdlib only
```

- **Off-GPU corpus.** The index lives in RAM/disk (SQLite FTS5 by default). The
  model only ever sees the prompt plus a handful of retrieved chunks, so its GPU
  footprint is *constant* regardless of corpus size.
- **Lossless lexical retrieval.** The default backend is an exact-token inverted
  index (BM25/FTS5), so a rare identifier — a symbol, a clause number,
  `GRM-2026-XT-5590` — comes back byte-for-byte. No embedding compression to smear it away.
- **Zero dependencies.** Core is pure stdlib (same ethos as `tina4-python`). The
  optional dense backend needs no numpy — vector math is stdlib (`veclite.py`) and
  the encoder is a thin HTTP client to an embeddings server *you* already run.
- **Verification, not just retrieval.** Optional verified ranking sinks code that
  doesn't run; the answer-reduce read loop maps a weak reader over batches when the
  retrieved set overflows its window, rather than silently truncating.

## Quickstart

```python
from groundwire import Groundwire

gw = Groundwire(memory="sqlite_fts", k=5)
gw.ingest(open("big_manual.txt").read())            # or a list of docs
gw.ingest_code(open("router.py").read(), title="router.py")

# retrieval only — build your own prompt / call your own model:
for cid, text, score in gw.retrieve("where is the route registered?"):
    print(text)

# or let a reader answer over the retrieved chunks:
from groundwire.answer import make_generator
gw = Groundwire(reader=make_generator("api"), k=6)  # any OpenAI-compatible endpoint
print(gw.ask("what was the Q3 revenue figure?"))
```

`retrieve()` is the whole product; `ask()` is a convenience that runs a reader over
the hits. The corpus can be gigabytes — only `k` chunks ever reach the model.

## How it works

```
ingest → chunk → index (off-GPU, RAM/disk) → retrieve top-k → prompt → read
                        └─ the whole corpus lives here ─┘   └─ model sees ~k chunks ─┘
```

"Unlimited context" here is **retrieval substituting for attention**: the index
holds everything; the model reads only what matched. The ceiling isn't the model's
context length — it's disk.

### Backends (all off-GPU)

| backend       | store          | notes |
|---------------|----------------|-------|
| `sqlite_fts`  | disk (FTS5)    | **default.** BM25 over an inverted index; flat RAM, scales past 3M tokens. |
| `bm25`        | RAM            | pure-Python BM25; fastest queries; memory grows with corpus. |
| `dense`       | RAM (stdlib)   | semantic retrieval via an embeddings server you run; no numpy/torch. |
| `hybrid`      | fused          | Reciprocal-Rank-Fusion of lexical + dense. |
| `iterative`   | lexical        | pointer-walk retrieval for multi-hop chains. |

The trick that makes *lexical* work for code: the `fold()` tokenizer (lowercase +
strip accents + de-comma numbers + **split camelCase**), applied symmetrically to
both document and query — so a query for `field` reaches `IntegerField`, and
`24,601` is findable. Optionally, `Groundwire(rerank="dense")` reorders a lexical
pool by dense cosine (query embedded once, candidates cached by id) — dense
*discrimination* without embedding the whole corpus.

## What the numbers actually say

Measured with `python -m groundwire.harness` (network-free retrieval recall = "needle
chunk in top-k"). The headline is counterintuitive:

**Recall is essentially independent of corpus size — for lexical retrieval.** A
distinctive-identifier query holds 100% from 8K to 3M tokens; only latency moves.

| corpus | chunks | lexical recall | query latency |
|--------|--------|----------------|---------------|
| 8K     | 20     | 100%           | 0.2 ms |
| 512K   | 1,202  | 100%           | 1.0 ms |
| 1M     | 2,348  | 100%           | 1.9 ms |
| 3M     | 7,043  | 100%           | 5.3 ms |

Even **128 lexical distractors per needle** don't dent it — filler that doesn't
share the query's discriminative terms is invisible to an inverted index.

**Where recall is capped, it's the *query type*, not size** (each row is flat
across 8K–3M):

| query type                        | lexical | note |
|-----------------------------------|---------|------|
| distinctive identifier            | 100%    | the regime most code queries live in |
| paraphrase / synonym (NoLiMa)     | ~71%    | zero shared words — see below |
| multi-hop chain (k=5 → k=50)      | 14% → 100% | a top-k *coverage* limit, fixed by wider k or `iterative` |

**Dense retrieval is the opposite of lexical — it *decays* with size.** Measured
with a real `nomic-embed-text` server on the paraphrase (NoLiMa) task:

| corpus | lexical | dense (nomic-embed) |
|--------|---------|---------------------|
| 8K     | 71%     | **100%** |
| 64K    | 71%     | 57% |
| 256K   | 71%     | 43% |
| 1M     | 71%     | 43% |

Dense recovers the paraphrase *at small scale* then falls below lexical as filler
crowds the manifold. This is why Groundwire is **lexical-first** and treats dense as
a **reranker over a small lexical pool**, not a whole-corpus retriever: lexical
bounds a size-invariant candidate set; dense only reorders it, never fighting the
whole haystack. The residual gap — a zero-overlap paraphrase that never enters the
lexical pool — is best closed *lexically* at ingest/query time (fold, camelCase
split, synonym expansion), the only size-invariant lever.

## The answer-reduce read loop

The fixes above all *widen* the retrieved set (multi-hop wants a big `k`, hybrid
widens the pool, code chunks run long) — until it overflows a weak reader's window
and gets silently truncated, dropping the very chunks a wider `k` was meant to
catch. So when the chunks don't fit, `ask()` **maps the question over budget-sized
batches and reduces by verified-answer selection** instead of truncating:

```python
gw = Groundwire(reader=make_generator("api", num_ctx=8192), k=40, verify=my_verifier)
gw.ask("…")   # auto-batches when retrieved chunks exceed the reader's window
```

The reducer prefers a candidate the retrieved text actually *grounds* (present in
the batch it came from); pass a custom `verify(question, answer, chunks) -> score`
to plug in your own signal (e.g. "the code boots"). Validated end-to-end against a
live 7B reader. See `groundwire/mapreduce.py`.

## The GPU edge

Full attention holds a KV cache in VRAM that grows linearly with context.
Retrieval keeps the corpus on RAM/disk and runs the model over the prompt plus a
few chunks — a *constant* footprint. Same Qwen2.5-7B, both ways
(`python -m groundwire.memory_model`):

| context | full-attention VRAM | GPU needed        | retrieval VRAM | ratio |
|---------|---------------------|-------------------|----------------|-------|
| 256K    | 29 GB               | 1× 48GB           | ~15 GB         | 1.9×  |
| 1M      | 69 GB               | 1× 80GB           | ~15 GB         | 4.5×  |
| 3M      | **175 GB**          | **4× 80GB (~256GB)** | **~15 GB**  | **11×** |

At 3M tokens a full-attention 7B needs a ~256GB four-GPU rig; retrieval fits on a
single 20GB card, and usually runs *faster* (no prefill over the whole context).
The honest trade is a capability give-back on tasks needing global reasoning across
the whole context — which is what the multi-hop / NoLiMa rows above quantify.

## Isn't this just RAG?

It shares one mechanic — retrieve text, put it in the prompt — and differs in what
matters. Classic RAG embeds documents into vectors and does approximate-nearest-
neighbour search for *semantically similar* chunks. Groundwire's default is
**lossless exact-token** retrieval, and it's positioned as a **context-window
substitute** (what the model would otherwise hold in KV), not a knowledge-base
feature you write into an app.

| | classic (dense) RAG | Groundwire (default) |
|---|---|---|
| representation | lossy: chunks → vectors | **lossless: exact tokens** in FTS5/BM25 |
| match | fuzzy semantic (cosine) | exact lexical — the token is *there*, verbatim |
| GPU at ingest/query | an embedding model | **none** for lexical; constant footprint |
| scaling | dense recall **decays** with corpus size | lexical recall **flat** to 3M |
| failure mode | embedding misses a paraphrase → silent miss | if the token exists it is found |

Groundwire isn't anti-embeddings — the `dense`/`hybrid` backends *are* embedding
retrieval, for the semantic tail. The point is the default and the headline rest on
exact-token retrieval, because that's what solves the long-context problem off-GPU.

## The NIAH harness

Groundwire ships the long-context recall harness it was validated with:

```bash
groundwire-bench --backend sqlite_fts --context-tokens 64000 256000 1000000
groundwire-bench --task nolima  --backend dense --encoder openai:nomic-embed-text
groundwire-bench --task multihop --backend iterative
python -m groundwire.harness --help
```

It reports, per context size: retrieval recall (needle in top-k), optional answer
recall (a reader extracts the value), and throughput (ingest tok/s, query latency).
Encoder specs: `ollama:MODEL`, `openai:MODEL` (any OpenAI-compatible endpoint incl.
vLLM/Azure), `hash:DIM` (offline). Config via `EMBED_URL` / `EMBED_MODEL`.

## Verified ranking (for code retrieval)

`Groundwire(verified_scorer=...)` scores each source once at ingest — canonical
framework source is trusted; docs/how-tos are boot-gated in an isolated subprocess
against a real framework instance — and sinks examples that *don't run* below ones
that do at query time (the query path only reads the cached score, staying pure
stdlib). Set `GROUNDWIRE_VERIFIED_RANK=0` for a clean A/B baseline.

## Experimental

Not part of the tested core; useful but rougher: `groundwire.proxy` (a transparent
drop-in Ollama proxy that injects retrieved context into every request),
`groundwire.server` (an OpenAI-compatible proxy with an import verify-and-correct
loop), `groundwire.mcp_server` (a Tina4-stack MCP server), and a desktop tray
(`groundwire.tray`, `pip install "tina4-groundwire[app]"`) / native macOS menu-bar
app (`macos/`).

## Development

```bash
python -m unittest discover tests      # 73 tests, network-free, seconds
```

Every fix ships with a deterministic regression test. See [CLAUDE.md](CLAUDE.md)
for the working constraints and [BENCHMARKS.md](BENCHMARKS.md) for the benchmark plan.

## Layout

```
groundwire/
  pipeline.py        # Groundwire: ingest/chunk/retrieve/ask, _stitch, verified ranking
  memory_systems.py  # backends + fold() tokenizer: bm25, sqlite_fts, dense, hybrid, iterative
  rerank.py          # DenseReranker: reorder a lexical pool by dense cosine, cached, off-GPU
  encoders.py        # stdlib HTTP embedding clients (Ollama / OpenAI-compatible)
  answer.py          # readers: regex_extract (GPU-free), qwen, api
  mapreduce.py       # answer-reduce read loop + map-reduce summarizer
  verified.py        # boot-gate verified ranking (rank by whether code runs)
  harness.py         # long-context NIAH recall/throughput harness + CLI
```

## License

MIT © Andre van Zuydam. Part of the [tina4stack](https://github.com/tina4stack).
