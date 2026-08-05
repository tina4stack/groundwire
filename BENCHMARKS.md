# Benchmarking plan — proving tina4-groundwire is actually good

To claim we built something worthwhile we have to (1) test on the *hard*,
recognized long-context suites — not just our own easy needle — and (2) compare
against real baselines on a shared scorecard. This document defines both, up
front, so every result is measured the same way from day one.

## 1. Why our current test is not enough (be honest)

Our built-in needle is a **single, lexically distinctive** fact. That is the
easiest possible case: frontier long-context models already score ~99.7% on
single-needle NIAH, and so do we. Single-needle NIAH no longer differentiates
anything. The moment you add multiple needles, paraphrasing, or reasoning,
scores fall hard — e.g. realistic multi-fact recall averages ~60% even for
strong models, and RULER runs 10–25 points below single-needle NIAH. So our
easy 100% is table stakes, not evidence.

## 2. Standard benchmarks to adopt

| Benchmark      | What it stresses                                             | Why it matters here |
|----------------|-------------------------------------------------------------|---------------------|
| **RULER**      | 13 synthetic tasks: multi-key / multi-value / multi-hop needles, variable tracking, aggregation | The credible successor to NIAH; separates retrieval from reasoning |
| **NoLiMa**     | Needles with **minimal lexical overlap** with the question (latent associative reasoning) at up to 128K | Kills pure lexical (BM25) shortcuts — the real test of whether we need dense/hybrid |
| **InfiniteBench** | Real + synthetic tasks averaging ~200K tokens (novel QA, code, math) | Long, realistic, beyond synthetic recall |
| **HELMET**     | Real-world mix: summarization, long-doc QA, many-shot ICL, RAG, citation, re-ranking | Application-level, not just recall |
| **LongBench / LOFT** | Multi-task long-context suites | Breadth / cross-checking |

Priority order for us: **RULER** (multi-needle recall, directly comparable to our
harness) → **NoLiMa** (the honesty check: does our lexical edge survive?) →
**InfiniteBench/HELMET** (realistic tasks).

## 3. Baselines to compare against

Group A — **full-context attention** (the thing we claim to beat on cost/memory):
- Open, self-hostable: Qwen2.5-7B-Instruct @128K (YaRN), Llama-3.1-8B @128K.
- Frontier references (published numbers, not run locally): Gemini 1.5/3 Pro
  (~1M ctx), GPT-4-class 128K, Claude 200K.

Group B — **linear / SSM** (constant memory, lossy recall):
- RWKV-7 (7.2B / 13.3B), Mamba-2. Expected to trail on exact recall — we want to
  *quantify* that gap on the same needles.

Group C — **retrieval baselines** (our own family — beat these too):
- Naive RAG: fixed-size chunking + a single dense retriever (FAISS + BGE).
- Lexical only: BM25 / Elasticsearch.
- Frameworks: LlamaIndex / LangChain default pipelines.

Group D — **ours**: hybrid (BM25 + dense) + RRF + reranker, feeding a Qwen2.5-7B
reader over retrieved chunks.

## 4. The scorecard (axes that decide "good")

Every system is measured on the *same* haystacks and questions:

| Axis | Metric |
|------|--------|
| Accuracy | retrieval recall@k **and** end-to-end answer accuracy |
| Robustness | accuracy under multi-needle / NoLiMa low-lexical / distractors |
| GPU memory | peak VRAM (GB) at target context |
| Max context | largest context reaching the recall bar |
| Ingest | tokens/sec to index/prefill |
| Query latency | ms/query and queries/sec |
| Cost | estimated $/query (or FLOPs/query) |
| Footprint | RAM + disk for the memory store |

## 5. Reference points already in the literature (targets to beat / match)

- Single-needle NIAH: Gemini 1.5 Pro ~**99.7%**; realistic multi-fact ~**60%**.
- RULER sits **10–25 pts below** single-needle NIAH; at 256K only frontier models
  stay above ~80%.
- A 1M-token full-attention request runs **~30–60× slower** and **~1000–1250×**
  the per-query cost of a RAG pipeline; RAG queries average ~1 s.
- Above ~200–400K, RAG over a focused chunk-set typically **outperforms** naive
  long-context for non-frontier / open models — which is precisely our target
  regime (open 7B, 1–3M tokens).

## 6. The claim we are trying to substantiate

> On open, self-hostable models at **1–3M tokens**, a hybrid-retrieval memory
> matches or beats full-context attention on recall while using a **fraction of
> the GPU memory** (only a 7B model on the card, memory store on RAM/disk) and
> **RAG-level latency and cost** — and beats SSM/linear models on exact recall.

If RULER + NoLiMa numbers hold that up against Groups A–C, we have something.
If NoLiMa exposes our lexical retriever, that tells us exactly where to invest
(dense + reranker) — which is itself a documented finding.

## 7. Protocol notes

- Fixed seeds, identical chunking, identical `k`, report mean ± std over ≥3 seeds.
- Separate **retrieval recall** (ceiling) from **answer accuracy** (reader-limited).
- Log VRAM via `nvidia-smi`/`torch.cuda.max_memory_allocated`; cost from token
  counts × current model pricing.
- Publish the exact haystack generator + seeds so numbers are reproducible.

## Sources

- HELMET — https://huggingface.co/blog/helmet
- HELM Long Context — https://crfm.stanford.edu/2025/09/29/helm-long-context.html
- Long Context vs RAG (evaluation) — https://arxiv.org/pdf/2501.01880
- U-NIAH (unified RAG + LLM NIAH eval) — https://dl.acm.org/doi/10.1145/3786609
- Long-context vs RAG production framework — https://tianpan.co/blog/2026-04-09-long-context-vs-rag-production-decision-framework
- Needle-in-haystack 2026 overview — https://www.digitalapplied.com/blog/long-context-retrieval-needle-in-haystack-2026

---

# Measured results — 2026-07-05 (RTX A4500 20GB, aatos server)

Real numbers on real hardware, not projections. Retrieval runs off-GPU (CPU/RAM);
only the reader sits on the card. Full method + harnesses in `examples/`.

## 1. Reader shootout — which model on the 20GB GPU? (`reader_shootout.py`)

Retrieval held FIXED (same 2.8M-token 4-book corpus, same retrieved chunks per
question); the reader is the only variable. Answer recall on identical context:

| Reader | Params / ctx | Answer recall | Latency | q/s @ conc-8 |
|--------|-------------|---------------|---------|--------------|
| Qwen2.5-3B-Instruct-AWQ (GPU 0, 8K ctx) | 3B / 8K | **9/10** | **491 ms** | **49** |
| tina4 coder (GPU 1, Qwen-AWQ, 200K ctx) | ~3B / 200K | 8/10 | 1106 ms | 30 |
| Qwen2.5-1.5B-Instruct (Metal, local) | 1.5B | 9/10 | — | — |

The small clean AWQ **wins on every axis**. A 1.5B instruct matched it on accuracy
— answer recall saturates well below 3B because retrieval already did the hard
part; the reader only reads ~2K tokens. **Instruct is required** (the grounding
contract is instruction-following); a base/general completion model won't follow it.

## 2. Throughput — the 20GB headroom (Qwen2.5-3B-AWQ, GPU 0)

| Concurrency | 1 | 8 | 32 | 64 |
|-------------|---|---|----|----|
| queries/sec | 2.6 | 61 | 101 | **121** |
| tokens/sec  | 18 | 430 | 706 | **850** |

121 q/s from a 3B on one 20GB card. Prompts stay ~2K tokens (retrieved context),
so the card is decode-bound on short answers. The AWQ **weights are ~2.5 GB**; the
rest of VRAM is optional KV-pool for concurrency. The production coder is slower
mainly because `max-model-len 200000` reserves KV space it rarely uses — the
full-attention tax retrieval removes.

## 3. Dense vs lexical — the honesty check (does retrieval need embeddings?)

Live `nomic-embed-text` (vLLM, with `search_query:`/`search_document:` prefixes).

| Test | bm25 (lexical) | dense | hybrid | bm25+rerank |
|------|----------------|-------|--------|-------------|
| Synthetic NoLiMa (zero-overlap needles), recall@5 | **71%** | 57% | 71% | — |
| Real books, paraphrase-hard, **N=30**, recall@5 | 67% | 63% | **73%** | 67% |
| Ingest cost (2.8M tokens) | **1.0 s** | 38.5 s | 39.0 s | 1.0 s |
| GPU / service dependency | **none** | embed model | embed model | embed (query-time) |

At **N=30** the verdict sharpens: **lexical alone (67%) ties dense alone (63%)**
at 38x lower ingest cost, so lexical is the right DEFAULT. But **hybrid (73%)
wins by ~6 points** — fusing lexical + dense catches needles neither finds
alone, so dense earns its cost when FUSED, not as a standalone retriever.
Reranking helps *ranking* (code src-top1 17→23, sec. 1) but not pool *recall*
(books unchanged at 67%): it reorders candidates, it can't recover a passage
lexical never retrieved. (`examples/paraphrase_honesty.py`)

Real "paraphrase-hard" questions keep distinctive anchors ("whaling voyage",
"prophet…three days") that BM25 — with query-side folding, plural stemming, and
LLM query expansion — catches. The narrow slice where dense still earns its cost:
genuinely novel private content with true zero-overlap paraphrase, where neither
lexical anchors nor the rewriter's parametric knowledge help. Now measured, not
assumed. (nomic needs its task prefixes; without them dense is worse still.)

## 4. Multi-hop — where single-shot retrieval genuinely fails

RULER-style variable tracking: a chain `aL = a(L-1) = ... = a0 = value` is
scattered through the haystack; the question names only `aL`. One retrieval by
`aL` finds the last *link*, not the value.

| hops | context | single-shot (bm25) | iterative (chain-walk) |
|------|---------|--------------------|------------------------|
| 2 | 64K | 14% | **100%** |
| 4 | 64K | 14% | **100%** |
| 4 | 500K | 14% | **100%** |
| 8 | 64K | 29% | **100%** |
| 12 | 64K | — | **100%** |

This is the honest counter-case: lexical single-shot tops out at ~14-29% on
multi-hop, and no amount of query expansion fixes it — the value shares nothing
with the question. `IterativeRetriever` follows the reference chain hop by hop
and hits 100%, holding even at 500K tokens. (A limit surfaced and was fixed:
the default `max_hops=8` collapsed to 14% at 8-hop chains — off-by-one at the
walk depth; raised to 24.) Takeaway: match the retriever to the task —
single-shot for direct recall, chain-walking for multi-hop.

## 5. Verdict

Off-GPU lexical retrieval + a small AWQ instruct reader on a 20GB card beats a
big-context coder on accuracy, latency, AND throughput — and correctly-configured
dense embeddings add no recall on real single-shot workloads (hybrid adds ~6pts;
multi-hop needs the iterative retriever). The context window is replaced, not the
model; the corpus never touches the GPU.

---

# Code-retrieval benchmarks + reproduction guide (for external testers)

The sections above cover the long-context / NIAH family. This part covers the
**code-retrieval** family — grounding a *weak* reader to write **import-correct**
framework code — and is written so **anyone can reproduce the suite** on a laptop,
without our hardware. If you only skim one thing, read the **integrity rules** at
the end: they are how you avoid reporting a benchmark artifact as a result.

## Reproduce it in seconds (no dataset, no model, no GPU)

Every harness ships a network-free self-test, and the core has 49 unit tests:

```bash
python3 examples/tina4_bakeoff.py --self-test   # the verify-loop mechanism
python3 examples/beam_eval.py     --self-test   # BEAM retrieval-recall harness
python3 -m unittest discover tests              # 49 deterministic core tests
```

These prove the harness *logic* with no external dependencies — run them first.

## Setup for the full runs

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install numpy datasets              # datasets only for the BEAM benchmark
# an OpenAI-compatible reader/embedder — local Ollama works:
ollama serve; ollama pull qwen2.5:7b; ollama pull nomic-embed-text
export LLM_URL=http://localhost:11434  LLM_MODEL=qwen2.5:7b
export EMBED_URL=http://localhost:11434/v1/embeddings
```

## A. Tina4 retrieval-QA — recall / grounding / answer accuracy

`examples/tina4_eval.py` over `examples/tina4_eval.jsonl` (30 Qs, 6 language
targets; gold tokens + anchors verified verbatim against the framework source).
Reports retrieval recall (gold in top-k), grounding (top-1 in the right scope),
answer accuracy (exact gold-token match — a floor), and adjudicated accuracy
(LLM judge on misses via `--judge`). Build the corpus first:

```bash
python3 examples/tina4_corpus.py --repos-root ~/src --save results/tina4_corpus.idx
python3 examples/tina4_eval.py --judge      # or --no-reader for retrieval-only
```

## B. Tina4 verify-loop bake-off — the differentiator (objective metric)

`examples/tina4_bakeoff.py` isolates the one thing commodity code-RAG tools
(Continue, Aider, LlamaIndex, Cursor, Cody) do **not** do: an import/symbol
**verify-and-correct loop** on a weak reader. Three conditions, same reader, same
retry budget, vary only the grounding layer — `floor` (no context) / `vanilla`
(context + generic retry = commodity RAG) / `groundwire` (context + the verifier's
*specific* feedback). Metric is **objective**: `check_imports(code, package) == []`
(does the code reference only REAL symbols?), AST-based ⇒ **Python target only**;
other targets fall back to gold-token accuracy.

```bash
python3 examples/tina4_bakeoff.py --repos-root ~/src --budget 3
```

**Claim it tests:** verification beats commodity RAG at a fixed weak reader —
holds iff `groundwire > vanilla > floor` on import-clean. Fairness is wired in: all
conditions capped at `--budget` reader calls (groundwire stops early when clean, so
it can only use *less* compute, never more).

## C. BEAM cross-system retrieval — groundwire vs the memory crowd (with caveats)

`examples/beam_eval.py` scores any engine's retrieval on the `Mohammadta/BEAM`
dataset by whether a question's rubric facts appear in the top-k — the **same
metric applied to every engine** (`--engine groundwire|mnemosyne`), two metric modes
(`--metric overlap|semantic`).

```bash
pip install datasets
python3 examples/beam_eval.py --engine groundwire    --metric semantic --scales 100K
python3 examples/beam_eval.py --engine mnemosyne --metric semantic --scales 100K
```

**Read the result honestly (see integrity rules):** the token-`overlap` metric
flatters lexical retrieval; the `semantic` metric flatters dense. They *bracket*
the truth — **do not report either alone as a winner**. On BEAM retrieval, groundwire
and mnemosyne are roughly at parity; **we do not claim a recall win.** BEAM tests
episodic *memory*, not context-grounding — it's a sanity check, not our scoreboard.

## Integrity rules for these benchmarks (learned the hard way)

1. **Control the variable you claim.** To compare *retrieval*, hold reader,
   chunking, prompt, and retry budget constant; vary only the grounding layer.
2. **Objective > judged.** Import-correctness is pass/fail and reproducible; an
   LLM judge is not comparable across judges. Prefer it where available.
3. **One metric can flatter one side.** Token-overlap → lexical; embedding-cosine
   → dense. Report both; treat them as a bracket.
4. **Watch chunk size.** A retrieval "win" was once entirely a chunk-granularity
   difference (groundwire's larger chunks vs mnemosyne's ~500-char memories).
   Normalize chunk spec, or declare it as a variable.
5. **Equal compute.** A verify-and-correct loop makes extra reader calls — give
   every condition the same retry budget, or you're measuring compute.
6. **Don't cross categories.** A context/grounding layer is not an episodic-memory
   system; scoring it on a memory benchmark (BEAM) is ill-posed.
7. **Validate reader-dependent numbers on the WEAKEST reader** (Qwen-7B), not a
   strong one — retrieval-change effects can flip sign with reader capability.

## Publish the command with the number

Every result lands in `results/`. When you publish a number, publish the
**command, reader model, chunk spec, and retry budget** alongside it — that's the
difference between a benchmark and a screenshot.
