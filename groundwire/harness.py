#!/usr/bin/env python3
"""
Long-context recall + throughput harness  (needle-in-a-haystack, up to 3M tokens)
=================================================================================

Goal: objectively measure whether a memory system can hit ~100% recall on a
1-3M token context -- and how fast it ingests and answers -- WITHOUT burning
GPU memory. The retrieval layer here runs on CPU/RAM/disk; a real deployment
only puts the small Qwen model on the GPU and pulls the few relevant chunks in.

It measures two things per context size:
  * accuracy  : retrieval recall  (needle chunk present in top-k)   -> the
                ceiling on any RAG/kNN answer recall
  * throughput: ingest tokens/sec, index build time, avg query latency

Backends live in memory_systems.py (bm25 / sqlite_fts / hash_dense). Swap in
real embeddings, FAISS, or an exact-KV kNN store without touching this file.

Quick start
-----------
    python3 niah_harness.py                       # 64K -> 1M sweep, sqlite_fts
    python3 niah_harness.py --context-tokens 64000 256000 1000000 2000000 3000000
    python3 niah_harness.py --backend bm25 --k 5
    python3 niah_harness.py --out results.json

"Tokens" are approximated as words * 1.3 (set --tokens-per-word, or install
tiktoken and pass --tiktoken for exact counts).
"""

from __future__ import annotations
import argparse
import json
import random
import time

from .memory_systems import make_backend, terms  # noqa: F401
from .answer import make_generator
from .tasks import build_task, make_distractors

TOKENS_PER_WORD = 1.3

FILLER = (
    "system model memory context window token vector index query retrieval "
    "attention layer cache stream buffer signal gradient tensor matrix kernel "
    "process thread latency bandwidth storage disk region cluster sequence "
    "engine module pipeline schedule worker packet channel session record "
    "value weight bias sample corpus document passage sentence pattern anchor "
    "north river valley harbor meadow forest lantern copper amber willow ridge "
    "quiet gentle rapid distant hollow narrow golden silver crimson violet "
    "morning evening autumn winter summer harvest orchard cottage meadow trail "
    "the a of to and in that with for as on by from into over under between "
    "carry hold reach gather scatter follow return open close begin settle "
    "market ledger cargo compass beacon tunnel bridge canyon plateau delta"
).split()

CODEWORDS = (
    "zephyr quartz mirth vellum cinder gossamer thistle onyx marrow fathom "
    "brindle lattice pyre halcyon sable tundra cobalt ember plinth nimbus"
).split()

TOPICS = (
    "vault archive relay beacon ledger conduit cipher lattice manifold registry"
).split()


def make_sentence(rng: random.Random) -> str:
    n = rng.randint(10, 20)
    s = " ".join(rng.choice(FILLER) for _ in range(n))
    return s[0].upper() + s[1:] + "."


def build_filler(target_tokens, rng, tokens_per_word):
    target_words = int(target_tokens / tokens_per_word)
    sentences, words = [], 0
    while words < target_words:
        s = make_sentence(rng)
        sentences.append(s)
        words += s.count(" ") + 1
    return sentences


def plant(sentences, supports, rng):
    """Insert (depth, sentence) supports at their target depths. Distractors are
    already in `sentences`; supports go in deepest-first so indices stay valid."""
    n = len(sentences)
    for depth, sent in sorted(supports, key=lambda x: -x[0]):
        idx = min(n, max(0, int(depth * n)))
        sentences.insert(idx, sent)
    return sentences


def chunk_by_sentences(sentences, max_words=350, overlap=1):
    """Pack whole sentences into chunks up to max_words. Sentences are never
    split, so a needle sentence always lands intact inside exactly one chunk."""
    chunks, cur, cur_words, cid = [], [], 0, 0
    for s in sentences:
        w = s.count(" ") + 1
        if cur and cur_words + w > max_words:
            chunks.append((cid, " ".join(cur)))
            cid += 1
            cur = cur[-overlap:] if overlap else []
            cur_words = sum(x.count(" ") + 1 for x in cur)
        cur.append(s)
        cur_words += w
    if cur:
        chunks.append((cid, " ".join(cur)))
    return chunks


def run_one(backend_name, target_tokens, depths, k, seed, tokens_per_word,
            chunk_words, overlap, task="single", hops=3, distractors=0,
            generator=None, backend_kwargs=None, reranker=None):
    rng = random.Random(seed)
    items = build_task(task, depths, rng, hops=hops)
    if not items:
        raise ValueError("no needles to plant: 'depths' is empty")

    t0 = time.perf_counter()
    sentences = build_filler(target_tokens, rng, tokens_per_word)
    if distractors:
        for it in items:
            for ds in make_distractors(it, distractors, rng):
                sentences.insert(rng.randint(0, len(sentences)), ds)
    supports = [s for it in items for s in it["supports"]]
    plant(sentences, supports, rng)
    chunks = chunk_by_sentences(sentences, chunk_words, overlap)
    gen_s = time.perf_counter() - t0

    # True unique token count, measured from the planted sentences BEFORE
    # chunking. Summing over chunks double-counts every overlap-carried sentence
    # (it appears in two chunks), inflating both the reported context size and
    # ingest_tok_per_s by the overlap fraction (~5% at overlap=1, ~10% at 2).
    approx_tokens = int(sum(s.count(" ") + 1 for s in sentences) * tokens_per_word)

    backend = make_backend(backend_name, **(backend_kwargs or {}))
    try:
        t0 = time.perf_counter()
        backend.ingest(chunks)
        ingest_s = time.perf_counter() - t0

        hits, ans_hits, per_depth, latencies = 0, 0, [], []
        for it in items:
            t0 = time.perf_counter()
            if reranker is not None:
                # lexical retrieves a wider pool; the reranker reorders it and
                # we cut to k. Latency includes the (cached) embedding cost.
                pool = backend.query(it["question"], k=max(k * 3, k))
                res = reranker(it["question"], pool)[:k]
            else:
                res = backend.query(it["question"], k=k)
            latencies.append(time.perf_counter() - t0)
            found = any(it["gold"] in text for _, text, _ in res)
            hits += int(found)
            answered = None
            if generator is not None:
                answered = it["gold"] in generator.generate(it["question"], res)
                ans_hits += int(answered)
            per_depth.append({"depth": it["depth"], "found": found,
                              "answered": answered})
    finally:
        # Always release backend resources (e.g. the sqlite_fts DB handle) so a
        # multi-context sweep can delete/reopen the store on the next iteration.
        backend.close()

    recall = hits / len(items)
    avg_q = sum(latencies) / len(latencies)
    return {
        "backend": backend_name,
        "task": task,
        "target_tokens": target_tokens,
        "approx_tokens": approx_tokens,
        "chunks": len(chunks),
        "num_needles": len(items),
        "distractors_per_needle": distractors,
        "recall": recall,
        "answer_recall": (ans_hits / len(items)) if generator is not None else None,
        "per_depth": per_depth,
        "gen_s": round(gen_s, 3),
        "ingest_s": round(ingest_s, 3),
        "ingest_tok_per_s": int(approx_tokens / ingest_s) if ingest_s else None,
        "avg_query_ms": round(avg_q * 1000, 2),
        "queries_per_s": int(1 / avg_q) if avg_q else None,
    }


def print_report(rows):
    print()
    print(f"{'backend':<11} {'ctx(tok)':>10} {'chunks':>7} {'recall':>7} "
          f"{'answer':>7} {'ingest':>8} {'tok/s':>9} {'q_ms':>7} {'q/s':>6}")
    print("-" * 82)
    for r in rows:
        ans = "  -  " if r["answer_recall"] is None else f"{r['answer_recall']*100:>5.1f}%"
        print(f"{r['backend']:<11} {r['approx_tokens']:>10,} {r['chunks']:>7,} "
              f"{r['recall']*100:>6.1f}% {ans:>7} {r['ingest_s']:>7.2f}s "
              f"{(r['ingest_tok_per_s'] or 0):>9,} {r['avg_query_ms']:>6.1f} "
              f"{(r['queries_per_s'] or 0):>6,}")
    print()
    # depth-resolved recall for the largest context (classic NIAH view)
    big = max(rows, key=lambda r: r["approx_tokens"])
    print(f"Depth-resolved recall @ ~{big['approx_tokens']:,} tokens "
          f"({big['backend']}):")
    for d in big["per_depth"]:
        bar = "OK " if d["found"] else "MISS"
        print(f"  depth {int(d['depth']*100):>3}% : {bar}")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--context-tokens", type=int, nargs="+",
                    default=[64000, 128000, 256000, 512000, 1000000])
    ap.add_argument("--backend", default="sqlite_fts",
                    choices=["bm25", "sqlite_fts", "hash_dense", "hybrid",
                             "dense", "iterative"])
    ap.add_argument("--encoder", default="ollama:nomic-embed-text",
                    help="dense encoder spec: ollama:MODEL | openai:MODEL | "
                         "hash:DIM (see groundwire/encoders.py). Uses EMBED_URL env.")
    ap.add_argument("--rerank", default="none", choices=["none", "dense"],
                    help="reorder a lexical top-(k*3) pool by dense similarity "
                         "before the top-k cut (query-time; embeddings cached "
                         "by chunk id). Retrieval stays lexical.")
    ap.add_argument("--rerank-encoder", default=None,
                    help="encoder spec for --rerank (defaults to --encoder). "
                         "Use hash:DIM to exercise rerank fully offline.")
    ap.add_argument("--task", default="single",
                    choices=["single", "multihop", "nolima"],
                    help="single (easy) | multihop (RULER) | nolima (low-lexical)")
    ap.add_argument("--hops", type=int, default=3,
                    help="chain length for the multihop task")
    ap.add_argument("--distractors", type=int, default=0,
                    help="confuser sentences per needle (hard mode)")
    ap.add_argument("--generator", default="none",
                    choices=["none", "regex_extract", "qwen", "api"],
                    help="score end-to-end answer recall with this reader "
                         "(api = your AWQ vLLM /v1/chat/completions)")
    ap.add_argument("--llm-url", default=None,
                    help="LLM endpoint base for --generator api (or LLM_URL env)")
    ap.add_argument("--llm-model", default=None,
                    help="LLM model name for --generator api (or LLM_MODEL env)")
    ap.add_argument("--depths", type=float, nargs="+",
                    default=[0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--chunk-words", type=int, default=350)
    ap.add_argument("--overlap", type=int, default=1)
    ap.add_argument("--tokens-per-word", type=float, default=TOKENS_PER_WORD)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None, help="write full results JSON here")
    args = ap.parse_args()

    gen_kwargs = {}
    if args.generator == "api":
        gen_kwargs = {"url": args.llm_url, "model": args.llm_model}
    generator = make_generator(args.generator, **gen_kwargs)
    backend_kwargs = {"encoder": args.encoder} if args.backend == "dense" else {}

    reranker = None
    if args.rerank == "dense":
        from .encoders import make_encoder
        from .rerank import DenseReranker
        reranker = DenseReranker(make_encoder(args.rerank_encoder or args.encoder))

    rows = []
    for ctx in args.context_tokens:
        r = run_one(args.backend, ctx, args.depths, args.k, args.seed,
                    args.tokens_per_word, args.chunk_words, args.overlap,
                    task=args.task, hops=args.hops,
                    distractors=args.distractors, generator=generator,
                    backend_kwargs=backend_kwargs, reranker=reranker)
        rows.append(r)
        ans = "" if r["answer_recall"] is None else \
            f"  answer={r['answer_recall']*100:.1f}%"
        print(f"[done] {args.backend}  ~{r['approx_tokens']:,} tok  "
              f"recall={r['recall']*100:.1f}%{ans}  "
              f"ingest={r['ingest_s']:.2f}s ({r['ingest_tok_per_s']:,} tok/s)  "
              f"query={r['avg_query_ms']:.1f}ms")

    print_report(rows)

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"config": vars(args), "results": rows}, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
