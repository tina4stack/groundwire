#!/usr/bin/env python3
"""
Reader shootout — which model should sit on the GPU?

Retrieval is held FIXED (same corpus, same retrieved chunks per question), so
the only variable is the reader model. Each reader gets identical context and
is scored on:
  * answer recall  — does a gold substring appear in the answer?
  * latency        — ms/query (sequential)
  * throughput     — queries/sec and output tokens/sec under concurrency,
                     the number that decides how many a 20GB GPU can serve

Readers are any OpenAI-compatible /v1/chat/completions endpoint (llama.cpp,
vLLM/AWQ, Ollama, a remote gateway). Define them with --reader name=URL[:model].

    python3 examples/reader_shootout.py \
        --reader llama3.2-3b=http://127.0.0.1:8100 \
        --reader gemma3-1b=http://127.0.0.1:8101 \
        --reader tina4-mamba=http://your-gpu-host:11440:tina4 \
        --concurrency 8
"""
import argparse
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from groundwire.pipeline import Groundwire
from groundwire.answer import make_generator
from groundwire.memory_systems import fold

CACHE = os.path.expanduser("~/.cache/groundwire/books")
BOOKS = {"War and Peace": "war_and_peace.txt", "KJV Bible": "kjv_bible.txt",
         "Moby-Dick": "moby_dick.txt", "Les Misérables": "les_miserables.txt"}

# question, list of acceptable gold substrings (answer recall = any match)
QUESTIONS = [
    ("What does Anna Pavlovna say Genoa and Lucca have become at the start of War and Peace?",
     ["estate", "Buonaparte", "Bonaparte"]),
    ("What is Pierre Bezukhov's problem as an illegitimate son regarding inheritance?",
     ["inherit", "illegitimate", "bastard", "legitimate"]),
    ("What did God create in the beginning according to Genesis?",
     ["heaven and the earth", "heaven and earth"]),
    ("What are the opening words of the narrator of Moby-Dick?",
     ["call me ishmael"]),
    ("What is the name of Captain Ahab's whaling ship in Moby-Dick?",
     ["pequod"]),
    ("What did the bishop give Jean Valjean after he stole the silver in Les Misérables?",
     ["candlestick"]),
    ("Who was swallowed by a great fish and stayed in its belly three days?",
     ["jonah"]),
    ("Which captain lost his leg to a white whale?",
     ["ahab"]),
    ("Who relentlessly pursues the convict Jean Valjean (prisoner 24601)?",
     ["javert"]),
    ("In Les Misérables, who is the young girl Fantine leaves in the care of the Thénardiers?",
     ["cosette"]),
]

SYSTEM = ("Answer the question using only the provided context passages from "
          "classic novels. Be concise — one or two sentences. If the answer "
          "is a name or phrase, state it directly.")


def load_book(fname):
    text = open(os.path.join(CACHE, fname), encoding="utf-8",
                errors="ignore").read()
    s = text.find("*** START OF")
    e = text.rfind("*** END OF")
    if s != -1:
        text = text[text.find("\n", s) + 1:]
    if e != -1:
        text = text[:text.rfind("*** END OF")]
    return text


def post(url, model, messages, max_tokens, timeout=120):
    body = json.dumps({"model": model, "messages": messages,
                       "max_tokens": max_tokens, "temperature": 0}).encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    usage = d.get("usage", {})
    return (d["choices"][0]["message"]["content"],
            usage.get("completion_tokens", 0))


def parse_reader(spec):
    """name=URL[:model]. The model is an optional trailing :segment, told
    apart from a :port by not being all-digits."""
    name, _, rest = spec.partition("=")
    scheme, sep, hostpart = rest.partition("://")
    segs = hostpart.split(":")
    model = ""
    if len(segs) > 1 and not segs[-1].isdigit():  # trailing non-numeric = model
        model = segs.pop()
    base = (scheme + sep + ":".join(segs)).rstrip("/")
    url = base if base.endswith("completions") else base + "/v1/chat/completions"
    return name, url, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", action="append", required=True,
                    help="name=URL[:model] (repeatable)")
    ap.add_argument("--rewriter", default=None,
                    help="URL:model used to expand queries (fixed for all "
                         "readers); omit for pure-lexical fixed retrieval")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=100)
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()

    readers = [parse_reader(r) for r in args.reader]

    # ---- fixed retrieval: build index, retrieve once per question --------- #
    expand = 0
    rewriter = None
    if args.rewriter:
        rname, rurl, rmodel = parse_reader("rw=" + args.rewriter)
        os.environ["LLM_URL"] = rurl.replace("/v1/chat/completions", "")
        os.environ["LLM_MODEL"] = rmodel
        from groundwire.answer import LLMRewriter
        rewriter = LLMRewriter()
        expand = 3
    mem = Groundwire(memory="bm25", k=args.k, expand=expand, rewriter=rewriter)
    t0 = time.time()
    for title, fname in BOOKS.items():
        mem.ingest(load_book(fname), title=title)
    print(f"indexed {mem._next:,} chunks (~2.8M tokens) in {time.time()-t0:.1f}s "
          f"| fixed retrieval, k={args.k}, expand={expand}\n")

    fixed = []
    for q, gold in QUESTIONS:
        hits = mem.retrieve(q)
        ctx = "\n\n".join(f"[{mem.source_of(c) or c}] {t}" for c, t, _ in hits)
        fixed.append((q, gold, ctx))

    def ask(url, model, ctx, q, max_tokens=None):
        return post(url, model, [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {q}"},
        ], max_tokens or args.max_tokens)

    # ---- accuracy + sequential latency ------------------------------------ #
    print(f"{'reader':16} {'answer-recall':>13} {'avg latency':>12} "
          f"{'tok/s':>7}")
    print("-" * 52)
    summary = []
    for name, url, model in readers:
        hits_ok, lat, toks = 0, [], 0
        for q, gold, ctx in fixed:
            t0 = time.time()
            try:
                ans, ct = ask(url, model, ctx, q)
            except Exception as e:
                ans, ct = f"ERROR {e}", 0
            dt = time.time() - t0
            lat.append(dt)
            toks += ct
            fa = fold(ans)
            if any(fold(g) in fa for g in gold):
                hits_ok += 1
        avg_lat = sum(lat) / len(lat)
        tok_s = toks / sum(lat) if sum(lat) else 0
        print(f"{name:16} {hits_ok:>6}/{len(fixed):<6} "
              f"{avg_lat*1000:>9.0f}ms {tok_s:>7.1f}")
        summary.append((name, url, model, hits_ok, avg_lat))

    # ---- throughput under concurrency ------------------------------------- #
    print(f"\nthroughput burst: {args.concurrency} concurrent requests "
          f"(one representative question)")
    print(f"{'reader':16} {'queries/sec':>12} {'tok/sec':>9} "
          f"{'wall':>7}")
    print("-" * 48)
    q, _, ctx = fixed[7]  # the Ahab question — short answer, all readers know it
    for name, url, model, _, _ in [(s[0], s[1], s[2], 0, 0) for s in summary]:
        def one(_i):
            t0 = time.time()
            try:
                _, ct = ask(url, model, ctx, q, max_tokens=48)
            except Exception:
                ct = 0
            return time.time() - t0, ct
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            res = list(ex.map(one, range(args.concurrency)))
        wall = time.time() - t0
        qps = args.concurrency / wall
        tps = sum(c for _, c in res) / wall
        print(f"{name:16} {qps:>12.2f} {tps:>9.1f} {wall:>6.1f}s")


if __name__ == "__main__":
    main()
