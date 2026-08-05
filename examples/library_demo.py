#!/usr/bin/env python3
"""
Library demo — MULTIPLE whole books in ONE memory index, questioned by a small
model. This is the multi-book stress test: ~2.7M tokens across four classics,
with deliberately ambiguous questions (a fish swallows a man in two of them...)
to see whether retrieval stays grounded in the right book.

    War and Peace        ~732K tokens
    KJV Bible            ~1.0M tokens
    Moby-Dick            ~270K tokens
    Les Misérables       ~690K tokens

The corpus never enters the model's context window. It lives in the retrieval
index (CPU/RAM/disk); per question we pull k passages and hand them to whatever
reader you run (Ollama, llama.cpp server, vLLM -- anything OpenAI-compatible).

    export LLM_URL=http://localhost:11434  LLM_MODEL=llama3.2:3b
    python3 examples/library_demo.py                    # bm25 + your model
    python3 examples/library_demo.py --no-reader        # retrieval only
    python3 examples/library_demo.py --backend sqlite_fts

Books are fetched once from Project Gutenberg (all public domain) and cached in
~/.cache/groundwire/books.
"""
import argparse
import os
import random
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from groundwire.pipeline import Groundwire
from groundwire.answer import make_generator
from groundwire.memory_systems import fold

CACHE = os.path.expanduser("~/.cache/groundwire/books")

# title -> (gutenberg id, filename)
BOOKS = {
    "War and Peace":  (2600, "war_and_peace.txt"),
    "KJV Bible":      (10,   "kjv_bible.txt"),
    "Moby-Dick":      (2701, "moby_dick.txt"),
    "Les Misérables": (135,  "les_miserables.txt"),
}

# (question, anchor phrase expected in retrieved text, expected book)
QUESTIONS = [
    # -- straightforward, one book each ------------------------------------ #
    ("What does Anna Pavlovna say about Genoa and Lucca at the start?",
     "Genoa", "War and Peace"),
    ("What is Pierre Bezukhov's situation as an illegitimate son?",
     "Bezukhov", "War and Peace"),
    ("What did God create in the beginning, according to Genesis?",
     "In the beginning", "KJV Bible"),
    ("What are the opening words of the narrator of Moby-Dick?",
     "Call me Ishmael", "Moby-Dick"),
    ("What is the name of Captain Ahab's whaling ship?",
     "Pequod", "Moby-Dick"),
    ("What did the bishop give Jean Valjean after he stole the silver?",
     "candlesticks", "Les Misérables"),
    ("What crime sent Jean Valjean to the galleys, and for how long?",
     "loaf of bread", "Les Misérables"),
    # -- cross-book traps --------------------------------------------------- #
    ("Who was swallowed by a great fish and stayed in its belly three days?",
     "Jonah", "KJV Bible"),          # trap: Moby-Dick is FULL of whales
    ("Which captain lost his leg to a white whale?",
     "Ahab", "Moby-Dick"),           # trap: plenty of captains in W&P
    ("Who pursues the convict known by the number 24601?",
     "Javert", "Les Misérables"),    # trap: numbers/police in other books
]


def fetch(book_id, fname):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, fname)
    if not os.path.exists(path):
        url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
        print(f"  fetching #{book_id} -> {path}", flush=True)
        urllib.request.urlretrieve(url, path)
    text = open(path, encoding="utf-8", errors="ignore").read()
    s = text.find("*** START OF")
    e = text.rfind("*** END OF")
    if s != -1:
        text = text[text.find("\n", s) + 1:]
    if e != -1:
        text = text[:text.rfind("*** END OF")]
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="bm25",
                    choices=["bm25", "sqlite_fts", "dense", "hybrid"])
    ap.add_argument("--encoder", default=None, help="encoder spec for dense/hybrid")
    ap.add_argument("--no-reader", action="store_true")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--expand", type=int, default=0, metavar="N",
                    help="LLM query expansion: N keyword probes per question "
                         "(paraphrase fix with zero ingest cost)")
    ap.add_argument("--seed", type=int, default=None,
                    help="shuffle book ingestion order. Retrieval is content-"
                         "based, so the scorecard must NOT change across seeds "
                         "(full attention can't say the same: 'lost in the "
                         "middle' makes position matter)")
    ap.add_argument("--interleave", action="store_true",
                    help="slice books and interleave the slices, so adjacent "
                         "index positions cross books (hardest ordering)")
    args = ap.parse_args()

    reader = None if args.no_reader else make_generator("api", max_tokens=args.max_tokens)
    enc = None
    if args.backend in ("dense", "hybrid"):
        enc = args.encoder or "ollama:nomic-embed-text"
    mem = Groundwire(memory=args.backend, reader=reader, k=args.k, encoder=enc,
                  expand=args.expand)

    # -- build the ingestion plan: whole books, optionally shuffled/sliced --- #
    rng = random.Random(args.seed) if args.seed is not None else None
    order = list(BOOKS.items())
    if rng:
        rng.shuffle(order)

    total_words = 0
    units = []                     # (title, text) pieces, in ingestion order
    for title, (book_id, fname) in order:
        text = fetch(book_id, fname)
        total_words += len(text.split())
        if args.interleave:
            # slice on paragraph boundaries into ~24 pieces per book
            paras = text.split("\n\n")
            step = max(1, len(paras) // 24)
            units += [(title, "\n\n".join(paras[i:i + step]))
                      for i in range(0, len(paras), step)]
        else:
            units.append((title, text))
    if rng and args.interleave:
        rng.shuffle(units)

    mode = "interleaved slices" if args.interleave else "book order"
    shown = [t for t, _ in (units if args.interleave else order)]
    print(f"ingestion plan (seed={args.seed}, {mode}): "
          f"{' | '.join(shown[:8])}{' | ...' if len(shown) > 8 else ''}")

    print("building the library (index lives on CPU/disk — GPU untouched):")
    t0 = time.time()
    for title, text in units:
        mem.ingest(text, title=title)
    dt = time.time() - t0
    for title, (book_id, fname) in BOOKS.items():
        words = len(fetch(book_id, fname).split())
        print(f"  {title:<15} ~{words:>9,} words  ~{int(words*1.3):>9,} tokens",
              flush=True)
    tokens = int(total_words * 1.3)
    print(f"library total: ~{tokens:,} tokens in {mem._next:,} chunks, "
          f"ingested in {dt:.1f}s ({int(tokens/dt):,} tok/s)\n", flush=True)

    right_book = anchor_ok = 0
    for q, anchor, expect in QUESTIONS:
        t0 = time.time()
        hits = mem.retrieve(q)
        q_ms = (time.time() - t0) * 1000
        srcs = [mem.source_of(cid) or "?" for cid, _, _ in hits]
        top_book = srcs[0] if srcs else "?"
        found = any(fold(anchor) in fold(t) for _, t, _ in hits)
        right_book += (top_book == expect)
        anchor_ok += found

        print(f"Q: {q}")
        if args.expand:
            for probe in mem.memory.last_queries[1:]:
                print(f"   probe: {probe}")
        print(f"   sources: {srcs}  [{q_ms:.1f} ms]")
        print(f"   book {'OK' if top_book == expect else 'WRONG (wanted ' + expect + ')'}"
              f" | anchor {'OK' if found else 'MISSING (' + anchor + ')'}")
        if reader is not None:
            print(f"   answer: {mem.ask(q)}")
        print(flush=True)

    n = len(QUESTIONS)
    print(f"scorecard: top-1 right book {right_book}/{n}, "
          f"anchor in top-{args.k} {anchor_ok}/{n}")


if __name__ == "__main__":
    main()
