#!/usr/bin/env python3
"""
War and Peace long-context demo — load a whole book into the memory layer, then
ask real questions and get accurate text back, using YOUR model as the reader.

The point: the ~560k-word novel (~750k tokens) never enters the model's context
window. It lives in the retrieval index (CPU/disk); per question we pull the few
relevant passages and hand them to your AWQ/GGUF model. Accurate answers over a
book far larger than any KV cache would hold on a 20GB GPU.

Run on a machine that can reach your embedding + LLM servers:

    export EMBED_URL=http://localhost:8882   EMBED_MODEL=nomic-embed-text
    export LLM_URL=http://localhost:11440     LLM_MODEL=tina4-qwen-v7-awq
    python3 examples/war_and_peace_demo.py                 # dense + your model
    python3 examples/war_and_peace_demo.py --backend bm25 --no-reader   # retrieval only

--path FILE uses a local copy; otherwise the book is fetched from Project
Gutenberg (public domain, book #2600).
"""
import argparse
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from groundwire.pipeline import Groundwire
from groundwire.answer import make_generator

GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/2600/pg2600.txt"

# Real questions with a distinctive anchor phrase we can verify in retrieved text.
QUESTIONS = [
    ("How does the novel open — what does Anna Pavlovna say about Genoa and Lucca?",
     "Genoa"),
    ("Who is Anna Pavlovna Scherer?", "maid of honor"),
    ("What is Pierre Bezukhov's situation as an illegitimate son?", "Bezukhov"),
    ("Describe Prince Andrew Bolkonski's feelings about his wife.", "Bolkonski"),
    ("What happens at the name-day party for the Rostovs?", "Rostov"),
]


def load_book(path):
    if path and os.path.exists(path):
        text = open(path, encoding="utf-8", errors="ignore").read()
    else:
        print(f"fetching War and Peace from Project Gutenberg ...", flush=True)
        text = urllib.request.urlopen(GUTENBERG_URL, timeout=60).read().decode(
            "utf-8", errors="ignore")
    # strip Gutenberg header/footer
    s = text.find("*** START OF")
    e = text.find("*** END OF")
    if s != -1:
        text = text[text.find("\n", s) + 1:]
    if e != -1:
        text = text[:text.rfind("*** END OF")]
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=None, help="local War and Peace .txt")
    ap.add_argument("--backend", default="dense",
                    choices=["dense", "bm25", "sqlite_fts", "hybrid"])
    ap.add_argument("--encoder", default="openai:nomic-embed-text")
    ap.add_argument("--no-reader", action="store_true",
                    help="retrieval only; print the passage instead of a model answer")
    ap.add_argument("--k", type=int, default=4)
    args = ap.parse_args()

    book = load_book(args.path)
    words = book.split()
    print(f"loaded book: {len(book):,} chars, ~{len(words):,} words "
          f"(~{int(len(words)*1.3):,} tokens)\n")

    reader = None if args.no_reader else make_generator("api")
    enc = args.encoder if args.backend == "dense" else None
    mem = Groundwire(memory=args.backend, reader=reader, encoder=enc, k=args.k)

    print("indexing (this stays on CPU/disk — GPU untouched) ...", flush=True)
    mem.ingest(book)
    print("done. asking questions:\n")

    for q, anchor in QUESTIONS:
        hits = mem.retrieve(q)
        top = hits[0][1] if hits else ""
        found = any(anchor.lower() in t.lower() for _, t, _ in hits)
        print("Q:", q)
        if reader is None:
            snippet = " ".join(top.split()[:40])
            print(f"   top passage [{'anchor OK' if found else 'no anchor'}]: {snippet}...")
        else:
            print("   answer:", mem.reader.generate(q, hits))
        print()


if __name__ == "__main__":
    main()
