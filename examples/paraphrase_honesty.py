#!/usr/bin/env python3
"""
The honesty check at scale: does retrieval need embeddings?

BENCHMARKS.md's dense-vs-lexical verdict started at N=6. This runs a validated
~30-question paraphrase-hard set over the 4-book corpus (War and Peace, KJV
Bible, Moby-Dick, Les Misérables), comparing retrieval recall@k for:

    bm25            lexical only (folding + stemming, no model)
    dense           nomic-embed-text over the whole corpus
    hybrid          RRF of both
    bm25+rerank     lexical retrieve, dense reorder the pool (query-time only)

Every question is PARAPHRASED to minimise lexical overlap with its passage, and
every gold anchor is validated to exist in the corpus first (a stale question is
excluded, not counted as a miss). Needs the embed endpoint for dense/hybrid/rerank:

    export EMBED_URL=http://your-gpu-host:11435 EMBED_MODEL=nomic-embed-text
    python3 examples/paraphrase_honesty.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from groundwire.pipeline import Groundwire
from groundwire.memory_systems import fold
from examples.reader_shootout import load_book, BOOKS

# (paraphrased question with minimal word overlap, gold substring in the passage)
QUESTIONS = [
    # Moby-Dick
    ("Who is the sailor narrating the whaling voyage from its opening?", "ishmael"),
    ("What vessel does the one-legged captain command on his hunt?", "pequod"),
    ("Which obsessed commander lost a limb to the pale sea-beast?", "ahab"),
    ("What harpooneer from the South Seas shares Ishmael's room at the inn?", "queequeg"),
    ("What is the pale leviathan the captain is obsessed with destroying called?", "moby dick"),
    ("Which first mate urges caution against the captain's mad pursuit?", "starbuck"),
    # KJV Bible
    ("What did the deity fashion at the very start of everything?", "heaven and the earth"),
    ("Who was gulped down by a huge sea creature and stayed inside three days?", "jonah"),
    ("Which man was told to build a great vessel before the flood waters came?", "noah"),
    ("Who faced the giant warrior with only a sling and a stone?", "david"),
    ("Which garden held the tree whose fruit was forbidden to the first couple?", "eden"),
    ("Who led the enslaved people out of bondage across the parted waters?", "moses"),
    ("What did the deity say should exist first, bringing an end to darkness?", "let there be light"),
    # Les Misérables
    ("Which lawman relentlessly hunts the reformed ex-prisoner?", "javert"),
    ("What did the clergyman hand the thief so he'd escape arrest?", "candlestick"),
    ("What small girl is left with cruel innkeepers by her desperate mother?", "cosette"),
    ("Which dying woman sells her hair and teeth to support her child?", "fantine"),
    ("What is the surname of the greedy innkeeper couple who exploit everyone?", "thenardier"),
    ("Which young man mans the barricade and loves the girl raised by the convict?", "marius"),
    ("What is the prisoner number branded on the man who stole bread?", "24601"),
    # War and Peace
    ("At the opening soiree, whose estates are said to be mere family holdings now?", "genoa"),
    ("Which illegitimate heir unexpectedly inherits an enormous fortune?", "pierre"),
    ("Which thoughtful prince is wounded and gazes up at the vast sky at Austerlitz?", "andrew"),
    ("What young countess of the lively family captivates the men around her?", "natasha"),
    ("Which French emperor's invasion drives the novel's war chapters?", "napoleon"),
    ("Which ancient Russian general reluctantly commands against the invader?", "kutuzov"),
    ("What great city is abandoned and burns as the French army enters it?", "moscow"),
    # cross / trickier
    ("Which prophet ran from his calling by boarding a ship to Tarshish?", "jonah"),
    ("What tavern-keeper's daughter befriends the student at the barricade?", "eponine"),
    ("Which cabin boy aboard the whaler goes mad after nearly drowning?", "pip"),
]


def build(backend, enc=None, rerank=None):
    m = Groundwire(memory=backend, k=5, encoder=enc, rerank=rerank)
    for t, f in BOOKS.items():
        m.ingest(load_book(f), title=t)
    return m


def main():
    # validate anchors exist in the corpus (exclude stale ones)
    corpus = fold(" ".join(load_book(f) for f in BOOKS.values()))
    valid = [(q, g) for q, g in QUESTIONS if fold(g) in corpus]
    stale = [g for q, g in QUESTIONS if fold(g) not in corpus]
    if stale:
        print(f"excluded {len(stale)} stale anchors: {stale}")
    print(f"{len(valid)} validated paraphrase-hard questions\n")

    print("building indexes...", flush=True)
    t0 = time.time(); bm = build("bm25"); print(f"  bm25   {time.time()-t0:.1f}s", flush=True)
    t0 = time.time(); dn = build("dense", "openai:nomic-embed-text"); print(f"  dense  {time.time()-t0:.1f}s", flush=True)
    t0 = time.time(); hy = build("hybrid", "openai:nomic-embed-text"); print(f"  hybrid {time.time()-t0:.1f}s", flush=True)
    t0 = time.time(); rr = build("bm25", rerank="dense", enc="openai:nomic-embed-text")
    print(f"  bm25+rerank {time.time()-t0:.1f}s\n", flush=True)

    systems = [("bm25", bm), ("dense", dn), ("hybrid", hy), ("bm25+rerank", rr)]
    tot = {n: 0 for n, _ in systems}
    for q, gold in valid:
        for name, m in systems:
            hits = m.retrieve(q)
            if any(fold(gold) in fold(t) for _, t, _ in hits):
                tot[name] += 1

    n = len(valid)
    print("recall@5 over paraphrase-hard questions:")
    for name, _ in systems:
        bar = "#" * round(30 * tot[name] / n)
        print(f"  {name:14} {tot[name]:2}/{n}  {100*tot[name]/n:4.0f}%  {bar}")


if __name__ == "__main__":
    main()
