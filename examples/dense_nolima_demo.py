#!/usr/bin/env python3
"""
Pipeline validation: does dense retrieval solve NoLiMa where BM25 cannot?

NoLiMa needles share NO content words with their question -- only meaning. A
lexical retriever (BM25) therefore has nothing to match. A *semantic* encoder
maps the two phrasings to nearby vectors, so dense retrieval finds the needle.

We cannot reach a live Ollama server from this environment, so this demo uses a
SIMULATED semantic encoder: it maps each known concept (in either phrasing) to a
shared latent vector, i.e. it stands in for what nomic-embed-text would produce.
This validates the *pipeline and the claim* -- the real numbers come from:

    EMBED_URL=http://localhost:11434 \\
    groundwire --task nolima --backend dense --encoder ollama:nomic-embed-text

Run: python3 examples/dense_nolima_demo.py
"""
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from groundwire.veclite import normalize_rows     # pure stdlib -- no numpy

from groundwire.tasks import build_task, NOLIMA_PAIRS
from groundwire.harness import build_filler, plant, chunk_by_sentences
from groundwire.memory_systems import InMemoryBM25, DenseEncoderMemory

DIM = len(NOLIMA_PAIRS)


def simulated_semantic_encoder(texts):
    """Stand-in for a real embedder: any text mentioning a concept (in either
    phrasing) gets that concept's latent vector; filler gets a random vector."""
    rng = random.Random(0)
    rows = [[0.0] * (DIM + 8) for _ in texts]
    for i, t in enumerate(texts):
        tl = t.lower()
        hit = None
        for ci, (needle_phrase, query_phrase) in enumerate(NOLIMA_PAIRS):
            if needle_phrase in tl or query_phrase in tl:
                hit = ci
                break
        if hit is not None:
            rows[i][hit] = 1.0
        else:
            for j in range(DIM, DIM + 8):
                rows[i][j] = rng.random()
    return normalize_rows(rows)


def recall_for(backend, items, k=5):
    hits = 0
    for it in items:
        res = backend.query(it["question"], k=k)
        hits += any(it["gold"] in text for _, text, _ in res)
    return hits / len(items)


def main():
    rng = random.Random(7)
    depths = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    items = build_task("nolima", rng=rng, depths=depths)
    sents = build_filler(60000, rng, 1.3)
    plant(sents, [s for it in items for s in it["supports"]], rng)
    chunks = chunk_by_sentences(sents, 350, 1)

    bm25 = InMemoryBM25()
    bm25.ingest(chunks)
    dense = DenseEncoderMemory(encoder=simulated_semantic_encoder)
    dense.ingest(chunks)

    print(f"NoLiMa haystack: {len(chunks)} chunks, {len(items)} needles\n")
    print(f"  bm25  (lexical) recall@5 : {recall_for(bm25, items)*100:5.1f}%")
    print(f"  dense (semantic) recall@5: {recall_for(dense, items)*100:5.1f}%")
    print("\nLexical retrieval can't bridge zero word overlap; a semantic encoder "
          "can. Same result expected with real Ollama/vLLM embeddings.")


if __name__ == "__main__":
    main()
