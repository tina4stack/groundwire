"""Tiny stdlib vector ops — replaces numpy for groundwire's dense path.

The only thing numpy ever did here was arithmetic over embedding vectors that come
from the *remote* embedder (Ollama / vLLM): L2-normalize the rows, take dot products
(cosine, since the rows are unit vectors), and sort a small candidate pool by score.
The rerank pool is tens of vectors × ~768 dims — pure Python is well under a
millisecond and the embedding network call dominates anyway. Keeping this pure stdlib
makes groundwire (and the Tina4 `Context` subsystem that embeds it) truly zero-dependency.
"""
from __future__ import annotations
import math


def dot(a, b) -> float:
    """Dot product of two equal-length vectors."""
    return sum(x * y for x, y in zip(a, b))


def l2_normalize(vec) -> list[float]:
    """Unit-length copy of `vec` (zero vector is returned unchanged)."""
    n = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / n for x in vec]


def normalize_rows(rows) -> list[list[float]]:
    """L2-normalize each row. Accepts a single vector (1-D) or a matrix (2-D);
    always returns a list of unit rows — same contract the old numpy `_l2` had."""
    if rows and not isinstance(rows[0], (list, tuple)):
        rows = [rows]                       # a lone vector -> one row
    return [l2_normalize(r) for r in rows]


def cosine_scores(query, mat) -> list[float]:
    """Similarity of `query` to every row of `mat`. Both sides are assumed
    L2-normalized (the encoders emit unit rows), so cosine == dot."""
    return [dot(query, row) for row in mat]


def argsort_desc(scores) -> list[int]:
    """Indices of `scores` from highest to lowest (stable — ties keep input order,
    which is *more* deterministic than numpy's default quicksort argsort)."""
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
