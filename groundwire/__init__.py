"""Groundwire: retrieval + verification in front of an LLM.

Ground a weak reader in your own source — the corpus stays off-GPU, only the
top-k retrieved chunks reach the model. Part of the tina4stack.

    from groundwire import Groundwire

    gw = Groundwire(memory="sqlite_fts", k=5)
    gw.ingest(open("big_doc.txt").read())
    print(gw.retrieve("what was the Q3 revenue figure?"))   # or gw.ask(...) with a reader

Distributed on PyPI as `tina4-groundwire`.
"""
from .pipeline import Groundwire, chunk_text, chunk_code

__version__ = "0.1.0"
__all__ = ["Groundwire", "chunk_text", "chunk_code", "__version__"]
