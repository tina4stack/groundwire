"""Conversational memory as retrieval — groundwire's answer to long chat history.

Dumping a whole transcript into a small model overflows its window and corrupts
later turns. Instead, INDEX each turn and retrieve only the turns relevant to the
new message — bounded context, no matter how long the conversation. It's the same
primitive groundwire applies to source, applied to the dialogue itself (and the
original NIAH / BEAM use case: conversational-memory recall).

    from groundwire import ChatMemory

    mem = ChatMemory(k=3)
    mem.add_turn("How do I open a FireDAC connection?", "Use TFDConnection.Connected := True")
    mem.add_turn("What is the capital of France?", "Paris")
    # a follow-up retrieves only the RELEVANT prior turns, not the whole log:
    ctx = mem.context("and its connection parameters?")     # the FireDAC turn, not Paris

Default backend is in-memory BM25 (fast, no temp files) — ideal for a per-session
chat. Pass memory="dense"/"hybrid" or rerank="dense" for semantic recall.
"""
from __future__ import annotations

from .pipeline import Groundwire


class ChatMemory:
    """Index chat turns; retrieve the ones relevant to a new query."""

    def __init__(self, k: int = 3, memory: str = "bm25", encoder=None, rerank=None):
        self._gw = Groundwire(memory=memory, k=k, encoder=encoder, rerank=rerank)
        self.k = k
        self.turns: list[tuple[str, str]] = []      # (user, assistant)

    def add_turn(self, user: str, assistant: str = "") -> "ChatMemory":
        """Record and index one (user, assistant) exchange. Chainable."""
        i = len(self.turns) + 1
        self.turns.append((user or "", assistant or ""))
        self._gw.ingest(f"Q: {user or ''}\nA: {assistant or ''}".strip(),
                        title=f"turn {i}")
        return self

    def add_turns(self, turns) -> "ChatMemory":
        """Bulk-add an iterable of (user, assistant) pairs. Chainable."""
        for u, a in turns:
            self.add_turn(u, a)
        return self

    def retrieve(self, query: str, k: int | None = None):
        """Relevant prior turns as (id, text, score) — most relevant first, or []
        when the history is empty."""
        if not self.turns:
            return []
        return self._gw.retrieve(query, k=k or self.k)

    def context(self, query: str, k: int | None = None, sep: str = "\n\n") -> str:
        """The relevant prior turns joined into a context string ('' if none)."""
        return sep.join(t for _, t, _ in self.retrieve(query, k))

    def __len__(self):
        return len(self.turns)
