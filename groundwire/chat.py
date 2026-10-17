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

Every turn is stamped with when it happened (`add_turn(..., at=)`, default now), and
context() labels each recalled turn with that time so the model can tell "earlier
today" from "last month". Ask for `recency="prefer"` or `latest=True` when a newer
turn should outrank an older one that says the same thing ("I moved to Durban"
over "I live in Cape Town").
"""
from __future__ import annotations

from datetime import datetime

from .pipeline import Groundwire
from .temporal import to_epoch


class ChatMemory:
    """Index chat turns; retrieve the ones relevant to a new query."""

    def __init__(self, k: int = 3, memory: str = "bm25", encoder=None, rerank=None):
        self._gw = Groundwire(memory=memory, k=k, encoder=encoder, rerank=rerank)
        self.k = k
        self.turns: list[tuple[str, str]] = []      # (user, assistant)
        self.times: list[float] = []                 # epoch of each turn

    def add_turn(self, user: str, assistant: str = "", at=None,
                 subject=None) -> "ChatMemory":
        """Record and index one (user, assistant) exchange. `at` is when it
        happened (datetime / epoch / ISO string; default now); `subject` an
        optional fact key for latest-wins recall. Chainable."""
        i = len(self.turns) + 1
        ts = to_epoch(at)
        self.turns.append((user or "", assistant or ""))
        self.times.append(ts)
        self._gw.ingest(f"Q: {user or ''}\nA: {assistant or ''}".strip(),
                        title=f"turn {i}", at=ts, subject=subject)
        return self

    def add_turns(self, turns) -> "ChatMemory":
        """Bulk-add an iterable of (user, assistant) or (user, assistant, at)
        tuples. Chainable."""
        for t in turns:
            self.add_turn(*t)
        return self

    def retrieve(self, query: str, k: int | None = None, **time_opts):
        """Relevant prior turns as (id, text, score) hits — most relevant first,
        or [] when the history is empty. Each hit has `.at` (when it happened).
        `time_opts` pass through to Groundwire.retrieve: recency=, latest=,
        as_of=, recency_weight=, dup_threshold=."""
        if not self.turns:
            return []
        return self._gw.retrieve(query, k=k or self.k, **time_opts)

    def context(self, query: str, k: int | None = None, sep: str = "\n\n",
                timestamps: bool = True, fmt: str = "%Y-%m-%d %H:%M",
                **time_opts) -> str:
        """The relevant prior turns joined into a context string ('' if none).
        Each turn is labelled with when it happened, in local time
        ("[2026-10-17 14:02] Q: ..."); timestamps=False gives the bare turns."""
        hits = self.retrieve(query, k, **time_opts)
        if not timestamps:
            return sep.join(h[1] for h in hits)
        return sep.join(self._label(h, fmt) + h[1] for h in hits)

    @staticmethod
    def _label(hit, fmt):
        ts = getattr(hit, "ts", None)
        return "" if ts is None else f"[{datetime.fromtimestamp(ts).strftime(fmt)}] "

    def __len__(self):
        return len(self.turns)
