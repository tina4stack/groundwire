"""
The chat Session -- groundwire owning the before/after of every turn.

This is the load-bearing core the desktop app sits on. Per turn it:
  BEFORE  build the index over SANCTIONED paths, retrieve, and route the request
          (normal / quote / read / whole-book map-reduce), splicing the right
          context or span-offer into the messages;
  DISPATCH stream the chat through the SELECTED backend (local Ollama or cloud);
  AFTER   post-fill [[SPAN:…]] handles with verbatim bytes, append the audit
          footer, and persist the turn (with provenance) to history.

The backend is injected, so the whole pipeline unit-tests with a fake model and
no network. Streaming turns yield text deltas; turns that need the whole answer
(quote / map-reduce) buffer, post-process, then yield.
"""
from __future__ import annotations

import os

from .backends import Backend, OpenAICompatBackend
from .spans import SpanRegistry, QUOTE_INTENT, READ_INTENT
from .mapreduce import summarize
from . import proxy as P                     # reuse CONTEXT_HEADER, footers, sizing


class Session:
    def __init__(self, store, backends: dict, default_backend: str, k: int = 6):
        self.store = store
        self.backends = backends              # name -> Backend
        self.default_backend = default_backend
        self.k = k
        self.mem = None
        self.spans = SpanRegistry()
        self.reindex()

    # -- build the index over the sanctioned allowlist ----------------------- #
    def reindex(self, cloud: bool = False):
        """Rebuild retrieval index + span registry from enabled sanctioned
        paths. `cloud=True` withholds local-only paths (privacy guard)."""
        from .pipeline import Groundwire
        mem = Groundwire(memory="sqlite_fts", k=self.k)
        spans = SpanRegistry()
        for p in self.store.paths_for(cloud=cloud):
            if os.path.isdir(p["path"]):
                mem.ingest_repo(p["path"], prefix=p["scope"])
                spans.build_from_folder(p["path"])
        self.mem, self.spans = mem, spans
        return self

    def _is_cloud(self, backend: Backend) -> bool:
        return isinstance(backend, OpenAICompatBackend)

    # -- one turn (a generator of text deltas) ------------------------------- #
    def turn(self, conv_id, user_text: str, backend_name: str = None):
        backend = self.backends[backend_name or self.default_backend]
        # cloud selected -> reindex without local-only paths, so they never ship
        if self._is_cloud(backend):
            self.reindex(cloud=True)

        history = []
        if conv_id:
            conv = self.store.get_conversation(conv_id)
            history = [{"role": m["role"], "content": m["content"]}
                       for m in (conv["messages"] if conv else [])]
        msgs = history + [{"role": "user", "content": user_text}]

        for piece in self._route(user_text, msgs, backend):
            yield piece

    def _route(self, q, msgs, backend):
        cands = self.spans.resolve(q)

        # 1) READ a located span (summarize/explain chapter N): inject its text
        if cands and READ_INTENT.search(q):
            h, label = cands[0]
            num_ctx = min(P.READ_MAX_CTX, len((self.spans.text_of(h) or "")) // 4 + 1200)
            text = self.spans.text_of(h, cap_chars=(num_ctx - 1000) * 4)
            sysmsg = P.CONTEXT_HEADER + f"[{label}]\n{text}\n=== END CONTEXT ==="
            src = [(self.spans.source_of(h), label.split(" of ")[0], 0.0)]
            yield from self._stream([{"role": "system", "content": sysmsg}] + msgs,
                                    backend, footer=src, num_ctx=num_ctx)
            return

        # 2) whole-file map-reduce ("summarize the whole book")
        if READ_INTENT.search(q) and not cands:
            src = self.spans.resolve_whole(q)
            if src and len(self.spans.texts.get(src, "")) > (P.READ_MAX_CTX - 1200) * 4:
                yield from self._map_reduce(q, src, backend)
                return

        # 3) QUOTE verbatim (structural or content anchor): offer a handle, fill
        if QUOTE_INTENT.search(q):
            if not cands and self.mem is not None and self.mem._next:
                hits = self.mem.retrieve(q, k=1)
                if hits:
                    cid, text, _ = hits[0]
                    lbl = f"the passage matching your request (from {self.mem.source_of(cid) or '?'})"
                    cands = [(self.spans.register_text_span(lbl, text), lbl)]
            if cands:
                offer = self.spans.offer(cands)
                fallback = cands[0][0]
                # buffer, post-fill the handle with verbatim bytes
                whole = "".join(backend.chat(
                    [{"role": "system", "content": offer}] + msgs, stream=False))
                filled = self.spans.fill(whole)
                if filled == whole:                       # model fumbled the handle
                    span = self.spans.text_of(fallback)
                    if span:
                        filled = (whole.rstrip() + "\n\n" + span) if whole.strip() else span
                yield filled
                return

        # 4) normal: retrieve top-k, inject as grounding context, stream through
        block, sources = self._retrieve_block(q)
        send = ([{"role": "system", "content": block}] + msgs) if block else msgs
        yield from self._stream(send, backend, footer=sources)

    # -- helpers ------------------------------------------------------------- #
    def _retrieve_block(self, q):
        if self.mem is None or not self.mem._next:
            return None, []
        hits = self.mem.retrieve(q, k=self.k)
        if not hits:
            return None, []
        parts, sources = [], []
        for cid, text, score in hits:
            s = self.mem.source_of(cid)
            parts.append(f"[{s}]\n{text}" if s else text)
            sources.append((s or str(cid), cid, score))
        return P.CONTEXT_HEADER + "\n\n---\n\n".join(parts) + "\n=== END CONTEXT ===", sources

    def _stream(self, send, backend, footer=None, num_ctx=None):
        opts = {"num_ctx": num_ctx} if num_ctx else None
        for piece in backend.chat(send, stream=True, options=opts):
            yield piece
        if footer:
            yield P.sources_footer(footer)

    def _map_reduce(self, q, src, backend):
        segments = self.spans.coarse_segments(src)
        budget = (P.READ_MAX_CTX - 1200) * 4
        yield f"Summarizing {src} in {len(segments)} parts…\n\n"
        prog = []
        def call_llm(prompt):
            return backend.complete([{"role": "user", "content": prompt}],
                                    options={"num_ctx": P.READ_MAX_CTX})
        # progress is emitted between segments via the callback buffer
        result = {}
        def on_progress(m):
            prog.append(f"▸ {m}\n")
        final = summarize(call_llm, segments, budget, q, on_progress=on_progress)
        for line in prog:
            yield line
        yield "\n" + final + f"\n\n— groundwire ▸ map-reduce summary of {src}"

    # -- persistence convenience -------------------------------------------- #
    def ask_and_store(self, conv_id, user_text, backend_name=None) -> str:
        """Run a turn to completion, persist both messages, return the answer."""
        if conv_id is None:
            conv_id = self.store.new_conversation(user_text[:60], backend_name
                                                  or self.default_backend)
        self.store.add_message(conv_id, "user", user_text)
        answer = "".join(self.turn(conv_id, user_text, backend_name))
        self.store.add_message(conv_id, "assistant", answer,
                               model=backend_name or self.default_backend)
        return conv_id, answer
