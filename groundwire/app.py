"""
The chat Session -- groundwire owning the before/after of every turn.

This is the load-bearing core the desktop app sits on. Per turn it:
  BEFORE  retrieve over SANCTIONED paths and route the request (normal / quote /
          read / whole-book map-reduce), splicing the right context or
          span-offer into the messages;
  DISPATCH stream the chat through the SELECTED backend (local Ollama or cloud);
  AFTER   post-fill [[SPAN:…]] handles with verbatim bytes, append the audit
          footer, and persist the turn (with provenance) to history.

Two indexes are cached: a FULL one and a CLOUD-SAFE one (local-only paths
withheld). A turn picks by backend type, so selecting a cloud model never ships
a folder marked local-only -- and nothing is reindexed per request. The backend
is injected, so the whole pipeline unit-tests with a fake model and no network.
"""
from __future__ import annotations

import os

import re

from .backends import Backend, OpenAICompatBackend
from .spans import (SpanRegistry, QUOTE_INTENT, READ_INTENT, _NUMWORD,
                    _POS_REF, _CHAP_REF)
from .memory_systems import terms as _terms
from .mapreduce import summarize
from . import proxy as P                     # reuse CONTEXT_HEADER, footers, sizing

# Words that live in a QUOTE *request* but not in the passage itself -- they must
# not count as evidence that a retrieved chunk matches the request.
_QUOTE_SCAFFOLD = frozenset(
    "quote quotes quotation reproduce verbatim exact exactly text texts passage "
    "passages full entire whole complete word words wording print printed show "
    "shows give given line lines sentence sentences paragraph paragraphs please "
    "from starts start begins begin beginning opening excerpt".split())

# A POSITIONAL reference addresses a numbered/ordinal unit ("psalm 23", "john
# 3:16", "sonnet 18", "canto iv", "act two"). Quoting one is an ADDRESSING
# problem, not a search one: a lexical hit for "psalm 23" surfaces passages that
# MENTION psalm 23 (e.g. Acts 13:33, "in the second psalm"), never psalm 23
# itself. So when a positional quote can't be resolved to a structural span, we
# must NOT fabricate from a search hit -- we fall through to normal grounding.
# The division words are a generic literary/legal set, not corpus-specific data.
_POSITIONAL = re.compile(
    r"\b\d+\s*:\s*\d+\b"                             # verse/section refs: 3:16
    r"|\b\d+\b"                                      # any number: psalm 23
    r"|\b(?:chapter|canto|book|part|section|letter|verse|psalm|sonnet|hymn|"
    r"stanza|act|scene|paragraph|line|page|article|clause|figure|table)\s+"
    r"(?:[ivxlcdm]+|(?:" + _NUMWORD + r"))\b",       # roman/word ordinals
    re.I)


def _looks_positional(q: str) -> bool:
    return bool(_POSITIONAL.search(q))


def _quote_hit_matches(q: str, doc: str) -> bool:
    """Guard the CONTENT-anchored quote fallback (positional refs are handled by
    _looks_positional upstream). A k-best lexical hit becomes a VERBATIM quote
    only if the passage genuinely shares a salient term with the request --
    salient = alphabetic, length >= 3, not quote-scaffolding. On no match we
    return False and the turn falls through to normal grounded retrieval, whose
    CONTEXT header tells the model to admit it can't find the exact text rather
    than fabricate it."""
    salient = [t for t in _terms(q)
               if len(t) >= 3 and any(c.isalpha() for c in t)
               and t not in _QUOTE_SCAFFOLD]
    if not salient:                       # nothing distinctive to verify against
        return False
    dterms = set(_terms(doc))
    return any(t in dterms for t in salient)


# Words that surround a bare reference without turning it into a question.
_REF_FILLER = frozenset(
    "show me give given please read see get want the a an of in on from to and "
    "or book books chapter chapters verse verses section sections part parts "
    "canto cantos psalm psalms sonnet sonnets act acts scene scenes stanza line "
    "lines full whole complete entire text passage quote opening beginning start "
    "starts begin begins".split())


def _is_bare_reference(q: str) -> bool:
    """True when the query is essentially JUST a structural reference ('psalm 23',
    'chapter 5', 'canto iv') with no other question around it -- so we SHOW the
    passage verbatim rather than answer a question about it. 'who dies in chapter
    5' is NOT bare (it has 'who dies') and routes to inject-and-answer."""
    rest = _CHAP_REF.sub(" ", _POS_REF.sub(" ", q.lower()))
    return not [t for t in re.findall(r"[a-z]+", rest) if t not in _REF_FILLER]


class Session:
    def __init__(self, store, backends: dict, default_backend: str, k: int = 6):
        self.store = store
        self.backends = backends              # name -> Backend
        self.default_backend = default_backend
        self.k = k
        self._full = (None, SpanRegistry())   # (mem, spans) over all paths
        self._cloud = (None, SpanRegistry())  # (mem, spans) minus local-only
        self.reindex()

    # -- build the indexes over the sanctioned allowlist --------------------- #
    def _build(self, cloud: bool):
        from .pipeline import Groundwire
        mem = Groundwire(memory="sqlite_fts", k=self.k)
        spans = SpanRegistry()
        for p in self.store.paths_for(cloud=cloud):
            if os.path.isdir(p["path"]):
                mem.ingest_repo(p["path"], prefix=p["scope"])
                spans.build_from_folder(p["path"], prefix=p["scope"])
        return mem, spans

    def reindex(self):
        """Rebuild both indexes. Call after the sanctioned paths change (not per
        turn)."""
        self._full = self._build(cloud=False)
        self._cloud = self._build(cloud=True)
        return self

    def _index_for(self, backend: Backend):
        return self._cloud if isinstance(backend, OpenAICompatBackend) else self._full

    # -- one turn (a generator of text deltas) ------------------------------- #
    def turn(self, conv_id, user_text: str, backend_name: str = None):
        backend = self.backends[backend_name or self.default_backend]
        mem, spans = self._index_for(backend)
        history = []
        if conv_id:
            conv = self.store.get_conversation(conv_id)
            history = [{"role": m["role"], "content": m["content"]}
                       for m in (conv["messages"] if conv else [])]
        msgs = history + [{"role": "user", "content": user_text}]
        yield from self._route(user_text, msgs, backend, mem, spans)

    def _plan(self, q, msgs, mem, spans):
        """Decide the turn and build exactly what gets injected -- WITHOUT
        calling the model. Returns a plan dict that both _execute() (dispatch)
        and inspect() (audit) consume, so what you inspect is byte-identical to
        what runs."""
        cands = spans.resolve(q)
        # a BARE reference ("psalm 23") is shown verbatim; the SAME reference
        # inside a question ("what happens in chapter 5") is injected + answered.
        bare = bool(cands) and _is_bare_reference(q)

        # 1) READ a located span (a question ABOUT chapter N -- explain/summarize/
        #    who/why...): inject its text so the model reasons over it. Skipped
        #    for a bare reference or an explicit quote (handled verbatim below).
        if cands and not bare and not QUOTE_INTENT.search(q):
            h, label = cands[0]
            num_ctx = min(P.READ_MAX_CTX, len((spans.text_of(h) or "")) // 4 + 1200)
            text = spans.text_of(h, cap_chars=(num_ctx - 1000) * 4)
            ctx = P.CONTEXT_HEADER + f"[{label}]\n{text}\n=== END CONTEXT ==="
            return {"mode": "read", "context": ctx, "num_ctx": num_ctx,
                    "messages": [{"role": "system", "content": ctx}] + msgs,
                    "sources": [(spans.source_of(h), label.split(" of ")[0], 0.0)]}

        # 2) whole-file map-reduce ("summarize the whole book")
        if READ_INTENT.search(q) and not cands:
            src = spans.resolve_whole(q)
            if src and len(spans.texts.get(src, "")) > (P.READ_MAX_CTX - 1200) * 4:
                segs = spans.coarse_segments(src)
                return {"mode": "mapreduce", "mapreduce_src": src, "messages": msgs,
                        "context": f"(map-reduce over {len(segs)} segments of {src})",
                        "sources": [(src, "map-reduce", 0.0)]}

        # 3) QUOTE verbatim: an explicit quote verb runs a CONTENT-anchored
        #    lexical fallback; a bare structural/positional reference ("psalm 23")
        #    that already resolved is shown verbatim on its own.
        if (QUOTE_INTENT.search(q) and not cands and mem is not None
                and mem._next and not _looks_positional(q)):
            # take the top-ranked hit that ACTUALLY shares a salient term; never
            # fabricate a verbatim quote from a non-match. A POSITIONAL quote we
            # couldn't resolve structurally is deliberately NOT satisfied by
            # search -- it falls through to normal grounding instead.
            for cid, text, _ in mem.retrieve(q, k=5):
                if _quote_hit_matches(q, text):
                    lbl = f"the passage matching your request (from {mem.source_of(cid) or '?'})"
                    cands = [(spans.register_text_span(lbl, text), lbl)]
                    break
        if cands and (QUOTE_INTENT.search(q) or bare):
            fallback = cands[0][0]
            return {"mode": "quote", "offer": spans.offer(cands),
                    "fallback": fallback,
                    "context": spans.text_of(fallback),   # the verbatim bytes
                    "messages": [{"role": "system", "content": spans.offer(cands)}] + msgs,
                    "sources": [(spans.source_of(fallback) or "?",
                                 cands[0][1].split(" of ")[-1], 0.0)]}

        # 4) normal: retrieve top-k, inject as grounding context
        block, sources = self._retrieve_block(q, mem)
        return {"mode": "normal", "context": block,
                "messages": ([{"role": "system", "content": block}] + msgs)
                            if block else msgs, "sources": sources}

    def _execute(self, plan, backend):
        mode = plan["mode"]
        if mode == "mapreduce":
            yield from self._map_reduce_plan(plan, backend)
        elif mode == "quote":
            spans = None  # spans captured via plan; fill happens on the string
            whole = "".join(backend.chat(plan["messages"], stream=False))
            filled = self._fill_quote(whole, plan)
            yield filled
        else:  # read / normal
            yield from self._stream(plan["messages"], backend,
                                    footer=plan.get("sources"),
                                    num_ctx=plan.get("num_ctx"))

    def _route(self, q, msgs, backend, mem, spans):
        plan = self._plan(q, msgs, mem, spans)
        # quote/map-reduce need the live registry to fill/segment
        plan["_spans"] = spans
        yield from self._execute(plan, backend)

    def _fill_quote(self, whole, plan):
        spans = plan["_spans"]
        filled = spans.fill(whole)
        if filled == whole and plan.get("fallback"):     # model fumbled the handle
            span = spans.text_of(plan["fallback"])
            if span:
                filled = (whole.rstrip() + "\n\n" + span) if whole.strip() else span
        return filled

    def _map_reduce_plan(self, plan, backend):
        yield from self._map_reduce(plan["messages"][-1]["content"],
                                    plan["mapreduce_src"], backend, plan["_spans"])

    # -- audit: exactly what WOULD be injected, no model call ----------------- #
    def inspect(self, user_text: str, backend_name: str = None):
        """Return {mode, context, sources} for a query -- the real injected
        context, built by the same _plan() the turn uses, but the model is never
        called. This is how 'is the injected context accurate?' is answered by
        looking, not trusting."""
        backend = self.backends.get(backend_name or self.default_backend)
        mem, spans = self._index_for(backend) if backend else self._full
        plan = self._plan(user_text, [{"role": "user", "content": user_text}],
                          mem, spans)
        srcs = [{"source": s[0], "ref": s[1]} for s in plan.get("sources", [])]
        return {"mode": plan["mode"], "context": plan.get("context") or "",
                "sources": srcs}

    # -- helpers ------------------------------------------------------------- #
    def _retrieve_block(self, q, mem):
        if mem is None or not mem._next:
            return None, []
        hits = mem.retrieve(q, k=self.k)
        if not hits:
            return None, []
        parts, sources = [], []
        for cid, text, score in hits:
            s = mem.source_of(cid)
            parts.append(f"[{s}]\n{text}" if s else text)
            sources.append((s or str(cid), cid, score))
        return P.CONTEXT_HEADER + "\n\n---\n\n".join(parts) + "\n=== END CONTEXT ===", sources

    def _stream(self, send, backend, footer=None, num_ctx=None):
        opts = {"num_ctx": num_ctx} if num_ctx else None
        for piece in backend.chat(send, stream=True, options=opts):
            yield piece
        if footer:
            yield P.sources_footer(footer)

    def _map_reduce(self, q, src, backend, spans):
        segments = spans.coarse_segments(src)
        budget = (P.READ_MAX_CTX - 1200) * 4
        yield f"Summarizing {src} in {len(segments)} parts…\n\n"
        prog = []
        def call_llm(prompt):
            return backend.complete([{"role": "user", "content": prompt}],
                                    options={"num_ctx": P.READ_MAX_CTX})
        final = summarize(call_llm, segments, budget, q,
                          on_progress=lambda m: prog.append(f"▸ {m}\n"))
        for line in prog:
            yield line
        yield "\n" + final + f"\n\n— groundwire ▸ map-reduce summary of {src}"

    # -- persistence convenience -------------------------------------------- #
    def ask_and_store(self, conv_id, user_text, backend_name=None):
        """Run a turn to completion, persist both messages, return (id, answer)."""
        if conv_id is None:
            conv_id = self.store.new_conversation(user_text[:60], backend_name
                                                  or self.default_backend)
        self.store.add_message(conv_id, "user", user_text)
        answer = "".join(self.turn(conv_id, user_text, backend_name))
        self.store.add_message(conv_id, "assistant", answer,
                               model=backend_name or self.default_backend)
        return conv_id, answer
