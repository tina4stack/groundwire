"""
Map-reduce summarizer for spans too large to fit the model's context.

"Summarize the whole book" can't be one call -- the book dwarfs the window. So
we MAP a summary over each coarse segment (a book/part/chapter), then REDUCE the
segment summaries into one, recursing when a segment is itself too big or when
the collected summaries still overflow. The LLM is injected as `call_llm(prompt)
-> text`, so the engine is deterministic and unit-testable without a model.

    from groundwire.mapreduce import summarize
    final = summarize(call_llm, [("book 1", text1), ...],
                      budget_chars=24000, request="summarize War and Peace")

Segmentation itself lives in SpanRegistry.coarse_segments(); this module only
orchestrates the map/reduce over whatever segments it is handed.
"""
from __future__ import annotations

_MAP_PROMPT = (
    "You are summarizing one part of a larger work. Summarize the following "
    "section ({label}) faithfully and concisely, capturing its key events, "
    "characters, and ideas. Do not add information that is not present.\n\n"
    "=== {label} ===\n{text}\n=== END ===\n\nSummary of {label}:")

_REDUCE_PROMPT = (
    "Below are summaries of consecutive parts of a single work, in order. "
    "Combine them into one coherent summary that fulfils this request: "
    "\"{request}\". Preserve the order of events and do not invent anything.\n\n"
    "{text}\n\nCombined summary:")


def _windows(text: str, budget_chars: int, overlap: int = 200):
    """Split text into ~budget_chars windows on whitespace, with slight overlap
    so a sentence cut at a boundary still appears whole in one window."""
    text = text.strip()
    if len(text) <= budget_chars:
        return [text]
    out, i = [], 0
    while i < len(text):
        end = min(i + budget_chars, len(text))
        if end < len(text):
            sp = text.rfind(" ", i + budget_chars // 2, end)
            if sp != -1:
                end = sp
        out.append(text[i:end].strip())
        if end >= len(text):
            break
        i = max(end - overlap, i + 1)
    return out


def _map_one(call_llm, label: str, text: str, budget_chars: int, request: str):
    """Summarize one segment. If it overflows the budget, window it and reduce
    the window summaries first (recursion handles arbitrarily large segments)."""
    if len(text) <= budget_chars:
        return call_llm(_MAP_PROMPT.format(label=label, text=text))
    parts = [(f"{label} part {i + 1}", w)
             for i, w in enumerate(_windows(text, budget_chars))]
    mapped = [(lbl, _map_one(call_llm, lbl, w, budget_chars, request))
              for lbl, w in parts]
    return _reduce(call_llm, mapped, budget_chars, request)


def _reduce(call_llm, summaries, budget_chars: int, request: str) -> str:
    """Combine (label, summary) pairs into one summary. If they don't all fit in
    one reduce call, batch them, reduce each batch, then reduce the results
    (hierarchical reduce)."""
    if len(summaries) == 1:
        return summaries[0][1]
    combined = "\n\n".join(f"[{lbl}]\n{s}" for lbl, s in summaries)
    if len(combined) <= budget_chars:
        return call_llm(_REDUCE_PROMPT.format(request=request, text=combined))
    batches, cur, size = [], [], 0
    for lbl, s in summaries:
        if cur and size + len(s) > budget_chars:
            batches.append(cur)
            cur, size = [], 0
        cur.append((lbl, s))
        size += len(s) + len(lbl) + 4
    if cur:
        batches.append(cur)
    if len(batches) == len(summaries):        # nothing grouped -> force progress
        batches = [summaries[:len(summaries) // 2 or 1],
                   summaries[len(summaries) // 2 or 1:]]
        batches = [b for b in batches if b]
    reduced = [(f"batch {i + 1}", _reduce(call_llm, b, budget_chars, request))
               for i, b in enumerate(batches)]
    return _reduce(call_llm, reduced, budget_chars, request)


def summarize(call_llm, segments, budget_chars: int, request: str,
              on_progress=None) -> str:
    """Map a summary over each (label, text) segment, then reduce to one.
    `call_llm(prompt) -> text`. `budget_chars` should be ~ the model context in
    chars (num_ctx * ~4) minus prompt overhead. `on_progress(message)` fires
    before each top-level segment and before the reduce -- the proxy streams
    these to the client so a long run shows 'summarizing book 3/15…'."""
    segments = [(l, t) for l, t in segments if (t or "").strip()]
    if not segments:
        return ""
    n = len(segments)
    mapped = []
    for i, (lbl, text) in enumerate(segments, 1):
        if on_progress:
            on_progress(f"summarizing {lbl} ({i}/{n})")
        mapped.append((lbl, _map_one(call_llm, lbl, text, budget_chars, request)))
    if len(mapped) > 1 and on_progress:
        on_progress("combining the summaries")
    return _reduce(call_llm, mapped, budget_chars, request)


# --------------------------------------------------------------------------- #
# Answer-reduce: the QA counterpart of summarize().                           #
#                                                                             #
# Retrieval keeps the injected context small (k chunks), but the fixes for    #
# the recall ceilings all WIDEN it -- multi-hop wants a big k, hybrid widens   #
# the pool, code chunks run long -- until the retrieved set overflows a weak   #
# reader's window and gets silently truncated (dropping the very chunk you     #
# widened k to catch). So when the chunks don't fit, we MAP the question over  #
# rank-ordered, budget-sized batches and REDUCE by verified-answer selection   #
# instead of truncating. Unlike summarize() (map a *summary*, reduce by        #
# stitching prose), this maps the *question* and reduces by picking the        #
# candidate the retrieved text actually supports.                             #
# --------------------------------------------------------------------------- #

# Answers a weak reader emits when the batch it saw doesn't contain the answer.
# These lose to any real candidate during reduce (order-preserving otherwise).
_REFUSALS = frozenset((
    "", "i don't know", "i dont know", "idk", "no answer", "not found",
    "none", "n/a", "na", "unknown", "not in the context",
    "not in context", "the context does not", "cannot answer", "can't answer",
))


def _is_answer(s: str) -> bool:
    """A candidate is a real answer if it's non-empty and not a stock refusal."""
    t = (s or "").strip().lower().rstrip(".")
    return bool(t) and t not in _REFUSALS and not t.startswith("i don't know") \
        and not t.startswith("i cannot") and not t.startswith("not in the")


def _pack(chunks, budget_chars: int):
    """Greedy-pack (cid, text, score) chunks into rank-ordered batches, each
    <= budget_chars, never splitting a chunk. A lone chunk over budget gets its
    own batch (the reader truncates it) -- but _stitch() caps chunks at 2600
    chars, so any sane budget keeps whole chunks intact."""
    batches, cur, size = [], [], 0
    for ch in chunks:
        clen = len(ch[1]) + 2
        if cur and size + clen > budget_chars:
            batches.append(cur)
            cur, size = [], 0
        cur.append(ch)
        size += clen
    if cur:
        batches.append(cur)
    return batches


def _grounded(ans: str, batch) -> bool:
    """Default verification signal: is the answer actually present in the text
    the reader saw? A grounded candidate (the value really is in that chunk)
    beats one a weak reader may have hallucinated from a batch that lacked it."""
    a = (ans or "").strip().lower()
    if not a:
        return False
    hay = " ".join(t for _, t, _ in batch).lower()
    return a in hay


def _select(question, cands, verify):
    """Reduce candidate (batch, answer) pairs to one, in rank order.
    With a `verify(question, answer, batch) -> score`, keep the highest score
    (ties -> earliest = best-ranked). Without one, prefer a candidate grounded
    in its own batch, else the first real (non-refusal) answer."""
    real = [(b, a) for b, a in cands if _is_answer(a)]
    if not real:
        return cands[0][1] if cands else ""
    if verify is None:
        for b, a in real:
            if _grounded(a, b):
                return a
        return real[0][1]
    best_a, best_s = real[0][1], None
    for b, a in real:
        s = verify(question, a, b)
        if best_s is None or s > best_s:
            best_a, best_s = a, s
    return best_a


def answer(reader_generate, question, chunks, budget_chars: int,
           verify=None, on_progress=None) -> str:
    """Answer `question` from retrieved `chunks` when they overflow the reader's
    window. `reader_generate(question, batch) -> str` is injected (batch is a
    list of (cid, text, score) chunks), so this engine is model-free and
    unit-testable. `budget_chars` ~ the reader's context in chars minus prompt
    overhead. If everything fits in one batch this is exactly one reader call
    (identical to no loop); otherwise it maps over batches and reduces via
    `_select`. `verify` and `on_progress` are optional."""
    batches = _pack(chunks, budget_chars) if budget_chars else [list(chunks)]
    if len(batches) <= 1:
        return reader_generate(question, chunks)
    n = len(batches)
    cands = []
    for i, batch in enumerate(batches, 1):
        if on_progress:
            on_progress(f"reading batch {i}/{n}")
        cands.append((batch, reader_generate(question, batch)))
    if on_progress:
        on_progress("selecting the answer")
    return _select(question, cands, verify)
