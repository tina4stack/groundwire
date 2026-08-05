#!/usr/bin/env python3
"""
beam_eval.py -- test groundwire's RETRIEVAL on the BEAM conversational-memory
benchmark (Mohammadta/BEAM), the honest way.

BEAM's headline end-to-end numbers mix retrieval with an LLM judge AND with
per-ability oracle bypasses (regex date-math, negation injection, a
context->answer side-index). Those bypasses are a rule-engine for BEAM's ability
types, not memory -- and the judge is explicitly "not comparable across judges".
So chasing their 65% means rebuilding their product.

What IS a fair, reproducible fight for a retrieval layer is their Pure-Recall
axis: does the memory surface the evidence a question needs? We measure that
directly and deterministically -- no LLM, no judge, GPU-free:

    recall(question) = (# of the question's rubric facts whose content appears
                        in groundwire's top-k retrieved chunks) / (# rubric facts)

Aggregated per BEAM ability and overall. The honest bar to beat is BEAM's own
"RAG" line (32.3% end-to-end at 100K), comfortably -- not the 65% full-system.

    # real run (needs `datasets` + network; do this on the GPU box):
    python3 examples/beam_eval.py --scales 100K --sample 0 --k 6

    # logic check here (network-free, synthetic fixture):
    python3 examples/beam_eval.py --self-test

Only stdlib + groundwire (+ `datasets` for the real dataset). A reader+judge
end-to-end mode is intentionally out of scope for this file -- run that on the
GPU box against a real reader.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from groundwire.pipeline import Groundwire                       # noqa: E402

try:                                                     # match the index tokenizer
    from groundwire.memory_systems import terms as _terms    # noqa: E402
except Exception:                                        # pragma: no cover
    import re as _re
    def _terms(text):
        return _re.findall(r"[a-z0-9]+", (text or "").lower())


def _toks(text) -> set:
    """Folded content-token set, using groundwire's own tokenizer so the metric
    lines up with what the index actually matched on."""
    try:
        return set(_terms(text or ""))
    except Exception:
        return set(str(text or "").lower().split())


# --------------------------------------------------------------------------- #
# Schema handling -- BEAM rows vary (chat vs messages; probing_questions as a
# dict or an ast.literal_eval-able string). Be defensive on both.
# --------------------------------------------------------------------------- #
def _messages(row) -> list:
    """Return [(role, content), ...] from a BEAM row's chat/messages field.

    BEAM's `chat` is a list of SESSIONS, each itself a list of message dicts
    ({role, content, ...}) -- that nesting is the 'multi-session' structure.
    Flatten it. Also tolerate a flat list of dicts, or [role, content] pairs."""
    raw = row.get("chat") or row.get("messages") or []
    if isinstance(raw, str):
        try:
            raw = ast.literal_eval(raw)
        except Exception:
            return [("user", raw)]
    out = []

    def _add(m):
        if isinstance(m, dict):
            out.append((str(m.get("role", "user")), str(m.get("content", ""))))
        elif isinstance(m, (list, tuple)) and len(m) >= 2 \
                and not isinstance(m[0], (list, dict)):
            out.append((str(m[0]), str(m[1])))

    for item in raw:
        if isinstance(item, list):        # a session -> iterate its messages
            for m in item:
                _add(m)
        else:                             # already a message (dict/pair)
            _add(item)
    return out


def _probes(row) -> list:
    """Return [{ability, question, facts:[...]}, ...] from probing_questions.

    Each question object carries `rubric` (a list of expected facts) plus
    `ideal_answer`/`ideal_response`. We score recall against the rubric facts;
    if a probe has no rubric we fall back to the ideal answer as one fact."""
    pq = row.get("probing_questions")
    if isinstance(pq, str):
        try:
            pq = ast.literal_eval(pq)
        except Exception:
            return []
    probes = []

    def _emit(ability, obj):
        if not isinstance(obj, dict):
            return
        q = obj.get("question") or obj.get("probe") or ""
        rubric = obj.get("rubric") or obj.get("facts") or []
        if isinstance(rubric, str):
            rubric = [rubric]
        facts = [str(f) for f in rubric if str(f).strip()]
        if not facts:
            ideal = obj.get("ideal_answer") or obj.get("ideal_response") or ""
            if ideal:
                facts = [str(ideal)]
        if q and facts:
            probes.append({"ability": ability, "question": q, "facts": facts})

    if isinstance(pq, dict):                     # {ability: qobj | [qobj, ...]}
        for ability, v in pq.items():
            if isinstance(v, list):
                for obj in v:
                    _emit(ability, obj)
            else:
                _emit(ability, v)
    elif isinstance(pq, list):                   # [qobj, ...] with ability tags
        for obj in pq:
            _emit(obj.get("ability", "?") if isinstance(obj, dict) else "?", obj)
    return probes


# --------------------------------------------------------------------------- #
def fact_recalled(fact: str, pool_tokens: set, hit: float) -> bool:
    """A rubric fact counts as recalled if at least `hit` of its content tokens
    appear in the retrieved pool -- a deterministic proxy for 'the evidence for
    this fact was surfaced'."""
    ft = _toks(fact)
    if not ft:
        return False
    need = max(1, math.ceil(hit * len(ft)))
    return len(ft & pool_tokens) >= need


def _embedder(url: str, model: str):
    """Batched, cached embedding client (stdlib HTTP; OpenAI-compat /v1/embeddings,
    e.g. local Ollama nomic-embed-text). Returns unit-normalized vectors so a
    dot product IS cosine. Used by the SEMANTIC scorer, which removes the
    token-overlap metric's structural bias toward lexical retrieval."""
    import urllib.request
    from groundwire.veclite import l2_normalize     # pure stdlib -- no numpy
    cache = {}

    def embed(texts):
        need = [t for t in dict.fromkeys(texts) if t not in cache]
        for i in range(0, len(need), 64):
            batch = need[i:i + 64]
            body = json.dumps({"model": model, "input": batch}).encode()
            req = urllib.request.Request(url, data=body, headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer ollama"})
            data = json.loads(urllib.request.urlopen(req, timeout=180).read())
            for t, item in zip(batch, data["data"]):
                cache[t] = l2_normalize([float(x) for x in item["embedding"]])
        return [cache[t] for t in texts]

    return embed


def make_scorer(metric: str, hit: float, embed=None, tau: float = 0.6):
    """Build a `scorer(fact, chunk_texts) -> bool` for the chosen metric.

    overlap  -- deterministic: >= `hit` of the fact's content tokens appear in
                the retrieved chunks. Favors LEXICAL retrieval (token match).
    semantic -- a fact counts if max cosine(embed(fact), embed(chunk)) >= `tau`
                over the top-k chunks. Paraphrase-robust; no lexical home field.
    Same scorer is applied to every engine, so the number reflects the engine."""
    if metric == "semantic":
        def score(fact, chunks):
            if not chunks:
                return False
            vecs = embed([fact] + list(chunks))
            fv = vecs[0]
            return any(float(fv @ cv) >= tau for cv in vecs[1:])
        return score

    def score(fact, chunks):
        pool = set()
        for c in chunks:
            pool |= _toks(c)
        return fact_recalled(fact, pool, hit)
    return score


def _mnemosyne_store(msgs):
    """Ingest a conversation into a FRESH mnemosyne store and return it.
    `extract=False` keeps storage embedding-only (no LLM); their recall() is a
    hybrid dense+FTS+keyword voice ensemble. Lets us score THEIR retrieval with
    OUR identical metric -- the only thing that differs is the engine."""
    import tempfile
    import pathlib
    from mnemosyne.core.memory import Mnemosyne
    db = pathlib.Path(tempfile.mkdtemp()) / "m.db"
    m = Mnemosyne(db_path=db)
    for role, content in msgs:
        m.remember(f"{role}: {content}", extract=False, extract_entities=False)
    return m


def eval_row(row, k: int, backend: str, scorer,
             rerank=None, encoder=None, engine: str = "groundwire") -> list:
    """Index one conversation with the chosen ENGINE, retrieve top-k per probe,
    and score each rubric fact with `scorer(fact, chunk_texts)`. Returns
    [(ability, recall_fraction), ...].

    engine="groundwire"    -> lexical (+ optional dense rerank via encoder/EMBED_URL)
    engine="mnemosyne" -> their hybrid dense+FTS+keyword recall()

    Retrieval + scoring are identical across engines, so the number reflects the
    retrieval engine, not the metric."""
    msgs = _messages(row)
    probes = _probes(row)
    if not msgs or not probes:
        return []

    if engine == "mnemosyne":
        store = _mnemosyne_store(msgs)

        def _chunks(q):
            return [h.get("content", "") for h in store.recall(q, top_k=k)]
    else:
        mem = Groundwire(memory=backend, k=k, rerank=rerank, encoder=encoder)
        mem.ingest([(f"msg{i}:{role}", f"{role}: {content}")
                    for i, (role, content) in enumerate(msgs)])

        def _chunks(q):
            return [text for _cid, text, _score in mem.retrieve(q, k=k)]

    scored = []
    for p in probes:
        chunks = _chunks(p["question"])
        n = sum(1 for f in p["facts"] if scorer(f, chunks))
        scored.append((p["ability"], n / len(p["facts"])))
    return scored


def aggregate(scored: list) -> dict:
    """Per-ability mean recall + overall mean over all probes."""
    by = {}
    for ability, r in scored:
        by.setdefault(ability, []).append(r)
    per = {a: sum(v) / len(v) for a, v in by.items()}
    overall = sum(r for _a, r in scored) / len(scored) if scored else 0.0
    return {"overall": overall, "per_ability": per, "n_probes": len(scored)}


def run(scale: str, sample: int, k: int, backend: str, scorer,
        rerank=None, encoder=None, engine: str = "groundwire") -> dict:
    from datasets import load_dataset            # imported lazily (network dep)
    try:
        ds = load_dataset("Mohammadta/BEAM", split=scale)
    except Exception:                            # fall back to config/split guess
        ds = load_dataset("Mohammadta/BEAM")[scale]
    rows = list(ds)
    if sample:
        rows = rows[:sample]
    scored = []
    for i, row in enumerate(rows):
        scored += eval_row(row, k, backend, scorer, rerank=rerank,
                           encoder=encoder, engine=engine)
        print(f"  [{scale}] conv {i + 1}/{len(rows)}  "
              f"probes so far: {len(scored)}", file=sys.stderr)
    return aggregate(scored)


# --------------------------------------------------------------------------- #
def self_test() -> int:
    """Network-free proof the harness ingests, retrieves, and scores. Uses a
    synthetic BEAM-shaped row: facts that ARE in the conversation must recall
    high; an abstention probe whose facts are ABSENT must recall low."""
    row = {
        "chat": [
            {"role": "user", "content": "I moved to Lisbon in March 2021."},
            {"role": "assistant", "content": "Nice, Lisbon is lovely in spring."},
            {"role": "user", "content": "My cat is named Basil and he is a tabby."},
            {"role": "assistant", "content": "Basil the tabby sounds delightful."},
            {"role": "user", "content": "I switched jobs to a fintech in 2023."},
        ],
        "probing_questions": {
            "information_extraction": {
                "question": "What is the user's cat's name and breed?",
                "rubric": ["the cat is named Basil", "Basil is a tabby"],
            },
            "temporal_reasoning": {
                "question": "When did the user move to Lisbon?",
                "rubric": ["moved to Lisbon in March 2021"],
            },
            "abstention": {
                "question": "What is the user's favourite programming language?",
                "rubric": ["the user's favourite language is Rust"],  # NOT in convo
            },
        },
    }
    scored = eval_row(row, k=4, backend="sqlite_fts",
                      scorer=make_scorer("overlap", 0.6))
    agg = aggregate(scored)
    per = agg["per_ability"]
    print(json.dumps(agg, indent=2))

    ok = True
    if per.get("information_extraction", 0) < 0.99:
        print("FAIL: in-corpus IE facts should recall ~1.0"); ok = False
    if per.get("temporal_reasoning", 0) < 0.99:
        print("FAIL: in-corpus temporal fact should recall ~1.0"); ok = False
    if per.get("abstention", 1) > 0.34:
        print("FAIL: absent fact should NOT be recalled"); ok = False
    print("PASS: harness ingests, retrieves, and scores recall correctly"
          if ok else "SELF-TEST FAILED")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="groundwire retrieval-recall on BEAM")
    ap.add_argument("--scales", default="100K",
                    help="comma list of BEAM splits: 100K,500K,1M")
    ap.add_argument("--sample", type=int, default=0,
                    help="conversations per scale (0 = all)")
    ap.add_argument("--k", type=int, default=6, help="top-k retrieved")
    ap.add_argument("--hit", type=float, default=0.6,
                    help="overlap metric: fraction of a fact's tokens required")
    ap.add_argument("--metric", default="overlap",
                    choices=["overlap", "semantic"],
                    help="overlap=token match (favors lexical); "
                         "semantic=embed fact+chunk cosine (paraphrase-robust)")
    ap.add_argument("--embed-url",
                    default="http://localhost:11434/v1/embeddings",
                    help="OpenAI-compat embeddings endpoint for --metric semantic")
    ap.add_argument("--embed-model", default="nomic-embed-text")
    ap.add_argument("--sem-tau", type=float, default=0.6,
                    help="cosine threshold for a semantic fact hit")
    ap.add_argument("--engine", default="groundwire",
                    choices=["groundwire", "mnemosyne"],
                    help="retrieval engine under test (same metric for both)")
    ap.add_argument("--backend", default="sqlite_fts")
    ap.add_argument("--rerank", default=None,
                    help='set "dense" to reorder the lexical pool by cached '
                         'dense cosine (needs --encoder + EMBED_URL)')
    ap.add_argument("--encoder", default=None,
                    help='e.g. "openai:nomic-embed-text" (with EMBED_URL set)')
    ap.add_argument("--out", default="results/beam_recall.json")
    ap.add_argument("--self-test", action="store_true",
                    help="run the network-free synthetic-fixture check and exit")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    embed = (_embedder(args.embed_url, args.embed_model)
             if args.metric == "semantic" else None)
    scorer = make_scorer(args.metric, args.hit, embed, args.sem_tau)

    results = {}
    for scale in [s.strip() for s in args.scales.split(",") if s.strip()]:
        tag = (f"sem-tau={args.sem_tau}" if args.metric == "semantic"
               else f"hit={args.hit}")
        print(f"== BEAM {scale} (engine={args.engine}, metric={args.metric}, "
              f"{tag}, k={args.k}) ==", file=sys.stderr)
        results[scale] = run(scale, args.sample, args.k, args.backend, scorer,
                             rerank=args.rerank, encoder=args.encoder,
                             engine=args.engine)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
