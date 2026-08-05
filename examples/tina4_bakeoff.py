#!/usr/bin/env python3
"""
tina4_bakeoff.py -- prove groundwire's DIFFERENTIATOR the honest way.

The competitive scan showed the retrieval half is commodity; what no mainstream
code-RAG tool (Continue, Aider, LlamaIndex, Cursor, Cody) reproduces is the
import/symbol VERIFY-AND-CORRECT loop on a WEAK local reader. This benchmark
isolates exactly that, with an OBJECTIVE metric (import-correctness -- does the
generated code reference REAL symbols?), not a judged score.

Claim (a), fully controlled -- same reader, same 30 questions, same retry
budget, vary ONLY the grounding layer:

  0. floor    -- reader alone, NO retrieval          (parametric-knowledge base)
  1. vanilla  -- retrieval + generic self-retry       (commodity code-RAG)
  2. groundwire   -- retrieval + the VERIFIER's specific feedback as correction

Fairness (the BEAM-scar lesson): every condition gets the SAME retry budget, so
we measure the *verifier's feedback*, not extra compute. The verifier is used as
the METRIC for all three; only condition 2 uses it as correction FEEDBACK.

Metric: import-correctness == (check_imports(code, package) == []). AST/symbol
based -> the Python target (tina4-python) is the objectively-verifiable subset;
other language targets fall back to gold-token accuracy (verification n/a).

    # build the corpus once (needs the Tina4 checkouts + is a GPU-box job):
    python examples/tina4_corpus.py --repos-root ~/src --save results/tina4_corpus.idx
    export LLM_URL=http://localhost:11434  LLM_MODEL=qwen2.5:7b   # the WEAK reader
    python examples/tina4_bakeoff.py --repos-root ~/src --budget 3

    # logic check here (network-free, no corpus/LLM/package):
    python examples/tina4_bakeoff.py --self-test
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from groundwire.memory_systems import fold                       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(HERE, "tina4_eval.jsonl")
DEFAULT_IDX = os.path.join(os.path.dirname(HERE), "results", "tina4_corpus.idx")

# Only the Python target can be AST/symbol import-verified. scope -> (package,
# source subpath under --repos-root) for building the symbol map.
VERIFIABLE = {"tina4-python": ("tina4_python", ["tina4-python"])}

# (name, use_context, feedback): the three controlled conditions.
CONDITIONS = [
    ("floor",   False, "generic"),   # reader alone
    ("vanilla", True,  "generic"),   # commodity RAG + generic self-retry
    ("groundwire",  True,  "verifier"),  # RAG + verifier feedback (the differentiator)
]

GENERIC_FB = ("Your previous code may reference imports or symbols that do not "
              "exist in this framework. Re-check every import and API against the "
              "framework and return corrected code only.")


def _code_prompt(question, scope):
    """Force the reader to commit to actual code WITH imports. A weak reader left
    to its own devices answers in prose and never emits an import the verifier can
    check -- so the verify loop can't be exercised at all."""
    lang = "Python" if scope == "tina4-python" else scope
    return (f"Write a complete, minimal {lang} code example -- INCLUDING all the "
            f"necessary import statements -- that does the following, using the "
            f"framework's real APIs:\n{question}\nReturn only code.")


def any_gold(text, gold):
    t = fold(text)
    return next((g for g in gold if fold(g) in t), None)


def run_condition(reader, verify, scope, question, chunks, budget,
                  use_ctx, feedback):
    """Generate an answer under one condition, up to `budget` reader calls.
    Every condition may self-correct `budget` times; only feedback=='verifier'
    gets the verifier's SPECIFIC problems ('X lives in module Y'). Returns
    (code, rounds_used). Stops early only when the verifier reports clean --
    that's convergence, and it can only make groundwire use LESS compute, not more."""
    ctx = chunks if use_ctx else []
    code = reader.generate(_code_prompt(question, scope), ctx)
    rounds = 1
    while rounds < budget:
        if feedback == "verifier":
            probs = verify(code, scope) if verify else None
            if probs is None:            # scope not verifiable -> no loop
                break
            if not probs:                # clean -> converged
                break
            fb = "Problems found by the framework verifier:\n" + "\n".join(probs)
        else:
            fb = GENERIC_FB
        q = (f"{question}\n\nYour previous attempt:\n{code}\n\n{fb}\n\n"
             f"Return corrected code only.")
        code = reader.generate(q, ctx)
        rounds += 1
    return code, rounds


def score(code, scope, gold, verify):
    """(import_clean_or_None, gold_hit). import_clean is None when the scope is
    not import-verifiable (non-Python) -- those count only toward gold accuracy."""
    probs = verify(code, scope) if verify else None
    import_clean = None if probs is None else (len(probs) == 0)
    return import_clean, bool(any_gold(code, gold))


def build_verifier(repos_root):
    """Verifier over the REAL package: builds the symbol map from source, then
    check_imports. Imported from server.py (the canonical verifier -- reading it,
    not modifying it). Returns None for non-Python scopes (can't AST-check)."""
    from groundwire import server as S                            # lazy: real run only
    built = set()

    def verify(code, scope):
        spec = VERIFIABLE.get(scope)
        if not spec:
            return None
        pkg, sub = spec
        if scope not in built:
            root = os.path.join(repos_root, *sub)
            S.SYMBOLS.clear()
            S.build_symbol_map(root, pkg)
            built.add(scope)
        return S.check_imports(code, pkg)

    return verify


def evaluate(data, reader, verify, budget):
    """Run every question through all three conditions; aggregate per condition."""
    agg = {name: {"n": 0, "vn": 0, "clean": 0, "gold": 0, "correct": 0,
                  "uses_pkg": 0, "rounds": 0}
           for name, _, _ in CONDITIONS}
    for it in data:
        scope, q, gold = it["scope"], it["question"], it["gold"]
        pkg = VERIFIABLE.get(scope, (None,))[0]  # "tina4_python" (None if not verifiable)
        chunks = it.get("_chunks", [])           # injected by retrieval (or test)
        for name, use_ctx, fb in CONDITIONS:
            code, rounds = run_condition(reader, verify, scope, q, chunks,
                                         budget, use_ctx, fb)
            clean, gold_hit = score(code, scope, gold, verify)
            a = agg[name]
            a["n"] += 1
            a["rounds"] += rounds
            a["gold"] += gold_hit
            if pkg and pkg in code:
                a["uses_pkg"] += 1
            if clean is not None:
                a["vn"] += 1
                a["clean"] += clean
                # correct-API: used the RIGHT gold symbol AND its imports resolve.
                # A vague answer (no gold) or a wrong-import answer scores 0.
                a["correct"] += 1 if (gold_hit and clean) else 0
    return agg


def report(agg):
    print(f"\n{'condition':<10} {'correct-API':>12} {'import-clean':>13} "
          f"{'gold-acc':>10} {'uses-pkg':>10} {'rounds':>7}")
    print("-" * 66)
    for name, _, _ in CONDITIONS:
        a = agg[name]
        correct = f"{a['correct']}/{a['vn']}" if a["vn"] else "n/a"
        clean = f"{a['clean']}/{a['vn']}" if a["vn"] else "n/a"
        gold = f"{a['gold']}/{a['n']}"
        print(f"{name:<10} {correct:>12} {clean:>13} {gold:>10} "
              f"{str(a['uses_pkg'])+'/'+str(a['n']):>10} {a['rounds']/a['n']:>7.1f}")
    print("-" * 66)
    print("correct-API = used the RIGHT gold symbol AND its imports resolve "
          "(vague/generic answers score 0) -- the discriminating metric.")
    print("uses-pkg    = answers that reference the package at all. If ~0, the")
    print("              reader never commits to imports, so the verify loop")
    print("              has nothing to correct.")
    print("claim (a) holds if:  groundwire > vanilla > floor  on correct-API.")


# --------------------------------------------------------------------------- #
def self_test() -> int:
    """Network-free proof of the three-condition logic + the verify loop, with a
    stub reader (a weak model that only fixes a bad import when handed SPECIFIC
    verifier feedback) and a stub verifier. Expected: floor=0, vanilla=0,
    groundwire=1 on import-clean -- i.e. the verifier's feedback is what fixes it."""

    def stub_verify(code, scope):
        if scope != "tina4-python":
            return None
        return [] if "BadThing" not in code else \
            ["'BadThing' is not in tina4_python.orm; it lives in tina4_python.core"]

    class StubReader:
        """A weak reader: without the verifier's SPECIFIC hint it emits a broken
        import; only when the prompt carries 'lives in tina4_python.core' does it
        correct itself. Generic 'try again' does NOT help it."""
        def generate(self, question, chunks):
            if "lives in tina4_python.core" in question:
                return "from tina4_python.core import Orm  # fixed"
            return "from tina4_python.orm import BadThing"

    data = [{"scope": "tina4-python",
             "question": "Import the ORM base class.",
             "gold": ["Orm"],
             "_chunks": [(0, "tina4_python.core defines Orm", 1.0)]}]
    agg = evaluate(data, StubReader(), stub_verify, budget=3)

    ok = True
    for name, expect in [("floor", 0), ("vanilla", 0), ("groundwire", 1)]:
        got = agg[name]["clean"]
        if got != expect:
            print(f"FAIL: {name} import-clean expected {expect}, got {got}")
            ok = False
    # groundwire must self-correct (2 rounds); vanilla/floor exhaust the budget.
    if agg["groundwire"]["rounds"] != 2:
        print(f"FAIL: groundwire should converge in 2 rounds, "
              f"got {agg['groundwire']['rounds']}"); ok = False
    report(agg)
    print("\nPASS: verifier feedback fixes the import; generic retry does not"
          if ok else "\nSELF-TEST FAILED")
    return 0 if ok else 1


def load_dataset(path):
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("//"):
                items.append(json.loads(line))
    return items


def main() -> int:
    ap = argparse.ArgumentParser(description="groundwire verify-loop bake-off on Tina4")
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--index", default=DEFAULT_IDX)
    ap.add_argument("--repos-root", default=os.path.expanduser("~/src"),
                    help="where the Tina4 checkouts live (for the symbol map)")
    ap.add_argument("--backend", default="sqlite_fts")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--budget", type=int, default=3,
                    help="max reader calls per question (SAME for every condition)")
    ap.add_argument("--scope", default=None,
                    help="only run one scope, e.g. tina4-python (the "
                         "import-verifiable target)")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(HERE), "results", "tina4_bakeoff.json"))
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    from groundwire.pipeline import Groundwire
    from groundwire.answer import make_generator

    if not os.path.exists(args.index):
        sys.exit(f"no corpus at {args.index} -- build it (GPU box):\n"
                 f"  python examples/tina4_corpus.py --repos-root {args.repos_root}"
                 f" --save {args.index}")
    reader = make_generator("api", max_tokens=args.max_tokens)
    mem = Groundwire(memory=args.backend, k=args.k)
    mem.load(args.index)
    verify = build_verifier(args.repos_root)

    data = load_dataset(args.data)
    if args.scope:
        data = [d for d in data if d.get("scope") == args.scope]
    for it in data:                              # attach retrieved context once
        hits = mem.retrieve(it["question"], scope=it["scope"])
        it["_chunks"] = [(cid, f"[{mem.sources.get(cid, '?')}] {t}", s)
                         for cid, t, s in hits]
    print(f"corpus {mem._next:,} chunks · reader {os.environ.get('LLM_MODEL','?')} "
          f"· budget {args.budget} · {len(data)} questions")

    agg = evaluate(data, reader, verify, args.budget)
    report(agg)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
