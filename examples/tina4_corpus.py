#!/usr/bin/env python3
"""
Build a Tina4 *programmer reference* corpus — every Tina4 target (python, js,
php, ruby, nodejs, delphi) in ONE groundwire index — and probe it with the kind of
questions a coding assistant actually asks. This is the retrieval side of a
"tina4.com programmer context": the source never enters the model's window; per
question we pull the few relevant chunks and cite the file they came from.

    python examples/tina4_corpus.py                 # build + evaluate
    python examples/tina4_corpus.py --save tina4.idx  # also snapshot the index
    python examples/tina4_corpus.py --ask "how do I define an ORM model?"

Point --repos-root at wherever your Tina4 checkouts live if they are not the
defaults below. Retrieval is lexical (sqlite_fts): flat RAM, instant ingest,
scales past the whole multi-repo corpus, and the snapshot is just data.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from groundwire.pipeline import Groundwire
from groundwire.answer import make_generator
from groundwire.memory_systems import fold

# scope label -> (repo root, optional include-exts override, subdirs to prefer)
# Roots are absolute so the tool works from any cwd. Override with --repos-root.
DEFAULT_ROOT = r"D:\projects"
REPOS = {
    "tina4-python": {"path": ["tina4-python"]},
    "tina4-php":    {"path": ["tina4-php"]},
    "tina4-ruby":   {"path": ["tina4-ruby"]},
    "tina4-nodejs": {"path": ["tina4-nodejs"]},
    # the reactive frontend framework lives under php/tina4/tina4-js; index its
    # hand-written src + examples, not the compiled dist/ bundles (auto-skipped).
    "tina4-js":     {"path": ["php", "tina4", "tina4-js"]},
    # Delphi: .pas/.dpr now flow through the code chunker (pipeline refinement).
    # SKIP_DIRS + dot-dir skipping already drop the vendored .venv and worktrees.
    "tina4delphi":  {"path": ["tina4delphi"]},
}

# (scope, question, anchor phrase we expect to see in a retrieved chunk)
# One representative "how do I / where is" probe per target — the corpus is good
# if the anchor shows up in the top-k AND the top hit is from the right repo.
PROBES = [
    ("tina4-python", "How do I define a route that responds to GET?", "get"),
    ("tina4-python", "How do I define an ORM model class?", "orm"),
    ("tina4-php",    "How is a route registered in tina4 php?", "route"),
    ("tina4-ruby",   "How do I start the tina4 ruby server?", "server"),
    ("tina4-nodejs", "How do I add a route handler in tina4 nodejs?", "route"),
    ("tina4-js",     "How do I create a reactive signal?", "signal"),
    ("tina4-js",     "How do I define a Tina4Element web component?", "component"),
    ("tina4delphi",  "How do I create a REST route with TTina4REST?", "rest"),
    ("tina4delphi",  "How does the HTML render / Frond template work?", "render"),
]


def resolve(root, spec):
    return os.path.join(root, *spec["path"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos-root", default=DEFAULT_ROOT,
                    help="base dir holding the Tina4 checkouts")
    ap.add_argument("--backend", default="sqlite_fts",
                    choices=["bm25", "sqlite_fts", "hybrid"])
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--save", default=None, metavar="PATH",
                    help="snapshot the built index to PATH (reload with --load)")
    ap.add_argument("--load", default=None, metavar="PATH",
                    help="skip building; load a prior snapshot and query it")
    ap.add_argument("--ask", default=None, metavar="QUESTION",
                    help="ad-hoc query against the corpus, then exit")
    ap.add_argument("--reader", action="store_true",
                    help="also answer --ask with an LLM (LLM_URL/LLM_MODEL)")
    args = ap.parse_args()

    reader = make_generator("api") if args.reader else None
    mem = Groundwire(memory=args.backend, reader=reader, k=args.k)

    if args.load:
        mem.load(args.load)
        print(f"loaded corpus: {mem._next:,} chunks, scopes={sorted(mem.repos)}")
    else:
        print("building Tina4 reference corpus (index on CPU/disk — GPU idle):")
        t0 = time.time()
        for scope, spec in REPOS.items():
            root = resolve(args.repos_root, spec)
            if not os.path.isdir(root):
                print(f"  ! {scope:<14} MISSING: {root}  (skipped)")
                continue
            before = mem._next
            # skip design/planning docs (plan/): they describe the framework in
            # the abstract, often mix languages, and crowd out concrete source.
            # A reference corpus wants real code + usage docs, not roadmaps.
            mem.ingest_repo(root, prefix=scope, skip_dirs={"plan"})
            print(f"  + {scope:<14} {mem._next - before:>6,} chunks  <- {root}")
        dt = time.time() - t0
        print(f"corpus: {mem._next:,} chunks across {len(mem.repos)} scopes "
              f"in {dt:.1f}s\n")
        if args.save:
            mem.save(args.save)
            print(f"saved snapshot -> {args.save}\n")

    if args.ask:
        print(f"Q: {args.ask}")
        if reader:
            print(mem.ask(args.ask))
        else:
            for cid, text, score in mem.retrieve(args.ask):
                src = mem.source_of(cid) or "?"
                print(f"  [{score:.3f}] {src}\n      {text[:160].strip()}...")
        return

    # -- evaluate retrieval quality against the per-target probes ------------- #
    right_repo = anchor_ok = 0
    for scope, q, anchor in PROBES:
        hits = mem.retrieve(q, scope=scope)
        srcs = [mem.source_of(cid) or "?" for cid, _, _ in hits]
        top = srcs[0] if srcs else "?"
        in_scope = top.startswith(scope + "/")
        found = any(fold(anchor) in fold(t) for _, t, _ in hits)
        right_repo += in_scope
        anchor_ok += found
        print(f"[{scope}] {q}")
        print(f"   top: {top}  {'OK' if in_scope else 'OFF-SCOPE'} | "
              f"anchor '{anchor}' {'OK' if found else 'MISSING'}")
    n = len(PROBES)
    print(f"\nscorecard: top-1 in correct repo {right_repo}/{n}, "
          f"anchor in top-{args.k} {anchor_ok}/{n}")


if __name__ == "__main__":
    main()
