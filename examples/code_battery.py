#!/usr/bin/env python3
"""
Anchor-scored code-retrieval battery — the harness treatment for the coding
use case. No eyeball scoring: every question carries VERBATIM anchors that
must surface in the retrieved chunks (retrieval recall), the file that should
be cited (source accuracy), and gold substrings for the reader's answer
(answer recall). Anchors are validated against the repos on disk first, so a
stale question fails loudly instead of skewing the score.

    export LLM_URL=... LLM_MODEL=...          # probes (+ reader with --reader)
    python3 examples/code_battery.py                      # retrieval only
    python3 examples/code_battery.py --reader             # + answer scoring
    python3 examples/code_battery.py --expand 0           # pure lexical
    python3 examples/code_battery.py --json results/code_battery.json

Compare runs by diffing the JSON — that's the regression story.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from groundwire.pipeline import Groundwire
from groundwire.answer import APIGenerator
from groundwire.memory_systems import fold
from groundwire.encoders import _post

HOME = os.path.expanduser("~")
REPOS = {
    "tina4-python": f"{HOME}/IdeaProjects/tina4-python",
    "tina4-php": f"{HOME}/IdeaProjects/tina4-php",
    "tina4-js": f"{HOME}/IdeaProjects/tina4-js",
}

# id, repo, question, expected source file (substring), anchors (verbatim in
# retrieved text), gold (substrings expected in the answer)
Q = [
 # -- tina4-python -------------------------------------------------------- #
 ("py-routeref", "tina4-python",
  "What methods can I chain on a route reference (RouteRef) in tina4-python to modify the route?",
  "core/router.py",
  ["def secure(", "def no_auth(", "def cache(", "def middleware("],
  ["secure", "no_auth", "cache", "middleware"]),
 ("py-orm-factories", "tina4-python",
  "List the ORM field factory functions available in tina4-python's orm fields module.",
  "orm/fields.py",
  ["def IntegerField(", "def BlobField(", "def NumericField("],
  ["IntegerField", "StringField", "BooleanField", "FloatField",
   "DateTimeField", "TextField", "BlobField", "NumericField"]),
 ("py-queue-produce", "tina4-python",
  "What exact parameters does Queue.produce accept in tina4-python?",
  "queue/__init__.py",
  ["def produce(self, topic"],
  ["topic", "data", "priority", "delay_seconds", "delay_until"]),
 ("py-queue-consume", "tina4-python",
  "What parameters does Queue.consume accept in tina4-python?",
  "queue/__init__.py",
  ["def consume(self, topic"],
  ["topic", "job_id", "poll_interval"]),
 ("py-queue-env", "tina4-python",
  "Which environment variable selects the queue backend in tina4-python?",
  "queue/",
  ["TINA4_QUEUE_BACKEND"],
  ["TINA4_QUEUE_BACKEND"]),
 ("py-connpool", "tina4-python",
  "How does tina4-python pool database connections?",
  "database/connection.py",
  ["class ConnectionPool"],
  ["ConnectionPool"]),
 ("py-fk-field", "tina4-python",
  "How do I declare a foreign key relationship in the tina4-python ORM?",
  "orm/fields.py",
  ["class ForeignKeyField"],
  ["ForeignKeyField"]),
 ("py-websocket", "tina4-python",
  "How do I register a websocket route in tina4-python?",
  "core/router.py",
  ["def websocket(cls, path"],
  ["websocket", "path", "handler"]),
 ("py-group", "tina4-python",
  "How do I group routes under a common prefix with middleware in tina4-python?",
  "core/router.py",
  ["def group(cls, prefix"],
  ["group", "prefix", "middleware"]),
 ("py-verbs", "tina4-python",
  "Which HTTP verb methods does the Router expose in tina4-python?",
  "core/router.py",
  ["def put(", "def patch(", "def delete("],
  ["put", "patch", "delete", "any"]),
 ("py-migration", "tina4-python",
  "Which class runs database migrations in tina4-python and what method executes them?",
  "migration/runner.py",
  ["class Migration:", "def migrate(self)"],
  ["Migration", "migrate"]),
 ("py-frond-render", "tina4-python",
  "How do I render a Frond template with data in tina4-python?",
  "frond/engine.py",
  ["def render(self, template: str"],
  ["render", "template", "data"]),
 ("py-auth-token", "tina4-python",
  "How do I create a JWT token with the Auth class in tina4-python?",
  "auth/__init__.py",
  ["def get_token(self, payload"],
  ["get_token", "payload", "expires_in"]),
 ("py-auth-valid", "tina4-python",
  "How do I validate a bearer token in tina4-python?",
  "auth/__init__.py",
  ["def valid_token(self, token"],
  ["valid_token"]),
 ("py-orm-save", "tina4-python",
  "How do I persist an ORM model instance to the database in tina4-python?",
  "orm/model.py",
  ["def save(self)"],
  ["save"]),
 ("py-orm-load", "tina4-python",
  "What parameters does the ORM model load method take in tina4-python?",
  "orm/model.py",
  ["def load(self, filter"],
  ["filter", "params"]),
 ("py-orm-select", "tina4-python",
  "How do I select records with pagination in the tina4-python ORM?",
  "orm/model.py",
  ["def select(cls, sql"],
  ["select", "limit", "offset"]),
 # -- tina4-php ----------------------------------------------------------- #
 ("php-router-add", "tina4-php",
  "What is the exact signature of Router::add in tina4-php?",
  "Tina4/Router.php",
  ["public static function add(string $method"],
  ["$method", "$path", "$handler", "$middleware", "$swaggerMeta",
   "$template", "self"]),
 ("php-router-get", "tina4-php",
  "How do I define a GET route in tina4-php using the current Router class?",
  "Tina4/Router.php",
  ["public static function get(string $path"],
  ["Router::get"]),
 ("php-auth-gettoken", "tina4-php",
  "How do I create a token with the Auth class in tina4-php?",
  "Tina4/Auth.php",
  ["public static function getToken(array $payload"],
  ["getToken", "payload"]),
 ("php-auth-validtoken", "tina4-php",
  "How do I validate a token in tina4-php?",
  "Tina4/Auth.php",
  ["public static function validToken(string $token"],
  ["validToken"]),
 ("php-orm-class", "tina4-php",
  "What base class do ORM models extend in tina4-php?",
  "Tina4/ORM.php",
  ["abstract class ORM"],
  ["ORM"]),
 # -- tina4-js ------------------------------------------------------------ #
 ("js-signal", "tina4-js",
  "How do I create a signal in tina4-js and what is the exact function signature?",
  "src/core/signal.ts",
  ["export function signal<T>(initial: T, label?: string)"],
  ["initial", "label", "Signal"]),
 ("js-computed", "tina4-js",
  "How do I create a computed (derived) signal in tina4-js?",
  "src/core/signal.ts",
  ["export function computed<T>(fn: () => T)"],
  ["computed"]),
 ("js-effect", "tina4-js",
  "How do I run a side effect when signals change in tina4-js?",
  "src/core/signal.ts",
  ["export function effect(fn: () => void)"],
  ["effect"]),
 ("js-route", "tina4-js",
  "How do I register a client-side route in tina4-js?",
  "src/router/router.ts",
  ["export function route(pattern: string"],
  ["route", "pattern"]),
 ("js-navigate", "tina4-js",
  "How do I navigate programmatically in tina4-js?",
  "src/router/router.ts",
  ["export function navigate(path: string"],
  ["navigate"]),
]

SYSTEM = ("You are a coding assistant. Answer using the provided source-code "
          "context from the user's installed libraries. Prefer the context "
          "over general knowledge. Cite file paths. Be concise and exact.")


def validate(questions):
    """Confirm every anchor exists verbatim (folded) somewhere in its repo,
    so the battery can't silently rot when the repos change."""
    valid, bad, cache = [], [], {}
    for item in questions:
        qid, repo, _, _, anchors, _ = item
        if repo not in cache:
            buf = []
            for dirpath, dirnames, filenames in os.walk(REPOS[repo]):
                dirnames[:] = [d for d in dirnames
                               if d not in Groundwire.SKIP_DIRS
                               and not d.startswith(".")]
                for fn in filenames:
                    if os.path.splitext(fn)[1].lower() in (
                            Groundwire.CODE_EXTS | Groundwire.DOC_EXTS):
                        try:
                            buf.append(open(os.path.join(dirpath, fn),
                                            encoding="utf-8",
                                            errors="ignore").read())
                        except OSError:
                            pass
            cache[repo] = fold("\n".join(buf))
        missing = [a for a in anchors if fold(a) not in cache[repo]]
        if missing:
            bad.append((qid, missing))
        else:
            valid.append(item)
    return valid, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="bm25")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--expand", type=int, default=3)
    ap.add_argument("--rerank", default=None, help="'dense' to rerank the pool")
    ap.add_argument("--reader", action="store_true",
                    help="also generate + score answers (uses LLM_URL)")
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--json", default=None, help="write per-question results")
    ap.add_argument("--repos", nargs="*", default=None,
                    help="restrict to these repo labels (isolation testing: "
                         "does a cohort fail on its own, or only in the mix?)")
    args = ap.parse_args()

    global REPOS
    questions = Q
    if args.repos:
        REPOS = {k: v for k, v in REPOS.items() if k in args.repos}
        questions = [q for q in Q if q[1] in REPOS]

    valid, bad = validate(questions)
    for qid, missing in bad:
        print(f"!! {qid}: anchors not found in repo, EXCLUDED: {missing}")

    mem = Groundwire(memory=args.backend, k=args.k, expand=args.expand,
                  rerank=args.rerank,
                  encoder="openai:nomic-embed-text" if args.rerank else None)
    t0 = time.time()
    for label, root in REPOS.items():
        mem.ingest_repo(root, prefix=label)
    print(f"ingested {mem._next:,} chunks from {len(REPOS)} repos "
          f"in {time.time()-t0:.1f}s | backend={args.backend} "
          f"expand={args.expand} k={args.k}\n")

    reader = APIGenerator(max_tokens=args.max_tokens) if args.reader else None
    rows, out = [], []
    for qid, repo, question, file_sub, anchors, gold in valid:
        t0 = time.time()
        hits = mem.retrieve(question)
        r_ms = (time.time() - t0) * 1000
        folded = [fold(t) for _, t, _ in hits]
        a_found = sum(1 for a in anchors if any(fold(a) in f for f in folded))
        src = mem.source_of(hits[0][0]) if hits else ""
        src_hit = file_sub in (src or "")
        ans_score, answer, g_ms = None, None, None
        if reader:
            ctx = "\n\n".join(f"[{mem.source_of(cid) or cid}]\n{t}"
                              for cid, t, _ in hits)
            t0 = time.time()
            out_llm = _post(reader.url, {
                "model": reader.model,
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user",
                     "content": f"Context:\n\n{ctx}\n\nQuestion: {question}"},
                ],
                "max_tokens": args.max_tokens, "temperature": 0,
            }, headers={"Authorization": f"Bearer {reader.api_key}"}
                if reader.api_key else {}, timeout=reader.timeout)
            answer = out_llm["choices"][0]["message"]["content"]
            g_ms = (time.time() - t0) * 1000
            fa = fold(answer)
            ans_score = sum(1 for g in gold if fold(g) in fa) / len(gold)
        rows.append((qid, a_found, len(anchors), src_hit, ans_score, r_ms, g_ms))
        out.append({"id": qid, "repo": repo, "anchors_found": a_found,
                    "anchors_total": len(anchors), "src_hit": src_hit,
                    "src": src, "answer_score": ans_score,
                    "retrieve_ms": round(r_ms, 1),
                    "generate_ms": round(g_ms, 1) if g_ms else None,
                    "answer": answer})
        mark = "OK " if a_found == len(anchors) and src_hit else \
               ("part" if a_found else "MISS")
        extra = f" ans={ans_score:.2f}" if ans_score is not None else ""
        print(f"{mark} {qid:22} anchors {a_found}/{len(anchors)} "
              f"src={'Y' if src_hit else 'n'}{extra}  [{r_ms:5.0f}ms]",
              flush=True)

    n = len(rows)
    full = sum(1 for r in rows if r[1] == r[2] and r[3])
    any_a = sum(1 for r in rows if r[1] > 0)
    srcs = sum(1 for r in rows if r[3])
    avg_r = sum(r[5] for r in rows) / n
    print(f"\nscorecard: full-hit {full}/{n} | any-anchor {any_a}/{n} | "
          f"src-top1 {srcs}/{n} | avg retrieve {avg_r:.0f}ms")
    if args.reader:
        scored = [r[4] for r in rows if r[4] is not None]
        avg_g = sum(r[6] for r in rows if r[6]) / len(scored)
        print(f"answer recall: {sum(scored)/len(scored):.2f} avg | "
              f"avg generate {avg_g:.0f}ms")
    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        json.dump({"config": vars(args), "results": out},
                  open(args.json, "w"), indent=1)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
