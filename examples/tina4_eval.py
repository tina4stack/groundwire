#!/usr/bin/env python3
"""
Prove groundwire accuracy on the Tina4 reference corpus with a LOCAL LLM reader.

This is an end-to-end retrieval-QA benchmark over real Tina4 source (six
language targets). The corpus never enters the model's window; per question we
retrieve the few relevant chunks and hand them to your local model (Ollama /
vLLM -- OpenAI-compatible). Three numbers come out:

  * retrieval recall : gold anchor phrase present in the top-k chunks
                       (the ceiling on any answer -- can the reader even see it?)
  * grounding        : top-1 chunk came from the correct repo/scope
  * answer accuracy  : a gold token appears in the model's answer -- exact
                       string match (a floor: correct paraphrases miss it)
  * adjudicated acc  : exact match, with an LLM judge as a LAST RESORT only on
                       the answers exact-match missed (--judge) -- so it costs
                       one extra LLM call per miss, not per question

Dataset: examples/tina4_eval.jsonl -- one JSON object per line with keys
{id, scope, question, gold:[...], anchor}. Gold tokens and anchors are verified
against the actual framework source.

    # 1) build the corpus snapshot once
    python examples/tina4_corpus.py --save results/tina4_corpus.idx
    # 2) point at your local model and run the eval
    export LLM_URL=http://localhost:11434  LLM_MODEL=llama3.2:3b
    python examples/tina4_eval.py                 # exact-match only (fastest)
    python examples/tina4_eval.py --no-reader     # retrieval-only (no LLM)
    python examples/tina4_eval.py --judge         # judge ONLY the misses (cheap)
    python examples/tina4_eval.py --judge-all     # judge every answer (slow; also
                                                  #   catches exact-match false positives)
    JUDGE_MODEL=qwen2.5:7b python examples/tina4_eval.py --judge   # stronger judge
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from groundwire.pipeline import Groundwire
from groundwire.answer import make_generator
from groundwire.memory_systems import fold

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(HERE, "tina4_eval.jsonl")
DEFAULT_IDX = os.path.join(os.path.dirname(HERE), "results", "tina4_corpus.idx")


def load_dataset(path):
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("//"):
                items.append(json.loads(line))
    return items


def any_gold(text, gold):
    """Case/whitespace-insensitive: does any gold token appear in text?"""
    t = fold(text)
    return next((g for g in gold if fold(g) in t), None)


_JUDGE_SYS = (
    "You are a STRICT grader for a coding-assistant benchmark. Given a question "
    "about a specific framework, the expected key identifiers, and a candidate "
    "answer, decide if the answer is correct. Mark CORRECT only if the answer "
    "(a) is about the NAMED framework and language -- not a different one -- and "
    "(b) names the right API/approach: the expected identifiers OR a genuinely "
    "equivalent correct one for that framework. Mark INCORRECT if it is vague, "
    "wrong, or written for the wrong language/framework. Reply with exactly one "
    "word: CORRECT or INCORRECT."
)


def judge_answer(post, url, model, scope, question, gold, answer, timeout=120):
    """LLM-as-judge verdict for one answer. Returns (is_correct, raw_verdict).
    Replaces brittle exact gold-token matching with a semantic grade, so a
    correct paraphrase ('inherits from the ORM base class') is not scored a
    miss. Deterministic (temperature 0). Uses the same OpenAI-compatible client
    as the reader, so it works against Ollama/vLLM with no new dependency."""
    user = (f"Framework: {scope}\nQuestion: {question}\n"
            f"Expected key identifiers (an equivalent correct API is also fine): "
            f"{', '.join(gold)}\nCandidate answer:\n{answer}\n\nVerdict:")
    out = post(url, {
        "model": model,
        "messages": [{"role": "system", "content": _JUDGE_SYS},
                     {"role": "user", "content": user}],
        "max_tokens": 8, "temperature": 0,
    }, timeout=timeout)
    verdict = out["choices"][0]["message"]["content"].strip()
    up = verdict.upper()
    # "INCORRECT" contains "CORRECT", so test the negative first.
    is_correct = "INCORRECT" not in up and "CORRECT" in up
    return is_correct, verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--index", default=DEFAULT_IDX,
                    help="prebuilt corpus snapshot (from tina4_corpus.py --save)")
    ap.add_argument("--backend", default="sqlite_fts")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--no-reader", action="store_true",
                    help="score retrieval only; skip the LLM answer check")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(HERE), "results", "tina4_eval.json"))
    ap.add_argument("--show", action="store_true",
                    help="print each question's retrieved sources + answer")
    ap.add_argument("--name-scope", action="store_true",
                    help="tell the reader which Tina4 target the question is "
                         "about (a real assistant knows its framework). "
                         "Isolates reader disambiguation from retrieval.")
    ap.add_argument("--judge", action="store_true",
                    help="LAST-RESORT LLM judge: trust the cheap exact gold-token "
                         "match, and only invoke the judge on answers that MISS "
                         "(where correct-paraphrase artifacts hide). One extra "
                         "LLM call per miss, not per question.")
    ap.add_argument("--judge-all", action="store_true",
                    help="thorough (slow): judge EVERY answer, so the judge can "
                         "also overturn exact-match FALSE POSITIVES (a gold token "
                         "present in an otherwise-wrong answer). One extra LLM "
                         "call per question.")
    ap.add_argument("--judge-model", default=None,
                    help="judge model (or JUDGE_MODEL env; defaults to the "
                         "reader model -- prefer a DIFFERENT/stronger model to "
                         "avoid self-grading bias)")
    ap.add_argument("--judge-url", default=None,
                    help="judge endpoint base (or JUDGE_URL env; defaults to "
                         "LLM_URL)")
    args = ap.parse_args()

    data = load_dataset(args.data)
    reader = None if args.no_reader else make_generator("api", max_tokens=args.max_tokens)
    mem = Groundwire(memory=args.backend, reader=reader, k=args.k)

    if os.path.exists(args.index):
        mem.load(args.index)
        src = f"loaded {args.index}"
    else:
        sys.exit(f"no corpus at {args.index} -- run: "
                 f"python examples/tina4_corpus.py --save {args.index}")
    model = os.environ.get("LLM_MODEL", "?") if reader else "(no reader)"
    print(f"corpus: {mem._next:,} chunks, {len(mem.repos)} scopes ({src})")
    print(f"reader: {model}   dataset: {len(data)} questions")

    # LLM judge (optional). Reuses the reader's stdlib HTTP client. Defaults to
    # the reader model/endpoint, but a distinct judge is preferred. Two modes:
    # last-resort (--judge, only adjudicate exact-match misses -- fast) and
    # thorough (--judge-all, grade every answer -- also catches false positives).
    judge_all = args.judge_all and reader is not None
    use_judge = (args.judge or judge_all) and reader is not None
    judge_post = judge_url = judge_model = None
    if use_judge:
        from groundwire.encoders import _post as judge_post
        base = (args.judge_url or os.environ.get("JUDGE_URL")
                or os.environ.get("LLM_URL") or "http://localhost:8000").rstrip("/")
        judge_url = base if base.endswith("completions") else base + "/v1/chat/completions"
        judge_model = args.judge_model or os.environ.get("JUDGE_MODEL") or model
        mode = "every answer" if judge_all else "only exact-match misses"
        print(f"judge : {judge_model}  ({mode})")
    print()

    recall = grounded = answered = adjudicated = 0
    judge_calls = 0
    per_scope = {}
    rows = []
    for it in data:
        scope, q, gold = it["scope"], it["question"], it["gold"]
        anchor = it.get("anchor", "")
        t0 = time.time()
        hits = mem.retrieve(q, scope=scope)
        r_ms = (time.time() - t0) * 1000
        srcs = [mem.source_of(cid) or "?" for cid, _, _ in hits]
        top = srcs[0] if srcs else "?"
        in_scope = top.startswith(scope + "/")
        # retrieval recall = is the answer actually IN the retrieved chunks?
        # A gold token present in a top-k chunk means the reader can see the
        # answer. (The exact anchor phrase lives in one source file and is a
        # far stricter probe -- kept as a diagnostic, not the recall metric,
        # since docs legitimately answer 'how do I' questions too.)
        chunk_text = " ".join(t for _, t, _ in hits)
        found = bool(any_gold(chunk_text, gold)) \
            or (bool(anchor) and fold(anchor) in fold(chunk_text))
        anchor_file = bool(anchor) and fold(anchor) in fold(chunk_text)

        ans, hit_tok, a_ms = "", None, 0.0
        if reader is not None:
            # retrieval always uses the bare question (scope inferred). Only the
            # reader optionally gets the framework name -- that's the realistic
            # setting and it isolates 'can the reader use the right chunks' from
            # 'can retrieval find them'.
            rq = f"In the {scope} framework: {q}" if args.name_scope else q
            hits_lbl = [(cid, f"[{mem.sources[cid]}] {t}" if cid in mem.sources
                        else t, s) for cid, t, s in hits]
            t0 = time.time()
            ans = reader.generate(rq, hits_lbl)
            a_ms = (time.time() - t0) * 1000
            hit_tok = any_gold(ans, gold)

        # Judge only when it can change the verdict: in last-resort mode that's
        # exact-match MISSES (a miss might be a correct paraphrase); in thorough
        # mode, every answer (a hit might be a false positive).
        judge_ok, judge_verdict = None, None
        if use_judge and ans and (judge_all or not hit_tok):
            judge_ok, judge_verdict = judge_answer(
                judge_post, judge_url, judge_model, scope, q, gold, ans)
            judge_calls += 1

        # final adjudicated correctness. Thorough: judge is authoritative.
        # Last-resort: keep exact hits, rescue misses the judge approves.
        if judge_all:
            final_ok = bool(judge_ok)
        elif use_judge:
            final_ok = bool(hit_tok) or bool(judge_ok)
        else:
            final_ok = bool(hit_tok)

        recall += found
        grounded += in_scope
        answered += bool(hit_tok)
        adjudicated += final_ok
        st = per_scope.setdefault(scope, {"n": 0, "recall": 0, "ground": 0,
                                          "ans": 0, "adj": 0})
        st["n"] += 1
        st["recall"] += found
        st["ground"] += in_scope
        st["ans"] += bool(hit_tok)
        st["adj"] += final_ok

        rows.append({"id": it.get("id"), "scope": scope, "question": q,
                     "gold": gold, "anchor": anchor, "top_source": top,
                     "grounded": in_scope, "retrieval_hit": found,
                     "anchor_file_hit": anchor_file,
                     "answer": ans, "answer_hit": hit_tok,
                     "judge_hit": judge_ok, "judge_verdict": judge_verdict,
                     "final_correct": final_ok,
                     "retrieve_ms": round(r_ms, 1), "answer_ms": round(a_ms, 1)})

        flag = "OK " if (found and (reader is None or hit_tok)) else "-- "
        print(f"{flag}[{scope}] {q}")
        print(f"     top: {top}  {'' if in_scope else '(off-scope) '}"
              f"| retrieval {'HIT' if found else 'MISS'}", end="")
        if reader is not None:
            j = "" if judge_ok is None else \
                f" | judge {'OK' if judge_ok else 'NO'}"
            print(f" | answer {'HIT('+hit_tok+')' if hit_tok else 'MISS'}{j}"
                  f"  [{a_ms:.0f}ms]")
            if args.show and judge_verdict:
                print(f"     judge: {judge_verdict}")
            if args.show:
                print(f"     A: {ans.strip()[:200]}")
        else:
            print()

    n = len(data)
    print(f"\n{'='*60}")
    print(f"retrieval recall : {recall}/{n}  ({100*recall/n:.0f}%)")
    print(f"grounding (top-1): {grounded}/{n}  ({100*grounded/n:.0f}%)")
    if reader is not None:
        print(f"answer accuracy  : {answered}/{n}  ({100*answered/n:.0f}%)"
              f"   [exact gold-token match, reader: {model}]")
    if use_judge:
        mode = "judge-all" if judge_all else "exact + judge on misses"
        print(f"adjudicated acc  : {adjudicated}/{n}  ({100*adjudicated/n:.0f}%)"
              f"   [{mode}, judge: {judge_model}, {judge_calls} judge calls]")
    print(f"{'-'*60}")
    hdr = f"{'scope':<14} {'n':>3}  {'recall':>7} {'ground':>7} {'answer':>7}"
    if use_judge:
        hdr += f" {'adjud':>7}"
    print(hdr)
    for s, st in sorted(per_scope.items()):
        a = f"{st['ans']}/{st['n']}" if reader is not None else "-"
        line = (f"{s:<14} {st['n']:>3}  {st['recall']:>4}/{st['n']:<2} "
                f"{st['ground']:>4}/{st['n']:<2} {a:>7}")
        if use_judge:
            line += f" {st['adj']:>4}/{st['n']:<2}"
        print(line)

    summary = {
        "corpus_chunks": mem._next, "scopes": sorted(mem.repos),
        "reader": model, "judge": judge_model, "n": n,
        "judge_mode": ("judge-all" if judge_all else "last-resort") if use_judge else None,
        "judge_calls": judge_calls,
        "retrieval_recall": recall / n, "grounding": grounded / n,
        "answer_accuracy": (answered / n) if reader is not None else None,
        "adjudicated_accuracy": (adjudicated / n) if use_judge else None,
        "per_scope": per_scope, "rows": rows,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
