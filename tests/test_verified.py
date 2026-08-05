#!/usr/bin/env python3
"""Real tests for verified ranking -- rank retrieval by whether the code RUNS.

    python3 -m unittest tests.test_verified

No mocks: the scorer boot-gates each snippet in a REAL subprocess against a real (tiny,
self-contained) preamble, and the ranking test drives a real Groundwire index + FTS/BM25
retrieval. A broken example (calls a method that does not exist) must sink below a
runnable one that serialises the same objects correctly -- the select().to_array() class.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from groundwire.verified import verified_score, boot_gate, t1_parses, make_scorer
from groundwire.pipeline import Groundwire

# A self-contained framework stand-in: a real object with a real .to_dict() and NO .to_array().
# No tina4 dependency -- the boot-gate runs this verbatim in a subprocess, so a broken idiom
# fails truthfully (AttributeError) exactly as list.to_array() does against the real ORM.
PREAMBLE = '''
class _Item:
    def to_dict(self):
        return {"ok": 1}
items = [_Item(), _Item()]
def response(x, *a, **k):
    return x
def get(*a, **k):
    def _wrap(fn):
        return fn
    return _wrap
post = put = delete = patch = get
request = session = None
'''

CORRECT = "```python\nreturn response([i.to_dict() for i in items])\n```"
BROKEN = "```python\nreturn response([i.to_array() for i in items])\n```"   # to_array() doesn't exist
SYNTAX = "```python\nreturn response([i.to_dict() for i in items]\n```"     # unbalanced bracket


class TestVerifiedScore(unittest.TestCase):
    def test_runnable_scores_2(self):
        self.assertEqual(verified_score(CORRECT, is_doc=True, preamble=PREAMBLE), 2)

    def test_runtime_error_scores_0(self):
        # a list has no .to_array() -> AttributeError at run time -> broken (0)
        self.assertEqual(verified_score(BROKEN, is_doc=True, preamble=PREAMBLE), 0)

    def test_syntax_error_scores_0(self):
        self.assertFalse(t1_parses("return response([i.to_dict() for i in items]"))
        self.assertEqual(verified_score(SYNTAX, is_doc=True, preamble=PREAMBLE), 0)

    def test_pure_prose_is_neutral(self):
        self.assertEqual(verified_score("Just prose, no code fence.", is_doc=True, preamble=PREAMBLE), 2)

    def test_boot_gate_distinguishes_run_from_break(self):
        self.assertEqual(boot_gate("return response([i.to_dict() for i in items])", PREAMBLE), "boots")
        self.assertEqual(boot_gate("return response([i.to_array() for i in items])", PREAMBLE), "runtime-err")

    def test_make_scorer_trusts_source_boot_gates_docs(self):
        score = make_scorer(PREAMBLE)
        self.assertEqual(score("tina4_python/orm.py", BROKEN), 2)      # .py source: trusted, not gated
        self.assertEqual(score("docs/howto.md", BROKEN), 0)           # doc: boot-gated, broken


class TestVerifiedRanking(unittest.TestCase):
    QUERY = "serialize the items and return them in the response"

    def _index(self, scorer=True):
        mem = Groundwire(k=2, verified_scorer=make_scorer(PREAMBLE) if scorer else None)
        mem.ingest([("howto_broken.md",
                     "# Serialize items in a route\nReturn the items serialized:\n" + BROKEN),
                    ("howto_correct.md",
                     "# Serialize items in a route\nReturn the items serialized:\n" + CORRECT)])
        return mem

    def _order(self, mem):
        return [mem.source_of(cid) for cid, *_ in mem.retrieve(self.QUERY, k=2)]

    def test_sources_scored_at_ingest(self):
        mem = self._index()
        self.assertEqual(mem.verified.get("howto_correct.md"), 2)
        self.assertEqual(mem.verified.get("howto_broken.md"), 0)

    def test_broken_sinks_below_runnable(self):
        order = self._order(self._index())
        self.assertIn("howto_correct.md", order)
        self.assertIn("howto_broken.md", order)
        self.assertLess(order.index("howto_correct.md"), order.index("howto_broken.md"),
                        "runnable (score 2) example must out-rank the broken (score 0) one")

    def test_error_score_also_sinks(self):
        # a doc that ERRORS (score 1 -- e.g. a bad import) must ALSO sink below a runnable one.
        # The sink is "boots (2) vs not", NOT just "syntax-broken (0) vs not" -- a wrong import is
        # the load-bearing case (the model copies it), and it scores 1, not 0.
        badimport = "```python\nimport totally_not_a_real_module_zzz\nreturn response([i.to_dict() for i in items])\n```"
        self.assertEqual(verified_score(badimport, is_doc=True, preamble=PREAMBLE), 1)  # error (1), not broken (0)
        mem = Groundwire(k=2, verified_scorer=make_scorer(PREAMBLE))
        mem.ingest([("howto_badimport.md", "# Serialize items in a route\nReturn them:\n" + badimport),
                    ("howto_correct.md",   "# Serialize items in a route\nReturn them:\n" + CORRECT)])
        order = [mem.source_of(cid) for cid, *_ in mem.retrieve("serialize items and return in the response", k=2)]
        self.assertLess(order.index("howto_correct.md"), order.index("howto_badimport.md"),
                        "an erroring (score-1) example must ALSO sink below a runnable one")

    def test_env_off_switch_disables_sink(self):
        os.environ["GROUNDWIRE_VERIFIED_RANK"] = "0"
        try:
            mem = self._index()
            self.assertFalse(mem._verified_on)          # sink is a no-op...
            self.assertEqual(mem.verified.get("howto_broken.md"), 0)  # ...but scoring still ran
        finally:
            del os.environ["GROUNDWIRE_VERIFIED_RANK"]

    def test_no_scorer_is_noop(self):
        mem = self._index(scorer=False)
        self.assertEqual(mem.verified, {})              # nothing scored; ranking unchanged


if __name__ == "__main__":
    unittest.main()
