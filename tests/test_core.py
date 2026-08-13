#!/usr/bin/env python3
"""
Fast, network-free unit tests for the pure retrieval logic.

    python3 -m unittest discover tests        # or: python3 tests/test_core.py

No pytest, no models, no HTTP -- just the deterministic core: tokenizer
folding, stemming, code chunking, RRF fusion + phrase/def boosts, neighbor-pull
stitching, the import verifier, and groundedness. These guard every refinement
so a regression shows up in seconds instead of on the LLM-dependent batteries.
"""
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from groundwire.memory_systems import (
    fold, _light_stem, terms, InMemoryBM25, SqliteFTS, DenseEncoderMemory,
    MultiQueryRetriever)
from groundwire.pipeline import Groundwire, chunk_code
from groundwire.encoders import _prefixes_for, HashEncoder
from groundwire.rerank import DenseReranker


class TestFolding(unittest.TestCase):
    def test_diacritics(self):
        self.assertEqual(fold("Bezúkhov"), "bezukhov")

    def test_number_grouping(self):
        self.assertIn("24601", fold("prisoner 24,601"))
        self.assertNotIn("24,601", fold("24,601"))

    def test_camelcase_split(self):
        self.assertEqual(fold("ForeignKeyField"), "foreign key field")

    def test_terms_tokenizes_camel_and_number(self):
        t = terms("ForeignKeyField 24,601")
        self.assertIn("foreign", t)
        self.assertIn("field", t)
        self.assertIn("24601", t)


class TestStem(unittest.TestCase):
    def test_plural_folds(self):
        self.assertEqual(_light_stem("fields"), "field")

    def test_double_s_preserved(self):
        self.assertEqual(_light_stem("class"), "class")
        self.assertEqual(_light_stem("status"), "status")

    def test_short_word_untouched(self):
        self.assertEqual(_light_stem("is"), "is")


class TestChunkCode(unittest.TestCase):
    def test_splits_on_def_boundaries(self):
        src = "import os\n\ndef a():\n    return 1\n\ndef b():\n    return 2\n"
        chunks = chunk_code(src, path="m.py", max_lines=3)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(all(c[1].startswith("# file: m.py") for c in chunks))

    def test_path_indexed_in_body(self):
        chunks = chunk_code("def x(): pass", path="core/router.py")
        self.assertIn("core/router.py", chunks[0][1])


class TestBM25Get(unittest.TestCase):
    def test_get_by_id(self):
        b = InMemoryBM25()
        b.ingest([(5, "hello world"), (9, "foo bar")])
        self.assertEqual(b.get(5), "hello world")
        self.assertIsNone(b.get(99))


class TestFusion(unittest.TestCase):
    def _mq(self, chunks, rewriter=None):
        m = MultiQueryRetriever(InMemoryBM25(), rewriter=rewriter, n=3)
        m.ingest(chunks)
        return m

    def test_original_query_safety_floor(self):
        # a probe must never evict what the raw question found
        m = self._mq([(1, "router get post handler"),
                      (2, "unrelated filler text here")],
                     rewriter=lambda q, n: ["filler text"])
        ids = [c for c, _, _ in m.query("router handler", k=1)]
        self.assertIn(1, ids)

    def test_phrase_bonus_promotes_verbatim(self):
        m = self._mq([(1, "the ishmael genealogy begat begat begat ishmael"),
                      (2, "Call me Ishmael. Some years ago.")],
                     rewriter=lambda q, n: ["Call me Ishmael"])
        top = m.query("opening line", k=1)[0][0]
        self.assertEqual(top, 2)

    def test_offtopic_probe_gated(self):
        # a probe sharing no words with the question and not present verbatim
        # must not drag in an unrelated chunk
        m = self._mq([(1, "authentication token secure"),
                      (2, "sherlock holmes baker street")],
                     rewriter=lambda q, n: ["sherlock holmes"])
        ids = [c for c, _, _ in m.query("how do I secure a token", k=1)]
        self.assertEqual(ids, [1])


class TestNeighborPull(unittest.TestCase):
    def test_pulls_adjacent_code_chunk(self):
        m = Groundwire(memory="bm25", k=1)
        # three sequential chunks of one code file; only the middle is a hit
        m.ingest_code("def get(): pass\n" * 30 +
                      "\ndef put(): pass\n" +
                      "def delete(): pass\n", title="r.py", max_lines=10)
        hits = m.retrieve("put", scope=None)
        # neighbor-pull should surface put/delete even from a single top hit
        joined = " ".join(t for _, t, _ in hits)
        self.assertIn("def put", joined)

    def test_prose_not_pulled(self):
        m = Groundwire(memory="bm25", k=1)
        m.ingest("Sentence one about whales. " * 50, title="Moby-Dick")
        hits = m.retrieve("whales", scope=None)
        self.assertEqual(len(hits), 1)


class TestImportVerifier(unittest.TestCase):
    def setUp(self):
        import groundwire.server as S
        self.S = S
        S.SYMBOLS = {"pkg.orm.fields": {"IntegerField", "StringField"},
                     "pkg.queue": {"Queue"}}

    def test_valid_import_passes(self):
        self.assertEqual(
            self.S.check_imports("from pkg.orm.fields import IntegerField", "pkg"),
            [])

    def test_missing_module_flagged(self):
        p = self.S.check_imports("from pkg.middleware import Cors", "pkg")
        self.assertTrue(any("does not exist" in x for x in p))

    def test_wrong_module_hints_real_home(self):
        p = self.S.check_imports("from pkg.queue import IntegerField", "pkg")
        self.assertTrue(any("orm.fields" in x for x in p))

    def test_attribute_chain_flagged(self):
        p = self.S.check_imports("pkg.queue.nonexistent_fn()", "pkg")
        self.assertTrue(p)


class TestGroundedness(unittest.TestCase):
    def setUp(self):
        import groundwire.server as S
        self.g = S.groundedness

    def test_valid_paraphrase_not_flagged(self):
        s, f = self.g("Use Router.get to register a route.",
                      "Router.get registers a GET route with a path.")
        self.assertEqual(f, [])

    def test_offtopic_flagged(self):
        s, f = self.g("The blockchain validator synchronizes quantum ledger shards.",
                      "Router.get registers a route. Bishop gave candlesticks.")
        self.assertEqual(len(f), 1)


class TestPersistence(unittest.TestCase):
    def test_save_load_roundtrip(self):
        import tempfile
        m = Groundwire(memory="bm25", k=3)
        m.ingest_code("def get(): pass\ndef post(): pass\n", title="repo/r.py")
        m.repos.add("repo")
        before = [c for c, _, _ in m.retrieve("get", scope=None)]
        path = os.path.join(tempfile.gettempdir(), "lm_test.pkl")
        m.save(path)

        m2 = Groundwire(memory="bm25", k=3)
        m2.load(path)
        after = [c for c, _, _ in m2.retrieve("get", scope=None)]
        self.assertEqual(before, after)
        self.assertEqual(m2.source_of(before[0]), "repo/r.py")   # sources kept
        self.assertIn("repo", m2.repos)                          # repos kept
        os.remove(path)


class TestCapacityProtection(unittest.TestCase):
    def setUp(self):
        import groundwire.server as S
        self.S = S

    def test_overflow_signatures_match(self):
        for m in ["maximum context length is 8192 tokens. reduce the length",
                  "CUDA out of memory", "KV cache is full, cannot allocate"]:
            self.assertTrue(self.S._CAP_SIG.search(m))

    def test_benign_errors_not_capacity(self):
        for m in ["invalid api key", "model not found", "rate limit exceeded"]:
            self.assertFalse(self.S._CAP_SIG.search(m))

    def test_shrinks_and_recovers(self):
        S = self.S
        calls = {"n": 0, "budgets": [], "nhits": []}

        def fake(q, hits, mt, ctx_budget=None):
            calls["n"] += 1
            calls["budgets"].append(ctx_budget)
            calls["nhits"].append(len(hits))
            if calls["n"] < 3:
                raise S._CapacityError("maximum context length")
            return ("ok", {}, False)

        orig, S._grounded_answer = S._grounded_answer, fake
        orig_max, S.MAX_CTX_TOKENS = S.MAX_CTX_TOKENS, 4500
        try:
            out = S._answer_resilient("q", [(i, "c", 1.0) for i in range(6)], 60)
        finally:
            S._grounded_answer, S.MAX_CTX_TOKENS = orig, orig_max
        self.assertEqual(out[0], "ok")
        self.assertEqual(calls["budgets"], [4500, 2250, 1125])   # halving
        self.assertEqual(calls["nhits"], [6, 3, 1])              # dropping


class TestPrefixes(unittest.TestCase):
    def test_nomic_prefixes(self):
        self.assertEqual(_prefixes_for("nomic-embed-text"),
                         ("search_query: ", "search_document: "))

    def test_plain_model_no_prefix(self):
        self.assertEqual(_prefixes_for("some-random-model"), ("", ""))


class TestSqliteFTSFolding(unittest.TestCase):
    """SqliteFTS must fold the DOCUMENT side too, so code retrieval matches
    InMemoryBM25 instead of only prefix-matching opaque camelCase tokens."""

    DOCS = [(0, "class User(ORM):\n    id = IntegerField()"),
            (1, "def getToken(self): return jsonResponse(token)"),
            (2, "class Post(ORM):\n    author = ForeignKeyField(User)")]

    def _hits(self, backend, q):
        b = backend(); b.ingest(self.DOCS)
        out = [c for c, _, _ in b.query(q, k=3)]
        b.close()
        return out

    def test_noninitial_camel_component_matches_like_bm25(self):
        # 'field' is a non-initial component of IntegerField/ForeignKeyField --
        # unreachable by prefix, reachable once the body is folded.
        for q in ("integer field", "create a token", "foreign key to a model"):
            self.assertEqual(self._hits(SqliteFTS, q), self._hits(InMemoryBM25, q),
                             f"sqlite_fts diverged from bm25 on {q!r}")

    def test_definition_outranks_trivial_usage(self):
        b = SqliteFTS()
        b.ingest([(0, "def validateEmailAddress(x): pass"),
                  (1, "def validate(x): return x"),
                  (2, "email = getAddress()")])
        self.assertEqual(b.query("validateEmailAddress", k=1)[0][0], 0)
        b.close()

    def test_comma_grouped_number_reachable(self):
        b = SqliteFTS(); b.ingest([(0, "value 24,601 here")])
        self.assertEqual([c for c, _, _ in b.query("24601", k=1)], [0])
        b.close()

    def test_no_short_token_wildcard_pollution(self):
        # 'apps' must not drag in append/apply/application via a stem+prefix.
        b = SqliteFTS()
        b.ingest([(0, "the apps directory"), (1, "append helper"),
                  (2, "apply middleware"), (3, "application config")])
        self.assertEqual([c for c, _, _ in b.query("apps", k=4)], [0])
        b.close()

    def test_returns_raw_not_folded_text(self):
        b = SqliteFTS(); b.ingest([(0, "class ForeignKeyField(Field): pass")])
        self.assertEqual(b.query("foreign key", k=1)[0][1],
                         "class ForeignKeyField(Field): pass")
        self.assertEqual(b.get(0), "class ForeignKeyField(Field): pass")
        b.close()

    def test_save_load_roundtrip_refolds(self):
        import tempfile
        m = Groundwire(memory="sqlite_fts", k=3)
        m.ingest_code("class ForeignKeyField(Field): pass\n", title="repo/f.py")
        before = [c for c, _, _ in m.retrieve("foreign key", scope=None)]
        self.assertTrue(before)                       # reachable before save
        path = os.path.join(tempfile.gettempdir(), "lm_fts_test.pkl")
        m.save(path)
        m2 = Groundwire(memory="sqlite_fts", k=3); m2.load(path)
        after = [c for c, _, _ in m2.retrieve("foreign key", scope=None)]
        self.assertEqual(before, after)               # re-fold on restore
        self.assertIn("ForeignKeyField", m2.retrieve("foreign key")[0][1])
        os.remove(path)


class TestSqliteFTSIsolation(unittest.TestCase):
    def test_two_instances_do_not_clobber(self):
        a = SqliteFTS(); a.ingest([(1, "alpha")])
        b = SqliteFTS(); b.ingest([(2, "beta")])
        self.assertNotEqual(a.path, b.path)
        self.assertEqual([c for c, _, _ in a.query("alpha", k=1)], [1])
        a.ingest([(3, "gamma")])                       # used to be "readonly"
        self.assertEqual([c for c, _, _ in a.query("gamma", k=1)], [3])
        a.close(); b.close()

    def test_cross_thread_query(self):
        import threading
        b = SqliteFTS(); b.ingest([(0, "hello router get")])
        out, err = [], []

        def work():
            try:
                out.append([c for c, _, _ in b.query("router", k=1)])
            except Exception as e:       # ProgrammingError before the fix
                err.append(repr(e))
        t = threading.Thread(target=work); t.start(); t.join()
        self.assertEqual(err, [])
        self.assertEqual(out, [[0]])
        b.close()


class TestDenseEmptyBatch(unittest.TestCase):
    def test_empty_first_ingest_then_real(self):
        m = DenseEncoderMemory(encoder="hash:8")
        m.ingest([])                                   # used to poison self.mat
        m.ingest([(0, "hello world")])                 # used to crash in vstack
        # mat is a pure-stdlib list of rows now (veclite replaced numpy): 1 row x 8 dims
        self.assertEqual(len(m.mat), 1)
        self.assertEqual(len(m.mat[0]), 8)
        self.assertEqual([c for c, _, _ in m.query("hello", k=1)], [0])


class TestHashDeterminism(unittest.TestCase):
    def test_hashencoder_stable_across_processes(self):
        import subprocess
        # pure-stdlib argmax -- groundwire has no numpy dependency to lean on
        snippet = ("from groundwire.encoders import HashEncoder;"
                   "v=HashEncoder(64)(['getToken'])[0];"
                   "print(max(range(len(v)), key=v.__getitem__))")
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        outs = []
        for seed in ("0", "1"):
            env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONPATH=root)
            outs.append(subprocess.check_output(
                [sys.executable, "-c", snippet], env=env).strip())
        self.assertEqual(outs[0], outs[1])             # salt must not matter


class TestStitchUntitledDocs(unittest.TestCase):
    def test_untitled_docs_not_merged(self):
        m = Groundwire(memory="bm25", k=5); m.max_words = 2
        m.ingest("apple alpha.")                       # doc A, source None
        m.ingest("apple beta.")                        # doc B, source None
        hits = m.retrieve("apple", k=5)
        # no delivered span may contain BOTH docs' unique words
        for _, text, _ in hits:
            self.assertFalse("alpha" in text and "beta" in text,
                             f"stitched across untitled docs: {text!r}")


class TestMultiQueryK1(unittest.TestCase):
    def test_k1_returns_exactly_one_no_dup(self):
        m = MultiQueryRetriever(InMemoryBM25(),
                                rewriter=lambda q, n: ["getToken function"], n=3)
        m.ingest([(0, "router router router the add method wiring"),
                  (1, "router add method configuration router here"),
                  (2, "def getToken(): pass # getToken function defined")])
        res = m.query("router add method", k=1)
        cids = [c for c, _, _ in res]
        self.assertEqual(len(res), 1)
        self.assertEqual(len(cids), len(set(cids)))


class TestHarnessTokenCount(unittest.TestCase):
    def test_approx_tokens_independent_of_overlap(self):
        from groundwire.harness import run_one
        r0 = run_one("bm25", 3000, [0.5], 3, 7, 1.3, 40, 0)
        r2 = run_one("bm25", 3000, [0.5], 3, 7, 1.3, 40, 2)
        # overlap changes chunking, not the true token count -- so the reported
        # size must be identical (it double-counted overlap before the fix)
        self.assertEqual(r0["approx_tokens"], r2["approx_tokens"])

    def test_empty_depths_raises(self):
        from groundwire.harness import run_one
        with self.assertRaises(ValueError):
            run_one("bm25", 3000, [], 3, 7, 1.3, 40, 1)


class TestMakeDistractors(unittest.TestCase):
    class _CollidingRng:
        """randint for the entity draw collides on the first call, forcing the
        retry path; a distinct range serves _value()."""
        def __init__(self, suffix):
            self.suffix, self.n = suffix, 0

        def randint(self, a, b):
            if b <= 9999:
                self.n += 1
                return self.suffix if self.n == 1 else 1000 + self.n
            return 5000000

    def test_returns_exact_count_despite_collision(self):
        from groundwire.tasks import make_distractors
        item = {"entity": "brindle-3471", "topic": "vault"}
        out = make_distractors(item, 3, self._CollidingRng(3471))
        self.assertEqual(len(out), 3)                  # was 2 before the fix


class _RerankStub:
    """Deterministic offline encoder over a 3-word vocab; records what it
    embedded so tests can assert the reranker's call/cache behavior. No
    network."""
    VOCAB = ("alpha", "beta", "gamma")

    def __init__(self):
        self.doc_texts = []          # every document-side text embedded
        self.query_calls = 0

    def __call__(self, texts, is_query=False):
        from groundwire.veclite import normalize_rows   # pure stdlib -- no numpy
        if is_query:
            self.query_calls += 1
        else:
            self.doc_texts.extend(texts)
        rows = [[float(t.lower().count(w)) for w in self.VOCAB] for t in texts]
        return normalize_rows(rows)


class TestDenseReranker(unittest.TestCase):
    def test_reorders_by_dense_similarity(self):
        # lexical put the beta chunk LAST; the reranker (pure dense here) must
        # lift it to the top for a beta query
        rr = DenseReranker(_RerankStub(), blend=1.0)
        pool = [(0, "gamma gamma", 3.0), (1, "alpha alpha", 2.0),
                (2, "beta beta beta", 1.0)]
        self.assertEqual(rr("find the beta thing", pool)[0][0], 2)

    def test_candidate_embedded_at_most_once(self):
        from collections import Counter
        enc = _RerankStub(); rr = DenseReranker(enc)
        rr("about beta", [(0, "alpha", 1.0), (1, "beta", 0.9)])
        rr("about gamma", [(1, "beta", 1.0), (2, "gamma", 0.9)])   # cid 1 repeats
        self.assertEqual(set(enc.doc_texts), {"alpha", "beta", "gamma"})
        self.assertTrue(all(v == 1 for v in Counter(enc.doc_texts).values()))

    def test_query_embedded_once_across_passes(self):
        enc = _RerankStub(); rr = DenseReranker(enc)
        pool = [(0, "alpha", 1.0), (1, "beta", 0.9)]
        rr("same question", pool)
        rr("same question", pool)          # pool/wide re-rank of one question
        self.assertEqual(enc.query_calls, 1)

    def test_fallback_on_encoder_error(self):
        def boom(texts, is_query=False):
            raise RuntimeError("no embeddings endpoint")
        pool = [(0, "a", 1.0), (1, "b", 0.9)]
        self.assertEqual(DenseReranker(boom)("q", pool), pool)   # lexical order

    def test_empty_pool(self):
        self.assertEqual(DenseReranker(_RerankStub())("q", []), [])

    def test_cache_is_bounded(self):
        rr = DenseReranker(_RerankStub(), max_cache=2)
        for cid in range(5):
            rr("q", [(cid, "alpha", 1.0)])
        self.assertLessEqual(len(rr._cache), 2)

    def test_groundwire_wires_dense_reranker(self):
        m = Groundwire(memory="bm25", k=2, rerank=_RerankStub())   # callable -> encoder
        self.assertIsInstance(m.reranker, DenseReranker)
        m.ingest_code("def alpha(): pass\ndef beta(): pass\n", title="m.py")
        self.assertTrue(m.retrieve("beta", scope=None))        # end-to-end, no error


class _RecordingReader:
    """Stub reader that records the chunks it saw per call and extracts the
    7-digit value bound to the entity in the question (like RegexExtract). If
    its batch lacks the answer it 'refuses' -- so tests can prove the loop maps
    over batches and reduces to the batch that actually held the answer."""
    name = "recording"

    def __init__(self, context_chars=None):
        self.context_chars = context_chars
        self.saw = []                       # one entry per generate() call: #chunks

    def generate(self, question, chunks):
        self.saw.append(len(chunks))
        joined = " ".join(t for _, t, _ in chunks)
        m = re.search(r"entity ([a-z]+-\d+)", question.lower())
        if m:
            b = re.search(r"entity " + re.escape(m.group(1)) + r"\D*(\d{7})",
                          joined.lower())
            if b:
                return b.group(1)
        return "I don't know"


class TestAnswerReduceEngine(unittest.TestCase):
    """The map-reduce answer engine (groundwire.mapreduce.answer) -- model-free."""

    def setUp(self):
        from groundwire import mapreduce as MR
        self.MR = MR

    def _chunks(self, *texts):
        return [(i, t, 1.0) for i, t in enumerate(texts)]

    def test_pack_orders_and_never_splits(self):
        chunks = self._chunks("a" * 40, "b" * 40, "c" * 40)
        batches = self.MR._pack(chunks, budget_chars=100)   # ~2 chunks/batch
        self.assertEqual([len(b) for b in batches], [2, 1])
        flat = [c for b in batches for c in b]
        self.assertEqual(flat, chunks)                      # order + count preserved

    def test_oversized_chunk_gets_own_batch(self):
        chunks = self._chunks("x" * 500, "y" * 10)          # first > budget
        batches = self.MR._pack(chunks, budget_chars=100)
        self.assertEqual([len(b) for b in batches], [1, 1])

    def test_fits_is_a_single_call(self):
        r = _RecordingReader()
        chunks = self._chunks("short one", "short two")
        self.MR.answer(r.generate, "q", chunks, budget_chars=10_000)
        self.assertEqual(len(r.saw), 1)                     # one call, all chunks
        self.assertEqual(r.saw[0], 2)

    def test_overflow_maps_once_per_batch(self):
        r = _RecordingReader()
        chunks = self._chunks("a" * 60, "b" * 60, "c" * 60)
        self.MR.answer(r.generate, "q", chunks, budget_chars=80)  # 1 chunk/batch
        self.assertEqual(len(r.saw), 3)                     # mapped over 3 batches

    def test_reduce_skips_refusals_for_real_answer(self):
        # answer lives only in the 3rd batch; earlier batches refuse.
        chunks = self._chunks(
            "filler about ships",
            "more filler prose",
            "the secret code for entity falcon-4823 is 7654321 exactly")
        r = _RecordingReader()
        out = self.MR.answer(
            r.generate,
            "What is the code for entity falcon-4823?",
            chunks, budget_chars=25)                        # forces 1 chunk/batch
        self.assertEqual(out, "7654321")
        self.assertEqual(len(r.saw), 3)                     # all batches read, no early-exit

    def test_default_selects_grounded_over_ungrounded(self):
        # two real (non-refusal) candidates; only the second is present in its
        # batch text. Grounded selection must win even though it is later.
        cands = [
            ([(0, "no numbers here at all", 1.0)], "9999999"),    # hallucinated
            ([(1, "value is 1234567 in text", 1.0)], "1234567"),  # grounded
        ]
        self.assertEqual(self.MR._select("q", cands, verify=None), "1234567")

    def test_default_first_real_when_none_grounded(self):
        cands = [
            ([(0, "prose", 1.0)], "I don't know"),          # refusal -> skipped
            ([(1, "prose", 1.0)], "alpha"),                 # first real answer
            ([(2, "prose", 1.0)], "beta"),
        ]
        self.assertEqual(self.MR._select("q", cands, verify=None), "alpha")

    def test_custom_verify_picks_highest_score(self):
        cands = [
            ([(0, "t", 1.0)], "low"),
            ([(1, "t", 1.0)], "high"),
            ([(2, "t", 1.0)], "mid"),
        ]
        score = {"low": 1, "mid": 2, "high": 9}
        out = self.MR._select("q", cands, verify=lambda q, a, b: score[a])
        self.assertEqual(out, "high")

    def test_is_answer_rejects_refusals(self):
        self.assertFalse(self.MR._is_answer(""))
        self.assertFalse(self.MR._is_answer("  I don't know. "))
        self.assertFalse(self.MR._is_answer("not found"))
        self.assertTrue(self.MR._is_answer("7654321"))


class TestReadLoopPipeline(unittest.TestCase):
    """Groundwire.ask() engages the answer-reduce loop only when chunks overflow the
    reader's window, and is a plain single call otherwise (back-compat)."""

    def _index(self, reader):
        m = Groundwire(memory="bm25", k=6, reader=reader)
        for i in range(5):
            m.ingest(f"Chapter {i}: assorted filler prose about the sea. " * 3,
                     title=f"doc{i}")
        m.ingest("Record: the secret code for entity falcon-4823 is 7654321.",
                 title="answer-doc")
        return m

    def test_loop_engages_on_overflow_and_finds_answer(self):
        r = _RecordingReader()
        m = self._index(r)
        out = m.ask("What is the secret code for entity falcon-4823?",
                    reader_budget_chars=120)                # small -> must batch
        self.assertEqual(out, "7654321")
        self.assertGreater(len(r.saw), 1)                   # looped, not truncated

    def test_no_loop_when_it_fits(self):
        r = _RecordingReader()
        m = self._index(r)
        out = m.ask("What is the secret code for entity falcon-4823?",
                    reader_budget_chars=1_000_000)          # everything fits
        self.assertEqual(len(r.saw), 1)                     # exactly one call
        self.assertEqual(out, "7654321")

    def test_reader_context_chars_drives_budget(self):
        r = _RecordingReader(context_chars=120)             # reader declares window
        m = self._index(r)
        m.ask("What is the secret code for entity falcon-4823?")  # no explicit budget
        self.assertGreater(len(r.saw), 1)                   # loop used reader's window

    def test_no_budget_no_loop(self):
        r = _RecordingReader()                              # context_chars=None
        m = self._index(r)
        m.ask("What is the secret code for entity falcon-4823?")  # no budget anywhere
        self.assertEqual(len(r.saw), 1)                     # unchanged single call


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestChatMemory(unittest.TestCase):
    """Retrieved conversational memory (groundwire.ChatMemory)."""

    def test_retrieves_relevant_turn_not_the_whole_log(self):
        from groundwire import ChatMemory
        m = ChatMemory(k=1)
        m.add_turn("How do I open a FireDAC connection?",
                   "Use TFDConnection.Connected := True")
        m.add_turn("What is the capital of France?", "Paris")
        m.add_turn("How do I free a TFileStream?", "Call Free")
        ctx = m.context("and the FireDAC connection parameters?")
        self.assertIn("TFDConnection", ctx)      # the relevant turn is retrieved
        self.assertNotIn("Paris", ctx)           # the off-topic turn is not

    def test_empty_history(self):
        from groundwire import ChatMemory
        m = ChatMemory()
        self.assertEqual(m.retrieve("anything"), [])
        self.assertEqual(m.context("anything"), "")
        self.assertEqual(len(m), 0)

    def test_add_turns_and_len_chain(self):
        from groundwire import ChatMemory
        m = ChatMemory(k=2).add_turns([("a?", "alpha"), ("b?", "beta")])
        self.assertEqual(len(m), 2)
        self.assertTrue(m.retrieve("alpha"))
