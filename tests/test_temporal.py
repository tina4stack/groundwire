"""Temporal retrieval: timestamps, recency decay, latest-wins supersession, as_of.

Network-free and deterministic: every timestamp is explicit and `as_of` pins "now"
where decay depends on it.
"""
import os
import pickle
import sys
import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from groundwire import ChatMemory, Groundwire               # noqa: E402
from groundwire.temporal import (DAY, Hit, apply_recency, half_life_of,  # noqa: E402
                                 supersede, to_epoch)

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
NOW = T0 + timedelta(days=365)


def _gw(**kw):
    return Groundwire(memory="bm25", k=kw.pop("k", 3), **kw)


class TestTimestamps(unittest.TestCase):

    def test_to_epoch_forms(self):
        e = T0.timestamp()
        self.assertEqual(to_epoch(e), e)
        self.assertEqual(to_epoch(T0), e)
        self.assertEqual(to_epoch("2026-01-01T00:00:00Z"), e)
        self.assertAlmostEqual(to_epoch(None), time.time(), delta=5)
        self.assertAlmostEqual(to_epoch(timedelta(days=1)), time.time() - DAY, delta=5)
        with self.assertRaises(TypeError):
            to_epoch(True)

    def test_hit_carries_timestamp_and_unpacks_as_triple(self):
        gw = _gw().ingest("The ledger closes on Friday.", title="ops", at=T0)
        hits = gw.retrieve("when does the ledger close")
        self.assertEqual(len(hits), 1)
        cid, text, score = hits[0]                  # back-compat: still a 3-tuple
        self.assertEqual(hits[0].at, T0)
        self.assertEqual(hits[0].ts, T0.timestamp())
        self.assertEqual(gw.time_of(cid), T0)
        self.assertEqual(hits[0], (cid, text, score))

    def test_default_is_now(self):
        gw = _gw().ingest("Fresh fact about zebras.")
        self.assertAlmostEqual(gw.retrieve("zebras")[0].ts, time.time(), delta=5)

    def test_triples_and_ingest_code(self):
        gw = _gw().ingest([("a", "alpha apples", T0), ("b", "beta bananas")], at=NOW)
        gw.ingest_code("def gamma():\n    return 1\n", title="g.py", at=T0)
        by = {gw.source_of(h[0]): h.at for h in gw.retrieve("alpha beta gamma", k=5)}
        self.assertEqual(by["a"], T0)
        self.assertEqual(by["b"], NOW)
        self.assertEqual(by["g.py"], T0)

    def test_repo_files_take_their_mtime(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "notes.md")
            with open(p, "w") as f:
                f.write("The deploy window is Tuesday.")
            os.utime(p, (T0.timestamp(), T0.timestamp()))
            gw = _gw().ingest_repo(d, prefix="r")
            self.assertEqual(gw.retrieve("deploy window")[0].at, T0)

    def test_save_load_roundtrip_keeps_times(self):
        gw = _gw().ingest("Quartz is the office cat.", title="cat", at=T0,
                          subject="cat")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "snap.pkl")
            gw.save(path)
            back = _gw().load(path)
        self.assertEqual(back.retrieve("office cat")[0].at, T0)
        self.assertEqual(back.subjects, gw.subjects)
        self.assertEqual(back.title_versions, gw.title_versions)

    def test_load_old_snapshot_without_times(self):
        gw = _gw().ingest("Legacy snapshot text about owls.")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "old.pkl")
            gw.save(path)
            with open(path, "rb") as f:
                payload = pickle.load(f)
            for key in ("times", "subjects", "title_versions"):
                payload.pop(key)
            with open(path, "wb") as f:
                pickle.dump(payload, f)
            back = _gw().load(path)
        hit = back.retrieve("owls", recency="prefer", latest=True)[0]
        self.assertIsNone(hit.at)

    def test_hit_pickles(self):
        h = pickle.loads(pickle.dumps(Hit(1, "x", 0.5, 7.0)))
        self.assertEqual((tuple(h), h.ts), ((1, "x", 0.5), 7.0))


class TestRecency(unittest.TestCase):

    def _office(self):
        # two equally-relevant statements a year apart; the OLD one is ingested
        # first so lexical ties favour it
        return _gw(k=1).ingest([
            ("hr-2025", "Office: the office is located in Durban.", T0),
            ("hr-2026", "The office is located in Pretoria.", NOW),
        ])

    def test_off_by_default_unchanged(self):
        gw = self._office()
        self.assertEqual(gw.retrieve("where is the office located")[0][1],
                         "Office: the office is located in Durban.")

    def test_prefer_puts_newer_first(self):
        gw = self._office()
        hit = gw.retrieve("where is the office located", recency="prefer", as_of=NOW)[0]
        self.assertIn("Pretoria", hit[1])
        self.assertEqual(hit.at, NOW)

    def test_numeric_half_life_and_timedelta(self):
        gw = self._office()
        for hl in (7 * DAY, timedelta(days=7)):
            self.assertIn("Pretoria",
                          gw.retrieve("office located", recency=hl, as_of=NOW)[0][1])
        with self.assertRaises(ValueError):
            gw.retrieve("office", recency=-1)
        self.assertIsNone(half_life_of(None))

    def test_recency_cannot_lift_irrelevant_chunk(self):
        gw = _gw(k=1).ingest([
            ("old", "Invoice numbering: invoices are numbered INV-0001 upward, "
                    "invoice prefix INV, invoice sequence per year.", T0),
            ("new", "Lunch is at noon; invoice questions go to finance.", NOW),
        ])
        hit = gw.retrieve("invoice numbering prefix sequence",
                          recency="prefer", as_of=NOW)[0]
        self.assertIn("INV-0001", hit[1])

    def test_weight_zero_is_pure_relevance(self):
        pool = [(1, "a", 2.0), (2, "b", 1.0)]
        times = {1: 0.0, 2: 1000.0 * DAY}
        self.assertEqual(apply_recency(pool, times, DAY, weight=0.0, now=1000 * DAY),
                         pool)
        self.assertEqual([h[0] for h in apply_recency(pool, times, DAY, weight=1.0,
                                                      now=1000 * DAY)], [2, 1])

    def test_untimed_pool_is_noop(self):
        pool = [(1, "a", 1.0), (2, "b", 1.0)]
        self.assertEqual(apply_recency(pool, {}, DAY), pool)


class TestLatestWins(unittest.TestCase):

    def test_subject_key_supersedes(self):
        gw = _gw(k=3)
        gw.ingest("Andre lives in Cape Town near the sea.", title="m1", at=T0,
                  subject="andre.home")
        gw.ingest("Andre moved: home is now Durban.", title="m2", at=NOW,
                  subject="andre.home")
        gw.ingest("Andre's favourite tea is rooibos.", title="m3", at=T0,
                  subject="andre.tea")
        texts = [h[1] for h in gw.retrieve("where does Andre live home", latest=True)]
        self.assertTrue(any("Durban" in t for t in texts))
        self.assertFalse(any("Cape Town" in t for t in texts))
        # without latest, both versions come back
        texts = [h[1] for h in gw.retrieve("where does Andre live home")]
        self.assertTrue(any("Cape Town" in t for t in texts))

    def test_reingested_title_drops_old_version(self):
        gw = _gw(k=5)
        gw.ingest("Rate limit is 100 requests per minute. Burst is 20.",
                  title="api.md", at=T0)
        gw.ingest("Rate limit is 500 requests per minute. Burst is 50.",
                  title="api.md", at=NOW)
        texts = [h[1] for h in gw.retrieve("rate limit requests burst", latest=True)]
        self.assertTrue(texts and all("500" in t for t in texts))

    def test_near_duplicate_wording(self):
        gw = _gw(k=3).ingest([
            ("a", "Standup meeting happens daily at 09:00 in room Kestrel.", T0),
            ("b", "Standup meeting happens daily at 09:30 in room Kestrel.", NOW),
        ])
        texts = [h[1] for h in gw.retrieve("standup meeting room", latest=True)]
        self.assertEqual(len(texts), 1)
        self.assertIn("09:30", texts[0])
        # the wording test can be switched off
        self.assertEqual(len(gw.retrieve("standup meeting room", latest=True,
                                         dup_threshold=None)), 2)

    def test_keeps_best_rank_position(self):
        pool = [(1, "alpha beta gamma", 3.0), (2, "unrelated zeta", 2.0),
                (3, "alpha beta gamma", 1.0)]
        out = supersede(pool, {1: 1.0, 2: 1.0, 3: 2.0})
        self.assertEqual([h[0] for h in out], [3, 2])

    def test_untimed_never_displaces_timed(self):
        pool = [(1, "alpha beta gamma", 2.0), (2, "alpha beta gamma", 1.0)]
        self.assertEqual([h[0] for h in supersede(pool, {1: 5.0})], [1])


class TestAsOf(unittest.TestCase):

    def test_hides_later_facts(self):
        gw = _gw(k=3).ingest([
            ("v1", "The project codename is Heron.", T0),
            ("v2", "The project codename is Osprey.", NOW),
        ])
        then = gw.retrieve("project codename", as_of=T0 + timedelta(days=1))
        self.assertEqual([h[1] for h in then], ["The project codename is Heron."])
        both = gw.retrieve("project codename")
        self.assertEqual(len(both), 2)

    def test_as_of_with_reingested_title(self):
        gw = _gw(k=3)
        gw.ingest("Price is 10 dollars.", title="price", at=T0)
        gw.ingest("Price is 12 dollars.", title="price", at=NOW)
        then = gw.retrieve("price dollars", latest=True, as_of=T0 + timedelta(days=1))
        self.assertEqual([h[1] for h in then], ["Price is 10 dollars."])


class TestChatMemoryTime(unittest.TestCase):

    def test_turns_are_stamped_and_labelled(self):
        at = datetime(2026, 3, 4, 15, 30)                # naive = local time
        m = ChatMemory(k=1).add_turn("Where do I park?", "Bay 7", at=at)
        self.assertEqual(m.times[0], at.timestamp())
        ctx = m.context("park")
        self.assertTrue(ctx.startswith("[2026-03-04 15:30] Q: Where do I park?"))
        self.assertEqual(m.context("park", timestamps=False),
                         "Q: Where do I park? A: Bay 7")
        self.assertEqual(m.retrieve("park")[0].at.timestamp(), at.timestamp())

    def test_latest_turn_wins(self):
        m = ChatMemory(k=1)
        m.add_turn("My car is a blue Golf.", "Noted.", at=T0, subject="car")
        m.add_turn("My car is now a red Polo.", "Noted.", at=NOW, subject="car")
        self.assertIn("red Polo", m.context("what car do I drive", latest=True))
        self.assertIn("red Polo", m.context("what is my car", recency="prefer",
                                            as_of=NOW))

    def test_add_turns_accepts_triples(self):
        m = ChatMemory().add_turns([("a?", "alpha", T0), ("b?", "beta")])
        self.assertEqual(m.times[0], T0.timestamp())
        self.assertEqual(len(m), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
