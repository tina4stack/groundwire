"""Time-awareness for groundwire: when a fact was recorded, and which one is current.

Retrieval alone is time-blind -- when an old fact and its replacement both match a
query, relevance picks either. This module adds three opt-in tools, all pure stdlib
and all no-ops unless a caller asks for them:

- timestamps: every chunk carries the epoch it was recorded (``ingest(..., at=)``,
  default now; ``ingest_repo`` uses each file's mtime). Hits expose it as ``hit.at``
  so a caller can render "as of <date>".
- recency: ``retrieve(..., recency="prefer" | half_life_seconds)`` scales each hit's
  relevance by an exponential time decay. Multiplicative, so recency can only ever
  discount an old chunk by at most ``recency_weight`` -- an irrelevant chunk never
  wins just for being fresh.
- supersession: ``retrieve(..., latest=True)`` keeps only the newest of chunks that
  say the same thing -- same caller-supplied ``subject``, same document title
  re-ingested (an older version), or near-duplicate wording.

``as_of=`` answers "what did we know then": it hides chunks recorded later and
measures age from that moment instead of now.
"""
from __future__ import annotations

import time as _time
from datetime import date, datetime, timedelta, timezone

from .memory_systems import terms

DAY = 86400.0
DEFAULT_HALF_LIFE = 30 * DAY        # recency="prefer": a month-old chunk counts ~75%
DEFAULT_WEIGHT = 0.5                # the most recency can discount a chunk
DEFAULT_DUP_THRESHOLD = 0.6         # Jaccard of content terms for "same statement"

# function words carry no subject; leaving them in makes every pair of short
# prose chunks look like near-duplicates ("the X is in Y" vs "the X is in Z").
_STOP = frozenset("""a an and are as at be been but by can did do does for from had
has have he her his i if in into is it its me my no not of on or our she so than
that the their them then there these they this to us was we were what when where
which who will with would you your now also just""".split())


def to_epoch(at) -> float:
    """Normalise a timestamp to epoch seconds. Accepts None (now), int/float epoch,
    datetime (naive = local time, as datetime.timestamp() does), date (local
    midnight), timedelta (that long AGO), or an ISO-8601 string."""
    if at is None:
        return _time.time()
    if isinstance(at, bool):
        raise TypeError("at= must be a time, not a bool")
    if isinstance(at, (int, float)):
        return float(at)
    if isinstance(at, datetime):
        return at.timestamp()
    if isinstance(at, date):
        return datetime(at.year, at.month, at.day).timestamp()
    if isinstance(at, timedelta):
        return _time.time() - at.total_seconds()
    if isinstance(at, str):
        return datetime.fromisoformat(at.strip().replace("Z", "+00:00")).timestamp()
    raise TypeError(f"unsupported timestamp: {at!r}")


def to_datetime(ts):
    """Epoch -> timezone-aware UTC datetime (None stays None)."""
    return None if ts is None else datetime.fromtimestamp(ts, tz=timezone.utc)


def half_life_of(recency):
    """Resolve the `recency=` option to a half-life in seconds, or None (off)."""
    if recency is None or recency is False:
        return None
    if recency is True or recency == "prefer":
        return DEFAULT_HALF_LIFE
    if isinstance(recency, timedelta):
        recency = recency.total_seconds()
    if isinstance(recency, (int, float)) and recency > 0:
        return float(recency)
    raise ValueError(f"recency must be None, 'prefer' or a positive half-life, "
                     f"not {recency!r}")


class Hit(tuple):
    """A retrieval hit: still the (id, text, score) 3-tuple every caller unpacks,
    plus `.ts` (epoch seconds, or None if the chunk predates timestamps) and
    `.at` (the same as an aware UTC datetime) for rendering "as of <date>"."""

    def __new__(cls, cid, text, score, ts=None):
        h = super().__new__(cls, (cid, text, score))
        h.ts = ts
        return h

    def __reduce__(self):                    # pickle/copy keep the timestamp
        return (Hit, (self[0], self[1], self[2], self.ts))

    @property
    def at(self):
        return to_datetime(self.ts)


def _relevance(pool):
    """Each hit's relevance on a 0..1 scale relative to the best in the pool.
    Score ratios when every score is positive (BM25, RRF, cosine); otherwise rank
    position, since a ratio of mixed-sign scores means nothing."""
    scores = [h[2] for h in pool]
    if scores and all(isinstance(s, (int, float)) and s > 0 for s in scores):
        top = max(scores)
        return [s / top for s in scores]
    n = len(pool)
    return [(n - i) / n for i in range(n)]


def apply_recency(pool, times, half_life, weight=DEFAULT_WEIGHT, now=None):
    """Re-order `pool` by relevance x time decay. A chunk one half-life old keeps
    (1 - weight/2) of its relevance, a very old one (1 - weight); a chunk with no
    timestamp is treated as very old. Stable, and a no-op when no hit is timed."""
    if not pool or not any(times.get(h[0]) is not None for h in pool):
        return pool
    now = _time.time() if now is None else now
    weight = min(max(float(weight), 0.0), 1.0)
    rel = _relevance(pool)
    keyed = []
    for i, h in enumerate(pool):
        ts = times.get(h[0])
        decay = 0.0 if ts is None else 0.5 ** (max(0.0, now - ts) / half_life)
        keyed.append((rel[i] * ((1.0 - weight) + weight * decay), -i, h))
    keyed.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [h for _, _, h in keyed]


def content_terms(text) -> frozenset:
    """The subject-bearing terms of a chunk (stopwords dropped)."""
    return frozenset(t for t in terms(text or "") if t not in _STOP and len(t) > 1)


def jaccard(a, b) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def supersede(pool, times, subjects=None, sources=None, title_versions=None,
              threshold=DEFAULT_DUP_THRESHOLD, as_of=None):
    """Latest wins: among hits that say the same thing, keep only the newest, in the
    best rank any of them held. Same thing means, strongest signal first:
      1. the same caller-supplied `subject` (exact fact key -- e.g. "office address");
      2. the same document title, older ingest (the doc was re-ingested -- every
         chunk of the superseded version drops, not just look-alikes);
      3. near-duplicate wording: content-term Jaccard >= `threshold` (None = off).
    Untimed chunks never displace timed ones. Stable otherwise."""
    subjects = subjects or {}
    sources = sources or {}
    title_versions = title_versions or {}

    def ts_of(h):
        ts = times.get(h[0])
        return float("-inf") if ts is None else ts

    def newest_version(title):
        vs = [v for v in title_versions.get(title, ()) if as_of is None or v <= as_of]
        return max(vs) if vs else None

    kept, kept_terms, by_subject = [], [], {}
    for h in pool:
        cid = h[0]
        title = sources.get(cid)
        if title is not None and times.get(cid) is not None:
            newest = newest_version(title)
            if newest is not None and times[cid] < newest:
                continue                        # an older version of this document
        subj = subjects.get(cid)
        if subj is not None:
            j = by_subject.get(subj)
            if j is not None:
                if ts_of(h) > ts_of(kept[j]):
                    kept[j], kept_terms[j] = h, content_terms(h[1])
                continue
        t = content_terms(h[1])
        if threshold is not None:
            j = next((j for j, kt in enumerate(kept_terms)
                      if jaccard(t, kt) >= threshold), None)
            if j is not None:
                if ts_of(h) > ts_of(kept[j]):
                    old_subj = subjects.get(kept[j][0])
                    if old_subj is not None and by_subject.get(old_subj) == j:
                        del by_subject[old_subj]
                    kept[j], kept_terms[j] = h, t
                    if subj is not None:
                        by_subject[subj] = j
                continue
        if subj is not None:
            by_subject[subj] = len(kept)
        kept.append(h)
        kept_terms.append(t)
    return kept
