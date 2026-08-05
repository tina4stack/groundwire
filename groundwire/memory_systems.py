"""
Pluggable memory / retrieval backends for the long-context recall harness.

Each backend implements the MemorySystem interface:

    reset()                       -> clear stored state
    ingest(chunks)                -> index an iterable of (chunk_id, text)
    query(question, k) -> list    -> return top-k [(chunk_id, text, score)]

WHY THIS SHAPE
--------------
The harness measures *retrieval recall*: whether the chunk that contains the
planted needle is returned in the top-k. That number is the CEILING on the
end-to-end answer recall of any RAG / kNN-memory system built around an LLM.
If the retriever never surfaces the needle, no model on top can answer it.
So we isolate and stress the part that actually decides whether 100% recall
at 1-3M tokens is even possible -- and it runs entirely off-GPU.

BACKENDS (all stdlib, no downloads, ~zero GPU)
  - InMemoryBM25 : classic BM25 lexical retrieval held in RAM
  - SqliteFTS    : SQLite FTS5 full-text index backed by a file on disk
                   (low, ~constant RAM; scales to millions of tokens)
  - HashEmbedding: dense cosine over hashed bag-of-words vectors
                   (optional; pure stdlib) -- a stand-in for real embeddings

To plug in production components (sentence-transformers + FAISS, or an
exact-KV kNN store around your fine-tuned Qwen), see README.md.
"""

from __future__ import annotations
import math
import os
import re
import sqlite3
import tempfile
import threading
import unicodedata
import uuid
import zlib
from collections import Counter, defaultdict

_WORD_RE = re.compile(r"[a-z0-9]+")


_NUM_COMMA = re.compile(r"(?<=\d),(?=\d)")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def fold(text: str) -> str:
    """Lowercase + strip diacritics + join comma-grouped numbers + split
    camelCase, so 'Bezúkhov' matches 'Bezukhov', prisoner '24,601' matches
    '24601', and 'ForeignKeyField' matches 'foreign key field'. Without this
    the needle is unreachable no matter how good the retriever is: an ASCII
    tokenizer shreds accented names ('bezúkhov' -> 'bez','khov'), digit
    grouping splits ids ('24,601' -> '24','601'), and code identifiers become
    single opaque tokens. snake_case already splits for free."""
    text = _CAMEL.sub(" ", _NUM_COMMA.sub("", text)).lower()
    return unicodedata.normalize("NFKD", text) \
        .encode("ascii", "ignore").decode("ascii")


def _light_stem(t: str) -> str:
    """Fold simple plurals: strip one trailing s, never from ss/us/is endings
    (class, status, axis). QUERY-side only -- stemming the index dilutes the
    IDF of discriminative code tokens (measured: -2 full-hit on the battery),
    but a stemmed query token reaching an exact index token only connects
    ('fields' in the question finds 'Field' definitions)."""
    if len(t) > 3 and t.endswith("s") and not t.endswith(("ss", "us", "is")):
        return t[:-1]
    return t


def terms(text: str):
    """Accent-folded lowercase alphanumeric tokenizer. Keeps digits so numeric
    needle keys (e.g. entity '4417') remain strong, distinctive signals."""
    return _WORD_RE.findall(fold(text))


class MemorySystem:
    name = "base"

    def reset(self):
        raise NotImplementedError

    def ingest(self, chunks):
        raise NotImplementedError

    def query(self, question, k=5):
        raise NotImplementedError

    def close(self):
        """Release any OS resources (DB connections, file handles). Backends
        that hold none can rely on this no-op default."""
        pass


# --------------------------------------------------------------------------- #
# 1. BM25 in RAM
# --------------------------------------------------------------------------- #
class InMemoryBM25(MemorySystem):
    """Okapi BM25. Pure Python, held in RAM. Great baseline: on lexically
    distinctive needles it routinely hits 100% retrieval recall."""

    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.reset()

    def reset(self):
        self.ids = []
        self.texts = []
        self.tfs = []
        self.dls = []
        self.df = defaultdict(int)
        self._pos = {}
        self.N = 0
        self.avgdl = 0.0

    def ingest(self, chunks):
        for cid, text in chunks:
            tf = Counter(terms(text))
            self._pos[cid] = len(self.ids)
            self.ids.append(cid)
            self.texts.append(text)
            self.tfs.append(tf)
            self.dls.append(sum(tf.values()))
            for t in tf:
                self.df[t] += 1
        self.N = len(self.ids)
        self.avgdl = (sum(self.dls) / self.N) if self.N else 0.0

    def get(self, cid):
        """Fetch a chunk's text by id (None if absent) -- enables neighbor-pull."""
        i = self._pos.get(cid)
        return self.texts[i] if i is not None else None

    def _state(self):
        return {"ids": self.ids, "texts": self.texts, "tfs": self.tfs,
                "dls": self.dls, "df": dict(self.df), "pos": self._pos,
                "N": self.N, "avgdl": self.avgdl, "k1": self.k1, "b": self.b}

    def _restore(self, s):
        self.reset()
        self.ids, self.texts, self.tfs = s["ids"], s["texts"], s["tfs"]
        self.dls, self._pos = s["dls"], s["pos"]
        self.df = defaultdict(int, s["df"])
        self.N, self.avgdl, self.k1, self.b = s["N"], s["avgdl"], s["k1"], s["b"]

    def query(self, question, k=5):
        # query-side plural folding: emit the stemmed form ALONGSIDE the
        # original token, so 'fields' also reaches indexed 'field' without
        # touching index IDF
        q = terms(question)
        q += [s for s in (_light_stem(t) for t in q) if s not in q]
        idf = {
            t: math.log(1 + (self.N - self.df[t] + 0.5) / (self.df[t] + 0.5))
            for t in set(q)
            if t in self.df
        }
        scored = []
        k1, b, avgdl = self.k1, self.b, self.avgdl
        for i, (tf, dl) in enumerate(zip(self.tfs, self.dls)):
            s = 0.0
            for t in q:
                f = tf.get(t)
                if f and t in idf:
                    s += idf[t] * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl))
            if s > 0:
                scored.append((s, i))
        scored.sort(reverse=True)
        return [(self.ids[i], self.texts[i], s) for s, i in scored[:k]]


# --------------------------------------------------------------------------- #
# 2. SQLite FTS5 on disk  (the "database + disk, low RAM" answer)
# --------------------------------------------------------------------------- #
class SqliteFTS(MemorySystem):
    """SQLite FTS5 full-text index persisted to a file. RAM stays roughly
    constant regardless of corpus size because the index lives on disk -- this
    is the pragmatic way to hold a 3M-token haystack without touching the GPU."""

    name = "sqlite_fts"

    def __init__(self, path: str | None = None):
        # A unique per-instance file by default. A fixed shared path let two
        # SqliteFTS in one process clobber each other: the second's reset()
        # os.remove()d the DB the first was still using, so one corpus went
        # empty and the other crashed with "readonly database". Callers wanting
        # a stable location can still pass one explicitly.
        self._is_temp = path is None
        if path is None:
            path = os.path.join(tempfile.gettempdir(),
                                f"groundwire_fts_{uuid.uuid4().hex}.db")
        self.path = path
        self.conn = None
        # One connection shared across threads (check_same_thread=False) with a
        # lock serializing access -- a shared retriever is queried from worker
        # threads, which a same-thread-only connection would reject outright.
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        with self._lock:
            if self.conn is not None:
                self.conn.close()
            if os.path.exists(self.path):
                os.remove(self.path)
            self.conn = sqlite3.connect(self.path, check_same_thread=False)
            try:
                # `body` holds fold(text) so matching is symmetric with the
                # query side (and with InMemoryBM25): camelCase identifiers and
                # comma-grouped numbers split the SAME way on both sides, so a
                # query for 'field' reaches 'IntegerField'. `raw` (UNINDEXED)
                # keeps the original text to return to the reader.
                self.conn.execute(
                    "CREATE VIRTUAL TABLE chunks "
                    "USING fts5(cid UNINDEXED, raw UNINDEXED, body)"
                )
            except sqlite3.OperationalError as e:
                raise RuntimeError(
                    "SQLite was built without FTS5. Use the bm25 backend, or a "
                    "Python/SQLite with FTS5 enabled."
                ) from e

    def ingest(self, chunks):
        with self._lock:
            self.conn.executemany(
                "INSERT INTO chunks(cid, raw, body) VALUES (?, ?, ?)",
                ((str(cid), text, fold(text)) for cid, text in chunks),
            )
            self.conn.commit()

    def query(self, question, k=5):
        # Match the folded body on EXACT tokens (plus each token's light stem),
        # mirroring InMemoryBM25. The old 'tok*' prefix wildcard fired on junk
        # short tokens ('i*','a*','to*' -> issue/author/token), polluting the
        # ranking; folding both sides is what let it be dropped.
        toks = set()
        for t in terms(question):
            toks.add(t)
            toks.add(_light_stem(t))
        toks.discard("")
        if not toks:
            return []
        match = " OR ".join('"%s"' % t for t in sorted(toks))  # quoted -> safe
        with self._lock:
            rows = self.conn.execute(
                "SELECT cid, raw, bm25(chunks) AS s FROM chunks "
                "WHERE chunks MATCH ? ORDER BY s LIMIT ?",
                (match, k),
            ).fetchall()
        # sqlite bm25(): more negative = better; flip sign for a consistent
        # "higher is better" score.
        return [(int(cid), raw, -s) for cid, raw, s in rows]

    def get(self, cid):
        """Fetch a chunk's ORIGINAL text by id (None if absent) -- enables the
        code neighbor-pull in Groundwire._stitch so a definition split across a
        chunk boundary comes back whole."""
        with self._lock:
            row = self.conn.execute(
                "SELECT raw FROM chunks WHERE cid = ? LIMIT 1", (str(cid),)
            ).fetchone()
        return row[0] if row else None

    def _state(self):
        # Snapshot the ORIGINAL text; _restore re-folds on ingest. Lets a
        # persisted corpus reload via Groundwire.save()/load() the same way the RAM
        # backends do.
        with self._lock:
            rows = self.conn.execute("SELECT cid, raw FROM chunks").fetchall()
        return {"rows": [(int(cid), raw) for cid, raw in rows]}

    def _restore(self, s):
        self.reset()
        self.ingest(s["rows"])

    def close(self):
        # Release the connection so the next run (or reset) can delete the DB
        # file. On Windows an open handle locks the file, so this is required
        # for a multi-context sweep to not crash on os.remove. Also drop the
        # auto-created temp DB so unique-per-instance paths don't accumulate.
        with self._lock:
            if self.conn is not None:
                self.conn.close()
                self.conn = None
            if self._is_temp and os.path.exists(self.path):
                try:
                    os.remove(self.path)
                except OSError:
                    pass


# --------------------------------------------------------------------------- #
# 3. Hashed dense embedding (optional; pure stdlib) -- embedding stand-in
# --------------------------------------------------------------------------- #
class HashEmbedding(MemorySystem):
    """Cosine similarity over hashed bag-of-words vectors. A dependency-free
    stand-in for real dense retrieval so you can wire the dense code path and
    later swap in sentence-transformers / BGE + FAISS with no harness changes."""

    name = "hash_dense"

    def __init__(self, dim: int = 2048):
        self.dim = dim
        self.reset()

    def reset(self):
        self.ids = []
        self.texts = []
        self.mat = []  # list of row-normalized unit vectors (list[list[float]])

    def _vec(self, text):
        from .veclite import l2_normalize
        v = [0.0] * self.dim
        for t in terms(text):
            # zlib.crc32, not builtin hash(): hash() is salted per process
            # (PYTHONHASHSEED), so an index built in one process and queried in
            # another bucketed identical tokens differently -> similarity noise.
            v[zlib.crc32(t.encode()) % self.dim] += 1.0
        return l2_normalize(v)

    def ingest(self, chunks):
        for cid, text in chunks:
            self.ids.append(cid)
            self.texts.append(text)
            self.mat.append(self._vec(text))

    def query(self, question, k=5):
        if not self.mat:
            return []
        from .veclite import cosine_scores, argsort_desc
        q = self._vec(question)
        sims = cosine_scores(q, self.mat)
        idx = argsort_desc(sims)[:k]
        return [(self.ids[i], self.texts[i], float(sims[i])) for i in idx]


# --------------------------------------------------------------------------- #
# 4. Hybrid retrieval via Reciprocal Rank Fusion (lexical + dense/lexical)
# --------------------------------------------------------------------------- #
class HybridRetriever(MemorySystem):
    """Fuse several backends with Reciprocal Rank Fusion (RRF). This is how you
    hold recall when needles stop being lexically obvious: BM25 catches exact
    tokens, a dense backend catches paraphrases, RRF blends their rankings
    without tuning score scales."""

    name = "hybrid"

    def __init__(self, backends=None, encoder=None, rrf_k: int = 60):
        if backends is None:
            if encoder is not None or os.environ.get("EMBED_URL") \
                    or os.environ.get("EMBED_MODEL"):
                # the real thing: exact tokens (BM25) + meaning (dense)
                backends = [InMemoryBM25(), DenseEncoderMemory(encoder=encoder)]
            else:
                # no encoder available -- fuse the two lexical views
                backends = [InMemoryBM25(), SqliteFTS()]
        self.backends = backends
        self.rrf_k = rrf_k

    def reset(self):
        for b in self.backends:
            b.reset()

    def close(self):
        for b in self.backends:
            b.close()

    def ingest(self, chunks):
        chunks = list(chunks)  # replay into every backend
        for b in self.backends:
            b.ingest(chunks)

    def query(self, question, k=5):
        pool = max(k * 4, 20)
        fused = defaultdict(float)
        texts = {}
        for b in self.backends:
            for rank, (cid, text, _) in enumerate(b.query(question, k=pool)):
                fused[cid] += 1.0 / (self.rrf_k + rank + 1)
                texts[cid] = text
        top = sorted(fused.items(), key=lambda x: -x[1])[:k]
        return [(cid, texts[cid], score) for cid, score in top]


# --------------------------------------------------------------------------- #
# 5. Dense embeddings (our own HTTP encoders) -- the NoLiMa fix
# --------------------------------------------------------------------------- #
class DenseEncoderMemory(MemorySystem):
    """Dense retrieval via an embeddings server you own (Ollama / vLLM). This is
    what bridges NoLiMa-style low-lexical-overlap needles: it matches on meaning,
    not shared tokens. No third-party ML library -- see encoders.py (stdlib HTTP).

    `encoder` is any callable list[str] -> (N, d) L2-normalized array; pass a
    string spec ("ollama:nomic-embed-text", "openai:...", "hash:512") or leave it
    None to read EMBED_URL / EMBED_MODEL from the environment. Similarity is a
    plain dot product (veclite; vectors are normalized), so retrieval is dependency-
    light and the index is just an array you can save/load."""

    name = "dense"

    def __init__(self, encoder=None, batch_size: int = 64):
        from .encoders import make_encoder
        self.encode = make_encoder(encoder)
        self.batch_size = batch_size
        self.reset()

    def reset(self):
        self.ids = []
        self.texts = []
        self.mat = []   # list of embedding row vectors (list[list[float]])

    def _emb(self, texts, is_query):
        # pass is_query so asymmetric encoders (nomic/E5) apply the right
        # task prefix; plain callables that don't accept it still work
        try:
            return self.encode(texts, is_query=is_query)
        except TypeError:
            return self.encode(texts)

    def ingest(self, chunks):
        chunks = list(chunks)
        if not chunks:
            return          # nothing to embed. Never fabricate a zero-width block.
        bodies = [t for _, t in chunks]
        for cid, text in chunks:
            self.ids.append(cid)
            self.texts.append(text)
        for i in range(0, len(bodies), self.batch_size):
            for row in self._emb(bodies[i:i + self.batch_size], is_query=False):
                self.mat.append(list(row))

    def query(self, question, k=5):
        if not self.mat or not self.ids:
            return []
        from .veclite import cosine_scores, argsort_desc
        q = list(self._emb([question], is_query=True)[0])
        sims = cosine_scores(q, self.mat)
        idx = argsort_desc(sims)[:k]
        return [(self.ids[i], self.texts[i], float(sims[i])) for i in idx]

    def save(self, path):
        """Persist the index (embeddings + texts) — stdlib pickle, no numpy .npz."""
        import pickle
        with open(path, "wb") as fh:
            pickle.dump({"mat": self.mat, "ids": self.ids, "texts": self.texts},
                        fh, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path):
        import pickle
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        self.mat, self.ids, self.texts = d["mat"], list(d["ids"]), list(d["texts"])
        return self


# --------------------------------------------------------------------------- #
# 6. Iterative retrieval -- the multi-hop / variable-tracking fix
# --------------------------------------------------------------------------- #
class IterativeRetriever(MemorySystem):
    """Follow references across hops. After an initial retrieval, it scans the
    retrieved text for 'variable <name>' references it hasn't chased yet and
    issues follow-up queries, accumulating chunks. This turns a single-shot
    retriever (which finds only the last link of a chain) into one that walks the
    chain to the value -- the RULER variable-tracking fix. Model-free."""

    name = "iterative"

    _REF = re.compile(r"variable ([a-z0-9]+)")

    def __init__(self, base=None, max_hops: int = 24, per_hop: int = 5):
        self.base = base or SqliteFTS()
        self.max_hops = max_hops
        self.per_hop = per_hop

    def reset(self):
        self.base.reset()

    def ingest(self, chunks):
        self.base.ingest(chunks)

    def query(self, question, k=5):
        """Focused pointer-walk: follow ONE variable at a time to its definition
        ('X equal variable Y' -> chase Y; 'X be <value>' -> done). Querying a
        single variable avoids the cross-chain contamination that a broad
        multi-ref frontier suffers from."""
        collected = {}  # cid -> (text, score, hop)
        seeds = self._REF.findall(question)
        current = seeds[-1] if seeds else None
        visited = set()
        hop = 0
        while current and current not in visited and hop < self.max_hops:
            visited.add(current)
            res = self.base.query(f"variable {current}", k=self.per_hop)
            nxt = None
            for cid, text, score in res:
                eq = re.search(rf"variable {re.escape(current)} equal variable "
                               r"([a-z0-9]+)", text)
                be = re.search(rf"variable {re.escape(current)} (?:be|is|=) ", text)
                if eq or be:
                    collected.setdefault(cid, (text, score, hop))
                    nxt = eq.group(1) if eq else None
                    break
            else:
                if res:  # keep best even if no explicit definition matched
                    cid, text, score = res[0]
                    collected.setdefault(cid, (text, score, hop))
            current = nxt
            hop += 1
        ranked = sorted(collected.items(), key=lambda x: (-x[1][2], -x[1][1]))
        out = [(cid, t, s) for cid, (t, s, _) in ranked[:k]]
        if len(out) < k:  # pad from a direct query
            have = {cid for cid, _, _ in out}
            for cid, t, s in self.base.query(question, k=k):
                if cid not in have:
                    out.append((cid, t, s))
                if len(out) >= k:
                    break
        return out[:k]


# --------------------------------------------------------------------------- #
# 7. Multi-query expansion -- the paraphrase fix WITHOUT embeddings
# --------------------------------------------------------------------------- #
class MultiQueryRetriever(MemorySystem):
    """Bridge the vocabulary gap at QUERY time instead of ingest time. A
    rewriter (typically the same small LLM you already run as the reader)
    turns the question into a few keyword probes -- often recovering exact
    phrases from parametric knowledge ('opening words of Moby-Dick' ->
    'Call me Ishmael') -- and the rankings are RRF-fused.

    Contrast with dense embeddings: an embedding index costs GPU/compute over
    the WHOLE corpus at ingest (and re-embedding on model change); this costs
    one tiny LLM call per query and keeps ingest lexical-fast. `rewriter` is
    any callable (question, n) -> list[str]; keeping it a callable keeps this
    module model-free."""

    name = "multiquery"

    # words that never NAME a symbol -- generic question/code vocabulary
    _DEF_STOP = frozenset(
        "the and what how does can which where when who why exact exactly "
        "list all available module class function functions method methods "
        "def signature parameters params arguments accept accepts take takes "
        "environment variable current using use used with that this are is "
        "was were has have had for from into onto factory factories create "
        "define defined chain modify reference".split())

    # bidirectional code abbreviations -- a query-side synonym probe, no LLM.
    # 'auth' also searches 'authentication' and vice versa.
    _SYN = {}
    for _a, _b in [("auth", "authentication"), ("config", "configuration"),
                   ("db", "database"), ("param", "parameter"),
                   ("arg", "argument"), ("fn", "function"),
                   ("func", "function"), ("repo", "repository"),
                   ("init", "initialize"), ("env", "environment"),
                   ("msg", "message"), ("req", "request"), ("res", "response"),
                   ("attr", "attribute"), ("dir", "directory"),
                   ("ctx", "context"), ("mw", "middleware"),
                   ("ws", "websocket"), ("fk", "foreignkey")]:
        _SYN.setdefault(_a, set()).add(_b)
        _SYN.setdefault(_b, set()).add(_a)

    def _syn_probe(self, question):
        """Build one extra probe swapping known abbreviations for their long
        form (and back). Returns '' when the question has no such token."""
        toks = terms(question)
        hits = [s for t in toks for s in self._SYN.get(t, ())]
        return " ".join(hits)

    def __init__(self, base=None, rewriter=None, n: int = 3, rrf_k: int = 60,
                 orig_weight: float = 3.0, phrase_bonus: float = 0.5,
                 def_bonus: float = 0.3):
        self.base = base or InMemoryBM25()
        self.rewriter = rewriter
        self.n = n
        self.rrf_k = rrf_k
        # the user's own words are the strongest signal; probes only assist.
        # Equal-weight RRF lets three mediocre probes outvote a good question.
        self.orig_weight = orig_weight
        # a probe found VERBATIM in a chunk (a quote recovered from the
        # model's parametric knowledge) is near-certain evidence -- it must
        # beat bag-of-words scores ('Call me Ishmael' vs Bible genealogies
        # dense in 'Ishmael').
        self.phrase_bonus = phrase_bonus
        # a chunk DEFINING a symbol named in the question ('function add(')
        # outranks the many chunks merely USING it -- 'add' and 'router' are
        # low-IDF words, so tests/docs otherwise crowd out the source.
        self.def_bonus = def_bonus
        # introspection: what was actually searched. Thread-local so concurrent
        # requests on a shared retriever don't clobber each other's provenance.
        self._tl = threading.local()

    @property
    def last_queries(self):
        return getattr(self._tl, "queries", [])

    @last_queries.setter
    def last_queries(self, v):
        self._tl.queries = v

    def reset(self):
        self.base.reset()

    def close(self):
        self.base.close()

    def ingest(self, chunks):
        self.base.ingest(chunks)

    def get(self, cid):
        return self.base.get(cid) if hasattr(self.base, "get") else None

    def _state(self):
        return self.base._state()

    def _restore(self, s):
        self.base._restore(s)

    def query(self, question, k=5):
        queries = [question]
        syn = self._syn_probe(question)
        if syn:
            queries.append(syn)
        if self.rewriter is not None:
            try:
                extra = self.rewriter(question, self.n)
                queries += [q.strip() for q in extra if q and q.strip()]
            except Exception:
                pass  # expansion is best-effort; the raw question still runs
        self.last_queries = queries
        pool = max(k * 6, 30)
        fused = defaultdict(float)
        texts = {}
        orig_top = []            # the raw question's own best hits
        # include synonym expansions so the synonym probe isn't gated as
        # off-topic (it shares no surface tokens with the question)
        q_terms = set(terms(question)) | set(terms(syn))

        def phrase_pat(q):
            ql = fold(q).strip(".,;:!? ")
            if len(ql.split()) < 2:
                return None
            # word-boundary match, not substring: 'us marshals' must not
            # score inside 'famous marshals'
            return re.compile(r"(?<![a-z0-9])" + re.escape(ql) + r"(?![a-z0-9])")

        for qi, q in enumerate(queries):
            hits = self.base.query(q, k=pool)
            # An off-topic probe ('Sherlock Holmes' for a Les Mis question)
            # and an inspired one ('Call me Ishmael' for 'how does Moby-Dick
            # open?') BOTH share no words with the question. The difference
            # is verbatim corpus evidence. A no-overlap probe gets no
            # bag-of-words vote -- only its EXACT-phrase matches may score.
            if qi > 0 and not (set(terms(q)) & q_terms):
                pat = phrase_pat(q)
                if pat is None:
                    continue
                for cid, text, _ in hits:
                    if pat.search(fold(text)):
                        texts[cid] = text   # scored once by the phrase pass below
                continue
            w = self.orig_weight if qi == 0 else 1.0
            for rank, (cid, text, _) in enumerate(hits):
                fused[cid] += w / (self.rrf_k + rank + 1)
                texts[cid] = text
                if qi == 0 and rank < 2:
                    orig_top.append(cid)
        # verbatim-phrase pass over candidates for the on-topic probes too:
        # an exact multi-word match outranks any bag-of-words evidence
        for q in queries[1:]:
            pat = phrase_pat(q)
            if pat is None:
                continue
            for cid, text in texts.items():
                if pat.search(fold(text)):
                    fused[cid] += self.phrase_bonus
        # definition boost: candidate symbol names are the non-generic terms
        # of the question AND the probes (the probes usually name the actual
        # symbol: 'create a token' -> 'getToken'), plus consecutive pairs
        # (camelCase symbols fold to pairs: RouteRef -> 'route ref'). A chunk
        # that DEFINES one gets the bonus once.
        names = []
        for q in queries:
            syms = [t for t in terms(q)
                    if len(t) >= 3 and t not in self._DEF_STOP]
            names += syms + [f"{a} {b}" for a, b in zip(syms, syms[1:])]
        if names:
            def_pat = re.compile(
                r"(?:async def|def|function|fn|func|class|interface|trait)\s+(?:"
                + "|".join(re.escape(n) for n in set(names))
                + r")(?![a-z0-9])")
            for cid, text in texts.items():
                if def_pat.search(fold(text)):
                    fused[cid] += self.def_bonus
        ranked = [cid for cid, _ in sorted(fused.items(), key=lambda x: -x[1])]
        # safety floor: probes may ADD candidates but never evict what the
        # user's own words found. Keep the raw question's top hits in top-k.
        # De-dupe orig_top and clamp the fill count at 0: with k=1 both raw-top
        # cids can be evicted, making `k - len(keep)` negative -- `ranked[:-1]`
        # then returns almost EVERYTHING, so a k=1 query returned ~4 chunks with
        # a duplicate. max(0, ...) plus the trailing [:k] bound it back to k.
        seen = set()
        keep = [c for c in orig_top
                if c not in ranked[:k] and not (c in seen or seen.add(c))]
        top = (ranked[:max(0, k - len(keep))] + keep)[:k] if keep else ranked[:k]
        return [(cid, texts[cid], fused[cid]) for cid in top]


BACKENDS = {
    "bm25": InMemoryBM25,
    "sqlite_fts": SqliteFTS,
    "hash_dense": HashEmbedding,
    "hybrid": HybridRetriever,
    "dense": DenseEncoderMemory,
    "iterative": IterativeRetriever,
    "multiquery": MultiQueryRetriever,
}


def make_backend(name: str, **kwargs) -> MemorySystem:
    if name not in BACKENDS:
        raise ValueError(f"unknown backend '{name}'. choices: {list(BACKENDS)}")
    return BACKENDS[name](**kwargs)
