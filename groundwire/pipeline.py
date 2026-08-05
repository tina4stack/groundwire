"""
Groundwire — the orchestration layer that sits IN FRONT of your model.

It does not touch transformers internals (no attention/KV surgery). It wraps the
normal flow: ingest -> chunk -> embed/index -> retrieve -> prompt -> generate.
The model (HF transformers, vLLM AWQ, Ollama, or an API) is used unchanged as the
reader; groundwire only decides *what text* to put in the prompt.

    from groundwire.pipeline import Groundwire
    from groundwire.answer import make_generator

    mem = Groundwire(
        memory="sqlite_fts",                       # or "dense", "hybrid", "iterative"
        reader=make_generator("api"),              # your AWQ vLLM /v1/chat/completions
        k=5,
    )
    mem.ingest(open("big_doc.txt").read())          # or a list of docs
    print(mem.ask("what was the Q3 revenue figure?"))

Retrieval-only (no reader) returns the chunks, so you can build the prompt in your
own transformers/vLLM call:

    hits = Groundwire(memory="dense").ingest(docs).retrieve(question)
    context = "\\n\\n".join(t for _, t, _ in hits)
    # ... feed `context` + question into model.generate(...) or /v1/chat/completions
"""

from __future__ import annotations
import re

from .memory_systems import make_backend
from .harness import chunk_by_sentences

# Sentence boundary = end punctuation FOLLOWED BY whitespace/EOL, or a newline. The old
# pattern [^.!?\n]+[.!?]? split on EVERY '.', which shredded embedded code in doc/markdown
# chunks (db.fetch() -> "db. fetch()", orm/model.py -> "orm/model. py", User.all() ->
# "User. all()") and made tina4_context serve mangled idioms. Same rule as server.py's
# _SENT_SPLIT. Intra-token dots (method calls, module paths, file names) stay intact.
_SENT = re.compile(r"(?<=[.!?])\s+|\n+")
# chunk boundaries across languages: python defs/classes/decorators, php/js
# functions and methods, ts exports/interfaces, php traits, and Object Pascal
# unit/type/routine headers (Delphi is case-insensitive, so accept either case).
_TOPLEVEL = re.compile(
    r"^(async def |def |class |@\w"
    r"|\s*(?:public |private |protected |static |final |abstract )*function \w"
    r"|(?:export )?(?:default )?(?:abstract )?class "
    r"|interface |trait "
    r"|export (?:const|function|interface|type|async) "
    r"|[Uu]nit |[Pp]rocedure |[Ff]unction |[Cc]onstructor |[Dd]estructor "
    r"|[Tt]ype$)"
)


def chunk_text(text, max_words=350, overlap=1):
    sents = [s.strip() for s in _SENT.split(text) if s.strip()]
    return chunk_by_sentences(sents, max_words, overlap)


def chunk_code(text, path="", max_lines=60):
    """Chunk source code on top-level def/class/decorator boundaries instead
    of sentences (sentence chunking shreds code). Segments are packed up to
    max_lines, and every chunk starts with a '# file: <path>' line so the
    path's tokens are indexed -- 'where is the router?' should match
    core/router.py by name alone."""
    lines = text.splitlines()
    bounds = [i for i, l in enumerate(lines) if _TOPLEVEL.match(l)]
    if not bounds or bounds[0] != 0:
        bounds = [0] + bounds
    segments = [lines[a:b] for a, b in zip(bounds, bounds[1:] + [len(lines)])]

    chunks, cur = [], []
    for seg in segments:
        while len(seg) > max_lines:          # oversized segment: hard split
            if cur:
                chunks.append(cur)
                cur = []
            chunks.append(seg[:max_lines])
            seg = seg[max_lines:]
        if cur and len(cur) + len(seg) > max_lines:
            chunks.append(cur)
            cur = []
        cur += seg
    if cur:
        chunks.append(cur)

    header = f"# file: {path}" if path else None
    out = []
    for i, ch in enumerate(chunks):
        body = "\n".join(([header] if header else []) + ch)
        out.append((i, body))
    return out


class Groundwire:
    def __init__(self, memory="sqlite_fts", reader=None, k=5, encoder=None,
                 expand=0, rewriter=None, rerank=None, max_words=350, overlap=1,
                 verified_scorer=None, verify=None):
        kwargs = {"encoder": encoder} if memory in ("dense", "hybrid") else {}
        self.memory = make_backend(memory, **kwargs) if isinstance(memory, str) else memory
        # optional reranker: lexical retrieves a wide pool, a semantic scorer
        # reorders it before the top-k cut. DenseReranker embeds the query once
        # + not-yet-seen candidates (cached by id) at QUERY time -- no corpus
        # embedding, no persistent index, each chunk embedded at most once
        # across all queries. Cost model in groundwire/rerank.py. `rerank="dense"`
        # uses the configured encoder; a callable `rerank` is used AS the
        # encoder (back-compat). Falls back to lexical order if the endpoint is
        # unreachable, so it can only reorder a request, never break it.
        self.reranker = None
        if rerank == "dense" or (rerank and not isinstance(rerank, str)):
            from .encoders import make_encoder
            from .rerank import DenseReranker
            enc = rerank if callable(rerank) else make_encoder(encoder)
            self.reranker = DenseReranker(enc)
        if expand:
            # paraphrase robustness WITHOUT embedding the corpus: the reader's
            # own model rewrites the question into `expand` keyword probes
            from .answer import LLMRewriter
            from .memory_systems import MultiQueryRetriever
            self.memory = MultiQueryRetriever(
                self.memory, rewriter or LLMRewriter(), n=expand)
        self.reader = reader
        # optional verifier for the answer-reduce read loop (see ask()): a
        # callable verify(question, answer, chunks) -> comparable score, higher
        # = better. Used only when retrieved chunks overflow the reader's window
        # and the loop must pick among per-batch candidates. None -> the loop
        # falls back to grounded-in-source selection (see mapreduce._select).
        self.verify = verify
        self.k = k
        self.max_words = max_words
        self.overlap = overlap
        self.sources = {}       # chunk_id -> document title
        self.repos = set()      # labels registered by ingest_repo()
        self._next = 0
        # verified ranking: score each SOURCE once at ingest (offline, via the injected
        # scorer -- keeps groundwire framework-agnostic and zero-dep), read the cached score at
        # query time to sink code that doesn't RUN below code that does. Off unless a scorer
        # is supplied; GROUNDWIRE_VERIFIED_RANK=0 force-disables it for a clean A/B baseline.
        import os as _os
        self.verified_scorer = verified_scorer
        self.verified = {}      # document title -> 0 broken | 1 error | 2 boots
        self._verified_on = _os.environ.get("GROUNDWIRE_VERIFIED_RANK", "1") != "0"

    def _record_verified(self, title, text):
        """Score a source once (offline, at ingest). Failures never break ingest --
        an un-scoreable source is left neutral (never sunk)."""
        if not (self.verified_scorer and title) or title in self.verified:
            return
        try:
            self.verified[title] = int(self.verified_scorer(title, text))
        except Exception:
            self.verified[title] = 2  # neutral-high: unknown != broken

    def _verified_of(self, source):
        """Cached verified score for a chunk's source; neutral-high (2) if unscored."""
        return self.verified.get(source, 2)

    def ingest(self, docs, title=None):
        """Index documents. Accepts a string, a list of strings, or a list of
        (title, text) pairs; `title=` labels a single string. Titles become the
        source shown to the reader and returned by source_of(). Returns self."""
        if isinstance(docs, str):
            docs = [(title, docs)]
        batch = []
        for doc in docs:
            doc_title, text = doc if isinstance(doc, tuple) else (None, doc)
            self._record_verified(doc_title, text)
            for _, chunk in chunk_text(text, self.max_words, self.overlap):
                batch.append((self._next, chunk))
                if doc_title:
                    self.sources[self._next] = doc_title
                self._next += 1
        if batch:
            self.memory.ingest(batch)
        return self

    def ingest_code(self, text, title=None, max_lines=60):
        """Index source code, chunked on def/class boundaries (not sentences).
        `title` should be the file path -- it is indexed inside each chunk and
        becomes the citation source."""
        self._record_verified(title, text)
        batch = []
        for _, chunk in chunk_code(text, path=title or "", max_lines=max_lines):
            batch.append((self._next, chunk))
            if title:
                self.sources[self._next] = title
            self._next += 1
        if batch:
            self.memory.ingest(batch)
        return self

    CODE_EXTS = {".py", ".php", ".js", ".mjs", ".ts", ".rb",
                 # Object Pascal / Delphi (tina4delphi): sources, project, and
                 # form files. .dfm/.fmx are declarative but chunk fine as code.
                 ".pas", ".dpr", ".dpk", ".inc", ".dfm", ".fmx"}
    DOC_EXTS = {".md", ".txt", ".rst", ".twig", ".html"}
    # deploy/CLI/env answers live in config files, not .py sources
    CONFIG_EXTS = {".toml", ".yml", ".yaml"}
    # by exact basename; note .env/.env.local are deliberately ABSENT --
    # real env files can hold secrets and the index is sent to a model
    SPECIAL_FILES = {"dockerfile", "makefile", "docker-compose.yml",
                     "package.json", "composer.json",
                     ".env.example", ".env.sample"}
    SKIP_DIRS = {".git", "__pycache__", "node_modules", "vendor", "dist",
                 "build", "coverage", ".idea", ".venv", "venv", ".pytest_cache"}

    def ingest_repo(self, root, include=None, prefix=None, skip_dirs=None):
        """Walk a repository: code files through the code chunker, docs through
        the prose chunker, everything titled '<repo>/<relative path>'. This is
        the 'give your coding model the whole library' entry point.

        skip_dirs adds to the default SKIP_DIRS -- e.g. skip_dirs={'plan'} drops
        design/vision/spec docs that describe the framework in the abstract (and
        often mix languages), which otherwise crowd out concrete source and, for
        a small reader, invite wrong-language hallucination."""
        import os as _os
        from .extract import extract_text, EXTRACT_EXTS
        root = _os.path.abspath(root)
        label = prefix or _os.path.basename(root)
        self.repos.add(label)
        skip = self.SKIP_DIRS | set(skip_dirs or ())
        for dirpath, dirnames, filenames in _os.walk(root):
            dirnames[:] = [d for d in dirnames
                           if d not in skip and not d.startswith(".")]
            for fn in sorted(filenames):
                ext = _os.path.splitext(fn)[1].lower()
                special = fn.lower() in self.SPECIAL_FILES
                wanted = include if include is not None \
                    else self.CODE_EXTS | self.DOC_EXTS | self.CONFIG_EXTS \
                    | EXTRACT_EXTS
                if (ext not in wanted and not special) \
                        or fn.endswith(".min.js"):
                    continue
                path = _os.path.join(dirpath, fn)
                title = f"{label}/{_os.path.relpath(path, root)}"
                if ext in EXTRACT_EXTS:
                    # binary doc (Word/PDF): pull the text layer, then chunk it
                    # as prose. Skip silently if extraction yields nothing.
                    text = extract_text(path)
                    if text:
                        self.ingest(text, title=title)
                    continue
                try:
                    text = open(path, encoding="utf-8", errors="ignore").read()
                except OSError:
                    continue
                if ext in self.CODE_EXTS or ext in self.CONFIG_EXTS or special:
                    # configs go through the line-window chunker too --
                    # sentence chunking shreds YAML/Dockerfiles
                    self.ingest_code(text, title=title)
                else:
                    self.ingest(text, title=title)
        return self

    def source_of(self, chunk_id):
        """Title of the document a chunk came from (None if untitled)."""
        return self.sources.get(chunk_id)

    @staticmethod
    def _is_testlike(src):
        s = (src or "").lower()
        return any(p in s for p in ("/test", "test/", "test_", "_test.",
                                    ".test.", "/example", "example/"))

    @staticmethod
    def _is_guide(src):
        base = (src or "").lower().rsplit("/", 1)[-1]
        return base in ("llms.txt", "claude.md", "readme.md", "agents.md") \
            or "/docs/" in (src or "").lower()

    def retrieve(self, question, k=None, scope=None):
        """Query the memory with two stable reorderings of a 3x candidate
        pool:
        - repo scoping: a question naming an ingested repo is answered from
          that repo first (a multi-repo index otherwise answers python
          questions from php/js files sharing vocabulary). `scope` is the
          default repo when the question names none -- terse prompts
          ('ORM model User...') carry no repo signal at all;
        - source-over-tests: BM25 structurally favors usage-dense test
          chunks (a test mentions getToken dozens of times, the definition
          once), but the definition is the answer. Tests stay in the pool
          as usage examples -- below the source. Skipped when the question
          is itself about tests."""
        k = k or self.k
        pool = self.memory.query(question, k=k * 3)
        if self.reranker and pool:
            pool = self.reranker(question, pool)
        scoped = [r for r in self.repos if r.lower() in question.lower()] \
            or ([scope] if scope in self.repos else [])
        if scoped:
            def in_scope(h):
                src = self.sources.get(h[0]) or ""
                return any(src.startswith(r + "/") for r in scoped)
            mine = [h for h in pool if in_scope(h)]
            # A k*3 pool is often <k in-scope, so the delivered top-k backfills
            # with OTHER repos -- and a small reader then answers a python
            # question in php/js from the contaminating chunk. When we know the
            # scope, widen the pool and refill from in-scope first so the reader
            # sees one language. Cross-repo chunks stay only as backfill if the
            # scope genuinely can't fill k (keeps recall for thin repos). Note:
            # on a single-repo index every chunk is in-scope, so len(mine) < k
            # never fires and this is a no-op -- it only shapes multi-repo pools.
            if len(mine) < k:
                wide = self.memory.query(question, k=max(k * 12, 60))
                if self.reranker and wide:
                    wide = self.reranker(question, wide)
                seen = {h[0] for h in mine}
                mine += [h for h in wide if in_scope(h) and h[0] not in seen]
                rest = [h for h in pool if not in_scope(h)]
                rseen = {h[0] for h in rest}
                rest += [h for h in wide if not in_scope(h) and h[0] not in rseen]
            else:
                rest = [h for h in pool if not in_scope(h)]
            pool = mine + rest
        if self.repos and "test" not in question.lower():
            prim = [h for h in pool
                    if not self._is_testlike(self.sources.get(h[0]))]
            pool = prim + [h for h in pool if h not in prim]
        # one guide slot: the canonical idiom lives in the guide docs
        # (llms.txt, CLAUDE.md, README). Definitions keep the top ranks --
        # guides ranked first floods every query (measured: full-hit 8->2) --
        # but if no guide made top-k, the best one takes the LAST slot as
        # idiom context for the reader. With a scope active, prefer an in-scope
        # guide so the slot doesn't reintroduce another language's chunk
        # (no-op on a single-repo index -- every guide is already in-scope).
        if self.repos and len(pool) > k:
            top, rest = pool[:k], pool[k:]
            if not any(self._is_guide(self.sources.get(h[0])) for h in top):
                guides = [h for h in rest
                          if self._is_guide(self.sources.get(h[0]))]
                guide = next((h for h in guides if scoped and in_scope(h)), None) \
                    or (guides[0] if guides else None)
                if guide:
                    pool = top[:-1] + [guide] + [h for h in rest if h is not guide]
        # verified sink: a chunk whose SOURCE doesn't BOOT (verified score < 2 -- a syntax/runtime
        # break like select().to_array() OR an error like a wrong import path) drops below every
        # runnable chunk, so the most-relevant-but-broken example can never out-rank a working
        # idiom the reader will copy. "Boots" is the only safe-to-ground signal; both "broken" (0)
        # and "error" (1) are code the model must not copy. Stable within each group -- BM25/
        # structure order is preserved; non-booting sources move, kept as last backfill (never
        # dropped, so recall holds). No-op unless a scorer produced scores.
        if self.verified and self._verified_on:
            runs = [h for h in pool if self._verified_of(self.sources.get(h[0])) == 2]
            sunk = [h for h in pool if self._verified_of(self.sources.get(h[0])) != 2]
            if runs and sunk:
                pool = runs + sunk
        return self._stitch(pool, k)

    _CODE_SUFFIX = ("py", "php", "js", "mjs", "ts", "rb",
                    "pas", "dpr", "dpk", "inc", "dfm", "fmx")

    def _stitch(self, pool, k, pull=1, max_span_chars=2600):
        """Merge retrieved chunks that are NEIGHBORS in the same document into
        one continuous span. For CODE files, also pull up to `pull` adjacent
        chunks per side straight from the backend even when they weren't
        retrieved -- a definition split across a chunk boundary comes back
        whole. The span is CAPPED at max_span_chars: a merged chunk that grew
        unbounded (300+ lines) starves the reader, which only sees the first
        chunk or two once the context budget truncates. Capping keeps each
        delivered chunk focused so the reader receives several, not one giant
        one. Overflowing neighbors stay in the pool as their own chunks."""
        by_id = {h[0]: h for h in pool}
        getter = getattr(self.memory, "get", None)

        def text_of(cid):
            if cid in by_id:
                return by_id[cid][1]
            return getter(cid) if getter else None

        out, used, i = [], set(), 0
        while len(out) < k and i < len(pool):
            cid, text, score = pool[i]
            i += 1
            if cid in used:
                continue
            src = self.sources.get(cid)
            is_code = (src or "").rsplit(".", 1)[-1].lower() in self._CODE_SUFFIX
            span = [cid]
            span_chars = len(text)
            for step in (1, -1):            # extend forward, then back
                nxt, pulled = cid + step, 0
                # `src is not None` guards the untitled case: two untitled docs
                # both have source None, and None == None would merge chunks
                # ACROSS the document boundary into one blob (starving top-k).
                while src is not None \
                        and self.sources.get(nxt) == src and nxt not in used:
                    nt = text_of(nxt)
                    if nt is None or span_chars + len(nt) > max_span_chars:
                        break               # cap: keep the merged chunk focused
                    if nxt in by_id:        # retrieved neighbor: merge
                        span.append(nxt)
                    elif is_code and pulled < pull:
                        span.append(nxt)    # unretrieved code neighbor: pull it
                        pulled += 1
                    else:
                        break
                    span_chars += len(nt)
                    nxt += step
            span = sorted(set(span))
            used.update(span)
            if len(span) > 1:
                parts = [t for t in (text_of(c) for c in span) if t]
                # drop repeated '# file:' headers on the continuation chunks
                parts = parts[:1] + [p.split("\n", 1)[1]
                                     if p.startswith("# file:") and "\n" in p
                                     else p for p in parts[1:]]
                text = "\n".join(parts)
            out.append((span[0], text, score))
        return out

    def ask(self, question, k=None, reader_budget_chars=None):
        """Retrieve, then read with the model. Falls back to returning the
        joined context if no reader was configured. Retrieved chunks are
        labeled with their source title so the reader can stay grounded in
        (and cite) the right document.

        When the labeled chunks overflow the reader's context window, we don't
        truncate (which would silently drop lower-ranked chunks -- the ones a
        widened k was meant to catch). Instead we MAP the question over
        budget-sized batches and REDUCE by verified-answer selection (see
        groundwire.mapreduce.answer). The budget comes from `reader_budget_chars`,
        else the reader's own `context_chars`; if neither is set the loop is
        off and behavior is a single call, exactly as before."""
        hits = self.retrieve(question, k=k)
        labeled = [
            (cid, f"[{self.sources[cid]}] {t}" if cid in self.sources else t, s)
            for cid, t, s in hits
        ]
        if self.reader is None:
            return "\n\n".join(t for _, t, _ in labeled)
        budget = reader_budget_chars or getattr(self.reader, "context_chars", None)
        if budget and sum(len(t) for _, t, _ in labeled) > budget:
            from .mapreduce import answer as _answer
            return _answer(self.reader.generate, question, labeled, budget,
                           verify=self.verify)
        return self.reader.generate(question, labeled)

    # -- session persistence: the memory is data, not GPU state -------------- #
    def save(self, path):
        """Snapshot the whole index (backend state + the sources/repos maps
        needed for citation and scoping) to one pickle. For lexical backends
        ingest is ~instant, so this mainly matters for very large corpora or
        the dense backend (whose embeddings are expensive to recompute)."""
        import pickle
        if hasattr(self.memory, "_state"):        # lexical: data-only snapshot
            payload = {"kind": "state", "backend": self.memory._state()}
        elif hasattr(self.memory, "save"):        # dense: its own npz sidecar
            self.memory.save(path + ".npz")
            payload = {"kind": "npz"}
        else:
            raise NotImplementedError(
                f"{self.memory.name} backend has no save()")
        payload.update(sources={str(k): v for k, v in self.sources.items()},
                       repos=list(self.repos), next=self._next)
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        return self

    def load(self, path):
        import pickle
        with open(path, "rb") as f:
            payload = pickle.load(f)
        if payload["kind"] == "state":
            self.memory._restore(payload["backend"])
        else:
            self.memory.load(path + ".npz")
        self.sources = {int(k): v for k, v in payload["sources"].items()}
        self.repos = set(payload["repos"])
        self._next = payload["next"]
        return self
