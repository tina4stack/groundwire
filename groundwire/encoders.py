"""
Encoders: text -> vector. Our own tiny HTTP clients, stdlib only.

We deliberately do NOT depend on sentence-transformers / torch / HuggingFace --
those are the parts that change, break, and drag in gigabytes. An embedding is
just a vector from a model you already run. aatos already serves embeddings via
Ollama (nomic-embed-text) and a self-hosted vLLM /embeddings route, so we talk to
those over plain HTTP and own the whole path.

    make_encoder(spec) -> encode(list[str]) -> list[list[float]]  (L2-normalized rows)

spec forms:
    None                 -> read EMBED_URL / EMBED_MODEL from env, infer provider
    "ollama:MODEL"       -> Ollama  (EMBED_URL or http://localhost:11434)
    "openai:MODEL"       -> OpenAI-compatible /v1/embeddings (vLLM, Azure, etc.)
    "hash:DIM"           -> offline deterministic hashing (tests / plumbing only)
    a callable           -> used as-is

Pure stdlib — no numpy, no torch, no sentence-transformers. Vector math is in veclite.
"""

from __future__ import annotations
import json
import os
import urllib.request
import zlib

DEFAULT_OLLAMA = "http://localhost:11434"
DEFAULT_TIMEOUT = 60


def _post(url, payload, headers=None, timeout=DEFAULT_TIMEOUT):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _l2(rows):
    from .veclite import normalize_rows
    return normalize_rows(rows)


# nomic-embed-text (and other asymmetric E5-style encoders) are TRAINED with
# task prefixes; omitting them measurably hurts query<->document similarity.
# Applied by side (query vs document) so retrieval is asymmetric as intended.
NOMIC_PREFIXES = ("search_query: ", "search_document: ")


def _prefixes_for(model):
    m = (model or "").lower()
    if "nomic" in m:
        return NOMIC_PREFIXES
    if "e5" in m or "bge" in m:  # E5/BGE use query:/passage:-style prompts too
        return ("query: ", "passage: ")
    return ("", "")


class OllamaEncoder:
    """Ollama embeddings. Tries the batch /api/embed first, falls back to the
    legacy per-item /api/embeddings. Matches aatos' nomic-embed-text setup."""

    def __init__(self, model="nomic-embed-text", url=None, timeout=DEFAULT_TIMEOUT,
                 prefixes=None):
        self.model = model
        self.url = (url or os.environ.get("EMBED_URL") or DEFAULT_OLLAMA).rstrip("/")
        self.timeout = timeout
        self.q_prefix, self.d_prefix = prefixes or _prefixes_for(model)

    def __call__(self, texts, is_query=False):
        pre = self.q_prefix if is_query else self.d_prefix
        texts = [pre + t for t in texts] if pre else list(texts)
        try:
            out = _post(f"{self.url}/api/embed",
                        {"model": self.model, "input": texts},
                        timeout=self.timeout)
            if "embeddings" in out:
                return _l2(out["embeddings"])
        except Exception:
            pass  # fall back to legacy single-item endpoint
        vecs = []
        for t in texts:
            out = _post(f"{self.url}/api/embeddings",
                        {"model": self.model, "prompt": t}, timeout=self.timeout)
            vecs.append(out["embedding"])
        return _l2(vecs)


class OpenAIEmbeddingsEncoder:
    """OpenAI-compatible /v1/embeddings -- works with your vLLM route and Azure
    OpenAI. Batches natively. Reads API key from EMBED_API_KEY / OPENAI_API_KEY."""

    def __init__(self, model="nomic-embed-text", url=None, api_key=None,
                 timeout=DEFAULT_TIMEOUT, prefixes=None):
        self.model = model
        base = (url or os.environ.get("EMBED_URL") or "http://localhost:8000").rstrip("/")
        self.url = base if base.endswith("embeddings") else base + "/v1/embeddings"
        self.api_key = api_key or os.environ.get("EMBED_API_KEY") \
            or os.environ.get("OPENAI_API_KEY")
        self.timeout = timeout
        self.q_prefix, self.d_prefix = prefixes or _prefixes_for(model)

    def __call__(self, texts, is_query=False):
        pre = self.q_prefix if is_query else self.d_prefix
        texts = [pre + t for t in texts] if pre else list(texts)
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        out = _post(self.url, {"model": self.model, "input": texts},
                    headers=headers, timeout=self.timeout)
        # Order by each item's `index`, not response position: the OpenAI
        # embeddings API does NOT guarantee `data` comes back in input order
        # (that's what `index` is for). Parsing positionally would bind vectors
        # to the wrong chunks and silently corrupt the index.
        data = sorted(out["data"], key=lambda d: d.get("index", 0))
        rows = [d["embedding"] for d in data]
        return _l2(rows)


class HashEncoder:
    """Deterministic offline encoder (hashed bag-of-words). NOT semantic -- for
    testing the pipeline without any server. Real semantics come from Ollama/vLLM."""

    def __init__(self, dim=512):
        self.dim = dim

    def __call__(self, texts):
        import re

        texts = list(texts)
        rows = [[0.0] * self.dim for _ in texts]
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                # zlib.crc32, not builtin hash(): hash() is per-process salted
                # (PYTHONHASHSEED), so this "deterministic offline" encoder was
                # not actually deterministic across processes -- an index saved
                # in one process scored ~0 when queried in another. crc32 is
                # stable, keeping save()/load() and cross-process runs correct.
                rows[i][zlib.crc32(w.encode()) % self.dim] += 1.0
        return _l2(rows)


def make_encoder(spec=None):
    if callable(spec):
        return spec
    if spec is None:
        url = os.environ.get("EMBED_URL", "")
        model = os.environ.get("EMBED_MODEL", "nomic-embed-text")
        if "11434" in url or "/api/embed" in url or not url:
            return OllamaEncoder(model=model, url=url or None)
        return OpenAIEmbeddingsEncoder(model=model, url=url)
    scheme, _, rest = spec.partition(":")
    if scheme == "ollama":
        return OllamaEncoder(model=rest or "nomic-embed-text")
    if scheme in ("openai", "api"):
        return OpenAIEmbeddingsEncoder(model=rest or "nomic-embed-text")
    if scheme == "hash":
        return HashEncoder(dim=int(rest) if rest else 512)
    # bare model name -> use env-configured provider
    return make_encoder(None) if not rest else OllamaEncoder(model=spec)
