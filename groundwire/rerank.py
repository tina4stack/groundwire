"""
Dense reranker -- reorder a lexical candidate pool by semantic similarity to
the query, reusing the embeddings endpoint you already run. No separate rerank
model, no cross-encoder, and nothing on the reader's GPU.

WHY A RERANKER, AND WHAT IT COSTS
---------------------------------
Lexical retrieval (bm25 / sqlite_fts) is free -- millions of tokens/sec on CPU,
no model. But it ranks by shared tokens, so a paraphrase, or a definition buried
under many usages, can sit just below the top-k cut. A reranker re-scores the
SMALL top-(k*N) pool by meaning and lifts the right chunk into the delivered
top-k.

To compare by meaning you need vectors -- that is unavoidable. This reranker
keeps that cost at the floor and OFF the reader's GPU:

  * retrieval stays lexical                         -> 0 model calls
  * the QUERY is embedded once per query            -> 1 small call
  * each candidate chunk is embedded AT MOST ONCE   -> cached by chunk id and
    EVER                                               reused across queries and
                                                       across the pool/wide passes

So a benchmark of Q queries over a corpus pays to embed each distinct chunk
ONCE, not Q times, and never runs a second (cross-encoder) model. The embeddings
server is the same one the `dense`/`hybrid` backends already use; the reader GPU
is untouched. If no embeddings endpoint is reachable, __call__ returns the
lexical order unchanged (graceful no-op), so enabling rerank can never break a
request -- it can only reorder it.

Interface: a DenseReranker is CALLABLE -- reranker(question, pool) -> pool
reordered -- so it drops in exactly where an inline reorder would sit.
"""
from __future__ import annotations


class DenseReranker:
    """Blend the incoming lexical rank with dense cosine-to-query rank via
    Reciprocal Rank Fusion. `encoder` is any callable list[str] -> (N, d)
    L2-normalized array (an `is_query=` kwarg is used when the encoder accepts
    one, for asymmetric nomic/E5-style models). Document vectors are cached by
    chunk id (FIFO-bounded) so no chunk is embedded twice."""

    def __init__(self, encoder, blend=0.5, rrf_k=60, max_cache=50_000):
        self.encode = encoder
        self.blend = blend            # dense weight in the fusion (0 == lexical)
        self.rrf_k = rrf_k
        self.max_cache = max_cache
        self._cache = {}              # chunk id -> unit vector (document side)
        self._q_cache = (None, None)  # (question, vec): skip re-embedding the
                                      # query across the pool/wide re-rank passes

    def _emb(self, texts, is_query):
        # pass is_query so asymmetric encoders apply the right task prefix;
        # plain callables that don't accept the kwarg still work
        try:
            return self.encode(texts, is_query=is_query)
        except TypeError:
            return self.encode(texts)

    def _put(self, cid, vec):
        c = self._cache
        if cid in c:
            return
        if len(c) >= self.max_cache:      # FIFO evict -- bound long-running use
            c.pop(next(iter(c)))
        c[cid] = vec

    def _query_vec(self, question):
        if self._q_cache[0] == question:
            return self._q_cache[1]
        v = list(self._emb([question], is_query=True)[0])
        self._q_cache = (question, v)
        return v

    def __call__(self, question, pool):
        from .veclite import cosine_scores, argsort_desc
        if not pool:
            return pool
        try:
            # embed only the candidates we've never seen -- one batched call
            missing = [(cid, text) for cid, text, _ in pool
                       if cid not in self._cache]
            if missing:
                vecs = self._emb([t for _, t in missing], is_query=False)
                for (cid, _), v in zip(missing, vecs):
                    self._put(cid, list(v))
            q = self._query_vec(question)
            cand = [self._cache[cid] for cid, _, _ in pool]      # unit vectors (lists)
            sims = cosine_scores(q, cand)
            # fuse dense rank with the incoming lexical rank (pool order) via
            # RRF -- same math as the original inline reranker, so the validated
            # behavior is preserved; only the embedding cost changes.
            dense_order = argsort_desc(sims)
            drank = {idx: r for r, idx in enumerate(dense_order)}
            fused = [(self.blend / (self.rrf_k + drank[i] + 1)
                      + (1 - self.blend) / (self.rrf_k + i + 1), i)
                     for i in range(len(pool))]
            fused.sort(reverse=True)
            return [pool[i] for _, i in fused]
        except Exception:
            return pool               # any embed error -> keep the lexical order
