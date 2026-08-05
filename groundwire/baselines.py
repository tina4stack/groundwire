"""
Baseline systems for head-to-head comparison (see BENCHMARKS.md).

A "system" answers a question given the *full* corpus. We compare three families
on identical haystacks/questions and the shared scorecard:

  A. FullContextBaseline  -- stuff the whole context into a long-context model
  B. (SSM/linear)         -- run RWKV/Mamba over the whole context (external)
  C. NaiveRAGBaseline     -- single dense retriever + reader (RAG floor)
  D. our hybrid retriever + reader (see memory_systems.HybridRetriever + answer)

Group A/B need transformers+GPU; imports are lazy so this file stays importable.
REFERENCE_SCORES holds published numbers we are trying to match or beat.
"""

from __future__ import annotations


class System:
    """Answer a question given the full corpus (list of (chunk_id, text))."""
    name = "base"

    def answer(self, question: str, corpus) -> str:
        raise NotImplementedError


class NaiveRAGBaseline(System):
    """The RAG floor: one dense retriever, fixed chunks, a reader. Beating this
    is the minimum bar for our hybrid+rerank approach to justify itself."""

    name = "naive_rag"

    def __init__(self, backend=None, reader=None, k=5):
        from .memory_systems import HashEmbedding
        from .answer import RegexExtractGenerator

        self.backend = backend or HashEmbedding()
        self.reader = reader or RegexExtractGenerator()
        self.k = k

    def answer(self, question, corpus):
        self.backend.reset()
        self.backend.ingest(corpus)
        hits = self.backend.query(question, k=self.k)
        return self.reader.generate(question, hits)


class FullContextBaseline(System):
    """Group A: feed the entire corpus to a long-context model. This is the
    thing we claim to beat on GPU memory and cost -- it must hold a KV cache for
    the whole context. Requires transformers + a big-context model + GPU."""

    name = "full_context"

    def __init__(self, model="Qwen/Qwen2.5-7B-Instruct", max_new_tokens=32,
                 dtype="bfloat16"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(
            model, torch_dtype=getattr(torch, dtype), device_map="auto"
        )
        self.max_new_tokens = max_new_tokens

    def answer(self, question, corpus):
        context = "\n".join(t for _, t in corpus)
        msgs = [
            {"role": "system", "content": "Answer only from the context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ]
        prompt = self.tok.apply_chat_template(msgs, tokenize=False,
                                              add_generation_prompt=True)
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                  do_sample=False)
        return self.tok.decode(out[0][inputs.input_ids.shape[1]:],
                               skip_special_tokens=True).strip()


# Published reference points (see BENCHMARKS.md sources). Targets to match/beat.
REFERENCE_SCORES = {
    "single_needle_niah": {"gemini_1_5_pro": 0.997},
    "realistic_multi_fact_recall": {"strong_models_avg": 0.60},
    "ruler_gap_vs_niah_points": (10, 25),
    "full_ctx_1M_vs_rag": {"latency_x_slower": (30, 60), "cost_x_more": (1000, 1250)},
    "notes": "RAG typically wins above ~200-400K on open/non-frontier models.",
}
