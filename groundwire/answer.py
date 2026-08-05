"""
Answer-generation layer: turn retrieved chunks into an answer, so the harness
can score end-to-end *answer recall* (not just retrieval recall).

    Generator.generate(question, chunks) -> str

- RegexExtractGenerator: a GPU-free "reader" that extracts the value tied to the
  entity named in the question. Proves the retrieve -> read -> score pipeline and
  demonstrates the ceiling: answer recall <= retrieval recall.
- QwenGenerator: scaffold that runs your fine-tuned Qwen2.5-7B on the retrieved
  chunks. Optional (needs transformers + a GPU); import is lazy so the harness
  works without it.
"""

from __future__ import annotations
import re


class Generator:
    name = "base"

    def generate(self, question: str, chunks) -> str:
        raise NotImplementedError


class RegexExtractGenerator(Generator):
    """Deterministic reader. Finds the entity key in the question, then the
    value bound to that entity in the retrieved text. Robust to distractors as
    long as the correct chunk was retrieved -- so answer recall tracks the
    retriever, which is exactly the point we want to measure."""

    name = "regex_extract"

    _ENT = re.compile(r"entity ([a-z]+-\d+)")
    _SEVEN = re.compile(r"\b(\d{7})\b")

    def generate(self, question, chunks):
        joined = " ".join(t for _, t, _ in chunks).lower()
        m = self._ENT.search(question.lower())
        if m:
            ent = re.escape(m.group(1))
            bound = re.search(r"entity " + ent + r"\D*(\d{7})", joined)
            if bound:
                return bound.group(1)
        allv = self._SEVEN.findall(joined)
        return allv[-1] if allv else ""


class QwenGenerator(Generator):
    """Runs a Qwen2.5 instruct model over retrieved chunks. Only the model sits
    on the GPU; the 1-3M-token corpus never does. Requires transformers+torch."""

    name = "qwen"

    def __init__(self, model="Qwen/Qwen2.5-7B-Instruct", max_new_tokens=32,
                 device_map="auto", dtype="bfloat16"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(
            model, torch_dtype=getattr(torch, dtype), device_map=device_map
        )
        self.max_new_tokens = max_new_tokens

    def generate(self, question, chunks):
        ctx = "\n\n".join(t for _, t, _ in chunks)
        msgs = [
            {"role": "system",
             "content": "Answer only from the context. Reply with the exact value."},
            {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {question}"},
        ]
        prompt = self.tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )
        text = self.tok.decode(out[0][inputs.input_ids.shape[1]:],
                               skip_special_tokens=True)
        return text.strip()


class APIGenerator(Generator):
    """Reader backed by an OpenAI-compatible /v1/chat/completions endpoint --
    e.g. your AWQ vLLM docker running tina4-qwen-v7-awq. Our own stdlib HTTP
    client: no openai package, no torch. The model stays on your GPU; we only
    send it the prompt + retrieved chunks. Config via LLM_URL / LLM_MODEL."""

    name = "api"

    def __init__(self, url=None, model=None, api_key=None, max_tokens=64,
                 timeout=120, num_ctx=None, context_chars=None):
        import os

        from .encoders import _post

        self._post = _post
        base = (url or os.environ.get("LLM_URL") or "http://localhost:8000").rstrip("/")
        self.url = base if base.endswith("completions") else base + "/v1/chat/completions"
        self.model = model or os.environ.get("LLM_MODEL", "")
        self.api_key = api_key or os.environ.get("LLM_API_KEY") \
            or os.environ.get("OPENAI_API_KEY")
        self.max_tokens = max_tokens
        self.timeout = timeout
        # Declared context window, so Groundwire.ask() can map-reduce when the
        # retrieved chunks would overflow it (instead of the server silently
        # truncating). Prefer explicit context_chars; else derive from num_ctx
        # tokens (~4 chars/token) minus room for the prompt + generation. Env
        # LLM_NUM_CTX gives a deployment default. None -> loop off (unchanged).
        num_ctx = num_ctx or int(os.environ.get("LLM_NUM_CTX", "0") or 0)
        if context_chars is not None:
            self.context_chars = context_chars
        elif num_ctx:
            self.context_chars = max(1000, (num_ctx - max_tokens - 256) * 4)
        else:
            self.context_chars = None

    def generate(self, question, chunks):
        ctx = "\n\n".join(t for _, t, _ in chunks)
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system",
                 "content": "Answer only from the context. Reply with the exact value."},
                {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {question}"},
            ],
            "max_tokens": self.max_tokens,
            "temperature": 0,
        }
        out = self._post(self.url, payload, headers=headers, timeout=self.timeout)
        return out["choices"][0]["message"]["content"].strip()


class LLMRewriter:
    """Query rewriter for MultiQueryRetriever: asks the SAME small model you
    already run (Ollama / llama.cpp / vLLM -- OpenAI-compatible) for keyword
    probes a BM25 index can match. Taps the model's parametric knowledge to
    recover exact wording ('opening words of Moby-Dick' -> 'Call me Ishmael')
    for a ~1s query-time call -- no ingest-time embedding of the corpus."""

    _PROMPT = (
        "Write {n} different short search queries for a keyword search engine "
        "to find a passage in the original text that answers the question. "
        "Keep the distinctive names, places and numbers from the question in "
        "each query. If the question refers to a famous line or passage you "
        "know, give its exact words as one query. Use words likely to appear "
        "verbatim in the source. One query per line, no numbering, no quotes.\n"
        "Example question: How does A Tale of Two Cities begin?\n"
        "Example queries:\n"
        "It was the best of times, it was the worst of times\n"
        "Tale of Two Cities opening chapter\n"
        "best of times worst of times"
    )

    def __init__(self, url=None, model=None, api_key=None, timeout=60):
        self._gen = APIGenerator(url=url, model=model, api_key=api_key,
                                 max_tokens=96, timeout=timeout)

    def __call__(self, question, n=3):
        out = self._gen._post(self._gen.url, {
            "model": self._gen.model,
            "messages": [
                {"role": "system", "content": self._PROMPT.format(n=n)},
                {"role": "user", "content": question},
            ],
            "max_tokens": 96,
            "temperature": 0,
        }, headers={"Authorization": f"Bearer {self._gen.api_key}"}
            if self._gen.api_key else {}, timeout=self._gen.timeout)
        text = out["choices"][0]["message"]["content"]
        lines = []
        for ln in text.splitlines():
            ln = ln.strip().strip('"').lstrip("-*0123456789. ").strip()
            if ln and ln.lower() != question.lower():
                lines.append(ln)
        return lines[:n]


GENERATORS = {
    "regex_extract": RegexExtractGenerator,
    "qwen": QwenGenerator,
    "api": APIGenerator,
}


def make_generator(name: str, **kwargs) -> Generator:
    if name in (None, "none"):
        return None
    if name not in GENERATORS:
        raise ValueError(f"unknown generator '{name}'. choices: {list(GENERATORS)}")
    return GENERATORS[name](**kwargs)
