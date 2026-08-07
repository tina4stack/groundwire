"""
Model backends -- the swappable "which model answers" layer.

Groundwire's before/after pipeline (retrieve, inject, post-fill, map-reduce) is
backend-agnostic: it rewrites the messages going in and the answer coming out,
and dispatches the middle to a Backend. A Backend streams text deltas, so the UI
sees tokens as they arrive.

    OllamaBackend("127.0.0.1:11434", "qwen2.5:7b")          # native /api/chat
    OpenAICompatBackend("https://api.openai.com/v1", "gpt-4o", api_key=...)
    OpenAICompatBackend("https://generativelanguage.googleapis.com/v1beta/openai",
                        "gemini-2.0-flash", api_key=...)     # Gemini (OpenAI mode)

`chat(messages, stream=True)` yields text deltas; `stream=False` returns the
whole string. The NDJSON/SSE parsers are pure functions so they unit-test with no
network. Stdlib HTTP only -- no provider SDKs.
"""
from __future__ import annotations

import json
import urllib.request

DEFAULT_TIMEOUT = 600


# --------------------------------------------------------------------------- #
# pure stream parsers (unit-testable without a network)
# --------------------------------------------------------------------------- #
def iter_ndjson_deltas(lines):
    """Ollama /api/chat streaming: one JSON object per line; yield each
    message.content delta. Stops at the done frame."""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        piece = (obj.get("message") or {}).get("content", "")
        if piece:
            yield piece
        if obj.get("done"):
            break


def iter_sse_deltas(lines):
    """OpenAI-compatible /v1/chat/completions streaming (SSE): lines are
    'data: {json}' with choices[0].delta.content; '[DONE]' ends it."""
    for line in lines:
        line = line.strip()
        if not line or not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            obj = json.loads(data)
        except ValueError:
            continue
        choices = obj.get("choices") or [{}]
        delta = choices[0].get("delta") or {}
        piece = delta.get("content")
        if piece:
            yield piece


def _iter_http_lines(url, payload, headers, timeout):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 method="POST")
    req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    resp = urllib.request.urlopen(req, timeout=timeout)
    return resp  # iterating a response yields bytes lines


# --------------------------------------------------------------------------- #
# backends
# --------------------------------------------------------------------------- #
class Backend:
    """A model provider. `chat` streams text deltas (or returns a string)."""
    name = "base"
    model = None

    def chat(self, messages, *, stream=True, options=None):
        raise NotImplementedError

    def models(self):
        return [self.model] if self.model else []

    # convenience: the whole answer as one string (used by map-reduce)
    def complete(self, messages, options=None) -> str:
        return "".join(self.chat(messages, stream=False, options=options))


class OllamaBackend(Backend):
    name = "ollama"

    def __init__(self, host="127.0.0.1:11434", model=None, timeout=DEFAULT_TIMEOUT):
        self.host = host.rstrip("/")
        if not self.host.startswith("http"):
            self.host = "http://" + self.host
        self.model = model
        self.timeout = timeout

    def chat(self, messages, *, stream=True, options=None):
        payload = {"model": self.model, "messages": messages, "stream": stream}
        if options:
            payload["options"] = options
        resp = _iter_http_lines(self.host + "/api/chat", payload, None, self.timeout)
        if not stream:
            obj = json.loads(resp.read())
            return [obj.get("message", {}).get("content", "")]
        return iter_ndjson_deltas(l.decode("utf-8", "ignore") for l in resp)

    def models(self):
        try:
            with urllib.request.urlopen(self.host + "/api/tags",
                                        timeout=10) as r:
                data = json.loads(r.read())
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []


class OpenAICompatBackend(Backend):
    """Any OpenAI-compatible /v1/chat/completions endpoint: OpenAI, Gemini
    (v1beta/openai), Ollama Cloud, OpenRouter, Groq, vLLM, Azure."""
    name = "openai"

    def __init__(self, base_url, model, api_key=None, timeout=DEFAULT_TIMEOUT):
        base = base_url.rstrip("/")
        self.url = base if base.endswith("completions") else base + "/chat/completions"
        self.base = base
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def _headers(self):
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def chat(self, messages, *, stream=True, options=None):
        payload = {"model": self.model, "messages": messages, "stream": stream}
        if options and "num_ctx" in options:      # OpenAI has no num_ctx; drop it
            options = {k: v for k, v in options.items() if k != "num_ctx"}
        if options:
            payload.update(options)
        resp = _iter_http_lines(self.url, payload, self._headers(), self.timeout)
        if not stream:
            obj = json.loads(resp.read())
            return [obj["choices"][0]["message"]["content"]]
        return iter_sse_deltas(l.decode("utf-8", "ignore") for l in resp)

    def models(self):
        try:
            req = urllib.request.Request(self.base + "/models")
            for k, v in self._headers().items():
                req.add_header(k, v)
            with urllib.request.urlopen(req, timeout=10) as r:
                data = json.loads(r.read())
            return [m["id"] for m in data.get("data", [])]
        except Exception:
            return [self.model] if self.model else []


# --------------------------------------------------------------------------- #
# config -> backend  (keys come from the environment, never the config file)
# --------------------------------------------------------------------------- #
# ready-made cloud endpoints so a config entry is just {name, provider, model}
PROVIDERS = {
    "openai":  ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "gemini":  ("https://generativelanguage.googleapis.com/v1beta/openai",
                "GEMINI_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "groq":    ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "ollama_cloud": ("https://ollama.com/v1", "OLLAMA_API_KEY"),
}


def make_backend(entry: dict, get_key=None) -> Backend:
    """Build a Backend from a config entry:
        {"type": "ollama", "host": "127.0.0.1:11434", "model": "qwen2.5:7b"}
        {"type": "openai-compat", "base_url": ..., "model": ..., "api_key_env": ...}
        {"type": "gemini", "model": "gemini-2.0-flash"}          # provider preset
    `get_key(name)` resolves an API key (defaults to os.environ); keys are NEVER
    read from the config file itself."""
    import os
    get_key = get_key or (lambda k: os.environ.get(k))
    typ = entry.get("type", "ollama")
    if typ == "ollama":
        return OllamaBackend(entry.get("host", "127.0.0.1:11434"), entry["model"],
                             timeout=entry.get("timeout", DEFAULT_TIMEOUT))
    if typ in PROVIDERS:                       # gemini / openai / groq / ...
        base, key_env = PROVIDERS[typ]
        return OpenAICompatBackend(entry.get("base_url", base), entry["model"],
                                   api_key=get_key(entry.get("api_key_env", key_env)))
    if typ in ("openai-compat", "openai"):
        return OpenAICompatBackend(entry["base_url"], entry["model"],
                                   api_key=get_key(entry.get("api_key_env", "")))
    raise ValueError(f"unknown backend type: {typ}")
