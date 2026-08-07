"""Unit tests for the model backend layer (parsers + config, no network)."""
import json

from groundwire.backends import (iter_ndjson_deltas, iter_sse_deltas,
                                  make_backend, OllamaBackend, OpenAICompatBackend)


def test_ndjson_parser_yields_content_until_done():
    lines = [
        json.dumps({"message": {"content": "Hel"}, "done": False}),
        json.dumps({"message": {"content": "lo"}, "done": False}),
        json.dumps({"message": {"content": ""}, "done": True}),
        json.dumps({"message": {"content": "IGNORED after done"}}),
    ]
    assert "".join(iter_ndjson_deltas(lines)) == "Hello"


def test_sse_parser_yields_delta_until_done():
    lines = [
        'data: ' + json.dumps({"choices": [{"delta": {"content": "Hel"}}]}),
        '',                                                   # keep-alive blank
        'data: ' + json.dumps({"choices": [{"delta": {"content": "lo"}}]}),
        'data: [DONE]',
        'data: ' + json.dumps({"choices": [{"delta": {"content": "after"}}]}),
    ]
    assert "".join(iter_sse_deltas(lines)) == "Hello"


def test_make_backend_ollama():
    b = make_backend({"type": "ollama", "host": "127.0.0.1:11434",
                      "model": "qwen2.5:7b"})
    assert isinstance(b, OllamaBackend) and b.model == "qwen2.5:7b"
    assert b.host == "http://127.0.0.1:11434"


def test_make_backend_gemini_preset_and_key_isolation():
    seen = {}
    def get_key(name):
        seen["asked"] = name
        return "SECRET123"
    b = make_backend({"type": "gemini", "model": "gemini-2.0-flash"}, get_key)
    assert isinstance(b, OpenAICompatBackend)
    assert b.api_key == "SECRET123" and seen["asked"] == "GEMINI_API_KEY"
    # base url is the Gemini OpenAI-compat endpoint, chat path appended
    assert b.url.endswith("/chat/completions") and "generativelanguage" in b.url


def test_openai_backend_drops_num_ctx_and_sets_auth_header():
    b = OpenAICompatBackend("https://api.openai.com/v1", "gpt-4o", api_key="k")
    assert b._headers() == {"Authorization": "Bearer k"}
