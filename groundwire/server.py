#!/usr/bin/env python3
"""
groundwire server — an OpenAI-compatible endpoint that gives ANY model a memory.

Stand it up in front of your repos/documents; point any OpenAI-style client at
it. Each chat request is answered by: retrieve top-k chunks from the index ->
build a small grounded prompt -> forward to the upstream model (Ollama,
llama.cpp server, vLLM). The corpus never enters the model's context window.

This makes head-to-head comparison trivial — same client, same questions:

    :11434  your plain / fine-tuned model        (parametric knowledge only)
    :8899   groundwire + the same or smaller model  (grounded in the actual code)

Usage:
    python3 -m groundwire.server --repo ~/IdeaProjects/tina4-python \
        --repo ~/IdeaProjects/tina4-php --repo ~/IdeaProjects/tina4-js \
        --port 8899 --expand 3
    export LLM_URL=http://localhost:11434  LLM_MODEL=llama3.2:3b   # upstream

    curl -s localhost:8899/v1/chat/completions -d '{
      "messages": [{"role": "user", "content": "How do I define a GET route?"}]
    }'

Responses are standard chat.completion JSON plus a "groundwire" key with the
retrieved sources, so you can see WHY it answered what it did. Stdlib only.
"""

from __future__ import annotations
import argparse
import ast
import json
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .answer import APIGenerator
from .encoders import _post
from .memory_systems import fold
from .pipeline import Groundwire

SYSTEM_PROMPT = (
    "You are a coding assistant. Answer using the provided source-code and "
    "documentation context, which comes from the user's actual installed "
    "libraries. Prefer the context over your general knowledge when they "
    "disagree. Use import paths EXACTLY as they appear in the context or "
    "module map -- never invent module paths. Cite the file paths you used. "
    "Show concise code examples."
)

MEM: Groundwire = None          # populated in main()
UPSTREAM: APIGenerator = None
SCOPE = None                 # default repo label for terse questions
MAX_CTX_TOKENS = 5000        # context budget; keep under the reader's window
SYMBOLS = {}                 # "pkg.mod" -> set of top-level names
STATE = {"requests": 0, "corrected": 0}
_STATE_LOCK = threading.Lock()


def _bump(key, n=1):
    with _STATE_LOCK:
        STATE[key] = STATE.get(key, 0) + n
        return STATE[key]


# --------------------------------------------------------------------------- #
# Symbol map: the index doubles as an import VERIFIER. Generated code that
# imports a module/name we know doesn't exist gets one corrective retry with
# the real locations -- retrieval as verifier, not just prompter.
# --------------------------------------------------------------------------- #
def build_symbol_map(repo_root, package):
    """Walk <repo_root>/<package> and record every module's top-level names
    (defs, classes, assignments, re-exports) via ast."""
    pkg_dir = os.path.join(repo_root, package)
    for dirpath, dirnames, filenames in os.walk(pkg_dir):
        dirnames[:] = [d for d in dirnames
                       if d not in Groundwire.SKIP_DIRS and not d.startswith(".")]
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            rel = os.path.relpath(os.path.join(dirpath, fn), repo_root)
            mod = rel[:-3].replace(os.sep, ".")
            if mod.endswith(".__init__"):
                mod = mod[: -len(".__init__")]
            try:
                tree = ast.parse(open(os.path.join(dirpath, fn),
                                      encoding="utf-8", errors="ignore").read())
            except SyntaxError:
                continue
            names = set()
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                     ast.ClassDef)):
                    names.add(node.name)
                elif isinstance(node, ast.Assign):
                    for t in node.targets:
                        if isinstance(t, ast.Name):
                            names.add(t.id)
                elif isinstance(node, ast.AnnAssign) and \
                        isinstance(node.target, ast.Name):
                    names.add(node.target.id)
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    for a in node.names:
                        names.add((a.asname or a.name).split(".")[0])
            SYMBOLS[mod] = names


def symbol_map_text(package, max_lines=60, relevant=None):
    """Compact 'module: names' listing for prompt priming. When `relevant`
    (a set of module names) is given, those modules come FIRST and in full --
    the full 87-module map was ~2300 tokens, half the context budget, starving
    the reader. Prioritising the modules the retrieved chunks came from keeps
    the map small and on-point."""
    mods = sorted(SYMBOLS)
    if relevant:
        mods = [m for m in mods if m in relevant] + \
               [m for m in mods if m not in relevant]
    lines = []
    for mod in mods:
        pub = sorted(n for n in SYMBOLS[mod] if not n.startswith("_"))
        if pub:
            lines.append(f"{mod}: {', '.join(pub[:12])}")
    return "\n".join(lines[:max_lines])


# import-name capture must stop at the line end (or the closing paren of a
# parenthesised import) -- a greedy \s crossed blank lines and swallowed the
# next statement into the name list.
_IMP = re.compile(r"^[ \t]*(?:from\s+([\w\.]+)\s+import\s+"
                  r"(\([^)]*\)|[\w ,\t*]+)"
                  r"|import\s+([\w\.]+))", re.M)


def check_imports(code, package):
    """Verify every <package>.* import in generated code against the map.
    Returns a list of problem descriptions with real locations."""
    problems = []
    for m in _IMP.finditer(code):
        mod = m.group(1) or m.group(3)
        if not mod or not mod.startswith(package):
            continue
        names = [n.strip(" ()") for n in (m.group(2) or "").split(",")
                 if n.strip(" ()") and n.strip() != "*"]
        if mod not in SYMBOLS:
            hint = ""
            for n in names:
                homes = [k for k, v in SYMBOLS.items() if n in v]
                if homes:
                    hint += f" ('{n}' actually lives in {homes[0]})"
            problems.append(f"module '{mod}' does not exist{hint}")
            continue
        for n in names:
            if n in SYMBOLS[mod] or f"{mod}.{n}" in SYMBOLS:
                continue
            homes = [k for k, v in SYMBOLS.items() if n in v]
            where = f"; it lives in {homes[0]}" if homes \
                else "; no such symbol exists"
            problems.append(f"'{n}' is not in {mod}{where}")
    # dotted ATTRIBUTE references bypass import statements entirely
    # (tina4_python.mcp.tools.migration_create(...)): walk each chain to the
    # deepest real module and verify the next component exists there
    for m in re.finditer(rf"\b({re.escape(package)}(?:\.\w+)+)", code):
        full = m.group(1)
        if full in SYMBOLS:
            continue                # the whole path is a real module -- fine
        parts = full.split(".")
        for depth in range(len(parts) - 1, 0, -1):
            mod = ".".join(parts[:depth])
            if mod in SYMBOLS:
                nxt = parts[depth]
                if nxt not in SYMBOLS[mod] and f"{mod}.{nxt}" not in SYMBOLS:
                    homes = [k for k, v in SYMBOLS.items() if nxt in v]
                    where = f"; '{nxt}' lives in {homes[0]}" if homes \
                        else f"; '{nxt}' does not exist"
                    problems.append(f"'{m.group(1)}' is invalid{where}")
                break
        else:
            problems.append(f"module '{m.group(1)}' does not exist")
    return sorted(set(problems))


import builtins as _builtins
import sys as _sys
_BUILTIN_NAMES = set(dir(_builtins)) | {
    "__name__", "__file__", "__doc__", "__builtins__", "self", "cls", "__class__"}
# Only stdlib module names are flagged when used-without-import. This is the
# high-frequency "doesn't run" cause (datetime.now() with no `import datetime`)
# and is unambiguous. We deliberately do NOT flag bare CapitalCase model refs
# (e.g. ForeignKeyField(to=User)) — that is the framework's canonical form
# (string forward-refs also work), and flagging it made the corrective retry
# restructure the answer and drop the idiom (-2 features on the 57-bench).
_STDLIB_MODS = set(getattr(_sys, "stdlib_module_names", ())) | {
    "datetime", "json", "os", "re", "time", "uuid", "math", "random",
    "hashlib", "base64", "pathlib", "decimal", "asyncio"}


def _parses(src):
    try:
        ast.parse(src)
        return True
    except SyntaxError:
        return False


def check_undefined_names(code):
    """Flag names USED but never imported or defined anywhere in the snippet —
    catches the missing-import class (`datetime.now()` with no `import datetime`,
    `ForeignKeyField(to=UserAccount)` without importing UserAccount, undefined
    type hints). Complements check_imports, which only validates <package>.*.

    Deliberately conservative: it collects every name bound ANYWHERE in the file
    (no scope analysis) so a name defined anywhere is never flagged, and it bails
    out entirely on a star-import or a parse error. A false positive would fire a
    corrective retry that could worsen the answer, so precision beats recall."""
    # Isolate python: a reply mixes prose with several fenced blocks, some not
    # python (shell, output). Keep only python-tagged/untagged blocks that
    # PARSE, then union them so an import in any block binds the name. Never
    # feed prose or a non-python block to ast.parse (it would bail to []).
    tagged = re.findall(r"```([a-zA-Z]*)\n(.*?)```", code, re.S)
    cand = [b for tag, b in tagged if tag.lower() in ("", "py", "python")] \
        if tagged else [code]
    srcs = [b for b in cand if _parses(b)]
    if not srcs:
        return []
    try:
        tree = ast.parse("\n".join(srcs))
    except SyntaxError:
        return []
    bound, used = set(), {}

    def bind(t):
        if isinstance(t, ast.Name):
            bound.add(t.id)
        elif isinstance(t, (ast.Tuple, ast.List)):
            for e in t.elts:
                bind(e)
        elif isinstance(t, ast.Starred):
            bind(t.value)

    def args(a):
        for x in [*getattr(a, "posonlyargs", []), *a.args, *a.kwonlyargs]:
            bound.add(x.arg)
        if a.vararg:
            bound.add(a.vararg.arg)
        if a.kwarg:
            bound.add(a.kwarg.arg)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                bound.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                if a.name == "*":
                    return []          # unknowable names — don't risk a bad retry
                bound.add(a.asname or a.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            bound.add(node.name); args(node.args)
        elif isinstance(node, ast.Lambda):
            args(node.args)
        elif isinstance(node, ast.ClassDef):
            bound.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                bind(t)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
            bind(node.target)
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
            bind(node.target)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for it in node.items:
                if it.optional_vars:
                    bind(it.optional_vars)
        elif isinstance(node, ast.ExceptHandler):
            if node.name:
                bound.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            for n in node.names:
                bound.add(n)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            used.setdefault(node.id, node.lineno)

    return sorted({f"'{n}' is used but never imported — add `import {n}`"
                   for n in used
                   if n not in bound and n not in _BUILTIN_NAMES
                   and n in _STDLIB_MODS})


class _CapacityError(Exception):
    """Upstream rejected the request for capacity reasons -- context too long
    or the reader ran out of GPU memory. Recoverable by shrinking context."""


# signatures vLLM / llama.cpp / TGI emit for context-overflow or OOM
_CAP_SIG = re.compile(
    r"context length|maximum context|context window|too long|reduce the "
    r"(?:length|input)|out of memory|\boom\b|cuda out|no available memory|"
    r"kv cache|insufficient|cannot allocate", re.I)


def _generate(messages, max_tokens):
    import urllib.error
    headers = {"Authorization": f"Bearer {UPSTREAM.api_key}"} \
        if UPSTREAM.api_key else {}
    payload = {
        "model": UPSTREAM.model, "messages": messages,
        "max_tokens": max_tokens, "temperature": 0,
    }
    # Anti-repetition guard (vLLM sampling param). OFF by default: a global
    # penalty regresses code accuracy (code legitimately repeats tokens like
    # `Auth.`/`response.`, and penalizing them suppresses the correct idiom —
    # measured -4 features on the 57-bench at 1.1). Kept as a knob for the review
    # path only; set REPETITION_PENALTY>1 to re-enable.
    _rep = float(os.getenv("REPETITION_PENALTY", "1.0"))
    if _rep and _rep != 1.0:
        payload["repetition_penalty"] = _rep
    try:
        out = _post(UPSTREAM.url, payload, headers=headers, timeout=UPSTREAM.timeout)
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", "ignore")
        except Exception:
            pass
        # 413 is always capacity; 400/500 only if the body says so (a generic
        # 500 for other reasons must still surface as a real error)
        if e.code == 413 or (e.code in (400, 500) and _CAP_SIG.search(body)):
            raise _CapacityError(body[:200]) from e
        raise
    return out["choices"][0]["message"]["content"], out.get("usage", {})


_INSUFFICIENT = re.compile(
    r"\b(not (?:enough|sufficient) (?:context|information)|"
    r"(?:context|information) (?:does not|doesn't|is insufficient)|"
    r"cannot (?:find|determine|answer)|no (?:relevant )?(?:context|information)"
    r"(?: is)? (?:provided|available|found))", re.I)


# generic explanatory vocabulary -- present in valid paraphrase, not evidence
_GROUND_STOP = frozenset(
    "the a an and or but if then this that these those with from into onto for "
    "you your use used using can could would should will need needs want define "
    "defined create creates creating method function class import example code "
    "returns return value values object objects which where when what here there "
    "also provided context installed library libraries package based specify "
    "specified specifies triggered trigger made make makes allows allow allowing "
    "will called call calls request response route routes handler handlers path "
    "paths server client data type types given set sets pass passed passing "
    "following above below like such they them their its each any some more most "
    "have has had been being are was were not only just both either same other".split())
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def groundedness(answer, context):
    """Advisory hallucination signal: sentences whose distinctive content
    words are ALL absent from the retrieved context. Deliberately conservative
    -- a purely lexical check cannot tell valid paraphrase from fabrication
    without an NLI model, so it flags only near-total misses (a strong
    fabrication signal like 'wooden utensils' when the context says silver),
    accepting it will miss subtler ones. Complements the import verifier
    (which covers only imports). Code fences are exempt."""
    prose = re.sub(r"```.*?```", " ", answer, flags=re.S)   # drop code fences
    ctx = fold(context)
    flagged, checked = [], 0
    for sent in _SENT_SPLIT.split(prose):
        toks = [t for t in set(re.findall(r"[a-z0-9]{4,}", fold(sent)))
                if t not in _GROUND_STOP]
        if len(toks) < 3:                   # too few content words to judge
            continue
        checked += 1
        present = sum(1 for t in toks if t in ctx)
        if present == 0:                    # NOTHING distinctive traces to context
            flagged.append(sent.strip()[:120])
    score = round(1 - len(flagged) / checked, 2) if checked else 1.0
    return score, flagged


def _grounded_answer(question, hits, max_tokens, ctx_budget=None):
    """Build the budgeted context from `hits`, generate, and run the import-
    verification loop. Returns (content, usage, corrected). Raises
    _CapacityError if even the initial generation overflows -- the caller
    retries with a smaller ctx_budget; correction-round overflows are absorbed
    (keep the last good answer) rather than failing the request."""
    budget = ctx_budget or MAX_CTX_TOKENS
    est_tok = lambda s: len(s) / 3.5
    package = None
    map_text = ""
    if SYMBOLS:
        package = next(iter(SYMBOLS)).split(".")[0]
        # modules the retrieved chunks came from: 'repo/pkg/core/router.py'
        # -> 'pkg.core.router'. Prioritise these and cap the map so it doesn't
        # crowd out the actual code chunks (the reader needs those more).
        relevant = set()
        for cid, _, _ in hits:
            src = MEM.source_of(cid) or ""
            if src.endswith(".py"):
                mod = src.rsplit("/", 1)[0] if False else src
                m = src.split("/", 1)[-1] if "/" in src else src
                relevant.add(m[:-3].replace("/", "."))
        map_body = symbol_map_text(package, max_lines=22, relevant=relevant)
        map_text = f"Module map (relevant modules first):\n{map_body}\n\n"
    remaining = budget - est_tok(map_text)
    parts = []
    for cid, text, _ in hits:
        t = est_tok(text)
        if parts and t > remaining:
            break               # keep whole chunks; stop before overflow
        parts.append(f"[{MEM.source_of(cid) or cid}]\n{text}")
        remaining -= t
    context = map_text + "\n\n".join(parts)
    hard = int(budget * 3.5)             # backstop: one merged chunk can be huge
    if len(context) > hard:
        context = context[:hard] + "\n...[truncated]"
    system = SYSTEM_PROMPT
    rules = ""
    if SCOPE:
        # a terse prompt names no language; a multi-language model drifts
        # (and non-python answers bypass the import verifier entirely)
        rules = (f"Unless the question names another language or framework, "
                 f"answer for {SCOPE}"
                 + (f" (python package {package})" if package else "") + ". ")
        system += " " + rules
    # the grounding contract goes in the USER message too: upstream gateways
    # (e.g. a production path that injects its own system prompt) may replace
    # the system slot, but the user message always survives
    messages = [
        {"role": "system", "content": system},
        {"role": "user",
         "content": f"{rules}Answer from this context of the installed "
                    f"libraries; use import paths exactly as shown, never "
                    f"invented ones:\n\n{context}\n\nQuestion: {question}"},
    ]
    content, usage = _generate(messages, max_tokens)
    corrected = False
    if package:
        # verified correction loop: retries are RE-CHECKED (an unverified
        # retry can introduce a fresh hallucination). A fine-tuned coder is
        # STUBBORN -- it re-emits baked-in wrong idioms unless shown what
        # actually exists -- so the correction lists the real symbols of the
        # offending modules, not just "these don't exist". 3 rounds.
        for _ in range(3):
            problems = check_imports(content, package)
            if os.getenv("VERIFY_UNDEFINED", "1") == "1":
                problems = problems + check_undefined_names(content)
            if not problems:
                break
            # which modules did the coder reach into? show their real contents
            offending = set()
            for m in _IMP.finditer(content):
                mod = m.group(1) or m.group(3)
                if mod and mod.startswith(package):
                    offending.add(mod)
            for m in re.finditer(rf"\b({re.escape(package)}(?:\.\w+)+)", content):
                for d in range(len(m.group(1).split(".")), 0, -1):
                    cand = ".".join(m.group(1).split(".")[:d])
                    if cand in SYMBOLS:
                        offending.add(cand)
                        break
            avail = []
            for mod in sorted(offending):
                if mod in SYMBOLS:
                    real = sorted(n for n in SYMBOLS[mod] if not n.startswith("_"))
                    avail.append(f"{mod} provides ONLY: {', '.join(real[:20])}")
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content":
                             "Your answer has these problems:\n- "
                             + "\n- ".join(problems)
                             + ("\n\nThe real API:\n" + "\n".join(avail)
                                if avail else "")
                             + "\nRewrite so it runs: use ONLY real names (if a "
                               "class you wanted doesn't exist, use the methods "
                               "that do) and add an import for every name you "
                               "reference. Keep it concise."})
            try:
                content, usage = _generate(messages, max_tokens)
            except _CapacityError:
                break            # correction grew the prompt past the window;
                                 # keep the last valid answer instead of failing
            corrected = True
            _bump("corrected")
    return content, usage, corrected


def _answer_resilient(question, hits, max_tokens):
    """Call _grounded_answer, and on a capacity error (context overflow or
    reader OOM) shrink the context -- halve the token budget and drop the
    lowest-ranked half of the chunks -- and retry, down to a single top chunk.
    Only if even that overflows do we surface an error. This keeps one big
    retrieval or a tight reader window from hard-failing a request."""
    budget = MAX_CTX_TOKENS
    cur = hits
    last = None
    for _ in range(5):
        try:
            return _grounded_answer(question, cur, max_tokens, ctx_budget=budget)
        except _CapacityError as e:
            last = e
            _bump("shrunk")
            budget = max(600, budget // 2)
            cur = cur[:max(1, len(cur) // 2)]
    # floor: even one chunk at the min budget overflowed -- answer question-only
    try:
        return _grounded_answer(question, [], max_tokens, ctx_budget=600)
    except _CapacityError:
        raise last


def answer(question: str, k=None, max_tokens=512):
    t0 = time.time()
    base_k = k or MEM.k
    hits = MEM.retrieve(question, k=base_k, scope=SCOPE)
    retrieve_ms = (time.time() - t0) * 1000
    content, usage, corrected = _answer_resilient(question, hits, max_tokens)
    widened = False
    # adaptive retrieval: if the reader says the context is insufficient,
    # retry ONCE with a wider pool -- the fixed k under-served this question
    if _INSUFFICIENT.search(content):
        wider = MEM.retrieve(question, k=base_k * 2, scope=SCOPE)
        if len(wider) > len(hits):
            hits = wider
            content, usage, corrected = _answer_resilient(question, hits, max_tokens)
            widened = True
    sources = [MEM.source_of(cid) or str(cid) for cid, _, _ in hits]
    queries = getattr(MEM.memory, "last_queries", [question])
    if widened:
        _bump("widened")
    gscore, flagged = groundedness(content, " ".join(t for _, t, _ in hits))
    return {"content": content, "sources": sources, "queries": queries,
            "retrieve_ms": retrieve_ms, "usage": usage, "corrected": corrected,
            "grounding": gscore, "ungrounded": flagged}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_stream(self, content, meta, model_id):
        """Emit an OpenAI-style SSE stream. We generate fully first (so the
        import-verification correction loop still runs), then chunk the final
        text out -- protocol-compatible with streaming clients without
        sacrificing grounding. The retrieved sources ride on the first event."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        def event(delta, finish=None):
            chunk = {"id": model_id, "object": "chat.completion.chunk",
                     "model": f"groundwire+{UPSTREAM.model}",
                     "choices": [{"index": 0, "delta": delta,
                                  "finish_reason": finish}]}
            if delta.get("role"):     # attach provenance to the opening event
                chunk["groundwire"] = meta
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()

        event({"role": "assistant"})
        # chunk on whitespace so clients render token-by-token
        for i, tok in enumerate(content.split(" ")):
            event({"content": (tok if i == 0 else " " + tok)})
        event({}, finish="stop")
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def do_GET(self):
        if self.path == "/v1/models":
            return self._send(200, {"object": "list", "data": [
                {"id": "groundwire", "object": "model", "owned_by": "groundwire"}]})
        if self.path == "/health":
            return self._send(200, {"status": "ok",
                                    "chunks": MEM._next,
                                    "requests": STATE["requests"]})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path.rstrip("/") == "/reindex":
            # live refresh: rebuild the index from CONFIG and hot-swap it
            try:
                self.rfile.read(int(self.headers.get("Content-Length", 0) or 0))
                info = reindex()
                print(f"reindexed: {info}", flush=True)
                return self._send(200, {"status": "reindexed", **info})
            except Exception as e:
                return self._send(500, {"error": str(e)})
        if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
            return self._send(404, {"error": "not found"})
        try:
            body = json.loads(self.rfile.read(
                int(self.headers.get("Content-Length", 0))))
            question = ""
            for m in reversed(body.get("messages", [])):
                if m.get("role") == "user":
                    c = m.get("content", "")
                    question = c if isinstance(c, str) else " ".join(
                        s.get("text", "") for s in c if isinstance(s, dict))
                    break
            if not question:
                return self._send(400, {"error": "no user message"})
            t0 = time.time()
            r = answer(question, max_tokens=body.get("max_tokens", 512))
            req_id = _bump("requests")       # atomic: unique per concurrent req
            total_ms = (time.time() - t0) * 1000
            print(f"[{req_id}] {total_ms:6.0f}ms "
                  f"(retrieve {r['retrieve_ms']:.0f}ms"
                  f"{', import-corrected' if r['corrected'] else ''}"
                  f"{', ungrounded' if r['ungrounded'] else ''}"
                  f"{', stream' if body.get('stream') else ''}) "
                  f"{question[:60]!r} <- {sorted(set(r['sources']))}", flush=True)
            meta = {"sources": r["sources"], "queries": r["queries"],
                    "retrieve_ms": round(r["retrieve_ms"], 1),
                    "import_corrected": r["corrected"],
                    "grounding": r["grounding"], "ungrounded": r["ungrounded"]}
            if body.get("stream"):
                return self._send_stream(r["content"], meta,
                                         f"groundwire-{req_id}")
            self._send(200, {
                "id": f"groundwire-{req_id}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": f"groundwire+{UPSTREAM.model}",
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant",
                                         "content": r["content"]}}],
                "usage": r["usage"],
                "groundwire": meta,
            })
        except Exception as e:  # keep the server up; report the error
            self._send(500, {"error": str(e)})

    def log_message(self, *a):   # our own request log is printed in do_POST
        pass


CONFIG = {}                  # ingest config, replayed by /reindex


def build_memory():
    """Build a fresh index + symbol map from CONFIG. Returns (mem, symbols).
    Kept pure so /reindex can build a NEW one and swap it in atomically --
    in-flight requests keep serving the old index until the swap."""
    mem = Groundwire(memory=CONFIG["backend"], k=CONFIG["k"],
                  expand=CONFIG["expand"])
    for r in CONFIG["repos"]:
        mem.ingest_repo(r)
    for d in CONFIG["docs"]:
        mem.ingest(open(d, encoding="utf-8", errors="ignore").read(), title=d)
    symbols = {}
    if CONFIG["scope"] and CONFIG["package"]:
        root = next((r for r in CONFIG["repos"]
                     if os.path.basename(os.path.abspath(r)) == CONFIG["scope"]),
                    None)
        if root:
            global SYMBOLS
            saved, SYMBOLS = SYMBOLS, {}     # build_symbol_map fills module SYMBOLS
            build_symbol_map(os.path.abspath(root), CONFIG["package"])
            symbols, SYMBOLS = SYMBOLS, saved
    return mem, symbols


def reindex():
    """Rebuild and atomically swap MEM + SYMBOLS. Safe under concurrency:
    Python attribute rebind is atomic, and in-flight queries hold their own
    reference to the old index."""
    global MEM, SYMBOLS
    t0 = time.time()
    mem, symbols = build_memory()
    MEM, SYMBOLS = mem, symbols          # atomic swap
    return {"chunks": mem._next, "modules": len(symbols),
            "seconds": round(time.time() - t0, 1)}


def main():
    global UPSTREAM, SCOPE, MAX_CTX_TOKENS, MEM, SYMBOLS
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", action="append", default=[],
                    help="repository to ingest (repeatable)")
    ap.add_argument("--doc", action="append", default=[],
                    help="plain-text/markdown file to ingest (repeatable)")
    ap.add_argument("--backend", default="bm25")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--expand", type=int, default=3)
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--max-context-tokens", type=int, default=5000,
                    help="context budget; keep well under the reader's window")
    ap.add_argument("--scope", default=None,
                    help="default repo label for questions that name none")
    ap.add_argument("--package", default=None,
                    help="python package inside --scope to build the symbol "
                         "map / import verifier from (e.g. tina4_python)")
    args = ap.parse_args()

    SCOPE = args.scope
    MAX_CTX_TOKENS = args.max_context_tokens
    UPSTREAM = APIGenerator()          # LLM_URL / LLM_MODEL env
    CONFIG.update(repos=args.repo, docs=args.doc, backend=args.backend,
                  k=args.k, expand=args.expand, scope=args.scope,
                  package=args.package)
    t0 = time.time()
    MEM, SYMBOLS = build_memory()
    print(f"memory ready: {MEM._next:,} chunks in {time.time()-t0:.1f}s | "
          f"upstream {UPSTREAM.url} model={UPSTREAM.model!r} | "
          f"backend={args.backend} expand={args.expand} k={args.k} "
          f"scope={SCOPE} | symbol map {len(SYMBOLS)} modules", flush=True)

    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"listening on http://127.0.0.1:{args.port}/v1/chat/completions "
          f"(POST /reindex to refresh)", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
