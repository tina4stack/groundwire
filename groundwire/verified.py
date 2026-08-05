"""Verified ranking signal — rank retrieval by whether the code RUNS, not just matches.

The motivating failure: the most *relevant* chunk can be the most *wrong* (a doc teaching
`select().to_array()` outranks the correct idiom on token overlap). `verified` re-ranks so runnable
code wins.

ARCHITECTURE — computed OFFLINE, read at runtime:
  * Score computed at corpus-BUILD time (framework importable, subprocess boot-gate affordable),
    stored per source. The router only READS the score at query time — it stays pure stdlib.

Two tiers (worst code block wins):
  T1  parse — `ast.parse` the block wrapped in a function (so a route-body `return` is valid).
  T2  boot-gate — run the block in an ISOLATED subprocess against a real framework instance
      (timeout, no network) AND CALL any route handler it defines, so a broken idiom inside a
      `def handler(...)` is actually exercised. Catches runtime-type errors (`list.to_array()`)
      and hallucinated APIs that T1 can't see.

Score: 0 broken · 1 error · 2 boots. Higher = safer to ground an agent on.
"""
from __future__ import annotations
import ast
import re
import subprocess
import sys
import textwrap

_FENCE = re.compile(r"```(?:python|py)?\n(.*?)```", re.S)
_WEIGHT = {"boots": 2, "error": 1, "runtime-err": 0, "timeout": 1}
_STUBS = ("response", "get", "post", "put", "delete", "patch", "noauth", "secured",
          "description", "tags", "request", "session", "db", "QueryBuilder")


def extract_python(text: str, is_doc: bool) -> list[str]:
    """Doc chunk → fenced ```python blocks; source .py → the whole text as one block.
    Always run against the ORIGINAL source, never a retrieved chunk (the chunker mangles fences)."""
    return _FENCE.findall(text) if is_doc else [text]


def t1_parses(code: str) -> bool:
    body = textwrap.indent(code.strip() or "pass", "    ")
    try:
        ast.parse("def _example():\n" + body)
        return True
    except SyntaxError:
        return False


def _module_valid(code: str) -> bool:
    """True if `code` runs as a top-level module. Uses compile(), NOT ast.parse():
    ast.parse ACCEPTS a top-level `return`/`yield` ('return outside function' is a
    compile/symtable check, not a parse one), which would misclassify a bare route-body
    snippet (`return response(...)`) as a full module and then fail to exec it. compile()
    catches it, so bare bodies correctly fall through to the function-wrapped path."""
    try:
        compile(code, "<groundwire-verify>", "exec")
        return True
    except (SyntaxError, ValueError):
        return False


# After executing a block, CALL any route handler it defined (module-level defs), so the code in a
# `def handler(request, response): return response(...bad...)` actually runs and can fail truthfully.
_CALL_HANDLERS = f'''
import types as _t
for _n, _o in list(dict(globals()).items()):
    if isinstance(_o, _t.FunctionType) and getattr(_o, "__module__", "") == "__main__" \\
       and _n not in {_STUBS!r} and not _n.startswith("_"):
        try:
            _o(request, response)
        except TypeError:
            try: _o()
            except Exception: pass
'''


def boot_gate(code: str, preamble: str, timeout: int = 10) -> str:
    """Run `code` after `preamble` (framework setup) in an isolated subprocess; returns
    'boots' | 'runtime-err' | 'error' | 'timeout'. Handles both bare-body examples (a top-level
    `return`) and route-handler examples (which are defined and then CALLED)."""
    if _module_valid(code):
        core = code + "\n" + _CALL_HANDLERS            # statements/handlers run; handlers get called
    else:
        core = "def _e():\n" + textwrap.indent(code.strip() or "pass", "    ") + "\n_e()"
    script = (preamble + "\ntry:\n" + textwrap.indent(core, "    ") +
              "\nexcept AttributeError as _err:\n    import sys; sys.exit(2)"
              "\nexcept Exception as _err:\n    import sys; sys.exit(3)"
              "\nimport sys; sys.exit(0)\n")
    try:
        proc = subprocess.run([sys.executable, "-c", script],
                              capture_output=True, text=True, timeout=timeout)
        return {0: "boots", 2: "runtime-err", 3: "error"}.get(proc.returncode, "error")
    except subprocess.TimeoutExpired:
        return "timeout"


def verified_score(text: str, is_doc: bool, preamble: str) -> int:
    """0 broken · 1 error · 2 boots. Worst code block wins (one broken example taints the chunk).
    Pure prose (no code) is neutral-high (2)."""
    blocks = extract_python(text, is_doc)
    if not blocks:
        return 2
    worst = 2
    for block in blocks:
        if not t1_parses(block):
            worst = 0
            continue
        worst = min(worst, _WEIGHT[boot_gate(block, preamble)])
    return worst


def make_scorer(preamble: str, source_exts: tuple = (".py",)):
    """Build a per-source `verified_scorer` for `Groundwire(verified_scorer=...)`.

    Real framework SOURCE (`.py` etc.) is canonical → trusted (2), no boot-gate. Everything else
    (docs, markdown how-tos — where the broken idioms hide) is boot-gated against `preamble` (which
    already has any `__TINA4__` placeholder filled in). Signature is `(title, text) -> int`, called
    once per source at INGEST time (offline), so the query path only ever READS the cached score."""
    def score(title: str, text: str) -> int:
        if title and title.lower().endswith(source_exts):
            return 2
        return verified_score(text, is_doc=True, preamble=preamble)
    return score


# Framework preamble: a real Tina4 ORM + the common surface (response, db, request, QueryBuilder,
# permissive route-decorator + request stubs, common example model names). `__TINA4__` is filled by
# the caller once. A permissive `_Any` absorbs unknown-symbol chains so real docs don't false-fail,
# while REAL objects (Product = a real ORM model) still fail truthfully.
PYTHON_PREAMBLE = """
import sys, os, tempfile
sys.path.insert(0, "__TINA4__")
from tina4_python.orm import ORM, IntegerField, StringField, JSONField, bind_database
from tina4_python.database import Database
try:
    from tina4_python.query_builder import QueryBuilder
except Exception:
    QueryBuilder = None
bind_database(Database("sqlite:///" + os.path.join(tempfile.mkdtemp(), "v.db")))
class Product(ORM):
    table_name = "products"
    id = IntegerField(primary_key=True, auto_increment=True)
    name = StringField()
User = Order = Invoice = Article = Note = Widget = Customer = Post = Product
class _Any:
    def __getattr__(self, n): return self
    def __call__(self, *a, **k): return self
    def __getitem__(self, k): return self
    def __iter__(self): return iter([])
request = session = _Any()
def response(x, *a, **k): return x
def get(*a, **k):
    def _d(f): return f
    return _d
post = put = delete = patch = noauth = secured = description = tags = get
db = Product._get_db()
Product.create_table(); Product({"name": "x"}).save()
"""
