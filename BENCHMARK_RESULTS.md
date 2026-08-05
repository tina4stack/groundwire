# groundwire benchmark — Qwen2.5-7B-Instruct, 57-feature tina4-python suite

Question: can a **general** instruct model + groundwire retrieval match the
fine-tuned tina4 coder, with no per-release retraining? And for a *weak* reader,
does the forced-verification **proxy** or the self-directed **agentic MCP** win?

Scorer: `bench_validate.py` — resolves imports against installed `tina4_python`
and checks idiom tokens. PASS = imports resolve **and** idiom is canonical.

## Fair three-way (all on the same clean retrieval config)

| Condition | Score | Idiom fails | Import fails | Notes |
|-----------|:-----:|:-----------:|:------------:|-------|
| Qwen-7B alone (no retrieval) | **32%** (18/57) | high | — | baseline |
| Qwen-7B + groundwire **proxy** | **74%** (42/57) | 15 | **0** | forced verify+fix loop |
| Qwen-7B + groundwire **agentic** (MCP) | **60%** (34/57) | 15 | **8** | 7B self-drives the tools |

**proxy (74%) > agentic (60%) > alone (32%).**

### Why proxy beats agentic on a small model
Both reach the same idiom quality (15 idiom fails each). The gap is **imports**:
the proxy's server-side verify-and-correct loop catches every hallucinated
import (**0** import fails), while a 7B driving the MCP tools itself does **not
reliably call `verify_code`** before answering → **8** import hallucinations slip
through. Self-directed verification is a *strong-client* feature (great for
Claude); a weak reader needs the loop run **for** it. Original prediction held.

The agentic 60% was re-measured on the reverted (clean) code and landed at the
same 60% (16 of 57 generations differed run-to-run; aggregate stable), so the
agentic path is unaffected by the retrieval change below.

## Regression found & reverted: the usage/example slot

Commit `0913ef0` added a "usage/example" reserved slot to `retrieve()`. It was
shipped as a universal win on the strength of Llama-8B. It was not:

| Reader | without slot | with slot | Δ |
|--------|:-----------:|:---------:|:--:|
| Llama-8B (strong) | 82% | 84% | +2 |
| **Qwen-7B (weak)** | **74%** | **49%** | **−25** |

A strong reader treats an injected test/CLI example as helpful context; a weak
7B **copies the example's non-canonical style** instead of the package idiom
(FAIL_IDIOM 15→29). Net-harmful for the cheap-reader case groundwire exists to
serve, so it is **reverted** (neighbor-pull, source-over-tests, single guide
slot, def-boost, and the dense reranker are all retained). A future *opt-in*
version could serve strong readers.

**Lesson recorded:** validate retrieval changes on the **weakest** target
reader, not the strongest — the effect sign can flip with reader capability.
