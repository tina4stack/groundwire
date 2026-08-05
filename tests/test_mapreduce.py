"""Unit tests for the map-reduce summarizer engine (LLM function injected)."""
from groundwire.mapreduce import summarize


def test_map_then_reduce():
    calls = []
    def fake(prompt):
        calls.append(prompt)
        return f"S{len(calls)}"
    segs = [("book 1", "aaa"), ("book 2", "bbb"), ("book 3", "ccc")]
    out = summarize(fake, segs, budget_chars=10000, request="summarize the book")
    assert len(calls) == 4                 # 3 map calls + 1 reduce call
    assert out == "S4"
    assert "S1" in calls[-1] and "S3" in calls[-1]   # reduce sees the map outputs
    assert "book 1" in calls[0]            # a map prompt carries its segment label


def test_oversized_segment_is_windowed():
    calls = []
    def fake(p):
        calls.append(p)
        return "s"
    big = ("book 1", "word " * 3000)       # ~15000 chars, far over budget
    summarize(fake, [big], budget_chars=2000, request="sum")
    assert len(calls) > 2                   # split into windows -> many map calls


def test_hierarchical_reduce_when_summaries_overflow():
    calls = []
    def fake(p):
        calls.append(p)
        return "x" * 400                    # each summary is big
    segs = [(f"s{i}", "t") for i in range(20)]
    summarize(fake, segs, budget_chars=1500, request="sum")
    # 20 map calls + more than one reduce (summaries don't fit in one reduce)
    assert len(calls) > 21


def test_progress_callback_per_segment():
    msgs = []
    summarize(lambda p: "s", [("book 1", "a"), ("book 2", "b"), ("book 3", "c")],
              budget_chars=10000, request="sum", on_progress=msgs.append)
    # a progress message per top-level segment (with its label + N/total)
    assert any("book 1" in m and "1/3" in m for m in msgs)
    assert any("book 3" in m and "3/3" in m for m in msgs)


def test_single_segment_no_reduce_noise():
    calls = []
    def fake(p):
        calls.append(p)
        return "only"
    out = summarize(fake, [("doc", "small text")], budget_chars=10000, request="s")
    assert out == "only" and len(calls) == 1   # 1 fits -> map only, no reduce call
