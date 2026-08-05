"""Unit tests for the span registry (post-fill / [[SPAN:...]] handles)."""
from groundwire.spans import SpanRegistry, _to_int, QUOTE_INTENT, READ_INTENT

# synthetic book: a table of contents (empty-body headings) then real chapters.
BODY1 = "one " * 150
BODY2 = "The distinctive ALPHA_TWO_BODY marker sits here. " + "two " * 120
BODY3 = "three " * 150
BOOK = (
    "CONTENTS\nCHAPTER I\nCHAPTER II\nCHAPTER III\n\n"     # TOC: empty bodies
    f"CHAPTER I\n\n{BODY1}\n\n"
    f"CHAPTER II\n\n{BODY2}\n\n"
    f"CHAPTER III\n\n{BODY3}\n"
)


def _reg():
    return SpanRegistry().register_file("drop/synthetic.txt", BOOK)


def test_toc_entries_filtered():
    # 3 TOC entries have ~empty bodies and must be dropped; 3 real chapters kept
    reg = _reg()
    assert len(reg.units["drop/synthetic.txt"]["chapter"]) == 3


def test_resolve_and_fill_verbatim():
    reg = _reg()
    cands = reg.resolve("quote the full text of chapter 2")
    assert cands, "chapter 2 should resolve"
    handle = cands[0][0]
    filled = reg.fill(f"intro\n[[SPAN:{handle}]]")
    assert "ALPHA_TWO_BODY" in filled            # the real chapter-2 body
    assert "[[SPAN:" not in filled               # placeholder was replaced


def test_opening_is_truncated():
    reg = _reg()
    full = reg.fill("[[SPAN:%s]]" % reg.resolve("full text of chapter 3")[0][0])
    opening = reg.fill("[[SPAN:%s]]" % reg.resolve("how does chapter 3 start")[0][0])
    assert len(opening) < len(full)              # "opening" returns less
    assert opening in full or opening.split()[0] in full


def test_heading_line_excluded():
    reg = _reg()
    filled = reg.fill("[[SPAN:%s]]" % reg.resolve("chapter 1 full")[0][0])
    assert not filled.lstrip().upper().startswith("CHAPTER")


def test_unknown_handle_left_intact():
    reg = _reg()
    assert reg.fill("[[SPAN:nope-chapter-9]]") == "[[SPAN:nope-chapter-9]]"


def test_ordinal_counts_only_chapters():
    # BOOK/PART markers must not shift chapter numbering
    reg = SpanRegistry().register_file(
        "drop/x.txt",
        "BOOK ONE\n\n" + "b " * 200 + "\n\nCHAPTER I\n\n" + "one " * 150 +
        "\n\nCHAPTER II\n\n" + "TWO_MARK " + "two " * 150)
    handle = reg.resolve("chapter 2 full")[0][0]
    assert "TWO_MARK" in reg.fill(f"[[SPAN:{handle}]]")


def test_coarse_segments_prefers_books():
    # two BOOKs, each with chapters -> segment by BOOK (coarsest), TOC skipped
    txt = ("CONTENTS\nBOOK I\nBOOK II\n\n"                       # TOC (tiny)
           "BOOK I\n\nCHAPTER I\n\n" + "a " * 800 +
           "\n\nCHAPTER II\n\n" + "b " * 800 +
           "\n\nBOOK II\n\nCHAPTER I\n\n" + "c " * 1200)
    reg = SpanRegistry().register_file("drop/x.txt", txt)
    segs = reg.coarse_segments("drop/x.txt")
    assert len(segs) == 2                       # two books, not 3 chapters
    assert segs[0][0].startswith("book")
    assert "a a" in segs[0][1] and "b b" in segs[0][1]   # book I holds both chapters
    assert "c c" in segs[1][1]


def test_resolve_whole_file():
    reg = SpanRegistry()
    reg.register_file("drop/war_and_peace.txt", "CHAPTER I\n\n" + "x " * 500)
    reg.register_file("drop/notes.md", "small")
    assert reg.resolve_whole("summarize war and peace") == "drop/war_and_peace.txt"
    assert reg.resolve_whole("summarize the whole book") == "drop/war_and_peace.txt"
    assert reg.resolve_whole("what time is it") is None


def test_word_numbered_books_beat_section_junk():
    # books use WORD numbers ("BOOK ONE"); there is also a stray license-style
    # "SECTION" marker. Coarse segmentation must pick the BOOKS, not sections.
    txt = ("BOOK ONE\n\nCHAPTER I\n\n" + "a " * 900 +
           "\n\nCHAPTER II\n\n" + "b " * 900 +
           "\n\nBOOK TWO\n\nCHAPTER I\n\n" + "c " * 900 +
           "\n\nSECTION 1\n\nGutenberg license text " + "z " * 40)
    reg = SpanRegistry().register_file("drop/x.txt", txt)
    segs = reg.coarse_segments("drop/x.txt")
    assert [s[0].split()[0] for s in segs] == ["book", "book"]
    assert "a a" in segs[0][1] and "c c" in segs[1][1]


def test_coarse_segments_falls_back_to_document():
    reg = SpanRegistry().register_file("drop/plain.txt", "just some prose " * 50)
    segs = reg.coarse_segments("drop/plain.txt")
    assert len(segs) == 1 and "prose" in segs[0][1]


def test_roman_and_intent():
    assert _to_int("iv") == 4 and _to_int("Xii") == 12 and _to_int("3") == 3
    assert QUOTE_INTENT.search("how does chapter 2 start")
    assert QUOTE_INTENT.search("quote the passage verbatim")
    assert not QUOTE_INTENT.search("what is the theme of the novel")


def test_book_opening_without_chapter_number():
    # "how does <book> start" (no chapter number) -> chapter 1's opening
    reg = SpanRegistry().register_file(
        "drop/x.txt",
        "CHAPTER I\n\nOPENING_MARKER " + "a " * 200 +
        "\n\nCHAPTER II\n\n" + "b " * 200)
    for q in ("how does the book start", "the opening of x", "how does x begin"):
        cands = reg.resolve(q)
        assert cands, f"{q!r} should resolve the opening"
        assert "OPENING_MARKER" in reg.fill(f"[[SPAN:{cands[0][0]}]]"), q


def test_content_span_register_and_fill():
    # a content-anchored span (from retrieval, not structure) stores text
    # directly and fills verbatim, same as a structural span
    reg = SpanRegistry()
    h = reg.register_text_span("the passage about the bishop",
                               "verbatim CONTENT with candlesticks here")
    assert reg.text_of(h) == "verbatim CONTENT with candlesticks here"
    assert reg.fill(f"See:\n[[SPAN:{h}]]") == "See:\nverbatim CONTENT with candlesticks here"
    assert reg.source_of(h)          # has a (synthetic) source for the footer


def test_quote_vs_read_routing():
    # "summarize chapter 1" must route to READ (inject text), NOT quote (verbatim)
    assert READ_INTENT.search("summarize chapter 1")
    assert not QUOTE_INTENT.search("summarize chapter 1")
    # "quote chapter 1" is the opposite
    assert QUOTE_INTENT.search("quote chapter 1")
    assert not READ_INTENT.search("quote chapter 1")
    # "how does chapter 2 start" is a quote (verbatim opening), not a read
    assert QUOTE_INTENT.search("how does chapter 2 start")
    assert not READ_INTENT.search("how does chapter 2 start")
