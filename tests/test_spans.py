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


# --- colon-numbered divisions ("psalm 23", "john 3:16") -------------------- #
# Two named divisions whose bodies number as chapter:verse and reset to 1:1.
# Generic pattern (scripture, legal codes, specs) -- no bible-specific data.
_ALPHA = "\n".join(f"{c}:{v} alpha ch{c} v{v} lorem ipsum dolor sit amet."
                   for c in range(1, 4) for v in range(1, 9))
_PSALMS = "\n".join(f"{c}:{v} psalms ch{c} v{v} the field of green pastures."
                    for c in range(1, 26) for v in range(1, 7))
NUMBERED = f"The Book of Alpha\n\n{_ALPHA}\n\nThe Book of Psalms\n\n{_PSALMS}\n"


def _numreg():
    return SpanRegistry().register_file("lib/scripture.txt", NUMBERED)


def test_numbered_divisions_parsed():
    reg = _numreg()
    divs = reg.numbered["lib/scripture.txt"]
    assert len(divs) == 2                              # Alpha + Psalms
    assert divs[1]["primary"] == "psalms"
    assert 23 in divs[1]["chapters"]                   # Psalm 23 exists


def test_positional_quote_resolves_to_right_division():
    reg = _numreg()
    cands = reg.resolve("quote psalm 23")
    assert cands, "psalm 23 should resolve structurally"
    txt = reg.text_of(cands[0][0])
    assert "psalms ch23 v1" in txt                     # the RIGHT chapter
    assert "alpha" not in txt                          # not the other division


def test_positional_quote_is_case_insensitive():
    reg = _numreg()
    assert reg.resolve("Quote PSALM 23")               # uppercase resolves too


def test_verse_level_reference():
    reg = _numreg()
    cands = reg.resolve("quote alpha 2:3")
    assert cands
    txt = reg.text_of(cands[0][0])
    assert "alpha ch2 v3" in txt
    assert "ch2 v4" not in txt                          # exactly that verse


def test_unmatched_division_does_not_resolve():
    reg = _numreg()
    assert not reg.resolve("quote leviticus 5")        # no such division here


def test_glued_and_punctuated_positional_forms():
    reg = _numreg()
    for q in ("psalm23", "Psalm23", "psalm-23", "psalm.23", "psalm 23"):
        cands = reg.resolve(q)
        assert cands, f"{q!r} should resolve"
        assert "psalms ch23 v1" in reg.text_of(cands[0][0]), q
    # a name that isn't a division must NOT resolve, even glued
    assert not reg.resolve("python3 tutorial")


def test_non_monotonic_colon_data_is_not_addressable():
    """Sports scores / ratios / clock times use 'N:M' but jump around -- they
    must NOT be mistaken for a numbered book, or 'results 3' would 'quote' a row."""
    res = [(1,1),(2,1),(3,0),(2,2),(1,1),(4,2),(3,3),(0,0),(2,1),(1,0),
           (3,1),(2,0),(1,2),(5,1),(2,2),(1,1),(3,2),(0,1),(4,4),(2,3),(1,1),(3,0)]
    scores = "League Results\n\n" + "\n".join(
        f"{h}:{a} home {h} away {a} match report." for h, a in res)
    reg = SpanRegistry().register_file("lib/scores.txt", scores)
    assert "lib/scores.txt" not in reg.numbered          # not detected as a book
    assert not reg.resolve("results 3")                  # nothing to quote
    assert not reg.resolve("league 3")


# --- label-aware structural refs (article/section nesting, roman, inline body) --
def _legal():
    parts = []
    for a in ("I", "II"):                                # Article > Section
        parts.append(f"Article {a}.")
        for s in range(1, 9):                            # Sections 1..8
            parts.append(f"Section {s}. provision for article {a} section {s} "
                         + "lorem ipsum dolor sit amet " * 20)
    return SpanRegistry().register_file("lib/code.txt", "\n\n".join(parts))


def test_bare_section_uses_printed_label_not_position():
    reg = _legal()
    c = reg.resolve("section 8")           # NOT the 8th marker -> the labelled S8
    assert c and "article I section 8" in reg.text_of(c[0][0])
    # inline body on the header line is captured in full (not dropped as a stub)
    assert reg.text_of(c[0][0]).lstrip().startswith("Section 8.")


def test_nested_reference_scopes_to_parent():
    reg = _legal()
    c = reg.resolve("article 2 section 8")           # Section 8 INSIDE Article II
    assert c and "article II section 8" in reg.text_of(c[0][0])
    # order-independent: "section 8 of article 2" means the same
    c2 = reg.resolve("section 8 of article 2")
    assert c2 and "article II section 8" in reg.text_of(c2[0][0])


def test_roman_and_word_labels_and_amendments():
    body = "rights retained by the people " * 25
    text = ("Amendment I.\n\n" + body
            + "\n\nAmendment II.\n\n" + body
            + "\n\nAmendment XIV.\n\n" + "equal protection of the laws " * 25)
    reg = SpanRegistry().register_file("lib/bill.txt", text)
    assert reg.resolve("amendment 14")               # arabic
    assert reg.resolve("amendment xiv")              # roman -> 14


def test_single_sequence_sonnet():
    text = "\n\n".join(f"Sonnet {n}\n\nverse for sonnet {n} " + "beauty truth " * 40
                       for n in range(1, 31))
    reg = SpanRegistry().register_file("lib/sonnets.txt", text)
    c = reg.resolve("sonnet 18")
    assert c and "verse for sonnet 18" in reg.text_of(c[0][0])
