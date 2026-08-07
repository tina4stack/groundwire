"""Session pipeline tests with a fake backend (no network)."""
from groundwire.backends import Backend
from groundwire.store import Store
from groundwire.app import Session


class FakeBackend(Backend):
    """Records the messages it's sent; returns a fixed answer. Because it never
    emits a [[SPAN:…]] handle, quote turns exercise the verbatim FALLBACK."""
    name = "fake"

    def __init__(self):
        self.seen = []

    def chat(self, messages, *, stream=True, options=None):
        self.seen.append((messages, options))
        return iter(["ANSWER"]) if stream else ["ANSWER"]

    def complete(self, messages, options=None):
        self.seen.append((messages, options))
        return "ANSWER"


DOC = ("CHAPTER I\n\nALPHA_ONE the first chapter body. " + "a " * 300 +
       "\n\nCHAPTER II\n\nBETA_TWO the second chapter body. " + "b " * 300)


def _session(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "book.txt").write_text(DOC, encoding="utf-8")
    store = Store(str(tmp_path / "g.db"))
    store.add_path(str(folder), "docs")
    fb = FakeBackend()
    return Session(store, {"fake": fb}, "fake"), fb, store


def test_normal_turn_injects_context_and_appends_footer(tmp_path):
    sess, fb, _ = _session(tmp_path)
    out = "".join(sess.turn(None, "what is in the first chapter body?"))
    assert "ANSWER" in out
    assert "— groundwire ▸ retrieved:" in out                # audit footer
    # the backend was handed a system CONTEXT block (grounding)
    sysmsgs = [m for m in fb.seen[-1][0] if m["role"] == "system"]
    assert sysmsgs and "CONTEXT" in sysmsgs[0]["content"]


def test_quote_turn_fills_verbatim_via_fallback(tmp_path):
    sess, fb, _ = _session(tmp_path)
    out = "".join(sess.turn(None, "quote the full text of chapter 2"))
    # fake never emitted the handle -> fallback patched the real chapter-2 bytes
    assert "BETA_TWO" in out
    # and the backend was NOT shown the passage (only a handle offer)
    assert "BETA_TWO" not in str(fb.seen[-1][0])


def test_read_turn_injects_the_span_text(tmp_path):
    sess, fb, _ = _session(tmp_path)
    "".join(sess.turn(None, "summarize chapter 1"))
    sysmsgs = [m for m in fb.seen[-1][0] if m["role"] == "system"]
    assert sysmsgs and "ALPHA_ONE" in sysmsgs[0]["content"]   # chapter 1 injected
    assert fb.seen[-1][1] and "num_ctx" in fb.seen[-1][1]     # context grown to fit


def test_inspect_reports_injected_context_without_calling_model(tmp_path):
    sess, fb, _ = _session(tmp_path)
    before = len(fb.seen)
    r = sess.inspect("what is in the first chapter body?")
    assert len(fb.seen) == before                 # model was NOT called
    assert r["mode"] == "normal"
    assert "ALPHA_ONE" in r["context"]            # the actual injected chunk text
    assert any(s["source"] for s in r["sources"])  # auditable sources

    q = sess.inspect("quote the full text of chapter 2")
    assert q["mode"] == "quote"
    assert "BETA_TWO" in q["context"]             # the verbatim bytes that fill in


def _bible_ish_session(tmp_path):
    """A corpus where a positional quote ('psalm 23') can't be satisfied: no
    chunk carries the word 'psalm', so a naive k=1 quote would post-fill the
    WRONG passage (the classic 'quote psalm 23 -> Acts 13 junk' bug)."""
    folder = tmp_path / "lib"
    folder.mkdir()
    (folder / "shepherd.txt").write_text(
        "23:1 The LORD is my shepherd; I shall not want. "
        + "green pastures still waters restoreth my soul " * 40, encoding="utf-8")
    # the trap: this distractor MENTIONS 'psalm' (as Acts 13:33 does) so a
    # term-overlap guard alone would still wrongly quote it for 'psalm 23'.
    (folder / "acts.txt").write_text(
        "13:33 as it is also written in the second psalm, Thou art my Son. "
        + "Israel Pilate Galilee resurrection sepulchre brethren " * 40,
        encoding="utf-8")
    store = Store(str(tmp_path / "g.db"))
    store.add_path(str(folder), "lib")
    return Session(store, {"fake": FakeBackend()}, "fake")


def test_quote_guard_rejects_unmatched_passage(tmp_path):
    sess = _bible_ish_session(tmp_path)
    # positional quote of a numbered unit we can't resolve structurally: even
    # though a distractor chunk MENTIONS 'psalm' (Acts 13:33), it must NOT be
    # post-filled as the verbatim quote. Falls through to normal grounding,
    # whose header tells the model to admit it could not find the passage.
    r = sess.inspect("quote psalm 23")
    assert r["mode"] == "normal"       # NOT a fabricated verbatim quote
    v = sess.inspect("quote john 3:16")
    assert v["mode"] == "normal"       # verse refs are positional too


def test_quote_guard_allows_matching_passage(tmp_path):
    sess = _bible_ish_session(tmp_path)
    r = sess.inspect("quote the passage about the shepherd")
    assert r["mode"] == "quote"                       # salient 'shepherd' matches
    assert "shepherd" in (r["context"] or "").lower()  # the RIGHT verbatim bytes


def _numbered_session(tmp_path):
    """A colon-numbered corpus (named divisions, chapter:verse) so a BARE
    positional reference resolves to a real span."""
    alpha = "\n".join(f"{c}:{v} alpha ch{c} v{v} lorem ipsum."
                      for c in range(1, 4) for v in range(1, 9))
    psalms = "\n".join(f"{c}:{v} psalms ch{c} v{v} green pastures."
                       for c in range(1, 26) for v in range(1, 7))
    folder = tmp_path / "lib"
    folder.mkdir()
    (folder / "scripture.txt").write_text(
        f"The Book of Alpha\n\n{alpha}\n\nThe Book of Psalms\n\n{psalms}\n",
        encoding="utf-8")
    store = Store(str(tmp_path / "g.db"))
    store.add_path(str(folder), "lib")
    return Session(store, {"fake": FakeBackend()}, "fake")


def test_bare_positional_reference_is_quoted(tmp_path):
    sess = _numbered_session(tmp_path)
    r = sess.inspect("psalm 23")               # bare -> show verbatim
    assert r["mode"] == "quote"
    assert "psalms ch23 v1" in (r["context"] or "")
    assert sess.inspect("Psalm 23")["mode"] == "quote"    # case-insensitive


def test_reference_inside_a_question_is_injected_not_quoted(tmp_path):
    sess = _numbered_session(tmp_path)
    r = sess.inspect("what happens in psalm 23")   # a question -> inject + answer
    assert r["mode"] == "read"
    assert "psalms ch23" in (r["context"] or "")   # the right passage injected


def test_ask_and_store_persists_both_messages(tmp_path):
    sess, _, store = _session(tmp_path)
    conv_id, answer = sess.ask_and_store(None, "what is in chapter one?")
    conv = store.get_conversation(conv_id)
    assert [m["role"] for m in conv["messages"]] == ["user", "assistant"]
    assert "ANSWER" in conv["messages"][1]["content"]
