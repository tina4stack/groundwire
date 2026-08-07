"""
Span registry -- addressable, verbatim passages the model can REFERENCE by a
handle instead of reproducing. The model emits `[[SPAN:war_and_peace-ch2]]`; the
proxy fills the exact stored bytes into the response afterward (post-fill). See
the "Post-fill" section of the README.

Why a separate registry (not just the retrieval index): quoting is an EXTRACTION
problem, not a search problem. We keep each file's exact text and expose named
spans over it -- whole files and structural units (chapters/cantos/books, and
Markdown headings) -- so a positional request ("how does chapter 2 start?") is a
deterministic lookup, and a large quote costs the model one handle token.

    reg = SpanRegistry()
    reg.register_file("drop/war_and_peace.txt", text)
    cands = reg.resolve("how does chapter 2 start?")   # [(handle, label), ...]
    prompt_block = reg.offer(cands)                     # goes in the system prompt
    answer = reg.fill(model_output)                     # [[SPAN:...]] -> verbatim
"""
from __future__ import annotations

import os
import re

# structural markers: CHAPTER/CANTO/BOOK/PART/LETTER + a number, each on its own
# line (Gutenberg/plain-text convention), and Markdown headings. The number may
# be arabic, roman, OR a spelled-out word/ordinal ("BOOK ONE", "PART SECOND") --
# classic novels number their top-level divisions in words.
_NUMWORD = (
    r"nineteenth|eighteenth|seventeenth|sixteenth|fifteenth|fourteenth|"
    r"thirteenth|twelfth|eleventh|tenth|ninth|eighth|seventh|sixth|fifth|"
    r"fourth|third|second|first|"
    r"nineteen|eighteen|seventeen|sixteen|fifteen|fourteen|thirteen|twelve|"
    r"eleven|twenty|thirty|forty|fifty|ten|nine|eight|seven|six|five|four|"
    r"three|two|one")
_STRUCT = re.compile(
    r"(?mi)^[ \t]*(chapter|canto|book|part|letter|section)\s+"
    r"([ivxlcdm]+|\d+|(?:" + _NUMWORD + r"))\b.*$")
_MD_HEADING = re.compile(r"(?m)^(#{1,6})[ \t]+(.+?)[ \t]*$")

_OPEN = re.compile(
    r"\b(start|starts|begin|begins|beginning|open|opening|"
    r"first (?:lines|words|paragraph|sentence))\b", re.I)
_FULL = re.compile(r"\b(full|entire|whole|complete|all of)\b", re.I)
# capture BOTH the unit word and the number, so "chapter 2" counts only chapter
# markers -- not the interleaved BOOK/PART markers.
_CHAP_REF = re.compile(
    r"\b(chapter|canto|part|book|section)\s+(\d+|[ivxlcdm]+)\b", re.I)
# "summarize the whole book / entire document / all of it"
_WHOLE = re.compile(r"\b(whole|entire|complete|all of|the book|the novel|"
                    r"the document|the text|everything)\b", re.I)
# wants the passage reproduced VERBATIM (-> offer a handle, post-fill exactly).
QUOTE_INTENT = re.compile(
    r"\b(quote|quotation|reproduce|verbatim|word[- ]for[- ]word|exact (?:text|"
    r"words|passage|wording)|full text|passage where|how does .* (?:start|begin|"
    r"open)|opening (?:lines|words|passage)|show me|give me the text|print)\b",
    re.I)
# wants to REASON over a passage (-> inject its text so the model can read it).
READ_INTENT = re.compile(
    r"\b(summari[sz]e|summar(?:y|ies)|synops|recap|gist|tl;?dr|explain|"
    r"analy[sz]e|describe|discuss|overview|paraphrase|what happens|what "
    r"happened|key (?:points|events|themes|ideas)|theme[s]? of)\b", re.I)

_OPEN_WORDS = 160          # how much "the opening of X" returns
_MIN_BODY = 400            # bodies shorter than this are table-of-contents /
                           # index entries (which duplicate every heading), not
                           # real chapters -- skip them so ordinals line up

# COLON-NUMBERED corpora: named divisions ("The Book of Psalms") whose bodies
# number as "chapter:verse" and reset to "1:1" at each new division. Common far
# beyond scripture -- legal codes, contracts, specs ("section 4:2"), RFCs. We
# address them positionally: "psalm 23" -> chapter 23 of the Psalms division;
# "john 3:16" -> that one verse. Generic pattern, NOT bible-specific data.
_VERSE = re.compile(r"(?m)^[ \t]*(\d+)[ \t]*:[ \t]*(\d+)\b")
_MIN_VERSES = 20           # fewer than this -> not a numbered corpus
# words in a division HEADER that don't name it (drop so 'psalms' is the key)
_DIV_GENERIC = frozenset(
    "the of a an book books gospel gospels epistle epistles letter letters gene "
    "first second third fourth fifth general according saint st prophet prophets "
    "called moses testament old new gener".split())
# a positional reference in a query: "<name> <chapter>[:<verse>]"
_POS_REF = re.compile(r"\b([a-z][a-z]+)\s+(\d+)(?:\s*[:.]\s*(\d+))?\b", re.I)


def _to_int(tok: str):
    tok = tok.strip().lower()
    if tok.isdigit():
        return int(tok)
    vals = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
    if not tok or any(c not in vals for c in tok):
        return None
    total, prev = 0, 0
    for c in reversed(tok):
        v = vals[c]
        total += -v if v < prev else v
        prev = max(prev, v)
    return total or None


def _stem(title: str) -> str:
    base = os.path.splitext(os.path.basename(title))[0].lower()
    return re.sub(r"[^a-z0-9]+", "_", base).strip("_")


class SpanRegistry:
    def __init__(self):
        self.texts = {}          # source title -> full verbatim text
        self.stems = {}          # stem -> source title
        self.units = {}          # source -> {unit_type: [(body_start, end), ...]}
        self.rawmarks = {}       # source -> [(pos, unit_type), ...] all markers
        self.spans = {}          # handle -> (source, start, end, label)
        self.numbered = {}       # source -> [division dicts] (colon-numbered)

    def clear(self):
        self.__init__()

    # -- ingest -------------------------------------------------------------- #
    def register_file(self, title: str, text: str):
        """Index one file's text: store it, and record structural span ranges,
        keyed by unit type (chapter / book / part / heading) so "chapter N"
        counts only chapters. Span bodies EXCLUDE the heading line."""
        self.texts[title] = text
        self.stems[_stem(title)] = title
        marks = [(m.start(), m.end(), m.group(1).lower())
                 for m in _STRUCT.finditer(text)]
        if not marks:
            marks = [(m.start(), m.end(), "heading")
                     for m in _MD_HEADING.finditer(text)]
        self.rawmarks[title] = [(hs, typ) for hs, he, typ in marks]
        by_type = {}
        for i, (hstart, hend, typ) in enumerate(marks):
            nxt = marks[i + 1][0] if i + 1 < len(marks) else len(text)
            if nxt - hend < _MIN_BODY:      # TOC / index entry, not real content
                continue
            by_type.setdefault(typ, []).append((hend, nxt))  # body after heading
        if by_type:
            self.units[title] = by_type
        self._index_numbered(title, text)
        return self

    # -- colon-numbered division index (bible/legal/spec addressing) ---------- #
    def _index_numbered(self, title: str, text: str):
        """Detect a 'chapter:verse' numbered corpus and record, per named
        division, the byte range of each chapter and verse. A division begins at
        every '1:1' (numbering resets); its NAME is the nearest non-empty,
        non-verse line above it ('The Book of Psalms')."""
        verses = [(m.start(), m.end(), int(m.group(1)), int(m.group(2)))
                  for m in _VERSE.finditer(text)]
        if len(verses) < _MIN_VERSES:
            return
        # division start = each verse that is chapter 1, verse 1
        starts = [i for i, (_, _, c, v) in enumerate(verses) if c == 1 and v == 1]
        if not starts:
            return
        divs = []
        for si, vi in enumerate(starts):
            v_end = starts[si + 1] if si + 1 < len(starts) else len(verses)
            div_verses = verses[vi:v_end]
            first_pos = div_verses[0][0]
            end_pos = (verses[v_end][0] if v_end < len(verses) else len(text))
            name = self._name_above(text, first_pos)
            keys = {t for t in re.findall(r"[a-z]+", name.lower())
                    if len(t) >= 3 and t not in _DIV_GENERIC}
            primary = (re.findall(r"[a-z]+", name.lower()) or [""])[-1]
            # chapter ranges: from a chapter's first verse to the next chapter's
            chapters, vspans, order = {}, {}, []
            for j, (vs, ve, c, v) in enumerate(div_verses):
                nxt = (div_verses[j + 1][0] if j + 1 < len(div_verses) else end_pos)
                vspans[(c, v)] = (vs, nxt)
                if c not in chapters:
                    chapters[c] = [vs, nxt]
                    order.append(c)
                else:
                    chapters[c][1] = nxt
            divs.append({"name": name, "keys": keys, "primary": primary,
                         "chapters": {c: tuple(r) for c, r in chapters.items()},
                         "verses": vspans})
        if divs:
            self.numbered[title] = divs

    @staticmethod
    def _name_above(text: str, pos: int) -> str:
        for line in reversed(text[:pos].splitlines()):
            s = line.strip()
            if not s or re.match(r"^\d+\s*:\s*\d+", s):
                continue
            return s
        return ""

    # coarsest-first: a "summarize the whole book" segments by book, then part,
    # then chapter. 'section' sits AFTER 'chapter' -- in Gutenberg texts stray
    # "SECTION" markers are usually license boilerplate, not narrative divisions,
    # and shouldn't preempt the real chapters.
    _SEG_ORDER = ("book", "part", "canto", "chapter", "section", "letter",
                  "heading")
    _SEG_MIN = 1500          # a real coarse segment is at least this many chars

    def coarse_segments(self, source: str):
        """Split a source into the COARSEST structural units that carry real
        content (skipping table-of-contents runs), for map-reduce summarizing.
        Each unit spans to the next marker OF THE SAME TYPE, so a book contains
        its chapters. Falls back to the whole document if there's no structure."""
        text = self.texts.get(source, "")
        marks = self.rawmarks.get(source, [])
        if not marks:
            return [("(document)", text)] if text.strip() else []
        from collections import defaultdict
        by_type = defaultdict(list)
        for pos, typ in marks:
            by_type[typ].append(pos)
        for typ in self._SEG_ORDER:
            ps = sorted(by_type.get(typ, []))
            if len(ps) < 2:
                continue
            spans = [(ps[i], ps[i + 1] if i + 1 < len(ps) else len(text))
                     for i in range(len(ps))]
            big = [(s, e) for s, e in spans if e - s >= self._SEG_MIN]
            if len(big) >= 2:
                return [(f"{typ} {i + 1}", text[s:e].strip())
                        for i, (s, e) in enumerate(big)]
        return [("(document)", text)] if text.strip() else []

    def build_from_folder(self, watch_dir: str, prefix: str = "drop"):
        """Register every ingestable file under a folder (mirrors ingest_repo's
        file selection, including Word/PDF extraction). `prefix` scopes the file
        titles (match ingest_repo's prefix so span sources == retrieval sources)."""
        from .extract import extract_text, EXTRACT_EXTS
        SKIP = {".git", "__pycache__", "node_modules", ".venv", "venv"}
        for dp, dns, fns in os.walk(watch_dir):
            dns[:] = [d for d in dns if d not in SKIP and not d.startswith(".")]
            for fn in sorted(fns):
                ext = os.path.splitext(fn)[1].lower()
                path = os.path.join(dp, fn)
                title = f"{prefix}/" + os.path.relpath(path, watch_dir).replace("\\", "/")
                try:
                    if ext in EXTRACT_EXTS:
                        text = extract_text(path)
                    else:
                        text = open(path, encoding="utf-8", errors="ignore").read()
                except OSError:
                    text = None
                if text:
                    self.register_file(title, text)
        return self

    # -- resolve an intent to concrete handles ------------------------------- #
    def _pick_source(self, q: str, unit: str):
        """Which registered file does the question mean, for this unit type? A
        stem token in the question wins; else the only file with that unit; else
        the file with the most of that unit."""
        have = [s for s in self.units if self.units[s].get(unit)]
        for src in have:
            toks = [t for t in _stem(src).split("_") if len(t) > 3]
            if any(t in q for t in toks):
                return src
        if len(have) == 1:
            return have[0]
        return max(have, key=lambda s: len(self.units[s][unit])) if have else None

    def _materialize(self, source: str, start: int, end: int, slug: str,
                     label: str, opening: bool) -> tuple:
        text = self.texts[source]
        if opening:
            body = " ".join(text[start:end].split()[:_OPEN_WORDS])
            end = start + len(text[start:end][:len(body) + 1])
        handle = f"{_stem(source)}-{slug}"
        self.spans[handle] = (source, start, end, label)
        return handle, label

    def resolve(self, question: str):
        """Return [(handle, label)] for structural references in the question
        (chapter/book/part N, optionally 'the opening of'). Empty if none apply."""
        q = question.lower()
        out, seen = [], set()
        opening = bool(_OPEN.search(q)) and not _FULL.search(q)
        for m in _CHAP_REF.finditer(q):
            unit, num = m.group(1).lower(), _to_int(m.group(2))
            src = self._pick_source(q, unit)
            ranges = self.units.get(src, {}).get(unit) if src else None
            if not num or not ranges or num > len(ranges):
                continue
            start, end = ranges[num - 1]
            label = (f"the opening of {unit} {num} of {src}" if opening
                     else f"{unit} {num} of {src}")
            slug = f"{unit}-{num}" + ("-open" if opening else "")
            h, lab = self._materialize(src, start, end, slug, label, opening)
            if h not in seen:
                seen.add(h)
                out.append((h, lab))
        # colon-numbered positional refs ("psalm 23", "john 3:16")
        for h, lab in self.resolve_numbered(question):
            if h not in seen:
                seen.add(h)
                out.append((h, lab))
        if not out and opening:
            # "how does <book> start / the opening of <book>" with no chapter
            # number -> the first chapter's opening (the real narrative start).
            src = self._pick_source(q, "chapter")
            ranges = self.units.get(src, {}).get("chapter") if src else None
            if ranges:
                start, end = ranges[0]
                out.append(self._materialize(src, start, end, "opening",
                                             f"the opening of {src}", True))
        return out

    @staticmethod
    def _match_division(divs, name_tok: str):
        """Pick the division a query name-token means. Case-insensitive; matches
        exact key or a stem/prefix ('psalm' -> the 'Psalms' division), preferring
        a match on the division's PRIMARY (most distinctive) token."""
        def score(div):
            keys, primary = div["keys"], div["primary"]
            if name_tok in keys:
                return 3 if primary == name_tok else 2
            for k in keys:
                if k.startswith(name_tok) or name_tok.startswith(k):
                    prim = primary.startswith(name_tok) or name_tok.startswith(primary)
                    return 2 if prim else 1
            return 0
        best, best_s = None, 0
        for div in divs:
            s = score(div)
            if s > best_s:
                best, best_s = div, s
        return best

    def resolve_numbered(self, question: str):
        """Resolve positional refs in colon-numbered corpora: '<name> <chap>' or
        '<name> <chap>:<verse>' -> the exact chapter/verse span of the matching
        named division. Case-insensitive throughout. Empty if none apply."""
        out, seen = [], set()
        for m in _POS_REF.finditer(question):
            name_tok = m.group(1).lower()
            chap = int(m.group(2))
            verse = int(m.group(3)) if m.group(3) else None
            if len(name_tok) < 3 or name_tok in _DIV_GENERIC:
                continue
            for src, divs in self.numbered.items():
                div = self._match_division(divs, name_tok)
                if not div:
                    continue
                nm = div["primary"] or name_tok
                if verse is not None and (chap, verse) in div["verses"]:
                    s, e = div["verses"][(chap, verse)]
                    label, slug = f"{nm} {chap}:{verse} of {src}", f"{_stem(nm)}-{chap}-{verse}"
                elif verse is None and chap in div["chapters"]:
                    s, e = div["chapters"][chap]
                    label, slug = f"{nm} {chap} of {src}", f"{_stem(nm)}-{chap}"
                else:
                    continue
                handle = f"{_stem(src)}-{slug}"
                self.spans[handle] = (src, s, e, label)
                if handle not in seen:
                    seen.add(handle)
                    out.append((handle, label))
        return out

    # -- prompt offer + post-fill -------------------------------------------- #
    def offer(self, candidates) -> str:
        """A system-prompt block listing referenceable spans (labels only, never
        the text) and how to reference them."""
        if not candidates:
            return ""
        lines = [f"  [[SPAN:{h}]]  = {label}" for h, label in candidates]
        first = candidates[0][0]
        return (
            "The user wants a stored passage, available to you as a token below. "
            "Your ENTIRE reply must be: a lead-in of AT MOST 6 words that contains "
            "NO names, dates or facts (e.g. 'The book opens:' or 'Chapter 2 "
            "begins:'), then a newline, then the token EXACTLY as written -- and "
            "nothing else. groundwire substitutes the verbatim passage for the token "
            "after you reply. Never write the passage yourself, never add a "
            "sentence of your own, never say you cannot find it.\n"
            f"Reply EXACTLY like this: 'The passage:\\n[[SPAN:{first}]]'\n"
            "Token(s):\n" + "\n".join(lines))

    def text_of(self, handle: str, cap_chars: int = None):
        s = self.spans.get(handle)
        if not s:
            return None
        source, start, end, _ = s
        t = self.texts[source][start:end].strip()
        if cap_chars and len(t) > cap_chars:
            t = t[:cap_chars].rsplit(" ", 1)[0] + " …[truncated]"
        return t

    def source_of(self, handle: str):
        s = self.spans.get(handle)
        return s[0] if s else None

    def resolve_whole(self, question: str):
        """Which whole FILE does a 'summarize the whole book / summarize <name>'
        request mean? A stem token in the question wins; else 'the whole book'
        style phrasing -> the largest source. None if neither applies."""
        q = question.lower()
        for stem, src in self.stems.items():
            if any(t in q for t in stem.split("_") if len(t) > 3):
                return src
        if _WHOLE.search(q) and self.texts:
            return max(self.texts, key=lambda s: len(self.texts[s]))
        return None

    def register_text_span(self, label: str, text: str) -> str:
        """Register a span from verbatim text directly (not a file offset) --
        used for CONTENT-anchored quotes, where a retrieved passage becomes a
        referenceable span. Returns a stable handle. Stored under a synthetic
        source so text_of/fill/source_of work unchanged."""
        base = re.sub(r"[^a-z0-9]+", "-", (label or "content").lower()).strip("-")
        base = base[:40] or "content"
        handle, i = base, 1
        while (handle in self.spans
               and self.texts.get(f"__span__:{handle}") != text):
            i += 1
            handle = f"{base}-{i}"
        key = f"__span__:{handle}"
        self.texts[key] = text
        self.spans[handle] = (key, 0, len(text), label)
        return handle

    _PLACEHOLDER = re.compile(r"\[\[SPAN:([A-Za-z0-9._\-]+)\]\]")

    def fill(self, response: str) -> str:
        """Replace every [[SPAN:handle]] in a model response with its verbatim
        stored passage. Unknown handles are left as-is (nothing to fill)."""
        def sub(m):
            return self.text_of(m.group(1)) or m.group(0)
        return self._PLACEHOLDER.sub(sub, response)

    def has_placeholder(self, text: str) -> bool:
        return bool(self._PLACEHOLDER.search(text))
