"""
Text extraction for binary document formats, so the drop folder indexes real
documents -- Word and PDF -- not just plain text.

Design goal: keep the common case dependency-free. A .docx is a zip of XML, so
stdlib pulls its text with no install. PDF genuinely needs a parser; we use
pypdf if present and degrade gracefully (return None + a one-time note) if not,
rather than indexing binary garbage.

    from groundwire.extract import extract_text, EXTRACT_EXTS
    text = extract_text("report.pdf")     # None if unsupported / extraction failed

Add formats by extending _EXTRACTORS. Everything here returns plain text that
then flows through groundwire's normal prose chunker.
"""
from __future__ import annotations

import os
import zipfile
import xml.etree.ElementTree as ET

_W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_warned = set()


def _warn_once(key: str, msg: str):
    if key not in _warned:
        _warned.add(key)
        print(f"groundwire: {msg}")


def _docx_text(path: str) -> str | None:
    """Extract visible text from a .docx (Office Open XML) with stdlib only.
    A .docx is a zip; the body lives in word/document.xml as <w:p> paragraphs
    of <w:t> runs. Iterating <w:p> also captures table cells (which nest
    paragraphs), so tables come through as newline-separated rows of text."""
    try:
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            xml = z.read("word/document.xml")
    except (zipfile.BadZipFile, KeyError, OSError):
        return None
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return None
    paras = []
    for p in root.iter(_W_NS + "p"):
        runs = [t.text for t in p.iter(_W_NS + "t") if t.text]
        if runs:
            paras.append("".join(runs))
    return "\n".join(paras) or None


def _pdf_text(path: str) -> str | None:
    """Extract text from a PDF via pypdf (pure-Python, no system deps). Returns
    None if pypdf isn't installed (with a one-time hint) or the PDF is scanned
    images with no text layer -- OCR is deliberately out of scope here."""
    try:
        from pypdf import PdfReader
    except ImportError:
        _warn_once("pypdf", "install pypdf to index PDFs (pip install pypdf)")
        return None
    try:
        reader = PdfReader(path)
        pages = [(page.extract_text() or "") for page in reader.pages]
    except Exception:
        return None
    text = "\n".join(pages).strip()
    return text or None


# extension -> extractor. Extend here to add .pptx, .xlsx, etc.
_EXTRACTORS = {
    ".docx": _docx_text,
    ".pdf": _pdf_text,
}

# the set the ingest layer checks to route a file here instead of open()/read()
EXTRACT_EXTS = frozenset(_EXTRACTORS)


def extract_text(path: str) -> str | None:
    """Return extracted plain text for a supported binary doc, else None.
    None means 'not my job' (unsupported ext) OR 'couldn't extract' -- the
    caller should skip the file rather than index bytes."""
    ext = os.path.splitext(path)[1].lower()
    fn = _EXTRACTORS.get(ext)
    return fn(path) if fn else None
