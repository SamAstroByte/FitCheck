"""PDF text extraction and text cleaning."""
import re

import pdfplumber


def extract_text_from_pdf(source):
    """Extract text from a PDF. `source` is a file path or a file-like object."""
    parts = []
    with pdfplumber.open(source) as pdf:
        for page in pdf.pages:
            parts.append(page.extract_text() or "")
    return "\n".join(parts)


_URL = re.compile(r"https?://\S+|www\.\S+")
_EMAIL = re.compile(r"\S+@\S+")
_PHONE = re.compile(r"(?<!\d)\+?\d[\d\s().-]{8,}\d(?!\d)")
_KEEP = re.compile(r"[^\w\s+#./%,;:-]")


def _strip_phone(m):
    # only drop things that really look like phone numbers (>= 10 digits),
    # so date ranges like "2019 - 2022" survive
    return " " if sum(c.isdigit() for c in m.group(0)) >= 10 else m.group(0)


def clean_text(text):
    """Light cleaning: drop URLs, emails and phone numbers, keep characters that
    matter for skills (+ # . / -), collapse whitespace but keep line breaks."""
    text = text.replace("\r", "\n")
    text = _URL.sub(" ", text)
    text = _EMAIL.sub(" ", text)
    text = _PHONE.sub(_strip_phone, text)
    text = _KEEP.sub(" ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()
