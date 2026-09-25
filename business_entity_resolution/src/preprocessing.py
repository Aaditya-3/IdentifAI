"""Country-independent, Unicode-aware normalization helpers."""
from __future__ import annotations

import re
import unicodedata

_SPACE = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w]+", flags=re.UNICODE)
_LEGAL = {
    r"\bcorporation\b": "corp", r"\bcorp\b": "corp", r"\bincorporated\b": "inc", r"\binc\b": "inc",
    r"\blimited\b": "ltd", r"\bltd\b": "ltd", r"\bprivate\b": "pvt", r"\bpvt\b": "pvt",
    r"\blimited liability company\b": "llc", r"\bllc\b": "llc", r"\bgmbh\b": "gmbh",
    r"\bsarl\b": "sarl", r"\bs\s*a\b": "sa",
}
_STREET = {r"\bstreet\b": "st", r"\bstrasse\b": "str", r"\broad\b": "rd", r"\bavenue\b": "ave", r"\bboulevard\b": "blvd"}


def _base(text: object) -> str:
    value = unicodedata.normalize("NFKD", str(text or "").casefold())
    chars = []
    previous_script = ""
    for ch in value:
        if unicodedata.combining(ch):
            if previous_script == "LATIN":
                continue
            chars.append(ch)
            continue
        chars.append(ch)
        if ch.isalpha():
            previous_script = "LATIN" if "LATIN" in unicodedata.name(ch, "") else "OTHER"
    value = "".join(chars)
    value = _PUNCT.sub(" ", value)
    return _SPACE.sub(" ", value).strip()


def normalize_name(text: object) -> str:
    value = _base(text)
    for pattern, replacement in _LEGAL.items():
        value = re.sub(pattern, replacement, value)
    return _SPACE.sub(" ", value).strip()


def normalize_address(text: object) -> str:
    value = _base(text)
    for pattern, replacement in _STREET.items():
        value = re.sub(pattern, replacement, value)
    return _SPACE.sub(" ", value).strip()


def composite_text(record: dict) -> str:
    return f"{normalize_name(record.get('business_name', ''))} {normalize_address(record.get('business_address', ''))}".strip()

