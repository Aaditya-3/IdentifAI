"""Language-agnostic normalization helpers that preserve the original fields."""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Mapping

_SPACE = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w]+", flags=re.UNICODE)
_NUMBER = re.compile(r"(?<!\w)\d+(?!\w)")

# Canonical forms retain the legal entity signal; core-name extraction removes
# recognized legal endings so the business name can be compared independently.
_LEGAL_SUFFIXES = {
    "corporation": "corp", "corp": "corp", "incorporated": "inc", "inc": "inc",
    "limited": "ltd", "ltd": "ltd", "private": "pvt", "pvt": "pvt",
    "limited liability company": "llc", "llc": "llc",
    "societe anonyme": "sa", "s a": "sa", "sa": "sa",
    "s a r l": "sarl", "sarl": "sarl", "s a s": "sas", "sas": "sas",
    "gesellschaft mit beschrankter haftung": "gmbh", "g m b h": "gmbh", "gmbh": "gmbh",
    "l l c": "llc", "p v t": "pvt", "l t d": "ltd", "i n c": "inc",
    "c o r p": "corp",
}
_LEGAL_PATTERN = re.compile(
    r"(?:^|\s)(" + "|".join(
        re.escape(term) for term in sorted(_LEGAL_SUFFIXES, key=len, reverse=True)
    ) + r")(?:\s|$)",
    flags=re.IGNORECASE,
)
_LEGAL_TAIL = re.compile(
    r"(?:\s+(?:" + "|".join(
        re.escape(term) for term in sorted(set(_LEGAL_SUFFIXES.values()), key=len, reverse=True)
    ) + r"))+$",
    flags=re.IGNORECASE,
)
_STREET = {
    "street": "st", "st": "st", "road": "rd", "rd": "rd",
    "avenue": "ave", "ave": "ave", "boulevard": "blvd", "strasse": "str",
}


def _base(text: object) -> str:
    value = unicodedata.normalize("NFKD", str(text or "").casefold())
    chars = []
    previous_script = ""
    for char in value:
        if unicodedata.combining(char):
            if previous_script == "LATIN":
                continue
            chars.append(char)
            continue
        chars.append(char)
        if char.isalpha():
            previous_script = "LATIN" if "LATIN" in unicodedata.name(char, "") else "OTHER"
    value = _PUNCT.sub(" ", "".join(chars))
    return _SPACE.sub(" ", value).strip()


def normalize_name(text: object) -> str:
    """Normalize Unicode and standardize recognized legal suffixes."""
    value = _base(text)

    def canonicalize(match: re.Match[str]) -> str:
        token = match.group(1)
        return " " + _LEGAL_SUFFIXES.get(token.casefold(), token.casefold()) + " "

    return _SPACE.sub(" ", _LEGAL_PATTERN.sub(canonicalize, " " + value + " ")).strip()


def core_name(text: object) -> str:
    """Return the normalized business name without trailing legal suffixes."""
    return _LEGAL_TAIL.sub("", normalize_name(text)).strip()


def normalize_address(text: object) -> str:
    """Normalize accents, punctuation, whitespace, and common street types."""
    value = _base(text)
    return _SPACE.sub(" ", re.sub(r"\b[\w]+\b", lambda m: _STREET.get(m.group().casefold(), m.group()), value)).strip()


def extract_address_numbers(text: object) -> str:
    """Return numeric address tokens in appearance order."""
    return " ".join(_NUMBER.findall(_base(text)))


def preprocess_record(record: Mapping[str, object]) -> dict:
    """Copy a record and add derived fields without changing its raw text."""
    enriched = dict(record)
    raw_name = record.get("business_name", "")
    raw_address = record.get("business_address", "")
    enriched["business_name_normalized"] = normalize_name(raw_name)
    enriched["business_name_core"] = core_name(raw_name)
    enriched["business_address_normalized"] = normalize_address(raw_address)
    enriched["business_address_numbers"] = extract_address_numbers(raw_address)
    return enriched


def preprocess_records(records: Iterable[Mapping[str, object]]) -> list[dict]:
    """Return records with derived columns while preserving all source values."""
    return [preprocess_record(record) for record in records]


def composite_text(record: Mapping[str, object]) -> str:
    return f"{normalize_name(record.get('business_name', ''))} {normalize_address(record.get('business_address', ''))}".strip()
