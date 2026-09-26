"""Language-agnostic normalization helpers that preserve the original fields."""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Mapping

_SPACE = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w]+", flags=re.UNICODE)
_NUMBER = re.compile(r"(?<!\w)\d+(?!\w)")
_POSTAL = re.compile(r"(?<!\d)\d{5,6}(?!\d)")

_LEGAL_SUFFIXES = {
    "corporation": "corp", "corp": "corp", "incorporated": "inc", "inc": "inc",
    "limited": "ltd", "ltd": "ltd", "private": "pvt", "pvt": "pvt",
    "limited liability company": "llc", "llc": "llc",
    "societe anonyme": "sa", "s a": "sa", "sa": "sa",
    "s a r l": "sarl", "sarl": "sarl", "s a s": "sas", "sas": "sas",
    "entreprise unipersonnelle a responsabilite limitee": "eurl",
    "e u r l": "eurl", "eurl": "eurl",
    "s a s u": "sasu", "sasu": "sasu", "s c i": "sci", "sci": "sci",
    "s n c": "snc", "snc": "snc", "e i": "ei", "ei": "ei",
    "e i r l": "eirl", "eirl": "eirl",
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
    "avenue": "av", "ave": "av", "av": "av",
    "boulevard": "bd", "blvd": "bd", "bd": "bd", "place": "pl", "pl": "pl",
    "chemin": "che", "che": "che", "residence": "res", "res": "res",
    "impasse": "imp", "imp": "imp", "rue": "rue", "strasse": "str",
}
_ADDRESS_ALIASES = {
    "bombay": "mumbai", "bangalore": "bengaluru", "calcutta": "kolkata",
    "madras": "chennai", "pondicherry": "puducherry", "trivandrum": "thiruvananthapuram",
}


def _base(text: object) -> str:
    raw = str(text or "").casefold().replace("œ", "oe").replace("æ", "ae")
    raw = raw.replace("&", " and ")
    value = unicodedata.normalize("NFKD", raw)
    chars = []
    previous_is_latin = False
    for char in value:
        if unicodedata.combining(char):
            if previous_is_latin:
                continue
            chars.append(char)
            continue
        chars.append(char)
        previous_is_latin = "a" <= char <= "z"
    value = _PUNCT.sub(" ", "".join(chars))
    return _SPACE.sub(" ", value).strip()


def normalize_name(text: object) -> str:
    value = _base(text)
    def canonicalize(match: re.Match[str]) -> str:
        token = match.group(1)
        return " " + _LEGAL_SUFFIXES.get(token.casefold(), token.casefold()) + " "
    return _SPACE.sub(" ", _LEGAL_PATTERN.sub(canonicalize, " " + value + " ")).strip()


def core_name(text: object) -> str:
    return _LEGAL_TAIL.sub("", normalize_name(text)).strip()


def normalize_address(text: object) -> str:
    value = _base(text)
    return _SPACE.sub(" ", re.sub(
        r"\b[\w]+\b", lambda m: _ADDRESS_ALIASES.get(
            m.group().casefold(), _STREET.get(m.group().casefold(), m.group())
        ), value,
    )).strip()


def extract_address_numbers(text: object) -> str:
    return " ".join(_NUMBER.findall(_base(text)))


def extract_postal_code(text: object) -> str:
    matches = _POSTAL.findall(_base(text))
    return matches[0] if matches else ""


def extract_acronym(text: object) -> str:
    tokens = [t for t in str(text or "").split() if t not in {"and", "of", "the", "&", "de", "la", "et"}]
    return "".join(t[0] for t in tokens if t) if len(tokens) >= 2 else ""


def composite_text(record: Mapping[str, object]) -> str:
    return f"{normalize_name(record.get('business_name', ''))} {normalize_address(record.get('business_address', ''))}".strip()


def preprocess_record(record: Mapping[str, object]) -> dict:
    enriched = dict(record)
    raw_name = record.get("business_name", "")
    raw_address = record.get("business_address", "")
    enriched["business_name_normalized"] = normalize_name(raw_name)
    enriched["business_name_core"] = core_name(raw_name)
    enriched["business_address_normalized"] = normalize_address(raw_address)
    enriched["business_address_numbers"] = extract_address_numbers(raw_address)
    return enriched


def preprocess_records(records: Iterable[Mapping[str, object]]) -> list[dict]:
    return [preprocess_record(record) for record in records]