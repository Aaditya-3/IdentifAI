"""Deterministic, offline normalization and lightweight address parsing helpers."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Mapping

_SPACE = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w]+", flags=re.UNICODE)
_NUMBER = re.compile(r"(?<!\w)\d+[A-Za-z]?(?!\w)")
_POSTAL = re.compile(r"(?<!\d)\d{5,6}(?!\d)")
_ADDRESS_SEPARATOR = re.compile(r"[,;|]+")
_LANDMARK_FILLERS = re.compile(
    r"\b(?:near|opposite|opp|beside|behind|next\s+to|in\s+front\s+of|close\s+to)\b",
    flags=re.IGNORECASE,
)

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


@dataclass(frozen=True)
class AddressComponents:
    """Generic address components that can be extracted without external lookup.

    City/street parsing is deliberately conservative: city is populated only when
    the source string provides a clear separator (comma/semicolon/pipe). This avoids
    pretending that the last token is always a city.
    """

    normalized: str
    street_number: str
    street: str
    city: str
    postal_code: str


def _base(text: object) -> str:
    raw = str(text or "").casefold().replace("œ", "oe").replace("æ", "ae")
    raw = raw.replace("&", " and ")
    value = unicodedata.normalize("NFKD", raw)
    chars: list[str] = []
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
    value = _LANDMARK_FILLERS.sub(" ", _base(text))
    return _SPACE.sub(
        " ",
        re.sub(
            r"\b[\w]+\b",
            lambda m: _ADDRESS_ALIASES.get(
                m.group().casefold(), _STREET.get(m.group().casefold(), m.group())
            ),
            value,
        ),
    ).strip()


def extract_address_numbers(text: object) -> str:
    return " ".join(_NUMBER.findall(_base(text)))


def extract_postal_code(text: object) -> str:
    matches = _POSTAL.findall(_base(text))
    return matches[0] if matches else ""


def extract_acronym(text: object) -> str:
    tokens = [
        token for token in str(text or "").split()
        if token not in {"and", "of", "the", "&", "de", "la", "et"}
    ]
    return "".join(token[0] for token in tokens if token) if len(tokens) >= 2 else ""


def parse_address_components(text: object) -> AddressComponents:
    """Parse conservative street-number/street/city/postal components.

    No external geocoder or country database is used. The parser intentionally
    leaves city blank when the raw address has no explicit delimiter that can
    support a defensible city boundary.
    """
    raw = str(text or "")
    normalized = normalize_address(raw)
    postal = extract_postal_code(normalized)
    raw_numbers = _NUMBER.findall(_base(raw))

    street_number = ""
    for number in raw_numbers:
        if number != postal:
            street_number = number
            break

    segments = [normalize_address(part) for part in _ADDRESS_SEPARATOR.split(raw) if normalize_address(part)]
    if not segments and normalized:
        segments = [normalized]

    def _strip_numeric(value: str) -> str:
        value = re.sub(rf"(?<!\w){re.escape(street_number)}(?!\w)", " ", value) if street_number else value
        value = re.sub(rf"(?<!\w){re.escape(postal)}(?!\w)", " ", value) if postal else value
        return _SPACE.sub(" ", value).strip()

    street = _strip_numeric(segments[0]) if segments else ""
    city = ""
    if len(segments) >= 2:
        city = _strip_numeric(segments[-1])
        if city == street and len(segments) > 2:
            city = _strip_numeric(segments[-2])

    # When the address contains only one component, keep the street body but do
    # not guess a city from arbitrary trailing tokens.
    return AddressComponents(
        normalized=normalized,
        street_number=street_number,
        street=street,
        city=city,
        postal_code=postal,
    )


def composite_text(record: Mapping[str, object]) -> str:
    return f"{normalize_name(record.get('business_name', ''))} {normalize_address(record.get('business_address', ''))}".strip()


def preprocess_record(record: Mapping[str, object]) -> dict:
    enriched = dict(record)
    raw_name = record.get("business_name", "")
    raw_address = record.get("business_address", "")
    components = parse_address_components(raw_address)
    enriched["business_name_normalized"] = normalize_name(raw_name)
    enriched["business_name_core"] = core_name(raw_name)
    enriched["business_address_normalized"] = components.normalized
    enriched["business_address_numbers"] = extract_address_numbers(raw_address)
    enriched["business_address_street_number"] = components.street_number
    enriched["business_address_street"] = components.street
    enriched["business_address_city"] = components.city
    enriched["business_address_postal"] = components.postal_code
    return enriched


def preprocess_records(records: Iterable[Mapping[str, object]]) -> list[dict]:
    return [preprocess_record(record) for record in records]
