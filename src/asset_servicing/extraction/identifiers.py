"""Identifier extraction by structure, then by label.

Layer 1 sweeps the whole page on shape alone — ISO 6166 for ISIN, the B3
ticker convention, the Receita Federal CNPJ format — so it never depends on
how a given issuer worded its notice. Layer 2 adds label anchors as
*reinforcement*: a hit raises confidence, an absence is not a failure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..config import load_config
from ..text import normalize


# --------------------------------------------------------------------------
# identifiers
# --------------------------------------------------------------------------


@dataclass
class Candidate:
    value: str
    anchored: bool = False
    context: str = ""


@dataclass
class IdentifierCandidates:
    isin: list[Candidate] = field(default_factory=list)
    ticker: list[Candidate] = field(default_factory=list)
    cnpj: list[Candidate] = field(default_factory=list)

    def best(self, kind: str) -> str | None:
        items: list[Candidate] = getattr(self, kind)
        if not items:
            return None
        # An anchored hit outranks a bare structural one.
        return sorted(items, key=lambda c: (not c.anchored,))[0].value


# Layer 1 — structure. From specification, not from the sample.
ISIN_ANCHORED = re.compile(r"ISIN[^A-Z0-9]{0,4}([A-Z]{2}[A-Z0-9]{9}\d)")
ISIN_STRUCTURAL = re.compile(r"(?<![A-Z0-9])([A-Z]{2}[A-Z0-9]{9}\d)(?![A-Z0-9])")
TICKER_STRUCTURAL = re.compile(r"(?<![A-Z0-9])([A-Z]{4}\d{1,2})(?![A-Z0-9])")
CNPJ_STRUCTURAL = re.compile(r"(\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2})")

#: Tokens that satisfy the ticker shape but never are one. Structural patterns
#: are permissive on purpose; the golden-record lookup is what ultimately
#: decides, and this only keeps obvious noise out of the candidate list.
TICKER_STOPWORDS = {"NIRE", "CNPJ", "ISIN"}


def _clean_for_ids(text: str) -> str:
    """Repair OCR artefacts that break identifier shapes, generically.

    Full-width parentheses and lost spaces are properties of scanned text, not
    of any one document, so this is normalization rather than a fix-up.
    """
    text = text.replace("（", " (").replace("）", ") ")
    return text.upper()


def find_identifiers(text: str, config_dir: str | None = None) -> IdentifierCandidates:
    anchors = load_config("anchors", config_dir)
    cleaned = _clean_for_ids(text)
    normalized = normalize(text)
    out = IdentifierCandidates()

    for match in ISIN_ANCHORED.finditer(cleaned):
        out.isin.append(Candidate(value=match.group(1), anchored=True))
    seen = {c.value for c in out.isin}
    for match in ISIN_STRUCTURAL.finditer(cleaned):
        if match.group(1) not in seen:
            out.isin.append(Candidate(value=match.group(1)))

    # Layer 2 — label anchors. Best-effort: a hit raises confidence, an absence
    # is not a failure, because layer 1 already swept the whole page.
    ticker_anchor_present = any(
        anchor in normalized for anchor in (anchors.get("ticker") or [])
    )
    for match in TICKER_STRUCTURAL.finditer(cleaned):
        value = match.group(1)
        if value[:4] in TICKER_STOPWORDS:
            continue
        out.ticker.append(Candidate(value=value, anchored=ticker_anchor_present))

    for match in CNPJ_STRUCTURAL.finditer(cleaned):
        out.cnpj.append(Candidate(value=_format_cnpj(match.group(1)), anchored=True))

    return out


def _format_cnpj(raw: str) -> str:
    digits = re.sub(r"\D", "", raw)
    if len(digits) != 14:
        return raw
    return f"{digits[:2]}.{digits[2:5]}.{digits[5:8]}/{digits[8:12]}-{digits[12:]}"
