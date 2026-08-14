"""Deterministic field extraction from the label→value table.

This is the ``--offline`` path, and it is a real fallback rather than a mock:
the repository runs end-to-end with no API key, and the tests exercise the
rules without billing anything. It also gives the LLM path something to be
checked against, the same way the regex prior checks the classifier.

It reads *less* than the model does — a value stated only in prose, in a
sentence the anchors do not cover, comes back as ``not_found`` instead of being
guessed. That is the intended failure: an absent field routes to a human, an
invented one corrupts the base.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rapidfuzz import fuzz

from ..config import load_config
from . import identifiers, layout
from ..models import EventType
from ..text import _squash, normalize
from .readers.base import Block

# A rota que produziu o valor era também uma constante de calibração — 0,97 para
# o valor ao lado de um rótulo exato, 0,80 para o recuperado de um padrão em
# prosa — e o número entrava na confiança do campo. A rota continua registrada,
# em `rationale`: "rótulo 'Data de pagamento'" e "padrão em prosa" dizem ao
# revisor de onde o valor veio, e dizem no recorte, que é onde ele confere. O
# decimal ao lado não acrescentava nada a isso e ainda competia, num `min`, com
# o autorrelato de um modelo, como se as duas grandezas fossem comparáveis.

# The year is `\d{4,}`, not `\d{4}`, for the same reason `_PERCENT` is greedy:
# this layer must hand the whole figure to repair.py rather than pre-trim it.
# Truncating "21/08/22026" to "21/08/2202" produces a *plausible-looking* date
# in the year 2202 and destroys the evidence that a digit was doubled.
DATE = r"(\d{1,2}\s+de\s+[a-zç]+\s+de\s+\d{4}|\d{1,2}[/.\-]\d{1,2}[/.\-]\d{4,}|\b\d{8}\b)"
_MONEY = re.compile(r"r\$\s*[\d.,]+", re.IGNORECASE)

# Deliberately greedy over separators. A tighter pattern like
# ``\d{1,3}(?:[.,]\d+)?%`` reads the OCR cell "...1.7,5%" as "7,5%" — it
# anchors on the second digit run and silently drops the leading "1", turning
# 17,5% into 7,5%. Narrowing a value here also robs repair.py of the context it
# needs to normalize: this layer should hand over the whole run of digits and
# separators and let normalization decide what they mean.
_PERCENT = re.compile(r"\d[\d.,\s]*%")
_INT = re.compile(r"\d+")


@dataclass
class RawField:
    name: str
    value_raw: str | None = None
    evidence_text: str | None = None
    block: Block | None = None
    rationale: str = ""


def _match_anchor(label: str, anchors: dict[str, list[str]]) -> str | None:
    """Longest anchor wins.

    Without that, ``valor bruto por acao`` and ``valor liquido por acao`` both
    match a short ``valor`` anchor and the two figures get swapped — a silent
    error that would survive every other check in the pipeline.

    Matching also runs space-blind, because OCR drops spaces constantly: the
    scanned notice yields ``Dataex-jCP``, which no anchor written as
    ``data ex-jcp`` would ever match.
    """
    normalized = normalize(label)
    squashed = _squash(label)
    best: tuple[str | None, int] = (None, 0)
    for field_name, candidates in anchors.items():
        for anchor in candidates or []:
            hit = anchor in normalized or (_squash(anchor) and _squash(anchor) in squashed)
            if hit and len(anchor) > best[1]:
                best = (field_name, len(anchor))
    if best[0]:
        return best[0]

    # Nothing matched exactly. OCR drops characters from labels as readily as
    # from values — the scanned notice yields "Data d pagamen." for "Data de
    # pagamento" — and an exact-only match turns a field that is plainly on the
    # page into a silent `not_found`. The bar is deliberately high: a loose
    # match here would assign the wrong field, which is worse than missing one.
    scored: tuple[str | None, float] = (None, 0.0)
    for field_name, candidates in anchors.items():
        for anchor in candidates or []:
            ratio = fuzz.ratio(_squash(anchor), squashed) / 100.0
            if ratio > scored[1]:
                scored = (field_name, ratio)
    return scored[0] if scored[1] >= 0.85 else None


def _first_date(text: str) -> str | None:
    match = re.search(DATE, normalize(text))
    return match.group(1) if match else None


def extract(doc, event_type: EventType, config_dir: str | None = None) -> dict[str, RawField]:
    anchors = load_config("anchors", config_dir)
    field_anchors = anchors.get("fields") or {}
    out: dict[str, RawField] = {}

    def put(name: str, value: str | None, block: Block | None, why: str) -> None:
        if not value or name in out:
            return
        out[name] = RawField(
            name=name,
            value_raw=value.strip(),
            evidence_text=(block.text if block else value).strip(),
            block=block,
            rationale=why,
        )

    # --- the label→value table -------------------------------------------
    for label_block, value_block in layout.label_value_pairs(doc.blocks):
        field_name = _match_anchor(label_block.text, field_anchors)
        if field_name is None:
            continue
        value = value_block.text.strip()

        if field_name == "ratio":
            numbers = _INT.findall(value)
            put("ratio_text", value, value_block, f"rótulo '{label_block.text.strip()}'")
            if len(numbers) >= 2:
                put("ratio_from", numbers[0], value_block, "primeiro número da proporção")
                put("ratio_to", numbers[1], value_block, "segundo número da proporção")
            continue

        if field_name in {"gross_per_share", "net_per_share", "cost_basis"}:
            money = _MONEY.search(value)
            put(
                field_name,
                money.group(0) if money else value,
                value_block,
                f"rótulo '{label_block.text.strip()}'",
            )
            put("currency", "BRL" if money else None, value_block, "valor grafado em R$")
            continue

        if field_name == "tax_rate":
            percent = _PERCENT.search(value)
            put(field_name, percent.group(0) if percent else value, value_block,
                f"rótulo '{label_block.text.strip()}'")
            continue

        if field_name.endswith("_date") or field_name in {"trading_start", "credit_date"}:
            date = _first_date(value)
            if date:
                put(field_name, date, value_block, f"rótulo '{label_block.text.strip()}'")
            else:
                # "A definir (vide aviso complementar)" is a *stated* absence,
                # which is a different fact from a missing field.
                put(field_name, value, value_block, f"rótulo '{label_block.text.strip()}'")
            continue

        put(field_name, value, value_block, f"rótulo '{label_block.text.strip()}'")

    # --- prose fallbacks --------------------------------------------------
    normalized_text = normalize(doc.raw_text)
    for field_name, templates in (anchors.get("prose") or {}).items():
        if field_name in out:
            continue
        for template in templates:
            match = re.search(template.replace("(DATE)", DATE), normalized_text)
            if match:
                block, _ = _locate_block(match.group(1), doc.blocks)
                put(field_name, match.group(1), block, f"padrão em prosa: '{template}'")
                break

    # --- identifiers ------------------------------------------------------
    ids = identifiers.find_identifiers(doc.raw_text, config_dir)
    for name, kind in (("isin", "isin"), ("ticker", "ticker"), ("cnpj", "cnpj")):
        value = ids.best(kind)
        if value:
            block, _ = _locate_block(value, doc.blocks)
            put(name, value, block, f"padrão estrutural de {kind.upper()}")

    # --- issuer and share class ------------------------------------------
    if doc.blocks:
        first = doc.blocks[0]
        put("issuer", first.text, first, "razão social no cabeçalho do aviso")

    share_class = _share_class(doc.raw_text)
    if share_class:
        put("share_class", share_class, None, "classe citada no texto")

    if "currency" not in out and "r$" in normalized_text:
        put("currency", "BRL", None, "valores grafados em R$")

    return out


def _locate_block(needle: str, blocks: list[Block]) -> tuple[Block | None, float]:
    from .provenance import locate

    return locate(needle, blocks)


def _share_class(text: str) -> str | None:
    normalized = normalize(text)
    if "preferencial" in normalized or re.search(r"\bpn\b", normalized):
        return "PN"
    if "ordinaria" in normalized or re.search(r"\bon\b", normalized):
        return "ON"
    return None
