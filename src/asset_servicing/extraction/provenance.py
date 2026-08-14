"""Pixel provenance: from a value back to the region of the page it was read from.

The enunciado asks that an operator be able to audit each value *without
reopening the document*. A page number is not enough for that; a cropped image
of the exact region, embedded in the exceptions report, is.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf

from ..models import BBox
from ..text import _squash, normalize
from .readers.base import Block

CROP_DPI = 200
CROP_PAD = 3.0


#: Um bloco que contém o próprio valor extraído é a origem dele, e não um
#: fragmento qualquer da evidência. Acima de 0,85 para vencer o casamento
#: achatado, abaixo de 1,0 que continua reservado à contenção exata.
_HOLDS_VALUE = 0.9


def locate(
    evidence: str, blocks: list[Block], value: str | None = None
) -> tuple[Block | None, float]:
    """Find the block an evidence span came from.

    Returns the block and a match quality in [0, 1]. Tries progressively looser
    comparisons; the caller decides what quality is acceptable, since a text
    layer and an OCR line deserve different bars.

    ``value`` desempata quando vários blocos são fragmentos da mesma evidência.
    Sem ele o desempate é por tamanho, e num par rótulo/valor o rótulo vence
    sempre: em `Data-base ("data com") 15/07/2026`, o rótulo tem 15 caracteres
    achatados e a data tem 8. O campo herdava então bbox, degrau e
    `corroborating_kinds` do bloco do rótulo — ou seja, a corroboração media a
    concordância dos leitores sobre as *palavras* "Data-base (data com)", e um
    dígito errado na data não a movia.
    """
    if not evidence or not evidence.strip():
        return None, 0.0

    target, squashed = normalize(evidence), _squash(evidence)
    if not squashed:
        return None, 0.0
    wanted = _squash(value) if value else ""

    best: tuple[Block | None, float] = (None, 0.0)
    for block in blocks:
        block_norm = normalize(block.text)
        if target and target in block_norm:
            return block, 1.0
        block_squashed = _squash(block.text)
        if squashed and block_squashed and squashed in block_squashed:
            if 0.85 > best[1]:
                best = (block, 0.85)
        elif block_squashed and squashed and block_squashed in squashed:
            # The value straddles several OCR lines; a partial anchor is still
            # a real anchor.
            score = 0.6 * (len(block_squashed) / len(squashed))
            if wanted and wanted in block_squashed:
                score = max(score, _HOLDS_VALUE)
            if score > best[1]:
                best = (block, score)
    return best


def crop(pdf_path: str | Path, bbox: BBox, out_path: str | Path, dpi: int = CROP_DPI) -> str | None:
    """Render the region behind a value to a PNG for the exceptions report."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with pymupdf.open(pdf_path) as pdf:
            page = pdf[bbox.page - 1]
            rect = pymupdf.Rect(
                max(page.rect.x0, bbox.x0 - CROP_PAD),
                max(page.rect.y0, bbox.y0 - CROP_PAD),
                min(page.rect.x1, bbox.x1 + CROP_PAD),
                min(page.rect.y1, bbox.y1 + CROP_PAD),
            )
            if rect.is_empty or rect.width < 1 or rect.height < 1:
                return None
            page.get_pixmap(dpi=dpi, clip=rect).save(out_path)
        return str(out_path)
    except Exception:
        # A missing snippet must never cost us the record it belongs to.
        return None
