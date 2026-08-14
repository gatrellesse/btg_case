"""Layout: the decisive values live in a label→value table, not in prose.

Treating a page as running text is where the pairing breaks. In doc 01 a linear
read separates ``Valor bruto por ação ordinária`` from ``R$ 0,4275000000``; in
the scanned doc 07 label and value arrive *fused* into one OCR line with leader
dots between them. Either way, reading order is a bad proxy for "these two
belong together".

Geometry is a good one. Labels and their values share a horizontal band, so
grouping blocks into rows by vertical overlap reconstructs the pairing without
depending on ruling lines, on a table model, or on how any particular issuer
words its labels.

``find_tables`` was tried first and rejected: these notices use borderless
tables, and the text strategy returns the whole page as a single table whose
row/col indices carry no meaning. Where a real table recogniser is warranted,
the tier-2 reader already supplies one.
"""

from __future__ import annotations

import re

from .readers.base import Block

#: A run of leader punctuation, or a wide gap, between a label and its value.
#: Keys on the typographic convention rather than on any label's wording — and
#: stays loose about *which* punctuation, because OCR renders the same printed
#: dot leader as ``....``, ``,,,,.,,`` or ``……`` depending on the engine.
_LEADER = re.compile(r"(?:\s*[.,·•…]{2,}\s*|\s{3,})")

#: A narrower gap only counts as a leader when what follows actually looks like
#: a value. Without that guard, two spaces inside prose would split sentences.
_GAP = re.compile(r"\s{2,}")
_VALUE_LIKE = re.compile(r"^(?:R\$|\d|[A-Z]{4}\d)", re.IGNORECASE)


def split_label_value(text: str) -> tuple[str, str] | None:
    """Split ``Data de pagamento ....2/08.2026`` into label and value."""
    parts = [p for p in _LEADER.split(text) if p.strip()]
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()

    # Fallback for engines that preserve spacing instead of the dots, e.g.
    # "IRRF na fonte  17,5%" — accepted only because the right side is a figure.
    if len(parts) == 1:
        pieces = _GAP.split(text.strip())
        if len(pieces) == 2 and _VALUE_LIKE.match(pieces[1].strip()):
            return pieces[0].strip(), pieces[1].strip()
    return None


def _split_fused(blocks: list[Block]) -> list[Block]:
    """Unfuse ``label ... value`` lines that OCR delivered as one block.

    The box is apportioned by character share — approximate, but it keeps each
    half pointing at roughly its own pixels, which is what the audit crop needs.
    """
    out: list[Block] = []
    for block in blocks:
        pair = split_label_value(block.text)
        if pair is None:
            out.append(block)
            continue
        label, value = pair
        total = max(len(label) + len(value), 1)
        cut = block.bbox.x0 + (block.bbox.x1 - block.bbox.x0) * (len(label) / total)
        for text, box in (
            (label, block.bbox.model_copy(update={"x1": cut})),
            (value, block.bbox.model_copy(update={"x0": cut})),
        ):
            out.append(
                Block(
                    text=text,
                    bbox=box,
                    level=block.level,
                    reader_kind=block.reader_kind,
                    role=block.role,
                    order=block.order,
                    alternatives=list(block.alternatives),
                    # partir rótulo e valor não muda quem leu a região
                    corroborating_kinds=list(block.corroborating_kinds),
                )
            )
    return out


def group_rows(blocks: list[Block], overlap: float = 0.5) -> list[list[Block]]:
    """Group blocks into visual rows by vertical overlap."""
    rows: list[list[Block]] = []
    for block in sorted(blocks, key=lambda b: (b.bbox.page, b.bbox.y0)):
        placed = False
        for row in rows:
            ref = row[-1]
            if ref.bbox.page != block.bbox.page:
                continue
            top = max(ref.bbox.y0, block.bbox.y0)
            bottom = min(ref.bbox.y1, block.bbox.y1)
            shared = bottom - top
            shortest = min(ref.bbox.y1 - ref.bbox.y0, block.bbox.y1 - block.bbox.y0)
            if shortest > 0 and shared / shortest >= overlap:
                row.append(block)
                placed = True
                break
        if not placed:
            rows.append([block])
    for row in rows:
        row.sort(key=lambda b: b.bbox.x0)
    return rows


def enrich(blocks: list[Block]) -> list[Block]:
    """Assign roles and (row, col) so label and value become neighbours."""
    blocks = _split_fused(blocks)
    for row_index, row in enumerate(group_rows(blocks)):
        # A lone block on a line is prose, not a cell.
        if len(row) < 2:
            continue
        for col_index, block in enumerate(row):
            block.role = "table_cell"
            block.row = row_index
            block.col = col_index
    return blocks


def label_value_pairs(blocks: list[Block]) -> list[tuple[Block, Block]]:
    """Every (label, value) pair the page offers.

    The leftmost cell of a row is the label and the rightmost its value; rows
    with more than two cells contribute the outermost pair, which is how these
    notices lay out a wrapped label beside a single figure.
    """
    pairs: list[tuple[Block, Block]] = []
    for row in group_rows([b for b in blocks if b.role == "table_cell"]):
        if len(row) >= 2:
            pairs.append((row[0], row[-1]))
    return pairs
