"""Text folding shared by every layer.

Separated from provenance because matching is not provenance: classification
folds markers, extraction folds anchors, validation folds evidence against
the source. All three need the same rules and none of them needs a page.

Used for matching only — the stored evidence is always the raw text.
"""

from __future__ import annotations

import re
import unicodedata

#: Typographic quotes and dashes carry no meaning here, but they differ between
#: how a notice is typeset and how a label is written in config. Folding them
#: is what lets an anchor `data ex-dividendos` match `Data “ex-dividendos”`.
_PUNCT_FOLD = str.maketrans({
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "‘": "'", "’": "'", "‚": "'",
    "–": "-", "—": "-", "−": "-",
    " ": " ",
})


def normalize(text: str) -> str:
    """Fold the differences that never carry meaning: case, accents, spacing,
    typographic punctuation.

    Used for matching only — the stored evidence is always the raw text.
    """
    text = unicodedata.normalize("NFKD", text.translate(_PUNCT_FOLD))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text).strip().lower()


def _squash(text: str) -> str:
    """Drop every separator. OCR loses spaces constantly, so a comparison that
    depends on them rejects readings that are in fact correct."""
    return re.sub(r"""[\s.,;:()\[\]/\\|"'—–-]""", "", normalize(text))

