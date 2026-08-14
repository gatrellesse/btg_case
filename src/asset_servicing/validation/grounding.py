"""Stage 4 — the anti-hallucination guard.

Every claimed evidence span must exist in the source text. This runs *before*
repair, against the raw reading: once a date has been normalized to
``2026-06-22`` it no longer appears verbatim anywhere, and checking against the
normalized form would quietly turn the guard off.

The bar differs by how the text was obtained, which is the whole point of
tracking ``ReaderKind``:

* text layer — strict. The characters are exactly what the PDF contains, so a
  span that does not match was not read, it was invented.
* OCR — probabilistic. The reader is expected to have mangled characters, so
  progressively looser comparisons apply and the match quality is carried
  forward rather than being turned into a pass/fail.
"""

from __future__ import annotations

from rapidfuzz import fuzz

from ..models import (
    AbsenceReason,
    ExtractedField,
    Grounding,
    GroundingStatus,
    ReaderKind,
)
from ..text import _squash, normalize

#: OCR confuses these routinely; folding them costs nothing and rescues
#: readings that are correct in substance.
_OCR_CONFUSIONS = str.maketrans({"0": "o", "1": "l", "5": "s", "8": "b", "|": "l"})


def _fold(text: str) -> str:
    return _squash(text).translate(_OCR_CONFUSIONS)


def ground_field(
    field: ExtractedField,
    source_text: str,
    fuzzy_min: float = 0.85,
) -> ExtractedField:
    # A field that is legitimately absent has nothing to ground. Saying so is
    # different from saying it could not be found.
    if field.value_raw is None:
        field.grounding = Grounding(
            status=(
                GroundingStatus.NOT_APPLICABLE
                if field.absence_reason
                in (AbsenceReason.STATED_UNDEFINED, AbsenceReason.NOT_APPLICABLE)
                else GroundingStatus.ABSENT
            ),
            score=0.0,
        )
        return field

    evidence = (field.evidence_text or field.value_raw or "").strip()
    if not evidence:
        field.grounding = Grounding(status=GroundingStatus.ABSENT, score=0.0)
        return field

    strict = field.reader_kind in (ReaderKind.TEXT_LAYER, None)
    source_norm = normalize(source_text)
    source_squashed = _squash(source_text)
    evidence_norm = normalize(evidence)
    evidence_squashed = _squash(evidence)

    status, score = GroundingStatus.ABSENT, 0.0
    if evidence_norm and evidence_norm in source_norm:
        status, score = GroundingStatus.EXACT, 1.0
    elif evidence_squashed and evidence_squashed in source_squashed:
        # Same characters, different spacing or punctuation — the reading is
        # right, the typography is not.
        status, score = GroundingStatus.NORMALIZED, 0.85
    elif not strict:
        folded_source, folded_evidence = _fold(source_text), _fold(evidence)
        if folded_evidence and folded_evidence in folded_source:
            status, score = GroundingStatus.NORMALIZED, 0.85
        else:
            similarity = fuzz.partial_ratio(evidence_squashed, source_squashed) / 100.0
            if similarity >= fuzzy_min:
                status, score = GroundingStatus.FUZZY, similarity

    field.grounding = Grounding(
        status=status,
        score=round(score, 4),
        matched_text=evidence if status != GroundingStatus.ABSENT else None,
        bbox=field.bbox,
    )

    # The value has to live inside the span that was cited for it. A correct
    # quotation attached to a value that is not in it is still a fabrication.
    if status != GroundingStatus.ABSENT and field.value_raw:
        if _squash(field.value_raw) and _squash(field.value_raw) not in _squash(evidence):
            field.notes.append(
                "valor não aparece dentro do trecho citado como evidência — verificar"
            )
    return field


def ground(fields: dict[str, ExtractedField], source_text: str, fuzzy_min: float = 0.85) -> None:
    for field in fields.values():
        ground_field(field, source_text, fuzzy_min)
