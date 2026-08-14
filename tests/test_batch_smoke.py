"""Invariants over the provided batch — and only invariants.

What this file deliberately does NOT assert is the disposition of each
document. Those verdicts are recorded in the README as a reference run:
evidence of how the pipeline behaves on one sample, not a contract. Freezing
them here would turn eight demonstration PDFs into the specification, and every
future improvement would look like a regression.

What must hold for *any* input is asserted instead: the batch completes, every
record validates against the schema, every value carries provenance, and
nothing is emitted without either evidence behind it or a stated reason for
having none.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asset_servicing.validation.evidence import load_thresholds  # noqa: E402
from asset_servicing.models import Disposition, EventRecord, GroundingStatus  # noqa: E402
from asset_servicing.pipeline import process_batch  # noqa: E402
from asset_servicing.validation.tools import Validators  # noqa: E402

DOCS = ROOT / "Case AI Dev - Envio" / "documents"
GOLDEN = ROOT / "Case AI Dev - Envio" / "golden_records" / "golden records.csv"

pytestmark = pytest.mark.skipif(
    not DOCS.exists() or not GOLDEN.exists(), reason="lote de demonstração ausente"
)


@pytest.fixture(scope="module")
def records() -> list[EventRecord]:
    thresholds = load_thresholds()
    return process_batch(
        sorted(DOCS.glob("*.pdf")),
        Validators(GOLDEN, thresholds),
        thresholds,
        client=None,  # deterministic path: no API key needed to run the suite
        workers=4,
    )


def test_every_document_produces_a_record(records):
    assert len(records) == len(list(DOCS.glob("*.pdf")))


def test_no_document_crashes_the_batch(records):
    failed = [r.document.file_name for r in records if r.triage.disposition == Disposition.FAILED]
    assert not failed, f"documentos com falha de processamento: {failed}"


def test_records_serialise_to_json(records):
    for record in records:
        payload = record.model_dump(mode="json")
        assert payload["document"]["file_name"]
        assert payload["event_type"]


def test_every_record_reaches_a_disposition(records):
    for record in records:
        assert record.triage.disposition in set(Disposition)


def test_every_extracted_value_carries_provenance(records):
    """A value an operator cannot trace is not auditable, however well it read."""
    for record in records:
        for name, field in record.fields.items():
            if field.value is None:
                continue
            assert field.evidence_text, f"{record.document.file_name}:{name} sem evidência"
            assert field.page is not None, f"{record.document.file_name}:{name} sem página"
            assert field.bbox is not None, f"{record.document.file_name}:{name} sem bbox"


def test_present_values_are_grounded_in_the_source(records):
    """The anti-hallucination guard: nothing is emitted that the document does
    not actually contain."""
    for record in records:
        for name, field in record.fields.items():
            if field.value is None:
                continue
            assert field.grounding.status != GroundingStatus.ABSENT, (
                f"{record.document.file_name}:{name} não foi encontrado no texto-fonte"
            )


def test_absent_values_explain_themselves(records):
    """Absence must be a stated fact, not a silent null."""
    for record in records:
        for name, field in record.fields.items():
            if field.value is not None:
                continue
            has_reason = (
                field.absence_reason is not None
                or any(r.candidates for r in field.repairs)
                or bool(field.notes)
            )
            assert has_reason, f"{record.document.file_name}:{name} ausente sem justificativa"


def test_every_record_runs_the_full_validator_set(records):
    """Coverage cannot depend on what a model chose to call."""
    expected = {
        "lookup_golden_record",
        "check_date_coherence",
        "check_amount_coherence",
        "check_event_type_consistency",
        "check_identifier_format",
        "check_required_fields",
    }
    for record in records:
        assert expected <= {v.tool for v in record.validations}


def test_flagged_records_carry_reasons_and_a_justification(records):
    for record in records:
        if record.triage.disposition == Disposition.AUTO:
            continue
        assert record.triage.reasons, f"{record.document.file_name} sinalizado sem motivo"
        assert record.triage.justification.strip()


def test_auto_records_have_nothing_outstanding(records):
    for record in records:
        if record.triage.disposition != Disposition.AUTO:
            continue
        assert not record.triage.reasons
        assert not record.triage.fields_for_review


def test_graph_and_linear_engines_agree(records):
    """Changing the orchestration must not change the outcomes.

    Both engines call the same extraction, validation and triage code — the
    graph rearranges who runs when, not what the rules decide. If these ever
    diverge in deterministic mode, the orchestration has grown business logic
    of its own, which is exactly what it must not do.
    """
    thresholds = load_thresholds()
    linear = process_batch(
        sorted(DOCS.glob("*.pdf")),
        Validators(GOLDEN, thresholds),
        thresholds,
        client=None,
        engine="linear",
        workers=4,
    )
    by_file = {r.document.file_name: r for r in linear}
    for record in records:
        other = by_file[record.document.file_name]
        assert record.event_type == other.event_type
        assert record.triage.disposition == other.triage.disposition


def test_scanned_document_is_routed_through_ocr(records):
    """Routing is by text-layer coverage, so a scan must not be read as text."""
    from asset_servicing.models import ReaderKind

    scanned = [r for r in records if "SCAN" in r.document.file_name.upper()]
    for record in scanned:
        assert ReaderKind.OCR_DET in record.document.reader_kinds
