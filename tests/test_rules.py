"""Correctness of the validation rules, on fixtures built in code.

No PDF is opened here, and none of the provided batch is referenced. That is
deliberate: the rules must be right about the *domain*, and a test that asserts
"doc 05 blocks" would only prove the pipeline still behaves the way it happened
to behave on one sample. Here a chronology violation is constructed directly,
so the test says what the rule means.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asset_servicing.validation.tools import Validators, cnpj_checksum_ok, isin_checksum_ok  # noqa: E402

GOLDEN = """emissor,cnpj,isin,ticker,classe,segmento_listagem,status
Companhia Alfa S.A.,11.111.111/0001-11,BRALFAACNOR1,ALFA3,ON,Novo Mercado,ativo
Companhia Beta S.A.,22.222.222/0001-22,BRBETAACNPR2,BETA4,PN,Nível 1,ativo
Companhia Gama S.A.,33.333.333/0001-33,BRGAMAACNOR3,GAMA3,ON,Tradicional,suspenso
"""


@pytest.fixture
def validators(tmp_path: Path) -> Validators:
    path = tmp_path / "golden.csv"
    path.write_text(GOLDEN, encoding="utf-8")
    return Validators(path, {"amount_tolerance": 1e-6, "date_window_years": 50})


# --------------------------------------------------------------------------
# reference base
# --------------------------------------------------------------------------


def test_lookup_matches_when_several_keys_agree(validators):
    result = validators.lookup_golden_record(
        issuer="Companhia Alfa S.A.", isin="BRALFAACNOR1", ticker="ALFA3", share_class="ON"
    )
    assert result["status"] == "MATCH"
    assert set(result["agreeing_keys"]) >= {"isin", "ticker"}


def test_lookup_survives_one_damaged_identifier(validators):
    """A scan can mangle the ISIN and the issuer should still resolve — that is
    the point of corroborating across keys instead of trusting one.

    The damage is still reported. Resolving *who* the issuer is and accepting a
    wrong ISIN are different things: the record is usable, and the operator is
    told which identifier disagrees with the registry.
    """
    result = validators.lookup_golden_record(
        issuer="Companhia Alfa S.A.", cnpj="11.111.111/0001-11", isin="BRALF4ACN0R1"
    )
    assert result["matched_row"]["ticker"] == "ALFA3"
    assert result["status"] == "MISMATCH"
    assert [m["field"] for m in result["mismatched_fields"]] == ["isin"]


def test_lookup_reports_absent_issuer_rather_than_guessing(validators):
    result = validators.lookup_golden_record(
        issuer="Companhia Omega S.A.", isin="BROMEGACNOR9", ticker="OMEG3"
    )
    assert result["status"] == "NOT_FOUND"
    assert result["reason_code"] == "ISSUER_NOT_IN_GOLDEN"


def test_lookup_flags_keys_pointing_at_different_issuers(validators):
    result = validators.lookup_golden_record(isin="BRALFAACNOR1", ticker="BETA4", cnpj="22.222.222/0001-22")
    assert result["status"] == "KEY_CONFLICT"


def test_lookup_blocks_inactive_issuer(validators):
    result = validators.lookup_golden_record(issuer="Companhia Gama S.A.", ticker="GAMA3")
    assert result["status"] == "INACTIVE"


def test_lookup_reports_share_class_disagreement(validators):
    result = validators.lookup_golden_record(
        issuer="Companhia Beta S.A.", ticker="BETA4", share_class="ON"
    )
    assert result["status"] == "MISMATCH"
    assert any(m["field"] == "share_class" for m in result["mismatched_fields"])


def test_single_key_is_not_enough(validators):
    result = validators.lookup_golden_record(ticker="ALFA3")
    assert result["status"] == "WEAK_MATCH"


# --------------------------------------------------------------------------
# chronology
# --------------------------------------------------------------------------


def test_payment_before_com_date_fails(validators):
    result = validators.check_date_coherence(
        approval_date="2026-06-01",
        com_date="2026-07-15",
        ex_date="2026-07-16",
        payment_date="2026-07-10",  # before entitlement even exists
    )
    assert result["status"] == "FAIL"
    assert result["reason_code"] == "DATE_INCOHERENCE"


def test_ex_date_must_follow_com_date(validators):
    result = validators.check_date_coherence(com_date="2026-06-12", ex_date="2026-06-12")
    assert result["status"] == "FAIL"


def test_coherent_chronology_passes(validators):
    result = validators.check_date_coherence(
        approval_date="2026-05-28",
        com_date="2026-06-12",
        ex_date="2026-06-15",
        payment_date="2026-07-03",
    )
    assert result["status"] == "PASS"


def test_missing_dates_are_skipped_not_failed(validators):
    """An absent payment date is a different finding from an impossible one."""
    result = validators.check_date_coherence(com_date="2026-06-23", ex_date="2026-06-24")
    assert result["status"] == "PASS"


# --------------------------------------------------------------------------
# amounts
# --------------------------------------------------------------------------


def test_gross_net_identity_holds(validators):
    result = validators.check_amount_coherence(
        gross_per_share="0.1738420000", tax_rate="0.175", net_per_share="0.1434196500"
    )
    assert result["status"] == "PASS"


def test_gross_net_identity_catches_a_wrong_digit(validators):
    result = validators.check_amount_coherence(
        gross_per_share="0.1738420000", tax_rate="0.175", net_per_share="0.1534196500"
    )
    assert result["status"] == "FAIL"
    assert result["reason_code"] == "AMOUNT_MISMATCH"


def test_amount_check_is_informational_without_a_rate(validators):
    result = validators.check_amount_coherence(gross_per_share="0.31", net_per_share=None)
    assert result["status"] == "INFO"


# --------------------------------------------------------------------------
# identifiers — informational only
# --------------------------------------------------------------------------


def test_identifier_checks_never_block(validators):
    """The synthetic reference base fails its own check digits. A pipeline that
    blocked on a checksum would reject most of the records it exists to serve,
    so these findings are reported and never enforced."""
    result = validators.check_identifier_format(
        isin="BRALFAACNOR1", cnpj="11.111.111/0001-11", ticker="ALFA3", share_class="ON"
    )
    assert result["status"] == "INFO"
    assert result["findings"]["isin_structure_ok"] is True


def test_ticker_suffix_and_share_class_disagreement_is_visible(validators):
    result = validators.check_identifier_format(ticker="ALFA3", share_class="PN")
    assert result["findings"]["ticker_class_consistent"] is False


def test_checksum_helpers_are_correct():
    assert isin_checksum_ok("US0378331005") is True
    assert isin_checksum_ok("US0378331006") is False
    assert cnpj_checksum_ok("09.888.999/0001-21") is True
    assert cnpj_checksum_ok("11.111.111/0001-11") is False


# --------------------------------------------------------------------------
# required fields
# --------------------------------------------------------------------------


def test_required_field_missing_is_reported(validators):
    result = validators.check_required_fields(event_type="jcp", present="issuer,isin,ticker")
    assert result["status"] == "FAIL"
    assert "com_date" in result["missing"]


def test_justified_absence_does_not_count_as_missing(validators):
    """A payment date the issuer explicitly deferred is present-as-deferred, not
    absent — the record still needs a human, but for a different reason."""
    result = validators.check_required_fields(
        event_type="jcp",
        present="issuer,isin,ticker,event_type_label,currency,gross_per_share,com_date,ex_date",
        absent="payment_date",
    )
    assert result["status"] == "PASS"


def test_unknown_event_type_falls_back_instead_of_raising(validators):
    result = validators.check_required_fields(event_type="reducao_de_capital", present="issuer")
    assert result["status"] == "PASS"


# --------------------------------------------------------------------------
# severity vs. disposition
# --------------------------------------------------------------------------


def test_lookup_reports_severity_separately_from_its_outcome(validators):
    """The lookup speaks two vocabularies and must emit both.

    Its domain outcome (MATCH / NOT_FOUND / …) is what an auditor reads; the
    confidence model needs PASS/WARN/FAIL. Emitting only the first meant every
    identity outcome fell through to the INFO default and scored exactly like a
    pass — an issuer absent from the reference base cost its fields nothing.
    """
    ok = validators.lookup_golden_record(issuer="Companhia Alfa S.A.", ticker="ALFA3")
    assert (ok["status"], ok["severity"]) == ("MATCH", "PASS")

    conflict = validators.lookup_golden_record(
        isin="BRALFAACNOR1", ticker="BETA4", cnpj="22.222.222/0001-22"
    )
    assert (conflict["status"], conflict["severity"]) == ("KEY_CONFLICT", "WARN")


def test_unknown_issuer_does_not_discredit_the_reading(validators):
    """Severity means doubt about the *reading*, not about the counterparty.

    An issuer missing from the reference base was transcribed perfectly well —
    the document says what it says. Scoring that as a weak reading dragged the
    record below the blocking floor and overrode the decision to send it to
    review, so the finding travels as a reason code instead.
    """
    absent = validators.lookup_golden_record(issuer="Companhia Omega S.A.", isin="BROMEGACNOR9")
    assert absent["status"] == "NOT_FOUND"
    assert absent["severity"] == "INFO"
    assert absent["reason_code"] == "ISSUER_NOT_IN_GOLDEN"
