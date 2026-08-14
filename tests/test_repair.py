"""Normalization and the refusal to resolve ambiguity by guessing."""

from __future__ import annotations

import sys
from datetime import date
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asset_servicing.formats import parse_decimal  # noqa: E402
from asset_servicing.validation import repair  # noqa: E402
from asset_servicing.models import (  # noqa: E402
    AbsenceReason,
    EventType,
    ExtractedField,
    ReaderKind,
)


def field(name: str, raw: str, **kwargs) -> ExtractedField:
    return ExtractedField(name=name, value_raw=raw, **kwargs)


# --------------------------------------------------------------------------
# dates
# --------------------------------------------------------------------------


def test_parses_the_formats_these_notices_use():
    assert repair.parse_date("15/06/2026")[0] == date(2026, 6, 15)
    assert repair.parse_date("28 de maio de 2026")[0] == date(2026, 5, 28)
    # OCR strips the separators entirely
    assert repair.parse_date("22062026")[0] == date(2026, 6, 22)


def test_impossible_date_is_not_coerced():
    assert repair.parse_date("31/02/2026")[0] is None


def test_malformed_year_is_offered_as_candidates_not_rewritten():
    """PP-OCRv6 renders the scanned payment date as ``21/08/22026``.

    Deleting a digit would make it parse cleanly, and that is exactly why it is
    refused here: normalization reformats, it never decides. The readings the
    document admits are recorded and only the disambiguation stage may choose
    between them, and only when the chronology leaves one standing.
    """
    assert repair.parse_date("21/08/22026")[0] is None

    f = field("payment_date", "21/08/22026")
    repair.repair_date(f)
    assert f.value is None
    candidates = next(r.candidates for r in f.repairs if r.candidates)
    assert "21/08/2026" in candidates and "21/08/2202" in candidates


def test_malformed_year_is_never_silently_truncated():
    """Reading 22026 as the year 2202 invents a plausible-looking date that no
    later check would flag."""
    parsed, _ = repair.parse_date("21/08/22026")
    assert parsed is None


def test_stated_deferral_is_recorded_as_such():
    f = field("payment_date", "A definir (vide aviso complementar)")
    repair.repair_date(f)
    assert f.value is None
    assert f.absence_reason == AbsenceReason.STATED_UNDEFINED


def test_ambiguous_ocr_day_is_left_unresolved():
    """The core rule: the document admits several readings and settles none, so
    the field records them all and goes to a human rather than picking one.

    The missing digit can be on either side. PP-OCRv4 read this same date as
    ``2/08.2026`` while PP-OCRv6 and a 400dpi re-render both read ``21/08`` —
    so a candidate set built only from *leading* digits (02, 12, 22) would have
    excluded the correct answer entirely.
    """
    f = field("payment_date", "2/08.2026", reader_kind=ReaderKind.OCR_DET, reading_level="baixa")
    repair.repair_date(f)
    assert f.value is None

    candidates = next(r.candidates for r in f.repairs if r.candidates)
    assert {"02/08/2026", "12/08/2026", "22/08/2026"} <= set(candidates)  # dígito perdido à esquerda
    assert "21/08/2026" in candidates  # dígito perdido à direita — o valor real
    assert all(1 <= int(c.split("/")[0]) <= 31 for c in candidates)


def test_corroborated_ocr_single_digit_day_is_accepted():
    """A suspeita é sobre uma leitura *sem corroboração*, não sobre dias curtos:
    onde o VLM leu a mesma coisa que o OCR, o dia de um dígito é o que o
    documento diz."""
    f = field("payment_date", "2/08/2026", reader_kind=ReaderKind.OCR_DET, reading_level="media")
    repair.repair_date(f)
    assert f.value == "02/08/2026"


def test_text_layer_is_never_treated_as_ambiguous():
    f = field("payment_date", "2/08/2026", reader_kind=ReaderKind.TEXT_LAYER, reading_level="alta")
    repair.repair_date(f)
    assert f.value == "02/08/2026"


# --------------------------------------------------------------------------
# numbers
# --------------------------------------------------------------------------


def test_ptbr_decimal_comma():
    assert repair.parse_ptbr_number("R$ 0,4275000000") == Decimal("0.4275000000")
    assert repair.parse_ptbr_number("R$ 50.000,00") == Decimal("50000.00")


def test_percentage_becomes_a_fraction():
    f = field("tax_rate", "17,5%")
    repair.repair_rate(f)
    assert parse_decimal(f.value) == Decimal("0.175")


def test_percentage_survives_ocr_noise():
    """``1.7,5%`` is how the scan rendered 17,5% — dropping the spurious
    thousands separator recovers the intended figure."""
    f = field("tax_rate", "...1.7,5%")
    repair.repair_rate(f)
    assert parse_decimal(f.value) == Decimal("0.175")


def test_rule_extractor_hands_repair_the_whole_figure():
    """Regression: the rule layer must not narrow a value before repair sees it.

    A tighter percent pattern matched "7,5%" inside the OCR cell "...1.7,5%",
    silently halving the withholding rate. It survived every other check —
    grounding passed, the figure looked plausible — and surfaced only because an
    independent extractor read 17,5% and the gross/net identity then failed.
    """
    from asset_servicing.extraction.rules import _PERCENT

    assert _PERCENT.search("...1.7,5%").group(0) == "1.7,5%"
    assert _PERCENT.search("17,5%").group(0) == "17,5%"

    f = field("tax_rate", _PERCENT.search("...1.7,5%").group(0))
    repair.repair_rate(f)
    assert parse_decimal(f.value) == Decimal("0.175")


# --------------------------------------------------------------------------
# identifiers
# --------------------------------------------------------------------------


def test_isin_recovered_from_ocr_debris():
    f = field("isin", "TLNR4（ISINBRTLNRACNPR2)")
    repair.repair_identifier(f)
    assert f.value == "BRTLNRACNPR2"


def test_cnpj_is_reformatted_from_bare_digits():
    f = field("cnpj", "33222111000144")
    repair.repair_identifier(f)
    assert f.value == "33.222.111/0001-44"


def test_every_change_is_recorded():
    f = field("cnpj", "33222111000144")
    repair.repair_identifier(f)
    assert f.repairs and f.repairs[0].value_from == "33222111000144"


# --------------------------------------------------------------------------
# derivation
# --------------------------------------------------------------------------


def test_structural_event_derives_ex_date_from_grouped_trading():
    fields = {
        "trading_start": ExtractedField(
            name="trading_start", value="2026-06-29", reading_level="muito_alta"
        ),
        "ex_date": ExtractedField(name="ex_date"),
    }
    repair.derive(fields, EventType.GRUPAMENTO)
    assert fields["ex_date"].value == "2026-06-29"
    assert fields["ex_date"].repairs[0].rule == "derivada_de_trading_start"


def test_cash_event_does_not_derive_an_ex_date():
    fields = {
        "trading_start": ExtractedField(name="trading_start", value="2026-06-29"),
        "ex_date": ExtractedField(name="ex_date"),
    }
    repair.derive(fields, EventType.DIVIDENDO)
    assert fields["ex_date"].value is None


def test_derivation_never_overwrites_a_stated_value():
    fields = {
        "trading_start": ExtractedField(name="trading_start", value="2026-06-29"),
        "ex_date": ExtractedField(name="ex_date", value="2026-06-15"),
    }
    repair.derive(fields, EventType.BONIFICACAO)
    assert fields["ex_date"].value == "2026-06-15"


# --------------------------------------------------------------------------
# a proporção: o sentido é do evento, não da ordem da frase
# --------------------------------------------------------------------------


def _par(de, para):
    return {
        "ratio_from": ExtractedField(name="ratio_from", value=de, value_raw=str(de)),
        "ratio_to": ExtractedField(name="ratio_to", value=para, value_raw=str(para)),
    }


def test_grupamento_vai_do_maior_para_o_menor():
    """10:1 — dez ações viram uma. A contagem cai."""
    from asset_servicing.models import EventType

    fields = _par(1, 10)
    repair.order_ratio(fields, EventType.GRUPAMENTO)
    assert (fields["ratio_from"].value, fields["ratio_to"].value) == (10, 1)
    assert any(r.rule == "ordem_da_proporcao" for r in fields["ratio_from"].repairs)


def test_bonificacao_vai_do_menor_para_o_maior():
    """O caso do doc 08: "1 ação nova para cada 20 ações". Lido como "de → para",
    o par (20, 1) descreveria um grupamento de 20:1 — o oposto econômico."""
    from asset_servicing.models import EventType

    fields = _par(20, 1)
    repair.order_ratio(fields, EventType.BONIFICACAO)
    assert (fields["ratio_from"].value, fields["ratio_to"].value) == (1, 20)


def test_desdobramento_tambem_aumenta_a_contagem():
    from asset_servicing.models import EventType

    fields = _par(3, 1)
    repair.order_ratio(fields, EventType.DESDOBRAMENTO)
    assert (fields["ratio_from"].value, fields["ratio_to"].value) == (1, 3)


def test_a_proporcao_ja_na_ordem_certa_nao_e_tocada():
    from asset_servicing.models import EventType

    fields = _par(10, 1)
    repair.order_ratio(fields, EventType.GRUPAMENTO)
    assert (fields["ratio_from"].value, fields["ratio_to"].value) == (10, 1)
    assert fields["ratio_from"].repairs == []


def test_evento_sem_proporcao_nao_e_reordenado():
    from asset_servicing.models import EventType

    fields = _par(20, 1)
    repair.order_ratio(fields, EventType.JCP)
    assert (fields["ratio_from"].value, fields["ratio_to"].value) == (20, 1)
