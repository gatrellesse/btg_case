"""Stage 5 — normalization, with every step recorded.

Runs *after* grounding, so the guard checks what the document actually says
rather than what we turned it into. Each change appends to ``field.repairs``
with the rule that made it, which keeps the transformation auditable and
reversible.

The rule that matters most: **ambiguity is never resolved by guessing.** When a
date could plausibly be several dates, the field records the candidates, keeps
no value at all and goes to a human. Picking the most likely one and moving on
is how a wrong payment date reaches a custody system.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from decimal import Decimal, InvalidOperation

from ..formats import format_date, format_decimal
from ..models import ExtractedField, ReaderKind, Repair

MONTHS = {
    "janeiro": 1, "fevereiro": 2, "marco": 3, "março": 3, "abril": 4,
    "maio": 5, "junho": 6, "julho": 7, "agosto": 8, "setembro": 9,
    "outubro": 10, "novembro": 11, "dezembro": 12,
}

DATE_FIELDS = {"approval_date", "com_date", "ex_date", "payment_date", "trading_start", "credit_date"}
MONEY_FIELDS = {"gross_per_share", "net_per_share", "cost_basis"}
RATE_FIELDS = {"tax_rate"}
INT_FIELDS = {"ratio_from", "ratio_to"}

_UNDEFINED = re.compile(r"\ba\s+definir\b|oportunamente|a\s+ser\s+divulgad", re.IGNORECASE)


# --------------------------------------------------------------------------
# dates
# --------------------------------------------------------------------------


def parse_date(raw: str) -> tuple[date | None, str | None]:
    """Return (date, rule) for the formats these notices actually use.

    Reformats only. A value this cannot read unambiguously returns ``None`` and
    is handled as a set of candidate readings — normalization never decides
    between them.
    """
    text = raw.strip().lower()

    # `(?!\d)` matters: without it, "21/08/22026" silently reads as the year
    # 2202. Truncating a malformed year is as much a fabrication as inventing
    # one, and it produces a plausible-looking date that no later check flags.
    match = re.search(r"(\d{1,2})\s*[/.\-]\s*(\d{1,2})\s*[/.\-]\s*(\d{4})(?!\d)", text)
    if match:
        day, month, year = (int(g) for g in match.groups())
        try:
            return date(year, month, day), "dmy_separators"
        except ValueError:
            return None, None

    match = re.search(r"(\d{1,2})\s*de\s+([a-zç]+)\s+de\s+(\d{4})", text)
    if match and match.group(2) in MONTHS:
        try:
            return date(int(match.group(3)), MONTHS[match.group(2)], int(match.group(1))), "dmy_extenso"
        except ValueError:
            return None, None

    # OCR strips separators: "22062026"
    match = re.search(r"\b(\d{2})(\d{2})(\d{4})\b", text)
    if match:
        day, month, year = (int(g) for g in match.groups())
        try:
            return date(year, month, day), "dmy_sem_separador"
        except ValueError:
            return None, None

    return None, None


def _day_is_ambiguous(raw: str, field: ExtractedField) -> list[str] | None:
    """A day that may have lost a digit to OCR.

    These notices print two-digit days. A single-digit day read por um OCR sem
    ninguém para confirmá-lo é suspeito, não apenas curto. O dígito que falta
    pode estar de qualquer lado — ``2`` admite 02, 12, 22 e 20–29 —, então todas
    as leituras que o documento comporta são oferecidas e nenhuma é escolhida
    aqui.

    A guarda era `source_score >= 0.85`, e o menor peso possível *era* 0,85:
    a condição nunca foi falsa desde que o consenso passou a ditar o score, e
    este ramo estava morto sem que nada acusasse. O degrau diz a mesma coisa e
    diz de verdade.
    """
    if field.reader_kind != ReaderKind.OCR_DET or field.reading_level != "baixa":
        return None
    match = re.search(r"(?<!\d)(\d)\s*[/.\-]\s*(\d{1,2})\s*[/.\-]\s*(\d{4})(?!\d)", raw.strip())
    if not match:
        return None
    day, month, year = match.groups()
    leading = [f"{p}{day}/{int(month):02d}/{year}" for p in ("0", "1", "2")]
    trailing = [f"{day}{s}/{int(month):02d}/{year}" for s in "0123456789" if int(f"{day}{s}") <= 31]
    seen: list[str] = []
    for candidate in leading + trailing:
        if candidate not in seen:
            seen.append(candidate)
    return seen


def _year_is_ambiguous(raw: str) -> list[str] | None:
    """A year with more than four digits admits several readings.

    ``21/08/22026`` could be 2026 with a doubled digit, or 2202, or 2206.
    Deleting one is a *choice*, not a reformatting — so the options are
    recorded and only the disambiguation stage may pick one, and only when the
    event's own chronology leaves exactly one standing.
    """
    match = re.search(r"(\d{1,2})\s*[/.\-]\s*(\d{1,2})\s*[/.\-]\s*(\d{5,})", raw.strip())
    if not match:
        return None
    day, month, digits = match.groups()
    out: list[str] = []
    for i in range(len(digits)):
        year = digits[:i] + digits[i + 1 :]
        if len(year) == 4:
            candidate = f"{int(day):02d}/{int(month):02d}/{year}"
            if candidate not in out:
                out.append(candidate)
    return out or None


def repair_date(field: ExtractedField) -> None:
    raw = field.value_raw or ""

    if _UNDEFINED.search(raw):
        from ..models import AbsenceReason

        field.value = None
        field.absence_reason = AbsenceReason.STATED_UNDEFINED
        field.repairs.append(
            Repair(field=field.name, rule="ausencia_declarada", value_from=raw, value_to=None)
        )
        return

    candidates = _year_is_ambiguous(raw) or _day_is_ambiguous(raw, field)
    if candidates:
        field.value = None
        field.repairs.append(
            Repair(
                field=field.name,
                rule="leitura_ambigua",
                value_from=raw,
                value_to=None,
                candidates=candidates,
            )
        )
        field.notes.append(
            f"leitura não decidida pelo documento; candidatos: {', '.join(candidates)}"
        )
        return

    parsed, rule = parse_date(raw)
    if parsed is None:
        field.notes.append(f"formato de data não reconhecido: {raw!r}")
        return
    field.value = format_date(parsed)
    if field.value != raw:
        field.repairs.append(
            Repair(field=field.name, rule=rule or "data", value_from=raw, value_to=field.value)
        )


# --------------------------------------------------------------------------
# numbers
# --------------------------------------------------------------------------


def parse_ptbr_number(raw: str) -> Decimal | None:
    """pt-BR convention: dot groups thousands, comma is the decimal mark."""
    text = re.sub(r"[^\d.,]", "", raw or "")
    if not text:
        return None
    text = text.replace(".", "").replace(",", ".")
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def repair_money(field: ExtractedField) -> None:
    value = parse_ptbr_number(field.value_raw or "")
    if value is None:
        field.notes.append(f"valor monetário não interpretável: {field.value_raw!r}")
        return
    field.value = format_decimal(value)
    field.repairs.append(
        Repair(field=field.name, rule="moeda_ptbr", value_from=field.value_raw, value_to=field.value)
    )


def repair_rate(field: ExtractedField) -> None:
    """``17,5%`` → ``0.175``. Also survives OCR noise like ``1.7,5%``, where
    dropping the thousands dot recovers the intended 17,5."""
    value = parse_ptbr_number(field.value_raw or "")
    if value is None:
        return
    field.value = format_decimal(value / Decimal(100), min_decimals=0)
    field.repairs.append(
        Repair(field=field.name, rule="percentual", value_from=field.value_raw, value_to=field.value)
    )


def repair_int(field: ExtractedField) -> None:
    match = re.search(r"\d+", field.value_raw or "")
    if not match:
        return
    field.value = int(match.group(0))
    if str(field.value) != (field.value_raw or "").strip():
        field.repairs.append(
            Repair(field=field.name, rule="inteiro", value_from=field.value_raw, value_to=str(field.value))
        )


# --------------------------------------------------------------------------
# identifiers
# --------------------------------------------------------------------------


def repair_identifier(field: ExtractedField) -> None:
    raw = (field.value_raw or "").strip()
    cleaned = raw.replace("（", "(").replace("）", ")")

    if field.name == "isin":
        match = re.search(r"([A-Z]{2}[A-Z0-9]{9}\d)", cleaned.upper().replace(" ", ""))
        value = match.group(1) if match else cleaned.upper().replace(" ", "")
    elif field.name == "ticker":
        match = re.search(r"([A-Z]{4}\d{1,2})", cleaned.upper().replace(" ", ""))
        value = match.group(1) if match else cleaned.upper().replace(" ", "")
    elif field.name == "cnpj":
        digits = re.sub(r"\D", "", cleaned)
        value = (
            f"{digits[:2]}.{digits[2:5]}.{digits[5:8]}/{digits[8:12]}-{digits[12:]}"
            if len(digits) == 14
            else cleaned
        )
    elif field.name == "currency":
        value = "BRL" if ("R$" in raw or "BRL" in raw.upper() or not raw) else raw.upper()
    elif field.name == "share_class":
        upper = cleaned.upper()
        value = "PN" if "PREFERENC" in upper or "PN" in upper else ("ON" if "ORDIN" in upper or "ON" in upper else cleaned)
    else:
        value = cleaned

    field.value = value
    if value != raw:
        field.repairs.append(
            Repair(field=field.name, rule=f"normaliza_{field.name}", value_from=raw, value_to=str(value))
        )


# --------------------------------------------------------------------------


def repair(fields: dict[str, ExtractedField]) -> None:
    for name, field in fields.items():
        if field.value_raw is None:
            continue
        if name in DATE_FIELDS:
            repair_date(field)
        elif name in MONEY_FIELDS:
            repair_money(field)
        elif name in RATE_FIELDS:
            repair_rate(field)
        elif name in INT_FIELDS:
            repair_int(field)
        elif name in {"isin", "ticker", "cnpj", "currency", "share_class"}:
            repair_identifier(field)
        else:
            field.value = field.value_raw.strip()


def order_ratio(fields: dict[str, ExtractedField], event_type) -> None:
    """Põe a proporção na ordem que o próprio evento determina.

    `ratio_from` e `ratio_to` são "de" e "para" — e o que é "de" e o que é
    "para" **depende do evento**, não da ordem em que os números aparecem na
    frase. Um grupamento consolida: a contagem de ações cai, então vai do maior
    para o menor (10:1 — dez viram uma). Bonificação e desdobramento aumentam a
    contagem, então vão do menor para o maior.

    Sem essa regra escrita, os dois extratores leem a mesma frase e discordam
    por leitura, não por valor: no doc 08, "1 ação nova para cada 20 ações", as
    regras devolveram (1, 20) pela ordem do texto e o modelo devolveu (20, 1)
    por "para cada 20 que você tem". A segunda é a mais perigosa: lida como "de
    → para", ela descreve um grupamento de 20:1, o oposto econômico de uma
    bonificação.

    A troca é registrada como reparo, com a razão, porque inverter um par de
    proporção sem deixar rastro é indistinguível de tê-lo lido invertido.
    """
    from ..models import RATIO_EVENTS, EventType

    if event_type not in RATIO_EVENTS:
        return
    de, para = fields.get("ratio_from"), fields.get("ratio_to")
    if de is None or para is None or de.value is None or para.value is None:
        return

    try:
        a, b = Decimal(str(de.value)), Decimal(str(para.value))
    except InvalidOperation:
        return
    if a == b:
        return

    cresce = event_type != EventType.GRUPAMENTO
    na_ordem = (a < b) if cresce else (a > b)
    if na_ordem:
        return

    sentido = "menor para maior" if cresce else "maior para menor"
    de.value, para.value = para.value, de.value
    de.value_raw, para.value_raw = para.value_raw, de.value_raw
    for campo, papel in ((de, "ratio_from"), (para, "ratio_to")):
        campo.repairs.append(
            Repair(
                field=papel,
                rule="ordem_da_proporcao",
                value_from=str(b if papel == "ratio_from" else a),
                value_to=str(campo.value),
            )
        )
        campo.notes.append(
            f"proporção reordenada: {event_type.value} vai do {sentido}"
        )


def derive(fields: dict[str, ExtractedField], event_type) -> None:
    """Fill fields that the event's own mechanics determine.

    For a grupamento or desdobramento there is no separately labelled "data
    ex": the first session trading in the new form *is* the ex date. Deriving
    it is a statement about how the event works, not a guess — so it inherits
    a leitura do campo de origem e é registrada como reparo, visível a quem
    audita o registro.
    """
    from ..models import RATIO_EVENTS

    if event_type not in RATIO_EVENTS:
        return

    ex_date, trading_start = fields.get("ex_date"), fields.get("trading_start")
    if trading_start is None or trading_start.value is None:
        return
    if ex_date is not None and ex_date.value is not None:
        return

    target = ex_date or ExtractedField(name="ex_date")
    target.value = trading_start.value
    target.value_raw = trading_start.value_raw
    target.evidence_text = trading_start.evidence_text
    target.page, target.bbox = trading_start.page, trading_start.bbox
    target.reader_kind = trading_start.reader_kind
    target.reading_level = trading_start.reading_level
    target.corroborating_kinds = list(trading_start.corroborating_kinds)
    target.grounding = trading_start.grounding
    target.absence_reason = None
    target.rationale = (
        "derivada do início da negociação: em evento estrutural, a primeira sessão "
        "negociando na nova forma é a data ex"
    )
    target.repairs.append(
        Repair(
            field="ex_date",
            rule="derivada_de_trading_start",
            value_from=None,
            value_to=str(trading_start.value),
        )
    )
    fields["ex_date"] = target

