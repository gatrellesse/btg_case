"""The six validators, as pure functions exposed to the model as tools.

The model decides which to call and writes the narrative; the verdict is
computed in ``triage.py`` from what these return. Tool name, arguments and
result all go into the record verbatim, so an operator can see not just the
conclusion but the evidence that produced it.

Keeping them deterministic is what makes the audit trail worth anything: run
the same record twice and the same tools return the same findings.
"""

from __future__ import annotations

import csv
import re
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

from ..text import normalize
from ..formats import format_date, parse_date, parse_decimal
from ..models import ReasonCode

# --------------------------------------------------------------------------
# checksums — INFO only, never blocking
# --------------------------------------------------------------------------


def isin_checksum_ok(isin: str) -> bool:
    """ISO 6166 check digit (Luhn over letter-expanded digits)."""
    isin = (isin or "").strip().upper()
    if len(isin) != 12 or not isin[-1].isdigit():
        return False
    expanded = "".join(str(ord(c) - 55) if c.isalpha() else c for c in isin[:-1])
    if not expanded.isdigit():
        return False
    total = 0
    for i, digit in enumerate(reversed([int(d) for d in expanded])):
        if i % 2 == 0:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return (10 - total % 10) % 10 == int(isin[-1])


def cnpj_checksum_ok(cnpj: str) -> bool:
    digits = [int(c) for c in re.sub(r"\D", "", cnpj or "")]
    if len(digits) != 14 or len(set(digits)) == 1:
        return False
    for size, weights in (
        (12, [5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]),
        (13, [6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]),
    ):
        remainder = sum(a * b for a, b in zip(digits[:size], weights)) % 11
        if digits[size] != (0 if remainder < 2 else 11 - remainder):
            return False
    return True


#: B3 suffix convention.
TICKER_CLASS = {"3": "ON", "4": "PN", "5": "PNA", "6": "PNB", "11": "UNIT"}

#: Identity outcome → severity, where severity means *doubt about the reading*.
#:
#: É a mesma separação de sempre, um nível acima: esta severidade fala da
#: *leitura* — "isto foi lido corretamente?" —, e não da identidade do ativo,
#: que é o que o resultado do lookup relata em `status`.
#:
#: So the two families separate cleanly:
#:
#: * MISMATCH, KEY_CONFLICT, WEAK_MATCH cast doubt on the *reading* — the
#:   document disagrees with an independent registry, or its identifiers
#:   disagree with each other. Something was probably misread. WARN.
#: * NOT_FOUND and INACTIVE say nothing about the reading. Doc 08's ISIN is
#:   transcribed perfectly; the issuer simply is not registered. Marcar o campo
#:   como mal lido seria tratar a *situação cadastral da contraparte* como
#:   problema de legibilidade. Esses achados viajam como reason codes, que é
#:   onde a disposição é decidida.
LOOKUP_SEVERITY = {
    "MATCH": "PASS",
    "MISMATCH": "WARN",
    "WEAK_MATCH": "WARN",
    "KEY_CONFLICT": "WARN",
    "NOT_FOUND": "INFO",
    "INACTIVE": "INFO",
}


# --------------------------------------------------------------------------


class Validators:
    """Bound validators. Constructed once per run with the reference base."""

    def __init__(self, golden_path: str | Path, thresholds: dict | None = None) -> None:
        self.golden = self._load_golden(golden_path)
        self.thresholds = thresholds or {}

    @staticmethod
    def _load_golden(path: str | Path) -> list[dict]:
        with open(path, encoding="utf-8-sig", newline="") as fh:
            return [dict(row) for row in csv.DictReader(fh)]

    # -- reference base --------------------------------------------------

    def lookup_golden_record(self, **kwargs) -> dict:
        """Resolve the issuer, tagging the outcome with a machine-readable severity.

        The domain vocabulary (MATCH / NOT_FOUND / …) is what an auditor wants
        to read, but a triagem precisa de PASS/WARN/FAIL. Returning only the
        former meant every outcome fell through to the INFO default and passed
        as harmless — an issuer absent from the reference base cost its identity
        fields nothing. Both are emitted now, and neither is inferred from the
        other.
        """
        result = self._lookup(**kwargs)
        result["severity"] = LOOKUP_SEVERITY.get(result.get("status", ""), "WARN")
        return result

    def _lookup(
        self,
        issuer: str | None = None,
        cnpj: str | None = None,
        isin: str | None = None,
        ticker: str | None = None,
        share_class: str | None = None,
    ) -> dict:
        """Resolve the issuer against ``golden_records.csv``.

        Multi-key corroboration: each identifier is looked up independently and
        at least two must land on the same row. That is what survives a damaged
        ISIN on a scanned notice — ticker, CNPJ and name still agree — and what
        catches a transcription error that a single-key lookup would accept.

        The base carries identity only (issuer, cnpj, isin, ticker, class,
        segment, status): no dates, no amounts. It settles *who*, and nothing
        else.
        """
        keys = {"isin": isin, "ticker": ticker, "cnpj": cnpj, "issuer": issuer}
        matches: dict[str, int] = {}
        for key, value in keys.items():
            if not value:
                continue
            for index, row in enumerate(self.golden):
                if key == "issuer":
                    if normalize(row["emissor"]) == normalize(value):
                        matches[key] = index
                        break
                elif normalize(row[_column(key)]) == normalize(value):
                    matches[key] = index
                    break

        if not matches:
            return {
                "status": "NOT_FOUND",
                "reason_code": ReasonCode.ISSUER_NOT_IN_GOLDEN.value,
                "message": "nenhum identificador do aviso consta na base de referência",
                "keys_tried": {k: v for k, v in keys.items() if v},
            }

        rows = list(matches.values())
        winner = max(set(rows), key=rows.count)
        agreeing = [k for k, v in matches.items() if v == winner]
        conflicting = [k for k, v in matches.items() if v != winner]

        if conflicting:
            return {
                "status": "KEY_CONFLICT",
                "reason_code": ReasonCode.IDENTIFIER_KEY_CONFLICT.value,
                "message": "identificadores apontam para emissores diferentes na base",
                "agreeing_keys": agreeing,
                "conflicting_keys": conflicting,
                "matched_row": self.golden[winner],
            }

        if len(agreeing) < 2:
            return {
                "status": "WEAK_MATCH",
                "reason_code": ReasonCode.IDENTIFIER_WEAK_MATCH.value,
                "message": f"apenas '{agreeing[0]}' casou; corroboração exige ao menos 2 chaves",
                "matched_row": self.golden[winner],
                "agreeing_keys": agreeing,
            }

        row = self.golden[winner]
        if normalize(row.get("status", "")) not in ("ativo", "active"):
            return {
                "status": "INACTIVE",
                "reason_code": ReasonCode.ISSUER_INACTIVE.value,
                "message": f"emissor com status '{row.get('status')}'",
                "matched_row": row,
            }

        mismatched = []
        if share_class and normalize(row.get("classe", "")) != normalize(share_class):
            mismatched.append(
                {"field": "share_class", "document": share_class, "reference": row.get("classe")}
            )
        for key in ("isin", "ticker", "cnpj"):
            value = keys[key]
            if value and normalize(value) != normalize(row[_column(key)]):
                mismatched.append(
                    {"field": key, "document": value, "reference": row[_column(key)]}
                )

        if mismatched:
            return {
                "status": "MISMATCH",
                "reason_code": ReasonCode.IDENTIFIER_MISMATCH.value,
                "message": "emissor identificado, mas há campos divergentes",
                "matched_row": row,
                "mismatched_fields": mismatched,
                "agreeing_keys": agreeing,
            }

        return {
            "status": "MATCH",
            "message": "identidade confirmada na base de referência",
            "matched_row": row,
            "agreeing_keys": agreeing,
        }

    # -- coherence -------------------------------------------------------

    def check_date_coherence(
        self,
        approval_date: str | None = None,
        com_date: str | None = None,
        ex_date: str | None = None,
        payment_date: str | None = None,
        event_type: str | None = None,
    ) -> dict:
        """Chronology that any corporate action must satisfy.

        These come from how the events work, not from the sample: entitlement
        is fixed on the com date, the share trades ex the following session,
        and nobody is paid before the entitlement exists.
        """
        parsed = {
            "approval_date": _as_date(approval_date),
            "com_date": _as_date(com_date),
            "ex_date": _as_date(ex_date),
            "payment_date": _as_date(payment_date),
        }
        violations: list[dict] = []

        def compare(earlier: str, later: str, strict: bool, rule: str) -> None:
            a, b = parsed[earlier], parsed[later]
            if a is None or b is None:
                return
            ok = a < b if strict else a <= b
            if not ok:
                violations.append(
                    {"rule": rule, earlier: format_date(a), later: format_date(b)}
                )

        compare("approval_date", "com_date", False, "aprovação deve preceder ou igualar a data com")
        compare("com_date", "ex_date", True, "data ex deve ser posterior à data com")
        compare("ex_date", "payment_date", False, "pagamento não pode anteceder a data ex")
        compare("com_date", "payment_date", False, "pagamento não pode anteceder a data com")

        window = int(self.thresholds.get("date_window_years", 2))
        today = date.today()
        for name, value in parsed.items():
            if value and abs((value - today).days) > window * 365:
                violations.append(
                    {
                        "rule": f"data fora da janela de {window} anos",
                        name: format_date(value),
                    }
                )

        if violations:
            # The message is what an operator reads first, so it names the
            # broken rule and the two dates that break it — not a code.
            detail = "; ".join(
                f"{v['rule']} ({', '.join(f'{k}={x}' for k, x in v.items() if k != 'rule')})"
                for v in violations
            )
            message = f"cronologia impossível: {detail}"
        else:
            message = "cronologia coerente"

        return {
            "status": "FAIL" if violations else "PASS",
            "reason_code": ReasonCode.DATE_INCOHERENCE.value if violations else None,
            "violations": violations,
            "message": message,
            "dates_checked": {k: format_date(v) for k, v in parsed.items() if v},
        }

    def check_amount_coherence(
        self,
        gross_per_share: str | None = None,
        tax_rate: str | None = None,
        net_per_share: str | None = None,
    ) -> dict:
        """gross × (1 − rate) = net.

        Worth more than it looks: on a scanned notice where OCR scored the
        figures poorly, this identity holding to the last decimal is
        independent evidence that the digits were read correctly. Coherence
        rescues the amounts even while an adjacent date stays flagged.
        """
        gross, rate, net = (_as_decimal(v) for v in (gross_per_share, tax_rate, net_per_share))
        if gross is None or net is None:
            return {"status": "INFO", "message": "bruto ou líquido ausente; identidade não aplicável"}
        if rate is None:
            return {"status": "INFO", "message": "alíquota ausente; identidade não verificável"}

        expected = gross * (Decimal(1) - rate)
        delta = abs(expected - net)
        tolerance = Decimal(str(self.thresholds.get("amount_tolerance", 1e-6)))
        ok = delta <= tolerance
        return {
            "status": "PASS" if ok else "FAIL",
            "reason_code": None if ok else ReasonCode.AMOUNT_MISMATCH.value,
            "gross": str(gross),
            "rate": str(rate),
            "net_declared": str(net),
            "net_expected": str(expected),
            "delta": str(delta),
            "tolerance": str(tolerance),
        }

    def check_event_type_consistency(
        self,
        declared_type: str | None = None,
        title_prior: str | None = None,
        body_prior: str | None = None,
        markers: str | None = None,
    ) -> dict:
        """Compare the type against what the heading and the body each imply."""
        conflict = bool(title_prior and body_prior and title_prior != body_prior)
        disagrees = bool(body_prior and declared_type and body_prior != declared_type)
        codes = []
        if conflict:
            codes.append(ReasonCode.EVENT_TYPE_TITLE_CONFLICT.value)
        if disagrees:
            codes.append(ReasonCode.CLASSIFIER_DISAGREEMENT.value)
        return {
            "status": "WARN" if codes else "PASS",
            "reason_codes": codes,
            "declared_type": declared_type,
            "title_prior": title_prior,
            "body_prior": body_prior,
            "markers": markers,
            "message": (
                "cabeçalho e corpo do aviso indicam tipos diferentes; a substância prevalece"
                if conflict
                else "tipo consistente com os marcadores do corpo"
            ),
        }

    def check_identifier_format(
        self,
        isin: str | None = None,
        cnpj: str | None = None,
        ticker: str | None = None,
        share_class: str | None = None,
    ) -> dict:
        """Structural checks — **always INFO, never blocking.**

        In this reference base 11 of 13 ISINs and 7 of 8 CNPJs fail their check
        digits, because the data is synthetic. A pipeline that blocked on a
        checksum would reject most of its own reference base. Worse, the one
        CNPJ that *is* mathematically valid belongs to the issuer that is
        absent from the base entirely: a well-formed identifier is not a known
        counterparty, and only the reference base can answer that.
        """
        findings = {
            "isin_structure_ok": bool(isin and re.fullmatch(r"[A-Z]{2}[A-Z0-9]{9}\d", isin)),
            "isin_checksum_ok": isin_checksum_ok(isin) if isin else None,
            "cnpj_checksum_ok": cnpj_checksum_ok(cnpj) if cnpj else None,
            "ticker_pattern_ok": bool(ticker and re.fullmatch(r"[A-Z]{4}\d{1,2}", ticker)),
        }
        if ticker and share_class:
            suffix = re.sub(r"^[A-Z]{4}", "", ticker)
            implied = TICKER_CLASS.get(suffix)
            findings["ticker_class_consistent"] = (
                None if implied is None else implied == share_class.upper()
            )
        return {
            "status": "INFO",
            "findings": findings,
            "message": (
                "checagens estruturais são informativas: a base de referência é a "
                "única autoridade sobre identidade"
            ),
        }

    def check_required_fields(self, event_type: str, present: str, absent: str = "") -> dict:
        """Required-field check that distinguishes *kinds* of absence.

        A payment date the issuer explicitly deferred is not the same fact as a
        payment date that should be there and is not, and neither is the same
        as a field that does not apply to the event type at all.
        """
        from ..models import EventType, spec_for

        try:
            spec = spec_for(EventType(event_type))
        except ValueError:
            spec = spec_for(EventType.OUTRO)

        present_set = {f.strip() for f in (present or "").split(",") if f.strip()}
        excused = {f.strip() for f in (absent or "").split(",") if f.strip()}
        missing = [f for f in spec.required if f not in present_set and f not in excused]
        return {
            "status": "FAIL" if missing else "PASS",
            "reason_code": ReasonCode.MISSING_REQUIRED_FIELD.value if missing else None,
            "required": spec.required,
            "missing": missing,
            "excused_absences": sorted(excused),
            # Quais campos o achado atinge só se sabe depois de rodar — é o
            # resultado, não a chamada. As outras validações declaram os campos
            # que examinam de antemão; esta declara aqui, e sem isso o achado
            # saía sem alvo: o doc 03 dizia "campo obrigatório não veio" numa
            # tabela e, duas tabelas adiante, "Tipo declarado no aviso ·
            # ausente", as duas metades do mesmo fato sem nada ligando uma à
            # outra.
            "fields_affected": missing,
            "message": (
                f"campo(s) obrigatório(s) para {event_type} ausente(s): {', '.join(missing)}"
                if missing
                else "todos os campos obrigatórios do tipo vieram"
            ),
        }


def _column(key: str) -> str:
    return {"isin": "isin", "ticker": "ticker", "cnpj": "cnpj", "issuer": "emissor"}[key]


def _as_date(value: str | None) -> date | None:
    """pt-BR e ISO — o valor pode vir do reparo, do golden record ou do modelo."""
    return parse_date(value)


def _as_decimal(value) -> Decimal | None:
    """`1.000,00` e `1000.00` — vírgula presente decide quem é o milhar."""
    return parse_decimal(value)


# --------------------------------------------------------------------------
# tool declarations handed to the model
# --------------------------------------------------------------------------

TOOL_DECLARATIONS = [
    {
        "name": "lookup_golden_record",
        "description": (
            "Valida a identidade do emissor contra a base de referência "
            "golden_records.csv usando corroboração multi-chave (ISIN, ticker, CNPJ, "
            "razão social). A base contém apenas identidade — não tem datas nem valores."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "issuer": {"type": "string"},
                "cnpj": {"type": "string"},
                "isin": {"type": "string"},
                "ticker": {"type": "string"},
                "share_class": {"type": "string"},
            },
        },
    },
    {
        "name": "check_date_coherence",
        "description": (
            "Verifica a cronologia do evento: aprovação ≤ data com < data ex ≤ pagamento, "
            "e se as datas estão numa janela plausível. Datas em formato ISO."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "approval_date": {"type": "string"},
                "com_date": {"type": "string"},
                "ex_date": {"type": "string"},
                "payment_date": {"type": "string"},
                "event_type": {"type": "string"},
            },
        },
    },
    {
        "name": "check_amount_coherence",
        "description": (
            "Verifica a identidade bruto × (1 − alíquota) = líquido. Alíquota como "
            "fração decimal (0.175 para 17,5%)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "gross_per_share": {"type": "string"},
                "tax_rate": {"type": "string"},
                "net_per_share": {"type": "string"},
            },
        },
    },
    {
        "name": "check_event_type_consistency",
        "description": (
            "Compara o tipo classificado com o que o cabeçalho e o corpo do aviso "
            "indicam separadamente. Divergência entre título e substância é sinal."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "declared_type": {"type": "string"},
                "title_prior": {"type": "string"},
                "body_prior": {"type": "string"},
                "markers": {"type": "string"},
            },
        },
    },
    {
        "name": "check_identifier_format",
        "description": (
            "Checagens estruturais de ISIN/CNPJ/ticker e coerência entre sufixo do "
            "ticker e classe. SEMPRE informativo: nunca bloqueia um registro."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "isin": {"type": "string"},
                "cnpj": {"type": "string"},
                "ticker": {"type": "string"},
                "share_class": {"type": "string"},
            },
        },
    },
    {
        "name": "check_required_fields",
        "description": (
            "Confere os campos obrigatórios do tipo de evento, distinguindo ausência "
            "justificada (declarada pelo emissor ou inaplicável) de ausência real."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "event_type": {"type": "string"},
                "present": {"type": "string", "description": "campos preenchidos, separados por vírgula"},
                "absent": {"type": "string", "description": "ausências justificadas, separadas por vírgula"},
            },
            "required": ["event_type", "present"],
        },
    },
]
