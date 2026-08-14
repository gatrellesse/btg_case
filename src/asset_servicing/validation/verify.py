"""Stage 6 — run the validators, then have the model explain the result.

One deliberate departure from "let the agent decide". Every validator runs on
every record, deterministically, rather than only the ones a model chose to
call. Coverage of a compliance check cannot depend on whether a model thought
of it — a validator skipped is a check that silently never happened, and the
record would look clean.

The tools are still genuine function declarations, and the model may call them
while writing its justification. What it cannot do is change the verdict: that
is computed in ``triage.py`` from what the functions returned.
"""

from __future__ import annotations

from ..llm.base import LLMError
from ..llm.schemas import VerificationOutput
from ..models import (
    AbsenceReason,
    EventRecord,
    ReasonCode,
    Severity,
    ValidationResult,
    ValidationStatus,
)
from .tools import TOOL_DECLARATIONS, Validators

SYSTEM = """Você é um operador sênior de Asset Servicing redigindo a justificativa \
de um registro extraído de um aviso de evento corporativo.

Escreva em português, de forma objetiva e curta (no máximo 6 linhas), dizendo:
o que foi confirmado, o que ficou pendente e por quê, e o que um humano precisa \
decidir.

RESTRIÇÃO: use SOMENTE os reason codes fornecidos. Não invente códigos, não \
invente valores e não contradiga o resultado das validações — elas já foram \
executadas e são a verdade do registro. Você pode chamar as tools para conferir \
um detalhe, mas o veredito não é seu."""


def _status(payload: dict | str | None) -> ValidationStatus:
    """Severity of a tool result, preferring an explicit ``severity`` field.

    A validator may report a domain-specific outcome in ``status`` (the lookup
    says MATCH / NOT_FOUND / …), which is what the audit trail should show. A
    triagem precisa de PASS/WARN/FAIL, então uma ferramenta com vocabulário
    próprio declara a severidade em separado em vez de deixá-la ser adivinhada —
    adivinhar fazia todo resultado de identidade cair em INFO, indistinguível de
    um pass.
    """
    if isinstance(payload, dict):
        raw = payload.get("severity") or payload.get("status")
    else:
        raw = payload
    return {
        "PASS": ValidationStatus.PASS,
        "FAIL": ValidationStatus.FAIL,
        "WARN": ValidationStatus.WARN,
        "INFO": ValidationStatus.INFO,
    }.get((raw or "INFO").upper(), ValidationStatus.INFO)


def _codes(result: dict) -> list[ReasonCode]:
    raw = [result.get("reason_code")] + list(result.get("reason_codes") or [])
    out = []
    for code in raw:
        if not code:
            continue
        try:
            out.append(ReasonCode(code))
        except ValueError:
            continue
    return out


def run_validators(record: EventRecord, validators: Validators) -> list[ValidationResult]:
    fields = record.fields
    value = lambda name: (fields[name].value if name in fields else None)  # noqa: E731

    present = [n for n, f in fields.items() if f.value is not None]
    # "Missing" and "present but unreadable" are different findings and get
    # different treatment. A date the OCR could not resolve *is* on the page —
    # AMBIGUOUS_DATE_PARSE already routes it to a human — so counting it as a
    # missing required field would block a record that only needs someone to
    # look at one crop.
    excused = [
        n
        for n, f in fields.items()
        if f.value is None
        and (
            f.absence_reason in (AbsenceReason.STATED_UNDEFINED, AbsenceReason.NOT_APPLICABLE)
            or any(r.candidates for r in f.repairs)
        )
    ]

    calls = [
        (
            "lookup_golden_record",
            {
                "issuer": value("issuer"),
                "cnpj": value("cnpj"),
                "isin": value("isin"),
                "ticker": value("ticker"),
                "share_class": value("share_class"),
            },
            validators.lookup_golden_record,
            ["issuer", "cnpj", "isin", "ticker"],
        ),
        (
            "check_date_coherence",
            {
                "approval_date": value("approval_date"),
                "com_date": value("com_date"),
                "ex_date": value("ex_date"),
                "payment_date": value("payment_date"),
                "event_type": record.event_type.value,
            },
            validators.check_date_coherence,
            ["approval_date", "com_date", "ex_date", "payment_date"],
        ),
        (
            "check_amount_coherence",
            {
                "gross_per_share": value("gross_per_share"),
                "tax_rate": value("tax_rate"),
                "net_per_share": value("net_per_share"),
            },
            validators.check_amount_coherence,
            ["gross_per_share", "net_per_share"],
        ),
        (
            "check_event_type_consistency",
            {
                "declared_type": record.event_type.value,
                "title_prior": (
                    record.classification.prior.title_prior.value
                    if record.classification.prior.title_prior
                    else None
                ),
                "body_prior": (
                    record.classification.prior.body_prior.value
                    if record.classification.prior.body_prior
                    else None
                ),
                "markers": ", ".join(
                    h.matched_text for h in record.classification.prior.hits if h.definitional
                )[:300],
            },
            validators.check_event_type_consistency,
            ["event_type"],
        ),
        (
            "check_identifier_format",
            {
                "isin": value("isin"),
                "cnpj": value("cnpj"),
                "ticker": value("ticker"),
                "share_class": value("share_class"),
            },
            validators.check_identifier_format,
            ["isin", "cnpj", "ticker"],
        ),
        (
            "check_required_fields",
            {
                "event_type": record.event_type.value,
                "present": ",".join(present),
                "absent": ",".join(excused),
            },
            validators.check_required_fields,
            [],
        ),
    ]

    results: list[ValidationResult] = []
    for name, args, fn, affected in calls:
        try:
            payload = fn(**args)
        except Exception as exc:  # noqa: BLE001
            # A validator that throws is a finding about the validator, not a
            # reason to lose the document.
            payload = {"status": "WARN", "error": f"{type(exc).__name__}: {exc}"}
        results.append(
            ValidationResult(
                tool=name,
                args={k: v for k, v in args.items() if v is not None},
                status=_status(payload),
                detail=payload,
                reason_codes=_codes(payload),
                # A lista declarada na chamada diz o que a ferramenta *examina*;
                # uma ferramenta cujos alvos só existem depois de rodar diz no
                # resultado. Sem essa segunda via, o achado de campo obrigatório
                # chegava à triagem sem alvo e virava achado de documento, sem
                # nomear o campo que faltou.
                fields_affected=payload.get("fields_affected") or affected,
                message=payload.get("message", "") or "",
            )
        )
    return results


def justify(record: EventRecord, validators: Validators, client=None) -> str:
    """Have the model turn reason codes into something an operator reads."""
    if client is None or not client.available():
        return _fallback_justification(record)

    lines = [
        f"Documento: {record.document.file_name}",
        f"Tipo classificado: {record.event_type.value} "
        + ("(unânime entre as modalidades)" if record.classification.unanimous
           else "(SEM unanimidade entre as modalidades)"),
        f"Disposição calculada: {record.triage.disposition.value}",
        "",
        "Resultado das validações (já executadas, determinísticas):",
    ]
    for validation in record.validations:
        lines.append(f"  - {validation.tool}: {validation.status.value} — {validation.message}")
        if validation.detail.get("violations"):
            lines.append(f"      violações: {validation.detail['violations']}")
        if validation.detail.get("missing"):
            lines.append(f"      obrigatórios ausentes: {validation.detail['missing']}")
    lines += ["", "Reason codes atribuídos (use apenas estes):"]
    for reason in record.triage.reasons:
        lines.append(f"  - {reason.code.value} [{reason.severity.value}] {reason.message}")
    if record.triage.fields_for_review:
        lines.append(f"Campos para conferência: {', '.join(record.triage.fields_for_review)}")

    try:
        response = client.with_tools(
            "\n".join(lines),
            TOOL_DECLARATIONS,
            {
                "lookup_golden_record": validators.lookup_golden_record,
                "check_date_coherence": validators.check_date_coherence,
                "check_amount_coherence": validators.check_amount_coherence,
                "check_event_type_consistency": validators.check_event_type_consistency,
                "check_identifier_format": validators.check_identifier_format,
                "check_required_fields": validators.check_required_fields,
            },
            system=SYSTEM,
        )
        for call in response.tool_calls:
            record.validations.append(
                ValidationResult(
                    tool=call["name"],
                    args=call["args"],
                    status=_status(call["result"] or {}),
                    detail=call["result"] or {},
                    reason_codes=_codes(call["result"] or {}),
                    message="chamada adicional solicitada pelo modelo durante a justificativa",
                )
            )
        text = (response.text or "").strip()
        return text or _fallback_justification(record)
    except LLMError:
        return _fallback_justification(record)


def _fallback_justification(record: EventRecord) -> str:
    """Readable without a model. The reason codes already carry the meaning."""
    if not record.triage.reasons:
        return (
            f"Registro de {record.event_type.value} consistente: identidade confirmada na base "
            "de referência, cronologia coerente e todo campo crítico sustentado por mais de "
            "uma leitura. Liberado sem intervenção humana."
        )
    blockers = [r for r in record.triage.reasons if r.severity == Severity.BLOCK]
    lead = (
        f"Registro de {record.event_type.value} **não pode seguir** para a base: "
        + "; ".join(r.message for r in blockers)
        + "."
        if blockers
        else f"Registro de {record.event_type.value} extraído, com pontos que exigem "
        "conferência humana antes da liberação."
    )
    parts = [lead]
    if record.triage.fields_for_review:
        parts.append(
            f"Conferir {len(record.triage.fields_for_review)} campo(s): "
            f"{', '.join(record.triage.fields_for_review)}."
        )
    return " ".join(parts)
