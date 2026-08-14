"""Stage 7b — the verdict.

Computed in code from validator results and from the facts about each field —
quem leu, o que a base de referência disse, o valor apareceu mesmo no documento.
The model writes the narrative afterwards, from the reason codes produced here;
it never selects them. That ordering is the whole point — a justification that
can invent its own grounds is not an audit trail.

Three dispositions:

* ``AUTO``    — nothing to look at. This is the straight-through rate.
* ``REVIEW``  — a human decides. The record is emitted complete.
* ``BLOCKED`` — the record must not reach downstream systems as it stands.

The line between the last two is not severity of doubt but *what a human can
do about it*. A blurry date is REVIEW: someone opens the crop and reads it. An
impossible chronology is BLOCKED: no amount of looking fixes a payment that
precedes its own entitlement date.
"""

from __future__ import annotations

from .evidence import is_critical
from ..extraction.consensus import PIXEL_DERIVED as PIXEL_DERIVED_KINDS, weakest
from ..models import (
    AbsenceReason,
    Disposition,
    EventRecord,
    EventType,
    GroundingStatus,
    ReasonCode,
    Severity,
    Triage,
    TriageReason,
    spec_for,
)


def _reason(code: ReasonCode, message: str, field: str | None = None) -> TriageReason:
    return TriageReason.of(code, message, field)


def triage(record: EventRecord, thresholds: dict) -> Triage:
    reasons: list[TriageReason] = []
    for_review: list[str] = []

    # --- what the validators found ---------------------------------------
    #
    # Um achado vale para **todo** campo que a ferramenta examinou, não só para o
    # primeiro. A cronologia impossível do doc 05 implica as quatro datas: só a
    # primeira levava o motivo, e as outras três apareciam no relatório com "não
    # aceito" e nenhuma explicação — ou pior, com o código de limiar, que era
    # consequência e não causa.
    for validation in record.validations:
        alvos = validation.fields_affected or [None]
        for code in validation.reason_codes:
            mensagem = validation.message or validation.detail.get("message", "") or code.value
            for alvo in alvos:
                reasons.append(_reason(code, mensagem, field=alvo))
                # um campo com achado não é um campo aceito
                if alvo:
                    for_review.append(alvo)

    # --- classification ---------------------------------------------------
    if record.event_type == EventType.OUTRO:
        reasons.append(
            _reason(
                ReasonCode.UNKNOWN_EVENT_TYPE,
                "tipo de evento fora do catálogo; extraída apenas a base comum",
            )
        )
    if not record.classification.agrees_with_prior:
        reasons.append(
            _reason(
                ReasonCode.CLASSIFIER_DISAGREEMENT,
                f"prior determinístico indicou "
                f"{record.classification.prior.top.value if record.classification.prior.top else '—'}"
                f" e o modelo classificou {record.event_type.value}",
            )
        )

    # --- field by field ---------------------------------------------------
    required = set(spec_for(record.event_type).required)

    for name, field in record.fields.items():
        critical = is_critical(name, thresholds)

        if field.value is None:
            if field.absence_reason == AbsenceReason.STATED_UNDEFINED:
                reasons.append(
                    _reason(
                        ReasonCode.FIELD_STATED_UNDEFINED,
                        f"'{name}' declarado como indefinido pelo próprio emissor",
                        field=name,
                    )
                )
                for_review.append(name)
            elif any(r.candidates for r in field.repairs):
                candidates = next(r.candidates for r in field.repairs if r.candidates)
                reasons.append(
                    _reason(
                        ReasonCode.AMBIGUOUS_DATE_PARSE,
                        f"'{name}' ilegível o bastante para admitir mais de uma leitura: "
                        f"{', '.join(candidates)}",
                        field=name,
                    )
                )
                for_review.append(name)
            elif field.absence_reason != AbsenceReason.NOT_APPLICABLE:
                # Chegou aqui: os extratores não acharam e a varredura final,
                # relendo as três leituras inteiras, também não. A ausência
                # deixou de ser "ninguém procurou direito" e passou a ser um
                # fato sobre o documento — e um fato que o operador tem de ver,
                # obrigatório ou não. Um obrigatório já bloqueia pelo
                # `check_required_fields`, que o nomeia; repetir aqui seria
                # dizer a mesma coisa duas vezes na mesma linha.
                if name not in required:
                    reasons.append(
                        _reason(
                            ReasonCode.FIELD_NOT_FOUND,
                            f"'{name}' não aparece em nenhuma das leituras do documento",
                            field=name,
                        )
                    )
                for_review.append(name)
            # O que não se aplica ao tipo do evento não é achado: é o schema por
            # evento funcionando.
            continue

        # An unfounded value is a fabrication, not a weak reading.
        if field.grounding.status == GroundingStatus.ABSENT and critical:
            reasons.append(
                _reason(
                    ReasonCode.UNGROUNDED_CRITICAL_FIELD,
                    f"'{name}' não foi encontrado no texto-fonte do documento",
                    field=name,
                )
            )
            for_review.append(name)
            continue

        # Corroboração é um fato sobre a evidência, e é a única pergunta feita
        # aqui: *quantos mecanismos independentes leram isto e concordaram?*
        #
        # A base de referência não entra nesta conta. Ela responde outra coisa —
        # *este ativo é quem o aviso diz que é?* — e as duas verificações vão
        # para o relatório lado a lado, sem uma anular a outra: um ISIN lido só
        # pelo OCR continua lido só pelo OCR depois de bater com a base, e o
        # operador que abre o recorte está conferindo a leitura, não a
        # identidade.
        #
        # A camada de texto sozinha não entra: ela devolve a string do programa
        # que gerou o PDF, não um palpite sobre tinta.
        if field.corroborating_kinds and set(field.corroborating_kinds) <= PIXEL_DERIVED_KINDS:
            if len(field.corroborating_kinds) == 1:
                lido_por = field.corroborating_kinds[0].value
                reasons.append(
                    _reason(
                        ReasonCode.NO_CORROBORATION,
                        f"'{name}' sustentado por um único mecanismo de pixel "
                        f"({lido_por}); nenhuma outra leitura confirma o valor",
                        field=name,
                    )
                )
                for_review.append(name)

    # --- disposition ------------------------------------------------------
    severities = {r.severity for r in reasons}
    if Severity.BLOCK in severities:
        disposition = Disposition.BLOCKED
    elif Severity.REVIEW in severities:
        disposition = Disposition.REVIEW
    else:
        disposition = Disposition.AUTO

    # Deduplicate while keeping the first occurrence's wording.
    seen: set[tuple] = set()
    unique: list[TriageReason] = []
    for reason in reasons:
        key = (reason.code, reason.field)
        if key not in seen:
            seen.add(key)
            unique.append(reason)

    return Triage(
        disposition=disposition,
        reasons=unique,
        # o registro vale o que vale sua leitura crítica mais fraca
        weakest_level=weakest(
            f.reading_level
            for nome, f in record.fields.items()
            if f.value is not None and is_critical(nome, thresholds)
        ),
        fields_for_review=sorted(set(for_review)),
    )
