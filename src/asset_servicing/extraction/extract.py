"""Stage 3 — per-type field extraction, with provenance attached on the way out.

Whichever path produced a value (model or rules), the pipeline immediately
looks the evidence back up in the source blocks. That is what supplies the
page, the bounding box and the reader that produced it — none of which the
model is asked to assert.
"""

from __future__ import annotations

from . import provenance, rules
from ..llm.base import LLMError
from ..llm.schemas import (
    CashEventExtraction,
    GenericEventExtraction,
    LLMField,
    RatioEventExtraction,
)
from ..models import (
    CASH_EVENTS,
    RATIO_EVENTS,
    AbsenceReason,
    EventType,
    ExtractedField,
    spec_for,
)

SYSTEM = """Você extrai campos de avisos de eventos corporativos brasileiros para \
uma base de custódia. Regras inegociáveis:

1. NUNCA invente um valor. Se o campo não está no documento, retorne value=null \
com o absence_reason apropriado.
2. `evidence` deve ser um trecho VERBATIM do documento — copiado, não parafraseado, \
não normalizado. Ele será conferido automaticamente contra o texto original e um \
trecho que não existir derruba o campo.
3. `value` deve vir como está escrito no aviso (ex.: "R$ 0,4275000000", "15/06/2026"). \
A normalização é feita depois, em código.
4. Distinga ausências: 'stated_undefined' quando o aviso diz que o dado será \
divulgado depois; 'not_applicable' quando o campo não faz sentido para o tipo de \
evento; 'not_found' quando deveria constar e não consta.
5. Proporção (`ratio_from`/`ratio_to`): o sentido é do EVENTO, não da ordem da frase. \
Grupamento consolida — a contagem de ações cai, vai do maior para o menor (10:1). \
Bonificação e desdobramento aumentam a contagem, vão do menor para o maior (1 nova para \
cada 20 → 1 e 20). Devolva os dois números; a ordem é conferida em código."""


def _schema_for(event_type: EventType):
    if event_type in CASH_EVENTS:
        return CashEventExtraction
    if event_type in RATIO_EVENTS:
        return RatioEventExtraction
    return GenericEventExtraction


def _prompt(text: str, event_type: EventType) -> str:
    spec = spec_for(event_type)
    return "\n".join(
        [
            f"Tipo de evento já classificado: {event_type.value}.",
            f"Campos obrigatórios para este tipo: {', '.join(spec.required)}.",
            "",
            "Extraia os campos do aviso abaixo.",
            "",
            "--- AVISO ---",
            text,
        ]
    )


def _attach_provenance(field: ExtractedField, doc) -> ExtractedField:
    """Anchor a value to the pixels it came from.

    The match quality is kept separate from the value: grounding.py turns it
    into a status, and a field whose evidence cannot be found at all is treated
    as a hallucination rather than a low-confidence read.
    """
    if not field.evidence_text:
        return field
    block, quality = provenance.locate(field.evidence_text, doc.blocks, field.value_raw)
    if block is None:
        return field
    field.page = block.bbox.page
    field.bbox = block.bbox
    field.reader_kind = block.reader_kind
    field.reading_level = block.level
    field.corroborating_kinds = list(block.corroborating_kinds)
    field.notes.append(f"evidência localizada com qualidade {quality:.2f}")
    if block.alternatives:
        field.notes.append(
            "leituras alternativas da mesma região: "
            + "; ".join(f"{name}={text!r}" for name, text in block.alternatives)
        )
    return field


def _from_llm_field(name: str, raw: LLMField, doc) -> ExtractedField:
    absence = None
    if raw.value is None and raw.absence_reason:
        try:
            absence = AbsenceReason(raw.absence_reason.strip().lower())
        except ValueError:
            absence = AbsenceReason.NOT_FOUND
    field = ExtractedField(
        name=name,
        value_raw=raw.value,
        evidence_text=raw.evidence,
        rationale=raw.rationale,
        absence_reason=absence,
    )
    return _attach_provenance(field, doc)


def _from_rules(doc, event_type: EventType, config_dir: str | None) -> dict[str, ExtractedField]:
    extracted = rules.extract(doc, event_type, config_dir)
    out: dict[str, ExtractedField] = {}
    for name, raw in extracted.items():
        field = ExtractedField(
            name=name,
            value_raw=raw.value_raw,
            evidence_text=raw.evidence_text,
            rationale=raw.rationale,
        )
        out[name] = _attach_provenance(field, doc)

    for name in spec_for(event_type).fields:
        out.setdefault(
            name,
            ExtractedField(
                name=name,
                absence_reason=AbsenceReason.NOT_FOUND,
                rationale="nenhuma âncora de rótulo casou para este campo",
            ),
        )
    return out


def extract(
    doc,
    event_type: EventType,
    client=None,
    config_dir: str | None = None,
) -> tuple[dict[str, ExtractedField], dict[str, ExtractedField]]:
    """Return (fields, extra_fields)."""
    if client is None or not client.available():
        return _from_rules(doc, event_type, config_dir), {}

    schema = _schema_for(event_type)
    try:
        response = client.structured(_prompt(doc.raw_text, event_type), schema, system=SYSTEM)
    except LLMError:
        fields = _from_rules(doc, event_type, config_dir)
        for field in fields.values():
            field.notes.append("extração determinística: LLM indisponível nesta execução")
        return fields, {}

    parsed = response.parsed
    fields: dict[str, ExtractedField] = {}
    extra: dict[str, ExtractedField] = {}

    for name, value in parsed.model_dump().items():
        if name == "extra_fields":
            continue
        raw = getattr(parsed, name)
        if isinstance(raw, LLMField):
            fields[name] = _from_llm_field(name, raw, doc)

    for named in getattr(parsed, "extra_fields", []) or []:
        extra[named.name] = _from_llm_field(named.name, named, doc)

    for name in spec_for(event_type).fields:
        fields.setdefault(
            name,
            ExtractedField(
                name=name,
                absence_reason=AbsenceReason.NOT_FOUND,
                rationale="campo não retornado pelo modelo",
            ),
        )
    return fields, extra
