"""Core schemas.

Two layers deliberately kept apart:

* ``llm/schemas.py`` — the small, flat shapes the model is asked to fill in.
* this module      — the enriched record the pipeline builds around them.

The model is never asked to supply confidence, bounding boxes or grounding:
those are measured, not claimed.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, model_validator


# --------------------------------------------------------------------------
# enums
# --------------------------------------------------------------------------


class EventType(StrEnum):
    DIVIDENDO = "dividendo"
    JCP = "jcp"
    BONIFICACAO = "bonificacao"
    GRUPAMENTO = "grupamento"
    DESDOBRAMENTO = "desdobramento"
    SUBSCRICAO = "subscricao"
    OUTRO = "outro"


#: Event families drive which schema and which coherence rules apply.
CASH_EVENTS = {EventType.DIVIDENDO, EventType.JCP}
RATIO_EVENTS = {EventType.BONIFICACAO, EventType.GRUPAMENTO, EventType.DESDOBRAMENTO}


class ReaderKind(StrEnum):
    """How a block of text was obtained. Governs retry policy and source score."""

    TEXT_LAYER = "text_layer"
    OCR_DET = "ocr_deterministic"
    OCR_VLM = "ocr_generative"


class GroundingStatus(StrEnum):
    EXACT = "exact"
    NORMALIZED = "normalized"
    FUZZY = "fuzzy"
    ABSENT = "absent"
    NOT_APPLICABLE = "not_applicable"


class AbsenceReason(StrEnum):
    """Why a field has no value. Absence is not automatically an error."""

    STATED_UNDEFINED = "stated_undefined"  # issuer wrote "a definir"
    NOT_APPLICABLE = "not_applicable"  # payment date on a grupamento
    NOT_FOUND = "not_found"  # genuinely missing


class ValidationStatus(StrEnum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    INFO = "info"  # never affects the verdict (checksums live here)


class Disposition(StrEnum):
    AUTO = "auto"
    REVIEW = "review"
    BLOCKED = "blocked"
    FAILED = "failed"  # the pipeline itself threw


class Severity(StrEnum):
    REVIEW = "review"
    BLOCK = "block"


class ReasonCode(StrEnum):
    # --- blocking ---
    DATE_INCOHERENCE = "DATE_INCOHERENCE"
    UNGROUNDED_CRITICAL_FIELD = "UNGROUNDED_CRITICAL_FIELD"
    ISSUER_INACTIVE = "ISSUER_INACTIVE"
    MISSING_REQUIRED_FIELD = "MISSING_REQUIRED_FIELD"
    # --- review ---
    # Identity findings escalate rather than block: the record is emitted
    # complete and an operator decides whether to register the issuer or reject
    # the notice. Nothing here is resolvable by reading the document again, but
    # it is resolvable by a person with access to the reference base.
    ISSUER_NOT_IN_GOLDEN = "ISSUER_NOT_IN_GOLDEN"
    IDENTIFIER_MISMATCH = "IDENTIFIER_MISMATCH"
    IDENTIFIER_KEY_CONFLICT = "IDENTIFIER_KEY_CONFLICT"
    #: Only one key corroborated. Distinct from a mismatch: nothing contradicts
    #: the base, the evidence is just thinner than the two-key rule wants.
    IDENTIFIER_WEAK_MATCH = "IDENTIFIER_WEAK_MATCH"
    EVENT_TYPE_TITLE_CONFLICT = "EVENT_TYPE_TITLE_CONFLICT"
    CLASSIFIER_DISAGREEMENT = "CLASSIFIER_DISAGREEMENT"
    UNKNOWN_EVENT_TYPE = "UNKNOWN_EVENT_TYPE"
    EXTRACTOR_DISAGREEMENT = "EXTRACTOR_DISAGREEMENT"
    #: Um único mecanismo de pixel leu a região e nada o contradisse nem o
    #: confirmou. É a pergunta *conseguimos ler isto?*, e ela não se mistura com
    #: a pergunta *este ativo é quem o aviso diz que é?*, que a base de
    #: referência responde: um ISIN lido só pelo OCR continua lido só pelo OCR,
    #: tenha ou não batido com a base. As duas verificações aparecem lado a lado
    #: no relatório, e nenhuma anula a outra.
    NO_CORROBORATION = "NO_CORROBORATION"
    AMBIGUOUS_DATE_PARSE = "AMBIGUOUS_DATE_PARSE"
    FIELD_STATED_UNDEFINED = "FIELD_STATED_UNDEFINED"
    AMOUNT_MISMATCH = "AMOUNT_MISMATCH"
    #: Nem os extratores nem a varredura final acharam o campo nas três
    #: leituras. Um campo obrigatório nessa situação já bloqueia pelo
    #: `check_required_fields`; este código é o que impede que a mesma ausência,
    #: num campo opcional, saia como se nada tivesse acontecido.
    FIELD_NOT_FOUND = "FIELD_NOT_FOUND"


class Layer(StrEnum):
    """Which stage of the pipeline produced a finding.

    An operator's first question about a flag is *who is telling me this* —
    a disagreement between readers and a failed chronology check are both
    "problems", but one is answered by looking at pixels and the other by
    calling the issuer.

    Havia uma quinta camada, `score`, e ela era só derivada: hospedava os
    códigos de limiar, que eram a consequência aritmética do que outra camada já
    tinha reportado. Uma consequência apresentada como achado é o que faz o
    operador procurar a causa no lugar errado — as quatro que restam são todas
    primárias.
    """

    CONSENSO = "consenso"
    CLASSIFICACAO = "classificacao"
    EXTRACAO = "extracao"
    VALIDACAO = "validacao"


class ErrorType(StrEnum):
    """What kind of problem it is, independent of which layer found it."""

    DIVERGENCIA = "divergencia"
    INCOERENCIA = "incoerencia"
    IDENTIDADE = "identidade"
    AUSENCIA = "ausencia"
    LEGIBILIDADE = "legibilidade"
    FABRICACAO = "fabricacao"
    PROCESSO = "processo"


#: Severity is a property of the reason, not of the document it came from.
REASON_SEVERITY: dict[ReasonCode, Severity] = {
    ReasonCode.DATE_INCOHERENCE: Severity.BLOCK,
    ReasonCode.UNGROUNDED_CRITICAL_FIELD: Severity.BLOCK,
    ReasonCode.ISSUER_INACTIVE: Severity.BLOCK,
    ReasonCode.MISSING_REQUIRED_FIELD: Severity.BLOCK,
    ReasonCode.ISSUER_NOT_IN_GOLDEN: Severity.REVIEW,
    ReasonCode.IDENTIFIER_MISMATCH: Severity.REVIEW,
    ReasonCode.IDENTIFIER_KEY_CONFLICT: Severity.REVIEW,
    ReasonCode.IDENTIFIER_WEAK_MATCH: Severity.REVIEW,
    ReasonCode.EVENT_TYPE_TITLE_CONFLICT: Severity.REVIEW,
    ReasonCode.CLASSIFIER_DISAGREEMENT: Severity.REVIEW,
    ReasonCode.UNKNOWN_EVENT_TYPE: Severity.REVIEW,
    ReasonCode.EXTRACTOR_DISAGREEMENT: Severity.REVIEW,
    ReasonCode.NO_CORROBORATION: Severity.REVIEW,
    ReasonCode.AMBIGUOUS_DATE_PARSE: Severity.REVIEW,
    ReasonCode.FIELD_STATED_UNDEFINED: Severity.REVIEW,
    ReasonCode.AMOUNT_MISMATCH: Severity.REVIEW,
    ReasonCode.FIELD_NOT_FOUND: Severity.REVIEW,
}

#: Origin is likewise a property of the code. Kept beside the severity table so
#: a new reason code cannot be added without deciding all three facts about it —
#: ``tests/test_audit.py`` asserts this covers every member of ``ReasonCode``.
REASON_ORIGIN: dict[ReasonCode, tuple[Layer, ErrorType]] = {
    ReasonCode.NO_CORROBORATION: (Layer.CONSENSO, ErrorType.AUSENCIA),
    ReasonCode.UNKNOWN_EVENT_TYPE: (Layer.CLASSIFICACAO, ErrorType.AUSENCIA),
    ReasonCode.CLASSIFIER_DISAGREEMENT: (Layer.CLASSIFICACAO, ErrorType.DIVERGENCIA),
    ReasonCode.EVENT_TYPE_TITLE_CONFLICT: (Layer.CLASSIFICACAO, ErrorType.DIVERGENCIA),
    ReasonCode.EXTRACTOR_DISAGREEMENT: (Layer.EXTRACAO, ErrorType.DIVERGENCIA),
    ReasonCode.UNGROUNDED_CRITICAL_FIELD: (Layer.EXTRACAO, ErrorType.FABRICACAO),
    ReasonCode.AMBIGUOUS_DATE_PARSE: (Layer.EXTRACAO, ErrorType.LEGIBILIDADE),
    ReasonCode.FIELD_STATED_UNDEFINED: (Layer.EXTRACAO, ErrorType.AUSENCIA),
    ReasonCode.FIELD_NOT_FOUND: (Layer.EXTRACAO, ErrorType.AUSENCIA),
    ReasonCode.DATE_INCOHERENCE: (Layer.VALIDACAO, ErrorType.INCOERENCIA),
    ReasonCode.AMOUNT_MISMATCH: (Layer.VALIDACAO, ErrorType.INCOERENCIA),
    ReasonCode.ISSUER_NOT_IN_GOLDEN: (Layer.VALIDACAO, ErrorType.IDENTIDADE),
    ReasonCode.IDENTIFIER_MISMATCH: (Layer.VALIDACAO, ErrorType.IDENTIDADE),
    ReasonCode.IDENTIFIER_KEY_CONFLICT: (Layer.VALIDACAO, ErrorType.IDENTIDADE),
    ReasonCode.IDENTIFIER_WEAK_MATCH: (Layer.VALIDACAO, ErrorType.IDENTIDADE),
    ReasonCode.ISSUER_INACTIVE: (Layer.VALIDACAO, ErrorType.IDENTIDADE),
    ReasonCode.MISSING_REQUIRED_FIELD: (Layer.VALIDACAO, ErrorType.AUSENCIA),
}

#: Report ordering: cause before consequence.
LAYER_ORDER: dict[Layer, int] = {
    Layer.CONSENSO: 0,
    Layer.CLASSIFICACAO: 1,
    Layer.EXTRACAO: 2,
    Layer.VALIDACAO: 3,
}


# --------------------------------------------------------------------------
# geometry and provenance
# --------------------------------------------------------------------------


class BBox(BaseModel):
    """Where on the page a value was read, in PDF points plus 0-1 normalized."""

    page: int
    x0: float
    y0: float
    x1: float
    y1: float
    page_width: float
    page_height: float

    @property
    def normalized(self) -> tuple[float, float, float, float]:
        w, h = self.page_width or 1.0, self.page_height or 1.0
        return (self.x0 / w, self.y0 / h, self.x1 / w, self.y1 / h)

    def union(self, other: BBox) -> BBox:
        return self.model_copy(
            update={
                "x0": min(self.x0, other.x0),
                "y0": min(self.y0, other.y0),
                "x1": max(self.x1, other.x1),
                "y1": max(self.y1, other.y1),
            }
        )


class Grounding(BaseModel):
    """Did this value actually appear in the source, and where?

    Deliberately separate from ``confidence``: a score answers "is the value
    right", grounding answers "does it exist at the claimed location". A
    reviewer needs both, and collapsing them into one number loses the second.
    """

    status: GroundingStatus = GroundingStatus.ABSENT
    score: float = 0.0
    matched_text: str | None = None
    bbox: BBox | None = None

    @property
    def factor(self) -> float:
        return {
            GroundingStatus.EXACT: 1.0,
            GroundingStatus.NORMALIZED: 0.85,
            GroundingStatus.FUZZY: 0.5,
            GroundingStatus.ABSENT: 0.0,
            GroundingStatus.NOT_APPLICABLE: 1.0,
        }[self.status]


class Repair(BaseModel):
    """One normalization step, recorded so it can be audited and reversed."""

    field: str
    rule: str
    value_from: str | None = None
    value_to: str | None = None
    candidates: list[str] = Field(default_factory=list)


class Reading(BaseModel):
    """Quem leu esta região da página, e o que isso vale.

    Houve aqui um `Uncertainty` com três floats — leitura, modelo e validação —
    e a regra `confiança = min(leitura, modelo) × validação`. Os três saíram.
    O da leitura era o único medido, e mesmo ele não era medida: `source_score`
    era uma consulta a uma tabela de quatro pesos indexada por *quem leu*, de
    modo que o número não carregava um grão de informação a mais que o degrau —
    só a escondia atrás de uma precisão de dois dígitos que a evidência não tem.

    O que sobra é o fato: os mecanismos que leram e concordaram, e a ferramenta
    que porventura reprovou o campo. Nada aqui se multiplica com nada.
    """

    #: muito_alta | alta | media | baixa — ver ``consensus.reading_level``.
    level: str = "baixa"
    #: Os mecanismos que leram a região *e concordaram* no valor.
    corroborating_kinds: list[ReaderKind] = Field(default_factory=list)
    #: Ferramentas de validação que reprovaram este campo. Não vira fator: o
    #: achado da ferramenta já é reportado com o código dela.
    cut_by: list[str] = Field(default_factory=list)


class GoldenCheck(BaseModel):
    """How this field fared against the reference base.

    A first-class fact because the enunciado asks for it by name. It previously
    existed only as the loose string "valor confirmado na base de referência"
    buried among the notes, which also left "did not match" and "is not an
    identity key at all" indistinguishable.
    """

    status: str = "nao_aplicavel"  # confere | diverge | ausente | nao_aplicavel
    expected: str | None = None


class FieldAudit(BaseModel):
    """As duas verificações independentes, uma ao lado da outra.

    `reading` responde *conseguimos ler isto?* e `golden` responde *este ativo é
    quem o aviso diz que é?*. São perguntas diferentes, respondidas por fontes
    diferentes, e é por isso que não se resolvem uma na outra: um ISIN lido só
    pelo OCR continua lido só pelo OCR depois de bater com a base, e um ISIN
    lido por três mecanismos continua ausente da base se a base não o tiver.
    """

    field_class: str = "default"
    critical: bool = False
    reading: Reading = Field(default_factory=Reading)
    golden: GoldenCheck = Field(default_factory=GoldenCheck)


class ExtractedField(BaseModel):
    name: str
    value_raw: str | None = None
    value: Any | None = None  # normalized by repair.py
    absence_reason: AbsenceReason | None = None

    # provenance
    evidence_text: str | None = None
    rationale: str | None = None
    page: int | None = None
    bbox: BBox | None = None
    snippet_path: str | None = None
    reader_kind: ReaderKind | None = None
    #: Os mecanismos que leram esta região e concordaram. É a resposta inteira
    #: para "quantas leituras independentes sustentam este valor" — a pergunta
    #: que o score colapsava num float onde um OCR sozinho e um valor
    #: corroborado saíam indistinguíveis.
    corroborating_kinds: list[ReaderKind] = Field(default_factory=list)
    #: O degrau da leitura — muito_alta | alta | media | baixa. Categórico de
    #: propósito: a diferença entre dois degraus não é percentual, é "a camada de
    #: texto confirmou" contra "só os pixels leram".
    reading_level: str = "baixa"
    #: Veio da varredura final (`extraction/sweep.py`), não de um extrator: o
    #: modelo releu as três leituras atrás dos campos que sobraram. O valor
    #: continua tendo de existir em alguma delas — o degrau acima sai de quantas
    #: o continham —, mas a origem muda o que o relatório mostra na coluna do
    #: recorte, porque não houve região casada, houve releitura.
    recovered: bool = False

    grounding: Grounding = Field(default_factory=Grounding)

    repairs: list[Repair] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    #: Derived at triage time by ``reporter.audit``. Optional so records written
    #: before it existed still load.
    audit: FieldAudit | None = None

    @property
    def is_present(self) -> bool:
        return self.value is not None or self.value_raw is not None


# --------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------


class MarkerHit(BaseModel):
    event_type: EventType
    marker: str
    weight: float
    definitional: bool
    matched_text: str


class ClassificationPrior(BaseModel):
    """Deterministic opinion, formed before any model call."""

    ranked: list[tuple[EventType, float]] = Field(default_factory=list)
    hits: list[MarkerHit] = Field(default_factory=list)
    title_prior: EventType | None = None
    body_prior: EventType | None = None

    @property
    def top(self) -> EventType | None:
        return self.ranked[0][0] if self.ranked else None

    @property
    def title_body_conflict(self) -> bool:
        return (
            self.title_prior is not None
            and self.body_prior is not None
            and self.title_prior != self.body_prior
        )


class Classification(BaseModel):
    """The two opinions and whether they agree.

    Sem número: o que decide é se as modalidades foram unânimes e se o resultado
    bate com o prior determinístico. `min(0.5 + margem/10, 0.95)` e
    `min(média, 0.55)` eram aritmética inventada sobre isso — davam um decimal a
    dois fatos binários que já estavam registrados aqui ao lado.
    """

    event_type: EventType
    subtype: str | None = None
    rationale: str = ""
    prior: ClassificationPrior = Field(default_factory=ClassificationPrior)
    agrees_with_prior: bool = True
    #: Todas as modalidades disponíveis votaram no mesmo tipo.
    unanimous: bool = True


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


class ValidationResult(BaseModel):
    """One tool call, recorded verbatim: name, arguments, and what came back.

    The verdict is computed from these, never from the model's prose.
    """

    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    status: ValidationStatus = ValidationStatus.INFO
    detail: dict[str, Any] = Field(default_factory=dict)
    reason_codes: list[ReasonCode] = Field(default_factory=list)
    fields_affected: list[str] = Field(default_factory=list)
    message: str = ""


class TriageReason(BaseModel):
    code: ReasonCode
    severity: Severity
    field: str | None = None
    message: str = ""
    layer: Layer | None = None
    error_type: ErrorType | None = None

    @model_validator(mode="after")
    def _origin_follows_the_code(self) -> TriageReason:
        """Origin is a property of the code, so never store a contradiction.

        A plain default would be worse than none: a record written before these
        fields existed would deserialize claiming to be whatever the default
        said, and a reason mislabelled as a validation finding sends a reviewer
        to the wrong place with full confidence.
        """
        layer, error_type = REASON_ORIGIN[self.code]
        if self.layer is None:
            self.layer = layer
        if self.error_type is None:
            self.error_type = error_type
        return self

    @classmethod
    def of(cls, code: ReasonCode, message: str = "", field: str | None = None) -> TriageReason:
        """The only way to build one: severity follows from the code too.

        There were two hand-written construction sites passing ``severity``
        explicitly — one place to look instead of two chances to drift.
        """
        return cls(
            code=code,
            severity=REASON_SEVERITY.get(code, Severity.REVIEW),
            field=field,
            message=message,
        )


class Triage(BaseModel):
    disposition: Disposition = Disposition.AUTO
    reasons: list[TriageReason] = Field(default_factory=list)
    justification: str = ""  # written from the codes, never inventing new ones
    #: O degrau de leitura mais fraco entre os campos críticos com valor. Era um
    #: `overall_confidence` — o mínimo de uma coluna de floats, que reunia num
    #: número só o campo que ninguém conseguiu ler e o campo que a cronologia
    #: contradisse, dois problemas sem nada em comum para quem vai resolvê-los.
    weakest_level: str = "baixa"
    fields_for_review: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------
# the record
# --------------------------------------------------------------------------


class DocumentInfo(BaseModel):
    file_name: str
    n_pages: int
    producer: str | None = None
    reader_kinds: list[ReaderKind] = Field(default_factory=list)
    escalated: bool = False
    escalation_note: str | None = None


class EventRecord(BaseModel):
    """One corporate action notice, structured, scored and traceable."""

    document: DocumentInfo
    event_type: EventType
    classification: Classification
    fields: dict[str, ExtractedField] = Field(default_factory=dict)
    validations: list[ValidationResult] = Field(default_factory=list)
    triage: Triage = Field(default_factory=Triage)
    extra_fields: dict[str, ExtractedField] = Field(default_factory=dict)
    error: str | None = None

    def field(self, name: str) -> ExtractedField | None:
        return self.fields.get(name)


# --------------------------------------------------------------------------
# per-type field contracts
# --------------------------------------------------------------------------

class FieldSpec(BaseModel):
    """O contrato de um tipo de evento: o que ele exige e o que ele admite.

    Fonte única, usada pela extração e pelo ``check_required_fields`` — uma lista
    só evita a deriva clássica em que o prompt pede um campo que o validador
    nunca confere.

    ``required`` e ``optional`` são enumerados **por tipo**, sem lista comum
    herdada. Herdar era o que produzia o defeito da moeda: `currency` valia para
    todo evento porque estava na base, e um grupamento — consolidação de ações,
    sem dinheiro trocando de mãos — bloqueava por não trazer moeda que o aviso
    não tinha por que ter. Repetir alguns nomes entre os tipos é barato; decidir
    campo a campo, por evento, é o que impede a regra de um vazar para outro.

    ``optional`` não é decoração: é o que se espera encontrar e cuja ausência não
    invalida o registro. Ele aparece nas tabelas do relatório como qualquer
    outro, para que o operador veja o que faltou sem que isso trave a liberação.
    """

    required: list[str]
    optional: list[str] = Field(default_factory=list)
    criticality: dict[str, str] = Field(default_factory=dict)

    @property
    def fields(self) -> list[str]:
        return [*self.required, *self.optional]


#: Identidade: quem é o ativo e que evento é este. Vale para qualquer tipo, e é
#: o que o enunciado cobra de todo registro — "emissor, ISIN, ticker, tipo de
#: evento". Repetido explicitamente em cada spec, não herdado.
_ID_REQ = ["issuer", "isin", "ticker", "event_type_label"]
_ID_OPT = ["cnpj", "share_class"]

FIELD_SPECS: dict[EventType, FieldSpec] = {
    # --- eventos de caixa: há dinheiro, logo há moeda ---------------------
    EventType.DIVIDENDO: FieldSpec(
        required=_ID_REQ + ["currency", "gross_per_share", "com_date", "ex_date"],
        # payment_date é opcional porque o emissor pode declará-la "a definir",
        # e isso é um aviso válido — vira FIELD_STATED_UNDEFINED, não ausência
        optional=_ID_OPT + ["approval_date", "payment_date", "tax_rate", "net_per_share"],
    ),
    EventType.JCP: FieldSpec(
        required=_ID_REQ + ["currency", "gross_per_share", "com_date", "ex_date"],
        optional=_ID_OPT + ["approval_date", "payment_date", "tax_rate", "net_per_share"],
    ),
    # --- eventos de proporção: a relação é o valor ------------------------
    EventType.BONIFICACAO: FieldSpec(
        # moeda é obrigatória aqui porque o custo unitário atribuído vem em R$ e
        # tem efeito fiscal
        required=_ID_REQ + ["ratio_from", "ratio_to", "com_date", "ex_date", "currency"],
        optional=_ID_OPT + ["approval_date", "ratio_text", "cost_basis",
                            "credit_date", "trading_start", "fraction_period"],
    ),
    EventType.GRUPAMENTO: FieldSpec(
        # sem moeda: consolidação de ações não move dinheiro, e exigi-la
        # bloquearia um registro correto por um campo que o aviso não tem
        required=_ID_REQ + ["ratio_from", "ratio_to", "com_date"],
        optional=_ID_OPT + ["approval_date", "ex_date", "ratio_text",
                            "trading_start", "fraction_period", "credit_date"],
    ),
    EventType.DESDOBRAMENTO: FieldSpec(
        required=_ID_REQ + ["ratio_from", "ratio_to", "com_date"],
        optional=_ID_OPT + ["approval_date", "ex_date", "ratio_text",
                            "trading_start", "fraction_period", "credit_date"],
    ),
    EventType.SUBSCRICAO: FieldSpec(
        required=_ID_REQ,
        optional=_ID_OPT + ["approval_date", "com_date", "ex_date", "currency"],
    ),
    # Tipo fora do catálogo: só a identidade é cobrável, porque não se sabe que
    # regra aplicar ao resto.
    EventType.OUTRO: FieldSpec(
        required=["issuer"],
        optional=_ID_OPT + ["isin", "ticker", "event_type_label",
                            "approval_date", "com_date", "ex_date", "currency"],
    ),
}


def spec_for(event_type: EventType) -> FieldSpec:
    return FIELD_SPECS.get(event_type, FIELD_SPECS[EventType.OUTRO])
