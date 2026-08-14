"""What the model is asked to return.

Deliberately small and flat. The model supplies a value and the verbatim span
it read it from — nothing else. Bounding box, grounding e quantos mecanismos
sustentam a leitura são *medidos* pelo pipeline, porque a afirmação de um modelo
sobre onde ele olhou não é evidência.

Nem sobre o quanto ele acertou: havia um `confidence` aqui, que o modelo
preenchia sobre si mesmo e que entrava na conta da confiança do campo. Um
autorrelato não é medida, e ninguém o calibrou contra acerto nenhum.

One schema per event family, which is what makes a grupamento's missing
currency a structural fact rather than a null to explain away.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class LLMField(BaseModel):
    value: str | None = Field(
        default=None,
        description="Valor exatamente como aparece no documento, sem normalizar. null se ausente.",
    )
    evidence: str | None = Field(
        default=None,
        description="Trecho VERBATIM do documento onde o valor foi lido. Nunca parafraseie.",
    )
    rationale: str | None = Field(default=None, description="Por que este trecho, em uma frase.")
    absence_reason: str | None = Field(
        default=None,
        description=(
            "Se value é null: 'stated_undefined' quando o próprio aviso diz que será definido "
            "depois; 'not_applicable' quando o campo não existe para este tipo de evento; "
            "'not_found' quando deveria estar e não está."
        ),
    )


class ClassificationOutput(BaseModel):
    event_type: str = Field(
        description="dividendo | jcp | bonificacao | grupamento | desdobramento | subscricao | outro"
    )
    subtype: str | None = Field(default=None, description="ex.: 'intercalar', 'inplit'")
    rationale: str = Field(
        description="Justifique pela SUBSTÂNCIA do ato (base legal, conta usada, efeito), "
        "e diga explicitamente se está contrariando o prior determinístico e por quê."
    )


class CashEventExtraction(BaseModel):
    """Dividendo, JCP — eventos com perna financeira."""

    issuer: LLMField
    cnpj: LLMField
    isin: LLMField
    ticker: LLMField
    share_class: LLMField
    gross_per_share: LLMField
    tax_rate: LLMField
    net_per_share: LLMField
    currency: LLMField
    approval_date: LLMField
    com_date: LLMField
    ex_date: LLMField
    payment_date: LLMField


class RatioEventExtraction(BaseModel):
    """Bonificação, grupamento, desdobramento — proporção, não valor."""

    issuer: LLMField
    cnpj: LLMField
    isin: LLMField
    ticker: LLMField
    share_class: LLMField
    ratio_from: LLMField
    ratio_to: LLMField
    ratio_text: LLMField
    approval_date: LLMField
    com_date: LLMField
    ex_date: LLMField
    trading_start: LLMField
    credit_date: LLMField
    fraction_period: LLMField
    cost_basis: LLMField


class NamedField(LLMField):
    name: str = Field(description="Nome do campo em snake_case")


class GenericEventExtraction(BaseModel):
    """Tipo fora do catálogo: extrai a base comum e preserva o resto.

    Sem este fallback, o primeiro aviso de redução de capital derruba o lote.
    """

    issuer: LLMField
    cnpj: LLMField
    isin: LLMField
    ticker: LLMField
    share_class: LLMField
    approval_date: LLMField
    com_date: LLMField
    ex_date: LLMField
    extra_fields: list[NamedField] = Field(default_factory=list)


class RecoveredFields(BaseModel):
    """A última varredura: só os campos que ninguém achou, e só os que existem.

    Um dicionário e não uma lista de campos fixos porque quais campos faltam
    muda por documento — pedir o schema inteiro convidaria o modelo a reescrever
    o que já foi extraído e conferido. O contrato é estreito de propósito: o que
    voltar fora do conjunto pedido é descartado em código, não negociado.
    """

    found: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Apenas os campos solicitados que você localizou, no formato "
            "{nome_do_campo: valor exatamente como escrito no documento}. "
            "Campo que você não encontrou NÃO entra no dicionário — não invente, "
            "não deduza e não devolva campo que não foi pedido."
        ),
    )


class VerificationOutput(BaseModel):
    """The narrative an operator reads. The verdict is computed elsewhere."""

    justification: str = Field(
        description="Resumo em PT-BR do que foi verificado e do que precisa de atenção humana. "
        "Use SOMENTE os reason codes fornecidos; não invente códigos novos."
    )
    cited_reason_codes: list[str] = Field(default_factory=list)
