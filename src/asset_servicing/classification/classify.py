"""Stage 2 — two independent opinions on the event type.

The regex prior forms one before any model runs; the LLM forms the other and is
shown the prior as evidence. Neither rules alone: agreement is what releases
the record, disagreement becomes a reason code and goes to a human.

This is what catches a notice *titled* one thing that *is* another. The title
reflects how an IR team labelled the file; the body reflects the act. When they
diverge, the tax treatment follows the act — and getting that backwards is a
tax error, not a typo.
"""

from __future__ import annotations

from . import markers
from ..llm.base import LLMError
from ..llm.schemas import ClassificationOutput
from ..models import Classification, ClassificationPrior, EventType

SYSTEM = """Você é um analista de Asset Servicing classificando avisos de eventos \
corporativos de companhias abertas brasileiras (padrão B3/CVM).

REGRA CENTRAL: classifique pela SUBSTÂNCIA do ato societário, nunca pelo título \
do documento. Um aviso intitulado "Distribuição de Dividendos" que remunera o \
capital próprio com base no patrimônio líquido, limitado pela TJLP e imputado ao \
dividendo obrigatório, é JCP (Lei 9.249/95, art. 9º) — o título está errado e a \
classificação correta é a que a lei determina, porque o tratamento tributário \
segue o ato, não o rótulo.

Distinções que importam:
- jcp: remuneração sobre patrimônio líquido, base legal Lei 9.249/95 art. 9º, \
limite pela TJLP, imputável ao dividendo obrigatório, retenção na fonte.
- dividendo: distribuição de lucro ou de reservas de lucros.
- bonificacao: capitalização de reservas com emissão de novas ações aos acionistas.
- grupamento: reduz a quantidade de ações sem alterar o capital social (inplit).
- desdobramento: aumenta a quantidade de ações sem alterar o capital social (split).
- subscricao: direito ou bônus de subscrição.
- outro: qualquer ato fora do catálogo acima.

NÃO use alíquotas como critério definitório: elas mudam com a legislação."""


def _prompt(text: str, prior: ClassificationPrior) -> str:
    lines = [
        "Classifique o evento corporativo do aviso abaixo.",
        "",
        "OPINIÃO DETERMINÍSTICA PRÉVIA (marcadores casados por regex, apenas como evidência):",
        f"  ranking: {[(t.value, s) for t, s in prior.ranked[:4]] or 'nenhum marcador casou'}",
        f"  prior do cabeçalho: {prior.title_prior.value if prior.title_prior else None}",
        f"  prior do corpo:     {prior.body_prior.value if prior.body_prior else None}",
    ]
    if prior.title_body_conflict:
        lines.append(
            "  ATENÇÃO: cabeçalho e corpo divergem. Decida pela substância descrita no corpo."
        )
    definitional = [h for h in prior.hits if h.definitional][:12]
    if definitional:
        lines.append("  marcadores definitórios encontrados:")
        for hit in definitional:
            lines.append(f"    - [{hit.event_type.value}] {hit.matched_text!r} (peso {hit.weight})")
    lines += [
        "",
        "Você pode contrariar o prior, mas se contrariar deve dizer por quê.",
        "",
        "--- AVISO ---",
        text,
    ]
    return "\n".join(lines)


def _from_prior(prior: ClassificationPrior) -> Classification:
    """No model available: the deterministic opinion stands on its own.

    Havia aqui um `min(0.5 + margem/10, 0.9)` traduzindo a distância para o
    segundo colocado em confiança. Não havia nada por trás da divisão por dez, e
    o que a triagem faz com o resultado é binário: o prior sozinho não é
    unanimidade entre modalidades, e isso já está dito em `unanimous`.
    """
    event_type = prior.top or EventType.OUTRO
    return Classification(
        event_type=event_type,
        # uma modalidade só não é unanimidade entre modalidades
        unanimous=False,
        rationale=(
            "Classificação determinística por marcadores definitórios "
            f"({', '.join(h.matched_text for h in prior.hits if h.definitional)[:200]})."
            if prior.hits
            else "Nenhum marcador definitório casou."
        ),
        prior=prior,
        agrees_with_prior=True,
    )


def classify(doc, client=None, config_dir: str | None = None) -> Classification:
    prior = markers.pre_classify(doc.raw_text, config_dir)

    if client is None or not client.available():
        return _from_prior(prior)

    try:
        response = client.structured(
            _prompt(doc.raw_text, prior), ClassificationOutput, system=SYSTEM
        )
        output: ClassificationOutput = response.parsed
        try:
            event_type = EventType(output.event_type.strip().lower())
        except ValueError:
            event_type = EventType.OUTRO
    except LLMError:
        # Falling back to the prior is strictly better than failing the
        # document: we still have a defensible classification and the record
        # records that it came from rules alone.
        result = _from_prior(prior)
        result.rationale += " (LLM indisponível nesta execução; prior determinístico usado.)"
        return result

    # Leitores independentes discordando é informação sobre o documento, e
    # nenhum dos dois é confiável o bastante para liberar sozinho.
    agrees = prior.top is None or prior.top == event_type

    return Classification(
        event_type=event_type,
        subtype=output.subtype,
        unanimous=agrees,
        rationale=output.rationale,
        prior=prior,
        agrees_with_prior=agrees,
    )
