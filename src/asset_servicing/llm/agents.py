"""Pydantic AI agents — the typed nodes of the graph.

Division of labour with LangGraph: the graph owns state and edges, Pydantic AI
owns what happens *inside* a node that talks to a model — output typed against
a schema, tools declared from signatures, retry on validation failure. They do
not overlap, which is what makes running both defensible rather than two
frameworks doing one job.

Provider-agnostic by construction: the model is a string
(``anthropic:claude-sonnet-5``, ``google-gla:gemini-2.5-flash``), so swapping
providers touches configuration and nothing else.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from pydantic_ai import Agent, ModelRetry

from . import trace
from .schemas import (
    CashEventExtraction,
    ClassificationOutput,
    GenericEventExtraction,
    RatioEventExtraction,
    RecoveredFields,
    VerificationOutput,
)
from ..models import CASH_EVENTS, RATIO_EVENTS, EventType

DEFAULT_MODEL = os.environ.get("AS_MODEL", "anthropic:claude-sonnet-5")

#: Which environment variable makes each provider prefix usable.
_PROVIDER_KEYS = {
    "anthropic": ("ANTHROPIC_API_KEY", "CLAUDE_API_KEY"),
    "google-gla": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    "google-vertex": ("GOOGLE_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
}


def model_available(model: str = DEFAULT_MODEL) -> bool:
    """True when the configured provider has a usable key.

    Checked up front so the graph can route to the deterministic path instead
    of discovering the problem one exception at a time, per document.
    """
    prefix = model.split(":", 1)[0]
    for variable in _PROVIDER_KEYS.get(prefix, ()):
        if os.environ.get(variable):
            return True
    return False


def load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader — no extra dependency for one file of key=value.

    Never overwrites a variable already set in the environment.
    """
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip().strip("'\""))
    except OSError:
        pass

    # The SDK reads ANTHROPIC_API_KEY; CLAUDE_API_KEY is a common local name
    # for the same secret. Aliasing avoids a key that is present but unused.
    if os.environ.get("CLAUDE_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY"):
        os.environ["ANTHROPIC_API_KEY"] = os.environ["CLAUDE_API_KEY"]


CLASSIFIER_INSTRUCTIONS = """Você classifica avisos de eventos corporativos de companhias \
abertas brasileiras (padrão B3/CVM).

REGRA CENTRAL: classifique pela SUBSTÂNCIA do ato societário, nunca pelo título do \
documento. Um aviso intitulado "Distribuição de Dividendos" que remunera o capital \
próprio com base no patrimônio líquido, limitado pela TJLP e imputado ao dividendo \
obrigatório, é JCP (Lei 9.249/95, art. 9º): o tratamento tributário segue o ato, não o \
rótulo.

Catálogo:
- jcp: remuneração sobre patrimônio líquido, base legal Lei 9.249/95 art. 9º, limite pela \
TJLP, imputável ao dividendo obrigatório.
- dividendo: distribuição de lucro ou de reservas de lucros.
- bonificacao: capitalização de reservas com emissão de novas ações.
- grupamento: reduz a quantidade de ações sem alterar o capital social (inplit).
- desdobramento: aumenta a quantidade de ações sem alterar o capital social (split).
- subscricao: direito ou bônus de subscrição.
- outro: qualquer ato fora do catálogo.

NÃO use alíquotas como critério definitório — elas mudam com a legislação."""

EXTRACTOR_INSTRUCTIONS = """Você extrai campos de avisos de eventos corporativos \
brasileiros para uma base de custódia. Regras inegociáveis:

1. NUNCA invente um valor. Ausente é value=null com o absence_reason correto.
2. `evidence` deve ser trecho VERBATIM do documento — copiado, não parafraseado. Ele é \
conferido automaticamente contra o texto original e um trecho inexistente derruba o campo.
3. `value` vem como está escrito ("R$ 0,4275000000", "15/06/2026"). Normalização é feita \
depois, em código.
4. Distinga ausências: 'stated_undefined' (o aviso diz que será divulgado depois), \
'not_applicable' (não faz sentido para o tipo), 'not_found' (deveria constar e não consta).
5. Proporção (`ratio_from`/`ratio_to`): o sentido é do EVENTO, não da ordem da frase. \
Grupamento consolida — a contagem de ações cai, vai do maior para o menor (10:1). \
Bonificação e desdobramento aumentam a contagem, vão do menor para o maior (1 nova para \
cada 20 → 1 e 20). Devolva os dois números; a ordem é conferida em código."""

REPORTER_INSTRUCTIONS = """Você é um operador sênior de Asset Servicing redigindo a \
justificativa de um registro extraído de um aviso de evento corporativo.

Escreva em português, objetivo e curto (no máximo 6 linhas): o que foi confirmado, o que \
ficou pendente e por quê, e o que um humano precisa decidir.

RESTRIÇÃO: use SOMENTE os reason codes fornecidos. Não invente códigos, não invente \
valores e não contradiga o resultado das validações — elas já foram executadas e são a \
verdade do registro."""


@dataclass
class ReporterDeps:
    """O que o registro de fato contém, para o validador conferir a narrativa.

    `allowed_codes` são os reason codes que a triagem atribuiu; `allowed_figures`
    são as datas e os montantes que os campos realmente têm. O relator não pode
    alterar dado — ele devolve só uma string —, mas pode *descrever* dado que
    não existe, e um resumo que cita 15/08 onde o registro diz 21/08 é falso do
    mesmo jeito que um valor adulterado seria.
    """

    allowed_codes: set[str]
    allowed_figures: set[str] = field(default_factory=set)


#: Datas e montantes são o que a narrativa erra de forma consequente. Contagens
#: ("3 campos") e percentuais em prosa não entram: o custo de um falso positivo
#: aqui é uma retentativa e depois o texto determinístico.
_FIGURE = re.compile(r"\b\d{1,2}/\d{1,2}/\d{4}\b|R\$\s*[\d.]+,\d+")


def _canon_figure(text: str) -> str:
    """Só os dígitos: `R$ 0,4275` e `0,4275000000` são a mesma afirmação, e a
    grafia do relator não deveria decidir se ele passa."""
    return re.sub(r"\D", "", text).rstrip("0") or "0"


def schema_for(event_type: EventType):
    if event_type in CASH_EVENTS:
        return CashEventExtraction
    if event_type in RATIO_EVENTS:
        return RatioEventExtraction
    return GenericEventExtraction


def build_text_classifier(model: str = DEFAULT_MODEL) -> Agent:
    return trace.wrap(Agent(
        model,
        output_type=ClassificationOutput,
        instructions=CLASSIFIER_INSTRUCTIONS,
        retries=2,
    ), "text_classifier")


def build_extractor(event_type: EventType, model: str = DEFAULT_MODEL) -> Agent:
    return trace.wrap(Agent(
        model,
        output_type=schema_for(event_type),
        instructions=EXTRACTOR_INSTRUCTIONS,
        retries=2,
    ), f"extractor:{event_type}")


SWEEPER_INSTRUCTIONS = """Você faz a última passada sobre um aviso de evento corporativo \
brasileiro cujos extratores já rodaram. Recebe as leituras do MESMO documento por três \
mecanismos independentes — camada de texto, OCR e VLM — e uma lista de campos que nenhum \
deles preencheu.

Sua única tarefa é procurar esses campos, e só esses, nas três leituras.

Regras inegociáveis:

1. Devolva APENAS campos da lista pedida. Qualquer outra chave é descartada.
2. Devolva APENAS o que você localizou de fato no texto de alguma das três leituras. \
O valor devolvido é conferido, literalmente, contra as três — o que não estiver em \
nenhuma delas é tratado como fabricação e jogado fora.
3. Copie o valor como ele aparece ("R$ 0,4275000000", "15/06/2026", "17,5%"). Não \
normalize, não converta, não complete.
4. Campo que você não achou simplesmente não entra no dicionário. Um dicionário vazio é \
uma resposta correta e comum: os três mecanismos podem ter lido o documento inteiro sem \
que o dado esteja lá.
5. As leituras divergem entre si — é por isso que são três. Se uma trouxer o campo e as \
outras não, devolva o que a que trouxe diz.

Pode haver uma segunda lista, (B): campos que já foram extraídos e que um único mecanismo \
sustenta. Ali a pergunta é outra — não é "qual é o valor", é "este valor aparece em mais \
alguma leitura?". Devolva o campo apenas se encontrar o MESMO valor em outra leitura, \
mesmo com ruído de OCR. Não corrija o valor: se o que você lê é outro dado, não devolva o \
campo."""


def build_field_sweeper(model: str = DEFAULT_MODEL) -> Agent:
    """A varredura final: os campos que sobraram, contra as três leituras."""
    return trace.wrap(Agent(
        model,
        output_type=RecoveredFields,
        instructions=SWEEPER_INSTRUCTIONS,
        retries=2,
    ), "field_sweeper")


def build_reporter(model: str = DEFAULT_MODEL) -> Agent:
    agent = Agent(
        model,
        output_type=VerificationOutput,
        deps_type=ReporterDeps,
        instructions=REPORTER_INSTRUCTIONS,
        retries=2,
    )

    @agent.output_validator
    def only_grounded_figures(ctx, output: VerificationOutput) -> VerificationOutput:
        """Recusa uma narrativa que cita número que o registro não tem.

        O relator não consegue adulterar dado — o nó devolve só `justification`
        e a disposição já foi calculada em código antes dele rodar. O que ele
        consegue é contar outra história sobre o mesmo dado, e é isso que o
        operador lê. Uma data inventada no resumo custa a mesma confiança que
        um campo errado custaria.
        """
        if not ctx.deps.allowed_figures:
            return output
        known = {_canon_figure(v) for v in ctx.deps.allowed_figures}
        invented = [
            f for f in _FIGURE.findall(output.justification)
            if _canon_figure(f) not in known
        ]
        if invented:
            raise ModelRetry(
                f"O resumo cita {invented}, que não está em nenhum campo do registro. "
                "Descreva apenas valores presentes, ou não cite número nenhum."
            )
        return output

    @agent.output_validator
    def only_assigned_codes(ctx, output: VerificationOutput) -> VerificationOutput:
        """Reject a justification that invents grounds.

        This is the reason the reporter is a Pydantic AI agent rather than a
        plain call: an audit trail whose narrative can cite reasons nobody
        found is worse than no narrative at all. A model that strays gets one
        retry with the error, then the deterministic fallback is used.
        """
        invented = [c for c in output.cited_reason_codes if c not in ctx.deps.allowed_codes]
        if invented:
            raise ModelRetry(
                f"Os códigos {invented} não foram atribuídos a este registro. "
                f"Use apenas: {sorted(ctx.deps.allowed_codes) or 'nenhum'}."
            )
        return output

    # Embrulhado só depois do validador: o decorator precisa do Agent real, e
    # assim a retentativa que o ModelRetry provoca também entra no registro.
    return trace.wrap(agent, "reporter")
