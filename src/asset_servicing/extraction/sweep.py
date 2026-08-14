"""Stage 5b — a última varredura, sobre as três leituras inteiras.

O que motiva o passo: um campo pode faltar no registro sem faltar no documento.
As três leituras discordam por construção, e a votação resolve **região a
região** ficando com uma versão de cada uma; o que perdeu vai para
`alternatives`, espalhado bloco a bloco. Depois, o extrator de regras precisa de
um rótulo reconhecível ao lado do valor, e o extrator do modelo lê o texto que
saiu da votação. Um dado que só o VLM leu, ou que ficou numa célula sem
geometria, atravessa os dois sem ser visto.

Então antes de a triagem contar as ausências, os campos que ninguém preencheu
voltam ao modelo — desta vez com as três leituras lado a lado, cada uma
identificada, e uma lista curta do que procurar.

Três guardas, todas em código, porque a resposta vem de um modelo:

1. **Só os campos pedidos.** Chave fora do conjunto ausente é descartada. A
   varredura preenche buraco; ela nunca reescreve o que já foi extraído,
   validado e conferido contra a base.
2. **Só o que está em alguma leitura.** O valor devolvido é procurado, literal e
   normalizado, nos textos do parser, do OCR e do VLM. Não estando em nenhum, é
   fabricação — descartada, e o campo continua ausente.
3. **O degrau sai de quem o continha.** Não é um degrau fixo: se o valor aparece
   no parser e no OCR, a leitura vale `muito_alta`; se só o VLM o tem, vale
   `baixa`, e a triagem levanta `NO_CORROBORATION` sozinha. É a mesma medida do
   resto do pipeline — quantos mecanismos independentes sustentam o valor —, e
   ela não muda de significado por o campo ter chegado aqui.

O que o modelo faz aqui é *procurar*, não decidir: qualquer afirmação dele sobre
quão confiável é o que achou seria descartada de todo modo.
"""

from __future__ import annotations

from ..models import AbsenceReason, ExtractedField, ReaderKind, spec_for
from ..text import _squash, normalize
from ..validation.repair import parse_ptbr_number
from .consensus import PIXEL_DERIVED as _PIXEL, reading_level

#: Ordem em que as leituras entram no prompt: da mais confiável para a menos,
#: e nomeadas, porque "o VLM leu isto e o OCR aquilo" é informação que o modelo
#: precisa para escolher — e que some se as três forem concatenadas.
_ORDEM = (ReaderKind.TEXT_LAYER, ReaderKind.OCR_DET, ReaderKind.OCR_VLM)

_ROTULO = {
    ReaderKind.TEXT_LAYER: "CAMADA DE TEXTO (a string que o programa gerador escreveu)",
    ReaderKind.OCR_DET: "OCR (PP-OCR sobre os pixels da página)",
    ReaderKind.OCR_VLM: "VLM (tokens gerados a partir da página)",
}


def uncorroborated_fields(fields: dict[str, ExtractedField]) -> list[str]:
    """Os campos que um único mecanismo de pixel sustenta.

    Mesma condição que a triagem usa para levantar `NO_CORROBORATION`, lida aqui
    do fato e não do achado — a triagem só roda depois.

    O que se pergunta ao modelo é diferente do caso ausente: o valor existe, foi
    lido e tem recorte. O que falta é segunda opinião, e ela pode existir sem o
    consenso ter enxergado: a votação casa regiões por sobreposição, então um
    leitor que devolveu a mesma linha com outro recorte, ou a célula sem
    geometria, fica de fora da região e não vota. O valor está no texto dele;
    não estava no lugar onde se olhou.

    Uma checagem literal já resolveria parte disso — e resolve, é a guarda de
    `corroborating_reads`. O que ela não resolve é ruído de OCR: `0,1124300000`
    contra `0,112430000`, `Alíquota` contra `Aliquota`. Ler os três textos e
    dizer se aquele valor está lá é exatamente o que um modelo faz melhor que
    um `in`.
    """
    return [
        name
        for name, field in fields.items()
        if field.value is not None
        and len(field.corroborating_kinds) == 1
        and set(field.corroborating_kinds) <= _PIXEL
    ]


def missing_fields(fields: dict[str, ExtractedField], event_type) -> list[str]:
    """Os campos que a varredura tem o que procurar.

    Ausência declarada pelo emissor (`a definir`) e campo que não se aplica ao
    tipo do evento ficam de fora: os dois são fatos afirmados sobre o documento,
    não buracos, e mandar o modelo procurá-los é convidá-lo a inventar o que o
    aviso disse não existir.
    """
    procuraveis = (None, AbsenceReason.NOT_FOUND)
    return [
        name
        for name in spec_for(event_type).fields
        if (field := fields.get(name)) is None
        or (field.value is None and field.absence_reason in procuraveis)
    ]


def build_prompt(
    reads: dict[ReaderKind, str],
    missing: list[str],
    confirm: dict[str, str] | None = None,
) -> str:
    """As duas perguntas, na mesma chamada e sobre o mesmo material.

    São perguntas diferentes — *onde está isto?* e *isto está aí?* — mas o
    material é o mesmo e a resposta tem a mesma forma, então separá-las em duas
    chamadas custaria o dobro para reler os mesmos três textos.
    """
    partes = []
    if missing:
        partes += [
            "(A) Campos que nenhum extrator preencheu. Procure cada um nas leituras abaixo:",
            ", ".join(missing),
            "",
        ]
    if confirm:
        partes += [
            "(B) Campos já extraídos, mas sustentados por um único mecanismo de leitura. "
            "Para cada um, confira se o MESMO valor aparece em alguma outra leitura — "
            "ainda que grafado com ruído de OCR (espaço a mais, acento perdido, zero "
            "sobrando no decimal). Devolva o valor como você o encontrou; se não "
            "encontrar em nenhuma outra leitura, não devolva o campo:",
        ]
        partes += [f"    {nome} = {valor!r}" for nome, valor in confirm.items()]
        partes.append("")
    for kind in _ORDEM:
        texto = (reads.get(kind) or "").strip()
        if not texto:
            continue
        partes += [f"--- LEITURA · {_ROTULO[kind]} ---", texto, ""]
    return "\n".join(partes)


def same_value(a: str, b: str) -> bool:
    """Duas grafias do mesmo valor.

    Tipada quando os dois lados são número — `R$ 0,1124300000` e `0,112430000`
    são a mesma quantia —, achatada quando não são: o OCR perde espaço e acento
    o tempo todo, e uma comparação que dependa deles rejeita leitura correta.
    """
    numero_a, numero_b = parse_ptbr_number(a or ""), parse_ptbr_number(b or "")
    if numero_a is not None and numero_b is not None:
        return numero_a == numero_b
    return _squash(normalize(a or "")) == _squash(normalize(b or ""))


def corroborating_reads(value: str, reads: dict[ReaderKind, str]) -> list[ReaderKind]:
    """Quais mecanismos contêm este valor.

    Comparação achatada — o OCR perde espaço o tempo todo e uma comparação que
    dependa deles rejeitaria leituras corretas —, e sobre o texto como lido, sem
    normalizar número ou data: aqui não se está votando entre grafias, está-se
    perguntando se o valor devolvido existe na página.
    """
    alvo = _squash(normalize(value))
    if not alvo:
        return []
    return [
        kind
        for kind in _ORDEM
        if (texto := reads.get(kind)) and alvo in _squash(normalize(texto))
    ]


def confirm(
    returned: dict[str, str],
    fields: dict[str, ExtractedField],
    pedidos: list[str],
    reads: dict[ReaderKind, str],
) -> list[tuple[str, str]]:
    """Segunda opinião para o que um mecanismo só sustentava.

    O que este passo pode fazer é **uma coisa**: acrescentar mecanismos à lista
    de quem corrobora, quando o valor que já está no registro aparece também na
    leitura de outro. Ele não escreve valor. Se o modelo devolver coisa
    diferente do que foi extraído, o campo fica como está e a divergência vira
    nota — o valor extraído passou por grounding, reparo e validação, e uma
    releitura de texto não é motivo para desfazer isso.

    O degrau é recalculado de quem contém o valor, como em todo o resto: dois
    mecanismos de pixel dão `media`, e aí a triagem já não levanta
    `NO_CORROBORATION` — não porque alguém decidiu perdoar o campo, mas porque
    a corroboração passou a existir e foi medida.
    """
    notas: list[tuple[str, str]] = []
    # Campo perguntado e não devolvido é resposta, não silêncio: o modelo leu as
    # três e não achou aquele valor em nenhuma outra. Sem a nota, o registro não
    # distingue "a varredura procurou e não achou" de "a varredura nem rodou" —
    # e as duas coisas pedem ações diferentes de quem revisa.
    for name in pedidos:
        if name not in (returned or {}) and name in fields:
            fields[name].notes.append(
                "varredura final não localizou este valor em nenhuma outra leitura"
            )

    for name, value in (returned or {}).items():
        if name not in set(pedidos):
            continue
        field = fields.get(name)
        if field is None or field.value is None or not str(value).strip():
            continue

        atual = field.value_raw or str(field.value)
        if not same_value(str(value), atual):
            notas.append((
                name,
                f"varredura leu {str(value)!r} onde o extrator leu {atual!r} — "
                "valor extraído mantido, divergência registrada",
            ))
            continue

        kinds = corroborating_reads(str(value), reads) or corroborating_reads(atual, reads)
        novos = [k for k in kinds if k not in field.corroborating_kinds]
        if not novos:
            notas.append((
                name,
                "varredura confirmou o valor, mas em nenhuma leitura além da que já o "
                "sustentava — segue sem corroboração",
            ))
            continue

        field.corroborating_kinds = sorted(
            set(field.corroborating_kinds) | set(kinds), key=_ORDEM.index
        )
        field.reading_level = reading_level(field.corroborating_kinds)
        quem = ", ".join(k.value for k in novos)
        field.notes.append(
            f"varredura final localizou o mesmo valor em: {quem} — a região não casou na "
            "votação, a leitura sim"
        )
    return notas


def accept(
    returned: dict[str, str], missing: list[str], reads: dict[ReaderKind, str]
) -> tuple[dict[str, ExtractedField], list[tuple[str | None, str]]]:
    """Aplica as três guardas. Devolve os campos aceitos e a trilha do descarte.

    Cada descarte volta com o campo a que se refere, para que a nota fique na
    linha certa do relatório: um descarte anônimo é indistinguível de nunca ter
    havido tentativa, e é exatamente isso que se quer poder auditar num passo
    que deixa o modelo escrever valor. Chave que nem campo é não tem linha — sai
    sem alvo, para o registro de erros do nó.

    Puro: não chama modelo, não toca em I/O. É aqui que mora a decisão, e é por
    isso que ela é testável sem chave de API.
    """
    pedidos = set(missing)
    aceitos: dict[str, ExtractedField] = {}
    notas: list[tuple[str | None, str]] = []

    for name, value in (returned or {}).items():
        if name not in pedidos:
            notas.append(
                (None, f"varredura devolveu '{name}', que não estava ausente — descartado")
            )
            continue
        if not value or not str(value).strip():
            continue

        kinds = corroborating_reads(str(value), reads)
        if not kinds:
            notas.append((
                name,
                f"varredura leu {str(value)!r} para este campo, e o valor não aparece em "
                "nenhuma das três leituras — descartado",
            ))
            continue

        nivel = reading_level(kinds)
        field = ExtractedField(
            name=name,
            value_raw=str(value).strip(),
            evidence_text=str(value).strip(),
            rationale="localizado na varredura final sobre as três leituras",
            reader_kind=kinds[0],
            reading_level=nivel,
            corroborating_kinds=kinds,
            recovered=True,
        )
        quem = ", ".join(k.value for k in kinds)
        field.notes.append(f"recuperado pela varredura final; presente em: {quem}")
        aceitos[name] = field

    return aceitos, notas
