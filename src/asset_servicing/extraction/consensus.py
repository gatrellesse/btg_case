"""Ingestão por consenso: três leituras da mesma região, comparadas.

Alternativa à cascata, não substituta. A cascata **escolhe** um leitor por
página e entrega o valor por conta dele; aqui os três leem a página inteira e o
bloco sai com *quantos mecanismos independentes concordaram sobre ele*. É a
diferença entre "o OCR leu isto" e "camada de texto e OCR leram a mesma coisa,
o VLM leu outra".

Ordem: ler → tipar → validar → votar. Cada passo existe por um caso medido:

- comparar texto cru inventa conflito onde não há (`0,1124300000` e
  `0,112430000` são o mesmo `Decimal`)
- reparar antes de votar **fabrica** concordância — o leitor nunca leu aquilo
- a estrutura resolve sozinha o que iria para humano: dos dois ISIN corrompidos
  pelo VLM no lote, ambos têm 11 caracteres onde a ISO 6166 exige 12

O que sai é `Block` normal, com as leituras vencidas em `alternatives` — o
campo já existe no contrato exatamente para isto: *"a reviewer should see both,
not just the winner"*.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pymupdf

from ..formats import parse_date, parse_decimal
from ..models import BBox, ReaderKind
from ..text import _squash
from .readers.base import Block, Reader, ReadResult

#: Taxa de erro por classe de mecanismo. Não são votos equivalentes: a camada de
#: texto devolve a string do próprio programa que gerou o PDF, o OCR classifica
#: tinta renderizada, o VLM gera tokens. Medido nos 8 documentos do lote: o VLM
#: corrompeu 2 dos 8 ISIN (um deles em PDF nativo, onde a string correta estava
#: na camada de texto), o OCR nenhum. Amostra pequena — a ordem importa mais que
#: os valores.
MECHANISM_ERROR: dict[ReaderKind, float] = {
    ReaderKind.TEXT_LAYER: 0.02,
    ReaderKind.OCR_DET: 0.10,
    ReaderKind.OCR_VLM: 0.25,
}

#: Mecanismos que leem os mesmos pixels erram junto — confundem o mesmo par de
#: glifos pelo mesmo motivo. Concordância entre eles vale menos que entre camada
#: de texto e pixel, que não compartilham modo de falha.
PIXEL_DERIVED = {ReaderKind.OCR_DET, ReaderKind.OCR_VLM}

#: Sobreposição mínima (interseção sobre a MENOR área) para dizer que dois
#: leitores falam da mesma região. Sobre a menor, e não IoU, porque um motor
#: devolve a tabela como bloco único e outro como oito células — IoU puniria a
#: diferença de granularidade como se fosse desacordo.
OVERLAP_MIN = 0.15

# --------------------------------------------------------------------------
# o portão: estrutura elimina antes de alguém votar
#
# Votar é o último recurso, não o primeiro. Um candidato que a especificação
# reprova não precisa de segunda opinião para ser descartado — e é justamente
# quando NÃO existe segunda opinião que isto importa. Foi o caso do doc 07: o
# OCR quebrou, sobrou só o VLM, e o ISIN de 11 caracteres virou o valor do
# registro porque não havia quem o contradissesse. Estrutura contradiz sozinha.
# --------------------------------------------------------------------------

#: 11 a 13 caracteres terminando em dígito: largo para capturar o ISIN
#: corrompido (que precisa aparecer para ser reprovado), estreito para não casar
#: com "PARTICIPACOES" nem "ACIONISTAS".
_ISIN_LIKE = re.compile(r"\b([A-Z]{2}[A-Z0-9]{8,10}[0-9])\b")
_DATE_LIKE = re.compile(r"\b(\d{1,2}/\d{1,2}/\d{4})\b")

#: A leitura eliminada sai do voto e **rebaixa um degrau**: um leitor ter
#: produzido lixo naquela região é informação sobre a região. Rebaixar, e não
#: multiplicar, mantém a escala categórica — um "0,81" não diz nada a mais que
#: "média" e finge uma precisão que não existe.
#:
#: Quando TODAS as leituras são reprovadas não há o que eliminar: o valor sai
#: assim mesmo, porque inventar não é opção, mas no degrau mais baixo para que a
#: triagem o encontre.


def _isin_structure_ok(code: str) -> bool:
    """ISO 6166, forma: 12 caracteres, prefixo de país, verificador numérico.

    Só a forma. O dígito verificador é INFO neste projeto — os identificadores
    do lote são sintéticos e quase nenhum fecha o mod 10, então vetar por
    checksum reprovaria a base inteira.
    """
    return len(code) == 12 and code[:2].isalpha() and code[-1].isdigit()


def structural_flaws(text: str) -> list[str]:
    """O que a especificação reprova neste texto, sem consultar mais ninguém."""
    flaws = []
    for match in _ISIN_LIKE.finditer(text.upper()):
        code = match.group(1)
        if not _isin_structure_ok(code):
            flaws.append(f"ISIN {code} tem {len(code)} caracteres, ISO 6166 exige 12")
    for match in _DATE_LIKE.finditer(text):
        if parse_date(match.group(1)) is None:
            flaws.append(f"data {match.group(1)} não existe no calendário")
    return flaws


#: Os quatro degraus, do mais forte ao mais fraco. A escala inteira: não há peso
#: por trás dela. Houve — 0,99 / 0,98 / 0,95 / 0,85, para alimentar
#: `confiança = min(leitura, modelo) × validação` — e o que a tabela de limiares
#: fazia com esses quatro números, cruzada com as quatro classes de campo, era
#: uma única distinção: `baixa` reprovava em campo crítico e de identidade, e
#: todo o resto passava. Dezesseis células para dizer o que a lista de quem leu
#: já dizia.
LEVELS = ("muito_alta", "alta", "media", "baixa")


def demote(level: str, degraus: int = 1) -> str:
    """Rebaixa na escala, sem sair dela."""
    return LEVELS[min(LEVELS.index(level) + degraus, len(LEVELS) - 1)]


def weakest(levels) -> str:
    """O degrau mais fraco do conjunto — o `min` que sobrou depois dos floats."""
    presentes = [lv for lv in levels if lv in LEVELS]
    return max(presentes, key=LEVELS.index) if presentes else LEVELS[-1]


#: Decimal em convenção pt-BR, com ou sem separador de milhar. Estreito de
#: propósito: casar ponto como decimal engoliria `33.222.111` de um CNPJ, e
#: canonizar inteiro solto faria `0001` e `001` virarem o mesmo — concordância
#: fabricada justamente em identificador, que é onde ela custa mais caro.
_DECIMAL_PTBR = re.compile(r"\d{1,3}(?:\.\d{3})+,\d+|\d+,\d+")


def _typed(match: re.Match) -> str:
    value = parse_decimal(match.group(0))
    return format(value.normalize(), "f") if value is not None else match.group(0)


def vote_key(text: str) -> str:
    """A forma sobre a qual se vota: tipada, não crua.

    Este é o passo *tipar* da ordem `ler → tipar → validar → votar`. Ele existia
    na documentação e não no código, e o custo apareceu medido no doc 07:

        OCR  'Valor bruto por ação PN ,,,,.,, R$ 0,1124300000'
        VLM  'Valor bruto por ação PN R$ 0,112430000'

    São o mesmo `Decimal`. Comparados como texto, viram desacordo, o campo perde
    a corroboração e vai para revisão — um humano gasta tempo para concluir que
    dois leitores leram o mesmo número.

    Tipar não é reparar. Reparar muda o valor (`10%` → `0,1`, `A definir` →
    ausente) e, feito antes do voto, **fabrica** concordância sobre algo que
    leitor nenhum leu. Aqui só se normaliza a grafia de um mesmo número; zeros
    à direita não carregam significado numérico.
    """
    return _squash(_DECIMAL_PTBR.sub(_typed, text))


def lend_cell_geometry(por_leitor: list[tuple[str, list[Block]]]) -> list[str]:
    """O VLM lê o conteúdo da tabela; o TableFormer mede as células.

    O `<otsl>` do doctags só localiza a tabela, então as células do VLM chegam
    com `geometry_pending` e a caixa da tabela inteira. Sem coordenada elas não
    se alinham com nada e o VLM fica de fora da votação justamente na tabela —
    onde estão todos os valores do evento.

    A transferência é por **posição no grid**, não por proporção: quando os dois
    leitores detectam a mesma tabela com o mesmo número de linhas e colunas, a
    célula (r, c) de um é a célula (r, c) do outro, e a caixa emprestada é a
    medida, não uma estimativa. Se as formas diferirem, não há correspondência
    confiável e nada é emprestado — a célula continua pendente e simplesmente
    não corrobora, que é o estado de hoje.

    Modifica os blocos no lugar e devolve o que foi feito, para a trilha.
    """
    doadores = [
        (nome, [b for b in bs if b.is_cell and not b.geometry_pending])
        for nome, bs in por_leitor
    ]
    notas: list[str] = []
    for nome, blocos in por_leitor:
        pendentes = [b for b in blocos if b.geometry_pending]
        if not pendentes:
            continue
        forma = (max(b.row for b in pendentes) + 1, max(b.col for b in pendentes) + 1)
        emprestados = 0
        for outro, celulas in doadores:
            if outro == nome or not celulas:
                continue
            if (max(c.row for c in celulas) + 1, max(c.col for c in celulas) + 1) != forma:
                continue
            medidas = {(c.row, c.col): c.bbox for c in celulas}
            for alvo in pendentes:
                caixa = medidas.get((alvo.row, alvo.col))
                # a tabela tem de ser a mesma região da página, não outra
                if caixa is None or _overlap(alvo.bbox, caixa) < OVERLAP_MIN:
                    continue
                alvo.bbox = caixa
                alvo.geometry_pending = False
                emprestados += 1
            if emprestados:
                notas.append(
                    f"{nome}: {emprestados} célula(s) de tabela receberam a geometria "
                    f"medida por {outro}"
                )
                break
        restantes = sum(1 for b in pendentes if b.geometry_pending)
        if restantes:
            notas.append(
                f"{nome}: {restantes} célula(s) sem geometria — nenhum leitor mediu "
                f"uma tabela {forma[0]}x{forma[1]} na mesma região; não votam"
            )
    return notas


#: Fração da altura do menor bloco que precisa coincidir para dois blocos serem
#: da mesma linha. Alto o bastante para não juntar linhas vizinhas de um
#: parágrafo, baixo o bastante para tolerar a diferença de baseline entre
#: motores que detectam a caixa de formas diferentes.
ROW_OVERLAP_MIN = 0.5


def _same_row(a: BBox, b: BBox) -> bool:
    alto = min(a.y1 - a.y0, b.y1 - b.y0)
    if alto <= 0:
        return False
    return (min(a.y1, b.y1) - max(a.y0, b.y0)) / alto >= ROW_OVERLAP_MIN


def rows(blocks: list[Block]) -> list[Block]:
    """Agrupa os blocos de um leitor em linhas, ordenando cada uma por `x0`.

    Existe porque granularidade não é desacordo. No doc 07 o OCR devolve a linha
    `Valor bruto por ação PN ,,,,.,, R$ 0,1124300000` e o VLM devolve dois
    blocos, `Valor bruto por ação PN` e `R$ 0,112430000`. O consenso casava o
    âncora com **um** bloco do outro leitor e comparava linha contra rótulo —
    diferente por construção, em toda linha de tabela que um leitor partisse e o
    outro não.

    `OVERLAP_MIN` já era intersecção sobre a menor área justamente para *achar* o
    par apesar da granularidade; o que faltava era igualá-la antes de comparar.
    Normalizar os dois lados para linha resolve simetricamente, sem depender de
    qual leitor virou referência.
    """
    restantes = sorted(blocks, key=lambda b: (b.bbox.y0, b.bbox.x0))
    saida: list[Block] = []
    while restantes:
        primeiro = restantes.pop(0)
        linha = [primeiro]
        restantes = [
            b for b in restantes
            if not (_same_row(primeiro.bbox, b.bbox) and linha.append(b) is None)
        ]
        if len(linha) == 1:
            saida.append(primeiro)
            continue
        linha.sort(key=lambda b: b.bbox.x0)
        caixa = linha[0].bbox
        for b in linha[1:]:
            caixa = caixa.union(b.bbox)
        saida.append(Block(
            text=" ".join(b.text for b in linha),
            bbox=caixa,
            # a linha inteira vale o que vale seu pedaço mais fraco
            level=weakest(b.level for b in linha),
            reader_kind=primeiro.reader_kind,
            role=primeiro.role,
            order=primeiro.order,
            corroborating_kinds=list(primeiro.corroborating_kinds),
        ))
    return sorted(saida, key=lambda b: (b.bbox.y0, b.bbox.x0))


def reading_level(kinds) -> str:
    """A força da leitura como categoria, não como número.

    O que decide não é o quanto um motor se acha seguro — é **quem** leu e se
    alguém confirmou. `combine` já era isso, mas devolvia `0,998` e `0,95`, e
    dois dígitos sugerem uma precisão que a evidência não tem: a diferença entre
    eles não é 4,8%, é "a camada de texto confirmou" contra "só os pixels".

    * **muito alta** — a camada de texto e pelo menos mais um mecanismo
    * **alta** — a camada de texto sozinha: é a string do programa que gerou o
      PDF, não um palpite sobre tinta
    * **média** — OCR e VLM concordando: dois leitores dos mesmos pixels, que
      erram junto
    * **baixa** — um único leitor de pixel, sem ninguém para contradizê-lo
    """
    kinds = set(kinds)
    if ReaderKind.TEXT_LAYER in kinds:
        return "muito_alta" if len(kinds) > 1 else "alta"
    if len(kinds) > 1:
        return "media"
    return "baixa"


def _overlap(a: BBox, b: BBox) -> float:
    x0, y0 = max(a.x0, b.x0), max(a.y0, b.y0)
    x1, y1 = min(a.x1, b.x1), min(a.y1, b.y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    smallest = min((a.x1 - a.x0) * (a.y1 - a.y0), (b.x1 - b.x0) * (b.y1 - b.y0))
    return inter / smallest if smallest > 0 else 0.0


@dataclass
class Reading:
    reader: str
    kind: ReaderKind
    block: Block


def resolve(readings: list[Reading]) -> tuple[Block, list[str]]:
    """Um bloco por região, com as leituras vencidas anexadas.

    Ordem: validar → votar. A estrutura elimina primeiro, e só os sobreviventes
    votam; assim um conflito que a especificação resolve sozinha não gasta
    revisão humana, e uma leitura única reprovada não passa por ser única.

    Concordância é medida sobre a `vote_key`: texto achatado — o OCR perde
    espaço o tempo todo, e uma comparação que dependa deles rejeita leituras que
    estão certas — e com os decimais tipados, para que a mesma quantia grafada
    com outro número de zeros não vire desacordo.
    """
    flaws = {id(r): structural_flaws(r.block.text) for r in readings}
    clean = [r for r in readings if not flaws[id(r)]]
    # se todas forem reprovadas não há o que eliminar: o valor sai no degrau
    # mais baixo em vez de sair como se ninguém tivesse notado
    all_flawed = not clean
    survivors = readings if all_flawed else clean
    eliminated = [r for r in readings if r not in survivors]

    notes = []
    for r in eliminated:
        notes.append(f"{r.reader} eliminado: {flaws[id(r)][0]}")
    if all_flawed:
        notes.append(
            f"nenhuma leitura passou na estrutura ({flaws[id(readings[0])][0]}) — "
            "valor mantido no degrau mais baixo"
        )

    groups: dict[str, list[Reading]] = {}
    for reading in survivors:
        groups.setdefault(vote_key(reading.block.text), []).append(reading)

    # vence o grupo com os mecanismos mais fortes, não o mais numeroso: dois
    # leitores de pixel concordando valem menos que a camada de texto sozinha
    best_key = max(groups, key=lambda k: -LEVELS.index(reading_level({r.kind for r in groups[k]})))
    winners = groups[best_key]
    kinds = {r.kind for r in winners}

    winner = min(winners, key=lambda r: MECHANISM_ERROR[r.kind]).block
    nivel = reading_level(kinds)
    if all_flawed:
        nivel = LEVELS[-1]
    elif eliminated:
        nivel = demote(nivel, len(eliminated))
    block = Block(
        text=winner.text,
        bbox=winner.bbox,
        level=nivel,
        reader_kind=min(kinds, key=lambda k: MECHANISM_ERROR[k]),
        role=winner.role,
        row=winner.row,
        col=winner.col,
        order=winner.order,
        # quem concordou, nominalmente — ver `Block.corroborating_kinds`
        corroborating_kinds=sorted(kinds, key=lambda k: MECHANISM_ERROR[k]),
    )
    # toda leitura que não venceu fica anexada, inclusive as concordantes: a
    # trilha tem de mostrar quem corroborou, não só quem divergiu
    block.alternatives = [
        (r.reader, r.block.text) for r in readings if r.block is not winner
    ]
    return block, notes


def _text_by_kind(results) -> dict[ReaderKind, str]:
    """O que cada mecanismo leu na página, em ordem de leitura."""
    textos: dict[ReaderKind, list[str]] = {}
    for _, resultado in results:
        for bloco in resultado.blocks:
            textos.setdefault(bloco.reader_kind, []).append(bloco.text)
    return {kind: "\n".join(linhas) for kind, linhas in textos.items()}


def read_page_consensus(
    page: pymupdf.Page, readers: list[Reader]
) -> tuple[list[Block], list[str], list[ReaderKind], dict[ReaderKind, str]]:
    """Roda cada leitor na página inteira e funde região a região.

    O quarto retorno é o texto **de cada leitor**, antes da fusão. A fusão
    escolhe um vencedor por região e o que os outros leram fica em
    `alternatives`, espalhado bloco a bloco — o que serve para auditar um valor,
    mas não para reler o documento. A varredura de campos ausentes precisa das
    três leituras inteiras e lado a lado: o campo que nenhum extrator achou
    costuma estar exatamente na leitura que perdeu a votação.
    """
    results: list[tuple[Reader, ReadResult]] = []
    notes: list[str] = []
    kinds: list[ReaderKind] = []

    for reader in readers:
        if not reader.available():
            notes.append(f"{reader.name} indisponível — não vota")
            continue
        result = reader.read(page)
        results.append((reader, result))
        if result.blocks:
            for block in result.blocks:
                if block.reader_kind not in kinds:
                    kinds.append(block.reader_kind)
        elif result.note and result.note.startswith("falhou"):
            # falha não é abstenção: o campo vai ficar com menos leitores e o
            # registro tem de dizer que foi por quebra, não por ausência de
            # conteúdo — senão degradação silenciosa vira "consenso"
            notes.append(f"{reader.name} FALHOU — {result.note[8:]}")
        else:
            # abstenção não é discordância: o parser devolve zero bloco num
            # scan, e isso não pode rebaixar o degrau de nada
            notes.append(f"{reader.name} não leu nada nesta página — abstém-se")

    if not results:
        return [], notes + ["nenhum leitor disponível"], kinds, {}

    # Capturado aqui, antes de qualquer poda: o que sai da votação por falta de
    # geometria continua sendo texto que alguém leu, e é justamente a célula de
    # tabela do VLM — onde estão os valores do evento. Para votar, uma leitura
    # sem coordenada não serve; para reler o documento atrás de um campo que
    # ninguém achou, serve.
    leituras = _text_by_kind(results)

    # Cada leitor é normalizado para linhas **para comparar**, e só para isso.
    # Um motor devolve a linha da tabela inteira e outro a parte em rótulo e
    # valor; sem igualar, a comparação é linha contra rótulo, diferente por
    # construção. Mas o que sai são os blocos finos originais, porque
    # granularidade é assunto de comparação e não de armazenamento: fundir de
    # verdade apagaria as caixas exatas das células e desfaria os pares
    # rótulo/valor de que o extrator de regras depende.
    # Antes de qualquer alinhamento: quem leu o conteúdo da célula mas não soube
    # situá-la pega a caixa de quem mediu. Sem isto o VLM não vota na tabela.
    notes.extend(lend_cell_geometry([(reader.name, result.blocks) for reader, result in results]))

    # O que continuou sem geometria sai da votação. A caixa que ele carrega é a
    # da tabela inteira, e mantê-lo faz estrago em vez de nada: as células se
    # fundem numa linha só, essa linha casa com todas as linhas da tabela, e um
    # defeito numa célula contamina o placar de todas. Medido no doc 03 — o VLM
    # corrompeu um ISIN, o empréstimo falhou por grid de forma diferente, e a
    # eliminação estrutural derrubou sete campos de uma vez.
    descartados = 0
    for _, result in results:
        antes = len(result.blocks)
        result.blocks = [b for b in result.blocks if not b.geometry_pending]
        descartados += antes - len(result.blocks)
    if descartados:
        notes.append(f"{descartados} célula(s) sem geometria ficaram fora da votação")

    por_leitor = [(reader, result.blocks, rows(result.blocks)) for reader, result in results]

    # a referência é quem leu mais regiões: é o único que garante cobrir a
    # página inteira nos três casos (nativo, escaneado, híbrido)
    reference_reader, _, reference_rows = max(por_leitor, key=lambda t: len(t[2]))
    merged: list[Block] = []

    for anchor in reference_rows:
        readings = [Reading(reference_reader.name, anchor.reader_kind, anchor)]
        for reader, _, linhas in por_leitor:
            if reader is reference_reader:
                continue
            match = max(
                (b for b in linhas if _overlap(anchor.bbox, b.bbox) >= OVERLAP_MIN),
                key=lambda b: _overlap(anchor.bbox, b.bbox),
                default=None,
            )
            if match is not None:
                readings.append(Reading(reader.name, match.reader_kind, match))
        linha, gate_notes = resolve(readings)
        notes.extend(gate_notes)

        # O placar é da linha; o recorte e o texto são dos blocos originais de
        # quem venceu, para que a proveniência continue apontando para a célula.
        origem = next(
            (bs for _, bs, _ in por_leitor if bs and bs[0].reader_kind == linha.reader_kind),
            [],
        )
        finos = [b for b in origem if _same_row(linha.bbox, b.bbox)] or [linha]
        for fino in finos:
            merged.append(Block(
                text=fino.text,
                bbox=fino.bbox,
                level=linha.level,
                reader_kind=fino.reader_kind,
                role=fino.role,
                row=fino.row,
                col=fino.col,
                order=fino.order,
                alternatives=list(linha.alternatives),
                corroborating_kinds=list(linha.corroborating_kinds),
            ))

    agreed = sum(1 for b in merged if b.alternatives)
    notes.append(
        f"consenso de {len(results)} leitura(s): {len(merged)} regiões, "
        f"{agreed} com mais de um leitor"
    )
    return merged, notes, kinds, leituras


@dataclass
class IngestConfig:
    """Quais leituras entram no consenso.

    Os três modos são o padrão porque é a configuração medida. Reduzir a lista
    é legítimo — `("parser", "ocr")` custa uma fração do tempo e ainda dá
    corroboração entre mecanismos disjuntos —, mas o degrau cai junto, e é essa
    a troca: velocidade contra quantas leituras independentes sustentam o valor.
    """

    modes: tuple[str, ...] = ("parser", "ocr", "vlm")


def default_readers(modes: tuple[str, ...] = ("parser", "ocr", "vlm")) -> list[Reader]:
    from .readers.docling_reader import DoclingReader

    return [DoclingReader(mode) for mode in modes]  # type: ignore[arg-type]
