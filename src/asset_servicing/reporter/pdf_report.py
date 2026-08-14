"""O relatório de exceções em PDF.

Escrito para ser trabalhado, não lido: o operador abre na capa, encontra o
emissor no índice, salta para o documento e decide campo a campo. Daí as três
decisões de forma:

* **Duas tabelas por documento — em revisão e aceitos — cuja soma é o schema
  inteiro do tipo de evento.** Mostrar só as exceções esconde o denominador: um
  documento com dois campos em revisão parece igual tendo dez ou trinta campos.
  A tabela de aceitos é o que dá tamanho ao problema. Duas, e não quatro: as
  tabelas de "não encontrados" saíram quando a varredura final passou a reler
  as três leituras atrás de cada campo ausente. Depois dela, um campo sem valor
  não é mais "ninguém procurou" — é um fato conferido sobre o documento, e o
  lugar dele é entre o que precisa de conferência.
* **Sem seção de motivos.** Ela repetia, em prosa, o que as linhas já dizem. A
  razão de um campo não ter sido aceito é uma coluna dele, ao lado do valor.
* **Uma coluna "ver" por campo, com link para o recorte.** Quando não há
  recorte, a célula diz por que não há — ausência de crop é informação sobre a
  evidência, e uma célula vazia a esconderia.

Sem dependência nova: `pymupdf.Story` diagrama HTML em PDF, com links internos e
imagens embutidas. O mesmo pymupdf que já lê os documentos.

Cada seção vira um `Story` próprio e os PDFs são costurados depois. Não é
preferência de estilo: no pymupdf 1.28, **um `<table>` em qualquer lugar antes de
um `page-break-before` faz o `Story.place()` nunca convergir** — o laço de
paginação roda para sempre. Medido e reduzido ao mínimo; nenhum contorno inline
funciona (`<p>` vazio, `<br/>`, quebra no próprio `<h2>`). Uma seção por `Story`
elimina a combinação, ao custo de refazer à mão os links do índice, que deixam
de ser resolvidos pelo `write_with_links` por atravessarem documentos.
"""

from __future__ import annotations

import hashlib
import html
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pymupdf

from . import drafted, labels
from ..extraction import provenance
from ..models import LAYER_ORDER, Disposition, EventRecord, spec_for

A4 = pymupdf.paper_rect("a4")
MARGEM = (40, 46, -40, -50)

#: Largura útil da página, para dimensionar os recortes embutidos.
UTIL = A4.width + MARGEM[0] + MARGEM[2]
#: Largura de exibição do recorte dentro da célula. Pequeno de propósito: o
#: recorte fica *ao lado* do valor, e quem quiser detalhe dá zoom no próprio
#: leitor de PDF — a imagem vai embutida a 200 dpi, então aguenta a ampliação.
VER_LARGURA = int(UTIL * 0.40)

AZUL = "#123f6d"

CSS = f"""
body {{ font-family: sans-serif; font-size: 8.5pt; color: #1a1a1a; }}
h2 {{ font-size: 13pt; margin-top: 0pt; margin-bottom: 1pt; color: {AZUL}; }}
h3 {{ font-size: 9.5pt; margin-top: 12pt; margin-bottom: 3pt; color: #333; }}
p  {{ margin-top: 2pt; margin-bottom: 2pt; }}

.faixa {{ background-color: {AZUL}; color: #ffffff; padding: 26px; }}
.faixa .t {{ font-size: 25pt; }}
.faixa .s {{ font-size: 10pt; color: #c8d8ea; }}

.meta {{ width: 100%; font-size: 10pt; margin-top: 26px; }}
.meta .k {{ color: #666; font-size: 8pt; }}
.meta .v {{ font-size: 13pt; }}

.sub {{ color: #666; font-size: 8pt; }}
.rot {{ color: #777; font-size: 7pt; }}

/* índice */
.emissor {{ font-size: 10pt; color: {AZUL}; margin-top: 11pt; margin-bottom: 1pt; }}
.grupo {{ font-size: 7.5pt; margin-left: 14px; margin-top: 5pt; margin-bottom: 0pt; }}
.rev {{ color: #a11; }}
.pronto {{ color: #176; }}
.entrada {{ margin-left: 30px; margin-top: 1pt; margin-bottom: 1pt; font-size: 8.5pt; }}

table {{ width: 100%; font-size: 7.5pt; }}
th {{ text-align: left; background-color: {AZUL}; color: #ffffff; font-size: 7pt;
     padding: 3px; }}
td {{ vertical-align: top; padding: 3px; }}
.no {{ color: #a11; }}
.ok {{ color: #176; }}
.achado {{ color: #a11; font-size: 8pt; }}
"""


def _esc(value) -> str:
    return html.escape(str(value if value is not None else "—"))


def batch_id(records: list[EventRecord], gerado: datetime) -> str:
    """Determinístico a partir do lote: mesmo conjunto, mesmo identificador.

    Um id aleatório por execução impediria dizer se dois relatórios falam do
    mesmo lote — que é a primeira pergunta ao arquivar um.
    """
    digest = hashlib.sha256(
        "|".join(sorted(r.document.file_name for r in records)).encode()
    ).hexdigest()[:6].upper()
    return f"AS-{gerado:%Y%m%d}-{digest}"


# --------------------------------------------------------------------------
# recortes
# --------------------------------------------------------------------------


def _crops(record: EventRecord, docs_dir: Path, out_dir: Path) -> dict[str, str]:
    """Um recorte por campo com região conhecida, relativo ao `Archive`.

    Gerados aqui, e não na extração, porque a extração só recortava o que ia
    para revisão — e a tabela de aceitos precisa poder ser conferida também.
    Auditar um valor aceito é exatamente o que o enunciado pede que seja
    possível sem reabrir o documento.

    O caminho sai **relativo** a `out_dir` porque é assim que o `Archive` do
    pymupdf resolve `<img src>`. Um caminho absoluto não levanta erro: a imagem
    simplesmente não aparece, e o relatório sai com a coluna "ver" apontando
    para o nada.
    """
    pdf_path = docs_dir / record.document.file_name
    if not pdf_path.exists():
        return {}
    stem = Path(record.document.file_name).stem
    out: dict[str, str] = {}
    for name, field in record.fields.items():
        if field.bbox is None:
            continue
        destino = out_dir / "crops" / stem / f"{name}.png"
        if destino.exists() or provenance.crop(pdf_path, _faixa(field.bbox), destino):
            out[name] = destino.relative_to(out_dir).as_posix()
    return out


#: Margem esquerda do texto nos avisos (63pt num A4 de 595), com folga.
_MARGEM_TEXTO = 50.0
#: Largura mínima da faixa, a partir da margem: cobre rótulo e valor de um par
#: (no doc 05 o rótulo fica em x 63–174 e o valor em 238–288) sem ir até a borda
#: direita, que espremeria a linha e devolveria o recorte ilegível — o problema
#: original com outra causa.
_FAIXA_MIN = 480.0


def _faixa(bbox):
    """A linha inteira, não só o pedaço que casou com a evidência.

    `layout.enrich` divide rótulo e valor em blocos separados, e a evidência
    frequentemente casa com o do rótulo — então o recorte de `com_date` saía
    mostrando `Data-base ("data com")` e não `15/07/2026`. Para conferir um
    valor o revisor precisa dos dois juntos: o rótulo diz o que o número é.

    Abrir na horizontal e manter a banda vertical resolve sem depender de
    corrigir a proveniência, que exigiria reprocessar o lote.

    A faixa é ancorada na margem do texto, não centrada na região: o rótulo fica
    sempre à esquerda, e centrar cortava `Código de negociação` ao meio quando a
    evidência casava com o valor, lá na direita.

    O corte contra a borda da página fica a cargo do `provenance.crop`, que já
    recorta contra `page.rect`: `page_width` vem 0.0 nos registros gravados, e
    calcular a partir dele daria uma faixa silenciosamente errada em qualquer
    página que não fosse A4.
    """
    x0 = min(bbox.x0, _MARGEM_TEXTO)
    return bbox.model_copy(update={"x0": x0, "x1": max(bbox.x1, x0 + _FAIXA_MIN)})


#: O que a coluna "ver" mostra quando o valor não foi casado a uma região da
#: página, e sim recuperado pela varredura final sobre as três leituras. Não há
#: recorte a apontar porque não houve rótulo casado com valor: houve releitura
#: do documento inteiro atrás de um campo que os extratores deixaram passar.
LLM_REASONING = "LLM-Reasoning"


def _porque_sem_recorte(field) -> str:
    """Célula da coluna 'ver' quando não há imagem. Nunca vazia."""
    if field is None:
        return "campo fora do registro"
    if getattr(field, "recovered", False):
        return f"{LLM_REASONING} — valor localizado relendo as três leituras do documento"
    if field.value is None and field.value_raw is None:
        motivo = getattr(field.absence_reason, "value", None)
        return {
            "stated_undefined": "sem recorte: o emissor declarou o valor indefinido",
            "not_applicable": "sem recorte: não se aplica a este tipo de evento",
        }.get(motivo, "sem recorte: nenhum extrator encontrou o campo")
    if field.bbox is None:
        return "sem recorte: valor extraído sem região localizada na página"
    return "sem recorte: falha ao renderizar a região"


# --------------------------------------------------------------------------
# tabelas
# --------------------------------------------------------------------------

#: A terceira coluna traz três verificações **independentes**, uma por linha:
#: quem leu o valor, se o valor aparece mesmo no documento, e o que a base de
#: referência diz sobre ele. Nenhuma resolve a outra — um ISIN lido só pelo OCR
#: continua lido só pelo OCR depois de bater com a base, e essa é justamente a
#: página do doc 07: três chaves de identidade confirmadas por um registro
#: externo, todas sustentadas por um único mecanismo de leitura. A confiança
#: num número só dizia as três coisas ao mesmo tempo e portanto nenhuma delas.
_COLUNAS = ("campo", "valor", "leitura / evidência / base", "por que não foi aceito",
            "ver — recorte da região de onde o valor foi lido")


def _linha(record: EventRecord, name: str, motivos: dict, crops: dict, anchor: str) -> str:
    field = record.fields.get(name)
    if field is None:
        return (f"<tr><td>{_esc(labels.campo(name))}</td><td>—</td><td>—</td>"
                f"<td>campo fora do registro</td><td class='rot'>—</td></tr>")

    valor = field.value if field.value is not None else (field.value_raw or "—")
    # Sem número: a leitura é o degrau — quem leu e quem confirmou. Um "0,97 de
    # 0,90" pedia ao operador que fizesse a comparação de cabeça para chegar à
    # mesma conclusão que a tabela em que a linha está já dá.
    # Sem marcador de reprovação: a tabela em que a linha está já diz se foi
    # aceita, e a coluna ao lado diz por que não. Um ⛔ ao lado de "muito alta"
    # dizia que a leitura falhou quando a leitura tinha sido ótima.
    conf = "—" if field.value is None else labels.nivel_curto(field.reading_level)

    # grounding: o valor aparece mesmo no documento, no lugar declarado
    evid = labels.evidencia(field.grounding.status)

    # e a segunda verificação, independente da primeira: o que a base diz. Ela
    # não opina sobre data nem valor, e "não é chave de identidade" é uma
    # resposta diferente de "não confere".
    base = labels.base_de_referencia(field.audit.golden.status) if field.audit else None

    porques = motivos.get(name, [])
    if porques:
        porque = "; ".join(porques)
    elif field.value is None:
        # Um campo sem valor não foi "aceito" — não houve o que aceitar. Marcá-lo
        # de verde ao lado dos que foram conferidos é a leitura errada exata: o
        # operador precisa saber se a ausência é do aviso ou da extração.
        if field.value_raw:
            # leu e não deu: dizer "nenhum extrator encontrou" seria falso, e o
            # que foi lido é a pista de por que não deu
            texto = f"lido como “{_esc(field.value_raw)}”, sem interpretação possível"
        else:
            texto = {
                "stated_undefined": "ausente — o emissor declarou indefinido",
                "not_applicable": "ausente — não se aplica a este tipo de evento",
            }.get(getattr(field.absence_reason, "value", None),
                  "ausente — nenhum extrator encontrou o campo")
        porque = f"<span class='rot'>{texto}</span>"
    else:
        porque = "<span class='ok'>aceito</span>"

    # O recorte vai embutido na própria linha, e não num apêndice: mandar o
    # operador para o fim do documento para conferir um valor desfaz justamente
    # o que a coluna existe para dar — o pixel ao lado do número. A imagem entra
    # pequena mas a 200 dpi, então dá zoom no leitor de PDF sem borrar.
    if name in crops:
        ver = f"<img src=\"{crops[name]}\" width=\"{VER_LARGURA}\"/>"
        if getattr(field, "recovered", False):
            # O recorte existe — a evidência foi localizada depois —, mas a
            # origem do valor continua sendo a releitura, e é ela que responde
            # "de onde isto veio". As duas coisas cabem na mesma célula.
            ver = f"<span class='rot'>{LLM_REASONING}</span><br/>{ver}"
    else:
        ver = f"<span class='rot'>{_esc(_porque_sem_recorte(field))}</span>"

    classe = "" if not porques else " class='no'"
    # as três verificações dividem uma célula: são fatos sobre o mesmo campo e
    # separá-las em colunas custava a largura que o recorte precisa para ser
    # legível. Empilhadas, continuam distintas — que é o que importa.
    base_linha = f"<br/><span class='rot'>{_esc(base)}</span>" if base else ""
    return (
        f"<tr><td width='11%'>{_esc(labels.campo(name))}</td>"
        f"<td width='14%'>{_esc(valor)}</td>"
        f"<td width='13%'>{conf}"
        f"<br/><span class='rot'>{evid}</span>{base_linha}</td>"
        f"<td width='21%'{classe}>{porque}</td>"
        f"<td width='42%'>{ver}</td></tr>"
    )


def _tabela(titulo: str, nomes, record, motivos, crops, ancoras) -> str:
    if not nomes:
        return f"<h3>{titulo}</h3><p class='rot'>nenhum campo nesta condição.</p>"
    cabecalho = "".join(f"<th>{c}</th>" for c in _COLUNAS)
    linhas = "".join(_linha(record, n, motivos, crops, ancoras[n]) for n in nomes)
    return f"<h3>{titulo}</h3><table><tr>{cabecalho}</tr>{linhas}</table>"


# --------------------------------------------------------------------------


def _por_campo(record: EventRecord) -> tuple[dict, list[str]]:
    """Motivos indexados por campo, e os que não pertencem a campo nenhum.

    Sem a segunda lista, retirar a seção de motivos perderia os achados de
    documento — tipo de evento em conflito, emissor fora da base.
    """
    do_campo: dict[str, list[str]] = defaultdict(list)
    do_documento: list[str] = []
    do_schema = set(spec_for(record.event_type).fields)
    for reason in record.triage.reasons:
        texto = labels.motivo(reason.code)
        if reason.field and reason.field in do_schema:
            do_campo[reason.field].append(texto)
        else:
            alvo = f"{labels.campo(reason.field)}: " if reason.field else ""
            do_documento.append(f"{alvo}{texto}")
    return do_campo, do_documento


def _capa(records, exceptions, gerado, bid) -> str:
    # As três contagens são disjuntas e somam o lote. Antes "Para revisar"
    # trazia todo não-liberado e "Bloqueados" vinha ao lado, contando o mesmo
    # documento duas vezes — um total maior que o lote.
    contagem = Counter(r.triage.disposition for r in records)
    linhas = [
        ("Identificador do lote", bid),
        ("Processado em", f"{gerado:%d/%m/%Y %H:%M} UTC"),
        ("Documentos no lote", str(len(records))),
        ("Exigem atuação humana", f"{len(exceptions)} de {len(records)}"),
    ]
    for disposition in (Disposition.BLOCKED, Disposition.REVIEW, Disposition.AUTO,
                        Disposition.FAILED):
        if contagem.get(disposition):
            linhas.append((labels.disposicao(disposition), str(contagem[disposition])))
    # duas por linha
    pares = [linhas[i:i + 2] for i in range(0, len(linhas), 2)]
    corpo = "".join(
        "<tr>" + "".join(
            f"<td width='50%'><p class='k'>{_esc(k)}</p><p class='v'>{_esc(v)}</p></td>"
            for k, v in par
        ) + "</tr>"
        for par in pares
    )
    return (
        f"<div class='faixa'><p class='t'>Relatório de Exceções</p>"
        f"<p class='s'>Eventos corporativos · Asset Servicing</p></div>"
        f"<table class='meta'>{corpo}</table>"
        f"<p>&nbsp;</p>"
        f"<p class='rot'>Cada documento traz o schema completo do seu tipo de evento, "
        f"dividido entre o que foi aceito e o que precisa de conferência. Cada campo "
        f"mostra quem leu o valor, se ele aparece no documento, o que a base de "
        f"referência diz sobre ele, a razão de não ter sido aceito e o recorte da "
        f"região de onde o valor foi lido.</p>"
    )


def _indice(por_emissor: dict) -> str:
    partes = ["<h2>Índice</h2>",
              "<p class='sub'>Por emissor. Clique numa linha para ir ao documento.</p>"]
    for emissor in sorted(por_emissor):
        partes.append(f"<p class='emissor'><b>{_esc(emissor)}</b></p>")
        # Bloqueado não é "para revisar": o registro não pode seguir como está,
        # e agrupar os dois escondia a diferença que decide o que o operador faz
        # primeiro.
        grupos = [
            ("Bloqueados", "rev", [e for e in por_emissor[emissor]
                                   if e["disposicao"] == Disposition.BLOCKED]),
            ("Para revisar", "rev", [e for e in por_emissor[emissor]
                                     if e["disposicao"] == Disposition.REVIEW]),
            ("Prontos", "pronto", [e for e in por_emissor[emissor]
                                   if e["disposicao"] == Disposition.AUTO]),
            ("Falha de processamento", "rev", [e for e in por_emissor[emissor]
                                               if e["disposicao"] == Disposition.FAILED]),
        ]
        for titulo, classe, grupo in grupos:
            if not grupo:
                continue
            partes.append(f"<p class='grupo {classe}'><b>{titulo}</b></p>")
            for entrada in grupo:
                # sem <a href>: o alvo está em outro Story, e o link é criado
                # depois da costura, em `_ligar_indice`
                partes.append(
                    f"<p class='entrada'>{_esc(entrada['nome'])}"
                    f"<span class='rot'>  ·  {_esc(entrada['arquivo'])}</span></p>"
                )
    return "".join(partes)


def _ausente(record: EventRecord, name: str) -> bool:
    """Sem valor utilizável — inclusive quando algo foi lido e não virou valor.

    `value_raw` sem `value` é o caso do doc 07: o OCR leu `/08/22026` para a
    data de pagamento e o reparo não conseguiu fazer disso uma data. O registro
    não tem valor, então não há o que aceitar; contar o `value_raw` como
    presença punha a linha entre os aceitos dizendo "ausente" na própria célula.
    """
    field = record.fields.get(name)
    return field is None or field.value is None


def _documento(record, crops, anchor) -> str:
    motivos, do_documento = _por_campo(record)
    schema = spec_for(record.event_type).fields
    obrigatorios = set(spec_for(record.event_type).required)

    # Sem valor é sempre "em revisão", esteja ou não em `fields_for_review`: o
    # que não se aplica ao tipo do evento não gera achado, mas também não foi
    # aceito — não houve o que aceitar. No doc 01 a alíquota de IR sumiu da
    # extração, caiu entre os aceitos e o documento saiu liberado sem motivo
    # nenhum: a ausência ficou invisível justamente por estar na tabela do que
    # deu certo.
    revisar = [
        n for n in schema if n in record.triage.fields_for_review or _ausente(record, n)
    ]
    aceitos = [n for n in schema if n not in revisar]
    ancoras = {n: f"{anchor}-{n}" for n in schema}

    partes = [
        f"<h2 id=\"{anchor}\">{_esc(_emissor(record))}</h2>",
        f"<p class='sub'>{_esc(labels.evento(record.event_type))} · "
        f"{_esc(labels.disposicao(record.triage.disposition))} · "
        f"{_esc(record.document.file_name)}</p>",
    ]
    if do_documento:
        # Seção própria, e não uma linha solta. O doc 03 sai com 13 de 13 campos
        # aceitos e mesmo assim vai para revisão, porque o conflito é sobre o
        # tipo do evento — que não é campo do schema e não cabe em nenhuma das
        # três tabelas. Uma linha discreta fazia a página parecer dizer que não
        # havia nada a fazer.
        linhas_ach = "".join(
            f"<tr><td width='16%'>{_esc(labels.camada(r.layer))}</td>"
            f"<td width='20%'>{_esc(labels.tipo_de_erro(r.error_type))}</td>"
            f"<td width='18%'>{_esc(labels.campo(r.field)) if r.field else '—'}</td>"
            f"<td width='46%' class='no'>{_esc(labels.motivo(r.code))}</td></tr>"
            for r in sorted(record.triage.reasons, key=lambda r: LAYER_ORDER[r.layer])
            if not (r.field and r.field in set(spec_for(record.event_type).fields))
        )
        partes.append(
            f"<h3>Achados do documento ({len(do_documento)}) — não ligados a um campo</h3>"
            f"<table><tr><th>camada</th><th>tipo</th><th>alvo</th>"
            f"<th>o que houve</th></tr>{linhas_ach}</table>"
        )
    # As três somam o schema inteiro do tipo: mostrar só as exceções esconderia
    # o denominador, e dois campos em revisão significam coisas diferentes num
    # documento de treze campos e num de trinta.
    n = len(schema)
    partes.append(_tabela(f"Em revisão ({len(revisar)} de {n})",
                          revisar, record, motivos, crops, ancoras))
    partes.append(_tabela(f"Aceitos ({len(aceitos)} de {n})",
                          aceitos, record, motivos, crops, ancoras))
    return "".join(partes)


def _emissor(record: EventRecord) -> str:
    field = record.fields.get("issuer")
    if field is not None and field.value:
        return str(field.value)
    return Path(record.document.file_name).stem


def _render(corpo: str, out_dir: Path) -> pymupdf.Document:
    """Uma seção, um Story, um PDF. Os links de dentro dela já saem resolvidos."""
    html_page = f"<html><head><style>{CSS}</style></head><body>{corpo}</body></html>"
    story = pymupdf.Story(html=html_page, archive=pymupdf.Archive(out_dir))
    return story.write_with_links(lambda rn, filled: (A4, A4 + MARGEM, None))


def _ligar_indice(doc: pymupdf.Document, indice_paginas: range, entradas: list[dict]) -> int:
    """Refaz os links do índice, que a costura desfez.

    A âncora é o nome do arquivo: é o único texto da linha garantidamente único
    no lote — dois avisos do mesmo emissor, tipo e data colidiriam no nome do
    índice. O retângulo é esticado até a margem para a linha inteira ser clicável.

    A busca desliga a preservação de ligaduras. Sem isso,
    `08_construtora_horizonte_bonificacao.pdf` não é encontrado: a fonte compõe
    o "fi" como um glifo só, e o índice sai com sete links para oito documentos —
    uma falha que não levanta erro nenhum, só perde uma linha clicável.
    """
    flags = pymupdf.TEXTFLAGS_SEARCH & ~pymupdf.TEXT_PRESERVE_LIGATURES
    feitos = 0
    for entrada in entradas:
        for numero in indice_paginas:
            page = doc[numero]
            achados = page.search_for(entrada["arquivo"], flags=flags)
            if not achados:
                continue
            rect = achados[0]
            rect.x0 = A4.x0 + MARGEM[0]
            page.insert_link({
                "kind": pymupdf.LINK_GOTO,
                "from": rect,
                "page": entrada["pagina"],
                "to": pymupdf.Point(0, 0),
            })
            feitos += 1
            break
    return feitos


def write_exceptions_pdf(
    records: list[EventRecord], docs_dir: str | Path, out_dir: str | Path
) -> Path:
    out_dir, docs_dir = Path(out_dir), Path(docs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gerado = datetime.now(timezone.utc)
    exceptions = [r for r in records if r.triage.disposition != Disposition.AUTO]
    bid = batch_id(records, gerado)

    ordem = {Disposition.BLOCKED: 0, Disposition.FAILED: 1, Disposition.REVIEW: 2}
    ordenados = sorted(records, key=lambda r: (ordem.get(r.triage.disposition, 9),
                                               _emissor(r)))

    por_emissor: dict[str, list[dict]] = defaultdict(list)
    entradas: list[dict] = []
    secoes: list[tuple[dict | None, str]] = []
    for i, record in enumerate(ordenados):
        redigido = drafted.from_pdf(docs_dir / record.document.file_name)
        entrada = {
            "nome": drafted.slug(record.event_type, redigido),
            "arquivo": record.document.file_name,
            "disposicao": record.triage.disposition,
            "pagina": 0,  # preenchido na costura
        }
        entradas.append(entrada)
        por_emissor[_emissor(record)].append(entrada)
        secoes.append((entrada, _documento(record, _crops(record, docs_dir, out_dir),
                                           f"doc{i}")))

    final = pymupdf.open()
    capa = _render(_capa(records, exceptions, gerado, bid), out_dir)
    final.insert_pdf(capa)
    capa.close()

    indice_inicio = final.page_count
    indice = _render(_indice(por_emissor), out_dir)
    final.insert_pdf(indice)
    indice_fim = final.page_count
    indice.close()

    for entrada, corpo in secoes:
        entrada["pagina"] = final.page_count
        secao = _render(corpo, out_dir)
        final.insert_pdf(secao)
        secao.close()

    _ligar_indice(final, range(indice_inicio, indice_fim), entradas)

    path = out_dir / "exceptions_report.pdf"
    final.set_metadata({"title": f"Relatório de Exceções · {bid}",
                        "subject": "Eventos corporativos — Asset Servicing"})
    final.save(path, garbage=3, deflate=True)
    final.close()
    return path
