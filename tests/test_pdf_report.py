"""O relatório em PDF: o que o operador abre.

Os testes de layout existem porque as falhas do `pymupdf.Story` são silenciosas
ou infinitas — uma imagem com caminho errado não levanta nada, ela só não
aparece; uma tabela antes de uma quebra de página trava o laço de paginação.
Nenhuma das duas apareceria numa revisão de código.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pymupdf
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asset_servicing.models import (  # noqa: E402
    AbsenceReason,
    Classification,
    Disposition,
    DocumentInfo,
    EventRecord,
    EventType,
    ExtractedField,
    Grounding,
    GroundingStatus,
    ReasonCode,
    Triage,
    TriageReason,
    spec_for,
)
from asset_servicing.reporter import drafted, pdf_report  # noqa: E402

DOCS = ROOT / "Case AI Dev - Envio/documents"


def campo(name, value, page=1, **kw) -> ExtractedField:
    from asset_servicing.models import BBox

    return ExtractedField(
        name=name, value=value, value_raw=str(value) if value else None,
        evidence_text=str(value) if value else None, page=page,
        reading_level="muito_alta",
        bbox=BBox(page=1, x0=60, y0=300, x1=300, y1=320,
                  page_width=595, page_height=842),
        grounding=Grounding(status=GroundingStatus.EXACT, score=1.0),
        **kw,
    )


def registro(**kw) -> EventRecord:
    fields = {
        n: campo(n, "valor") for n in spec_for(EventType.DIVIDENDO).fields
    }
    fields["com_date"] = campo("com_date", "15/07/2026")
    fields["net_per_share"] = ExtractedField(
        name="net_per_share", absence_reason=AbsenceReason.NOT_APPLICABLE
    )
    rec = EventRecord(
        document=DocumentInfo(file_name="01_energetica_vale_tiete_dividendo.pdf", n_pages=1),
        event_type=EventType.DIVIDENDO,
        classification=Classification(event_type=EventType.DIVIDENDO),
        fields=fields,
        triage=Triage(
            disposition=Disposition.BLOCKED,
            reasons=[TriageReason.of(ReasonCode.DATE_INCOHERENCE, "x", "com_date")],
            overall_confidence=0.29,
            fields_for_review=["com_date"],
        ),
    )
    for k, v in kw.items():
        setattr(rec, k, v)
    return rec


@pytest.fixture(scope="module")
def relatorio(tmp_path_factory):
    out = tmp_path_factory.mktemp("pdf")
    path = pdf_report.write_exceptions_pdf([registro()], DOCS, out)
    return path, pymupdf.open(path)


# --------------------------------------------------------------------------
# o que o Story quebra em silêncio
# --------------------------------------------------------------------------


def test_tabela_antes_de_quebra_de_pagina_nao_converge():
    """A razão de cada seção virar um `Story` próprio.

    Se uma versão futura do pymupdf corrigir isto, este teste falha e a costura
    manual pode ser revista — que é exatamente o aviso que se quer receber.
    """
    import io

    html = ("<html><body><p>a</p><table><tr><td>x</td></tr></table>"
            "<div style='page-break-before: always'></div><p>b</p></body></html>")
    story = pymupdf.Story(html=html)
    writer = pymupdf.DocumentWriter(io.BytesIO())
    more, paginas = 1, 0
    while more and paginas < 8:
        dev = writer.begin_page(pdf_report.A4)
        more, _ = story.place(pdf_report.A4 + pdf_report.MARGEM)
        story.draw(dev)
        writer.end_page()
        paginas += 1
    writer.close()
    assert more, "pymupdf passou a convergir: a costura por seções pode ser revista"


def test_os_recortes_aparecem_de_fato(relatorio):
    """Um `<img src>` que o Archive não resolve não levanta erro — a imagem
    simplesmente não é desenhada, e a coluna 'ver' aponta para o nada. Foi o que
    aconteceu com caminho absoluto."""
    _, doc = relatorio
    embutidas = sum(len(doc[i].get_images()) for i in range(doc.page_count))
    assert embutidas > 0


def test_todo_documento_tem_link_no_indice(relatorio):
    """`08_construtora_horizonte_bonificacao.pdf` não era encontrado pela busca:
    a fonte compõe o "fi" como um glifo só. O índice saía com um link a menos e
    nada acusava."""
    _, doc = relatorio
    assert len(doc[1].get_links()) == 1


def test_ligadura_nao_esconde_o_arquivo(tmp_path):
    rec = registro()
    rec.document.file_name = "08_construtora_horizonte_bonificacao.pdf"
    doc = pymupdf.open(pdf_report.write_exceptions_pdf([rec], DOCS, tmp_path))
    assert len(doc[1].get_links()) == 1


# --------------------------------------------------------------------------
# o conteúdo
# --------------------------------------------------------------------------


def test_a_capa_traz_lote_data_e_quantos_revisar(relatorio):
    _, doc = relatorio
    texto = doc[0].get_text()
    assert "Relatório de Exceções" in texto
    assert "AS-" in texto  # identificador do lote
    assert "1 de 1" in texto


def test_o_identificador_do_lote_e_estavel():
    """Um id aleatório por execução impediria dizer se dois relatórios falam do
    mesmo lote — a primeira pergunta ao arquivar um."""
    from datetime import datetime, timezone

    quando = datetime(2026, 8, 13, tzinfo=timezone.utc)
    recs = [registro()]
    assert pdf_report.batch_id(recs, quando) == pdf_report.batch_id(recs, quando)


def test_as_tabelas_somam_o_schema_do_tipo(relatorio):
    """Mostrar só as exceções esconde o denominador.

    Duas tabelas, e não quatro: depois que a varredura final passou a reler as
    três leituras atrás de cada campo ausente, um campo sem valor deixou de ser
    "ninguém procurou" e virou fato conferido — o lugar dele é entre o que
    precisa de conferência.
    """
    _, doc = relatorio
    texto = "".join(doc[i].get_text() for i in range(doc.page_count))
    total = len(spec_for(EventType.DIVIDENDO).fields)
    # com_date (em revisão) + net_per_share (sem valor)
    assert f"Em revisão (2 de {total})" in texto
    assert f"Aceitos ({total - 2} de {total})" in texto
    assert "Não encontrados" not in texto


def test_campo_ausente_sai_da_tabela_de_aceitos(relatorio):
    """Não houve o que aceitar.

    No doc 01 a alíquota de IR sumiu da extração, caiu entre os aceitos e o
    documento saiu liberado sem motivo nenhum. A ausência ficou invisível
    justamente por estar na tabela do que deu certo.
    """
    _, doc = relatorio
    texto = "".join(doc[i].get_text() for i in range(doc.page_count))
    # 'Valor líquido por ação' é o campo ausente do registro de teste
    assert texto.index("Valor líquido") < texto.index("Aceitos (")
    assert "ausente" in texto


def test_valor_recuperado_pela_varredura_diz_de_onde_veio():
    """A coluna do recorte responde "de onde isto veio". Para um campo que a
    varredura final recuperou não houve rótulo casado com valor, houve releitura
    do documento inteiro — e é isso que a célula tem de dizer."""
    rec = registro()
    rec.fields["tax_rate"] = ExtractedField(
        name="tax_rate", value="0,175", value_raw="17,5%", evidence_text="17,5%",
        reading_level="media", recovered=True,
    )
    corpo = pdf_report._documento(rec, {}, "d0")
    assert pdf_report.LLM_REASONING in corpo


def test_todo_campo_sem_valor_vai_para_revisao():
    """Obrigatório ou opcional, com achado ou sem: se não há valor, não houve o
    que aceitar, e a linha fica onde o operador olha."""
    rec = registro()
    rec.fields["isin"] = ExtractedField(name="isin")  # obrigatório, sem valor
    rec.triage.fields_for_review = ["com_date"]
    corpo = pdf_report._documento(rec, {}, "d0")
    # com_date (achado) + isin e net_per_share (sem valor)
    assert "Em revisão (3 de" in corpo
    assert "Não encontrados" not in corpo


def test_o_schema_e_enumerado_por_evento_sem_lista_herdada():
    """O defeito da moeda: `currency` valia para todo evento porque estava numa
    base comum, e um grupamento — que não move dinheiro — bloqueava por não
    trazer moeda que o aviso não tinha por que ter."""
    assert "currency" in spec_for(EventType.DIVIDENDO).required
    assert "currency" in spec_for(EventType.BONIFICACAO).required  # custo em R$
    assert "currency" not in spec_for(EventType.GRUPAMENTO).fields
    assert "currency" not in spec_for(EventType.DESDOBRAMENTO).fields


def test_campos_e_a_soma_de_obrigatorios_e_opcionais():
    for event in EventType:
        spec = spec_for(event)
        assert spec.fields == [*spec.required, *spec.optional]
        assert not (set(spec.required) & set(spec.optional)), event


def test_nenhum_score_no_relatorio(relatorio):
    """A confiança é o degrau — quem leu e quem confirmou. Um `0,97 de 0,90`
    pedia ao operador que fizesse a comparação de cabeça para chegar à mesma
    conclusão que a tabela em que a linha está já dá."""
    import re

    _, doc = relatorio
    corpo = "".join(doc[i].get_text() for i in range(doc.page_count))
    corpo = corpo.split("Correspondência")[0]
    # datas e valores extraídos do documento são dado, não score
    corpo = re.sub(r"\d{2}/\d{2}/\d{4}|\d+,\d{4,}|\d[\d.]*,\d{2}\b", "", corpo)
    assert not re.findall(r"\b0,\d\d\b", corpo), re.findall(r"\b0,\d\d\b", corpo)


def test_a_coluna_traz_o_degrau_e_a_evidencia(relatorio):
    _, doc = relatorio
    # o texto do PDF quebra a célula em linhas; o que importa é o conteúdo
    corpo = " ".join("".join(doc[i].get_text() for i in range(doc.page_count)).split())
    assert "muito alta" in corpo
    assert "muito alta exata" in corpo


def test_sem_secao_de_motivos(relatorio):
    """Ela repetia em prosa o que as linhas já dizem; a razão virou coluna."""
    _, doc = relatorio
    texto = "".join(doc[i].get_text() for i in range(doc.page_count))
    assert "Motivos" not in texto


def test_toda_linha_diz_algo_na_coluna_ver(relatorio):
    """Ausência de recorte é informação sobre a evidência; célula vazia a esconde."""
    _, doc = relatorio
    texto = "".join(doc[i].get_text() for i in range(doc.page_count))
    assert "ver" in texto and "sem recorte" in texto


# --------------------------------------------------------------------------
# a data de redação
# --------------------------------------------------------------------------


def test_data_de_redacao_sai_do_fim_do_aviso():
    """`São Paulo (SP), 28 de maio de 2026.` — exige a cidade antes para não
    capturar as datas do corpo, que são o evento e não a redação."""
    from datetime import date

    assert drafted.from_pdf(DOCS / "01_energetica_vale_tiete_dividendo.pdf") == date(2026, 5, 28)


def test_documento_escaneado_admite_nao_ter_a_data():
    """Não há camada de texto num scan. Inventar uma data para compor o nome do
    índice seria o pior resultado possível."""
    assert drafted.from_pdf(DOCS / "07_telecom_norte_jcp_SCAN.pdf") is None
    assert drafted.slug(EventType.JCP, None) == "jcp_sem-data"


def test_o_nome_do_indice_identifica_o_documento():
    from datetime import date

    assert drafted.slug(EventType.DIVIDENDO, date(2026, 5, 28)) == "dividendo_28-05-2026"
