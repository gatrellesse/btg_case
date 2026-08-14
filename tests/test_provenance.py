"""De qual região da página o valor veio.

O campo herda do bloco escolhido quatro coisas — recorte, nota de leitura,
corroboração e leitor. Escolher o bloco errado não erra só a imagem: erra a
nota, e a nota é o que decide se o campo passa no limiar.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asset_servicing.extraction.provenance import locate  # noqa: E402
from asset_servicing.extraction.readers.base import Block  # noqa: E402
from asset_servicing.models import BBox, ReaderKind  # noqa: E402


def bloco(text, x0, x1, level="muito_alta", kinds=None) -> Block:
    return Block(
        text=text,
        bbox=BBox(page=1, x0=x0, y0=336, x1=x1, y1=350,
                  page_width=595, page_height=842),
        level=level,
        reader_kind=ReaderKind.TEXT_LAYER,
        corroborating_kinds=kinds or [ReaderKind.TEXT_LAYER, ReaderKind.OCR_DET],
    )


#: A linha do doc 05 como o docling a devolve: duas células da tabela.
ROTULO = bloco('Data-base ("data com")', 63, 174)
VALOR = bloco("15/07/2026", 238, 288)
#: A evidência que o extrator devolve: a linha inteira, rótulo e valor.
EVIDENCIA = "Data-base ('data com')\n15/07/2026"


def test_sem_o_valor_o_desempate_e_por_tamanho():
    """O comportamento antigo, preservado como documentação do problema.

    Os dois blocos são fragmentos da evidência, e a nota `0,6 × tamanho`
    premia o mais comprido. Rótulo é sempre mais comprido que a data que ele
    rotula, então o rótulo vence *sempre* — não é um caso raro.
    """
    block, _ = locate(EVIDENCIA, [ROTULO, VALOR])
    assert block is ROTULO


def test_o_bloco_que_contem_o_valor_vence():
    block, quality = locate(EVIDENCIA, [ROTULO, VALOR], "15/07/2026")
    assert block is VALOR
    assert quality >= 0.9


def test_a_ordem_dos_blocos_nao_decide():
    for ordem in ([ROTULO, VALOR], [VALOR, ROTULO]):
        block, _ = locate(EVIDENCIA, ordem, "15/07/2026")
        assert block is VALOR


def test_a_nota_de_leitura_passa_a_ser_a_do_valor():
    """O ponto todo: a leitura media a concordância sobre as palavras
    "Data-base (data com)". Um dígito errado na data não a movia."""
    rotulo = bloco('Data-base ("data com")', 63, 174, level="muito_alta",
                   kinds=[ReaderKind.TEXT_LAYER, ReaderKind.OCR_DET])
    valor = bloco("15/07/2026", 238, 288, level="baixa",
                  kinds=[ReaderKind.OCR_DET])
    block, _ = locate(EVIDENCIA, [rotulo, valor], "15/07/2026")
    assert block.level == "baixa"
    assert block.corroborating_kinds == [ReaderKind.OCR_DET]


def test_evidencia_dentro_de_um_bloco_so_continua_valendo_1():
    """Quando o leitor devolveu a linha inteira num bloco só, ele já contém
    rótulo e valor — não há o que desempatar."""
    linha = bloco('Data-base ("data com") 15/07/2026', 63, 288)
    block, quality = locate('Data-base ("data com") 15/07/2026', [linha], "15/07/2026")
    assert block is linha
    assert quality == 1.0


def test_prosa_nao_muda():
    """Campos lidos de frase corrida têm um bloco só; o desempate não se aplica."""
    frase = bloco(
        "A posição acionária considerada será a do final do pregão de "
        "26 de junho de 2026 (data-base do grupamento).", 63, 533)
    block, _ = locate(frase.text, [frase], "26 de junho de 2026")
    assert block is frase


def test_valor_ausente_dos_blocos_cai_no_criterio_antigo():
    """Se nenhum bloco contém o valor, o desempate por tamanho ainda é o melhor
    palpite disponível — degradar para o comportamento anterior é o certo."""
    block, _ = locate(EVIDENCIA, [ROTULO, VALOR], "99/99/9999")
    assert block is ROTULO
