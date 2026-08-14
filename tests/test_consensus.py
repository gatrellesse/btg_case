"""O portão da ingestão: estrutura elimina antes de alguém votar."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asset_servicing.extraction import consensus as consensus_module  # noqa: E402
from asset_servicing.extraction.consensus import (  # noqa: E402
    LEVELS,
    Reading,
    reading_level,
    resolve,
    structural_flaws,
    weakest,
)
from asset_servicing.extraction.readers.base import Block  # noqa: E402
from asset_servicing.models import BBox, ReaderKind  # noqa: E402


def block(text: str) -> Block:
    return Block(
        text=text,
        bbox=BBox(page=1, x0=0, y0=0, x1=100, y1=10, page_width=595, page_height=842),
        reader_kind=ReaderKind.TEXT_LAYER,
    )


def reading(name: str, kind: ReaderKind, text: str) -> Reading:
    b = block(text)
    b.reader_kind = kind
    return Reading(reader=name, kind=kind, block=b)


# --------------------------------------------------------------------------
# estrutura
# --------------------------------------------------------------------------


def test_isin_curto_e_reprovado_pela_forma():
    """Os dois ISIN que o VLM corrompeu no lote têm 11 caracteres."""
    assert structural_flaws("TLNR4 (ISIN BRTLNRACPR2)")
    assert not structural_flaws("TLNR4 (ISIN BRTLNRACNPR2)")


def test_checksum_nao_reprova():
    """Os identificadores do lote são sintéticos e quase nenhum fecha o mod 10.
    Vetar por dígito verificador reprovaria a base de referência inteira."""
    assert not structural_flaws("ISIN BRTIETACNOR3")


def test_data_impossivel_e_reprovada():
    assert structural_flaws("Data de pagamento 31/02/2026")
    assert not structural_flaws("Data de pagamento 28/02/2026")


# --------------------------------------------------------------------------
# o voto
# --------------------------------------------------------------------------


def test_leitura_reprovada_sai_do_voto():
    """O VLM erra o ISIN, o OCR acerta: a estrutura decide sem segunda opinião
    precisar ser 'a maioria'."""
    out, notes = resolve([
        reading("docling:ocr", ReaderKind.OCR_DET, "TLNR4 (ISIN BRTLNRACNPR2)"),
        reading("docling:vlm", ReaderKind.OCR_VLM, "TLNR4 (ISIN BRTLNRACPR2)"),
    ])
    assert "BRTLNRACNPR2" in out.text
    assert any("eliminado" in n for n in notes)


def test_eliminado_rebaixa_um_degrau():
    """Um leitor ter produzido lixo naquela região é informação sobre a região."""
    limpo, _ = resolve([
        reading("docling:parser", ReaderKind.TEXT_LAYER, "ISIN BRTIETACNOR3"),
        reading("docling:ocr", ReaderKind.OCR_DET, "ISIN BRTIETACNOR3"),
    ])
    com_eliminado, _ = resolve([
        reading("docling:parser", ReaderKind.TEXT_LAYER, "ISIN BRTIETACNOR3"),
        reading("docling:ocr", ReaderKind.OCR_DET, "ISIN BRTIETACNOR3"),
        reading("docling:vlm", ReaderKind.OCR_VLM, "ISIN BRTIETACNR3"),
    ])
    assert limpo.level == "muito_alta"
    assert com_eliminado.level == "alta"


def test_no_piso_a_eliminacao_fica_na_trilha_e_nao_no_degrau():
    """O custo de ter quatro degraus: um leitor de pixel sozinho já está no
    fundo da escala, então eliminar um segundo não tem para onde rebaixar. A
    informação não se perde — ela vai para as notas, e a triagem já manda o campo
    para revisão por não haver corroboração nenhuma."""
    sozinho, _ = resolve([reading("docling:ocr", ReaderKind.OCR_DET, "ISIN BRTLNRACNPR2")])
    com_eliminado, notas = resolve([
        reading("docling:ocr", ReaderKind.OCR_DET, "ISIN BRTLNRACNPR2"),
        reading("docling:vlm", ReaderKind.OCR_VLM, "ISIN BRTLNRACPR2"),
    ])
    assert sozinho.level == com_eliminado.level == "baixa"
    assert any("eliminado" in n for n in notas)


def test_leitor_unico_reprovado_nao_passa_por_ser_unico():
    """O caso que motivou o portão: no doc 07 o OCR quebrou, sobrou só o VLM, e
    o ISIN de 11 caracteres virou o valor do registro porque não havia quem o
    contradissesse. Estrutura contradiz sozinha."""
    ruim, notes = resolve([reading("docling:vlm", ReaderKind.OCR_VLM, "ISIN BRTLNRACPR2")])
    assert ruim.level == LEVELS[-1]
    assert any("nenhuma leitura passou" in n for n in notes)
    # o valor sai mesmo assim: inventar não é opção, sinalizar é
    assert "BRTLNRACPR2" in ruim.text


def test_o_nivel_de_leitura_e_categorico():
    """O número dava falsa precisão: entre 0,998 e 0,95 não vão 4,8%, vai "a
    camada de texto confirmou" contra "só os pixels leram"."""
    T, O, V = ReaderKind.TEXT_LAYER, ReaderKind.OCR_DET, ReaderKind.OCR_VLM
    assert reading_level([T, O, V]) == "muito_alta"
    assert reading_level([T, O]) == "muito_alta"
    assert reading_level([T, V]) == "muito_alta"
    assert reading_level([T]) == "alta"
    assert reading_level([O, V]) == "media"
    assert reading_level([O]) == "baixa"
    assert reading_level([V]) == "baixa"


def test_a_escala_nao_tem_peso_por_tras():
    """A escala é a informação inteira. Havia pesos (0,99 · 0,98 · 0,95 · 0,85)
    para alimentar `min(leitura, modelo) × validação`; a conta saiu e eles com
    ela, e `weakest` é o `min` que sobrou — sobre degraus, não sobre floats."""
    assert weakest(["muito_alta", "baixa", "media"]) == "baixa"
    assert weakest(["muito_alta", "alta"]) == "alta"
    # sem leitura nenhuma, o degrau mais baixo: não se presume o que não se leu
    assert weakest([]) == "baixa"
    assert not hasattr(consensus_module, "LEVEL_WEIGHT")


def test_camada_de_texto_sozinha_vale_mais_que_dois_leitores_de_pixel():
    """Mecanismos disjuntos não compartilham modo de falha; dois leitores dos
    mesmos pixels erram junto, e por isso concordarem vale menos."""
    ordem = lambda ks: -LEVELS.index(reading_level(ks))  # noqa: E731
    assert ordem({ReaderKind.TEXT_LAYER}) > ordem({ReaderKind.OCR_DET, ReaderKind.OCR_VLM})


def test_concordancia_entre_mecanismos_disjuntos_e_a_mais_forte():
    assert reading_level({ReaderKind.TEXT_LAYER, ReaderKind.OCR_DET}) == "muito_alta"
    assert reading_level({ReaderKind.OCR_DET, ReaderKind.OCR_VLM}) == "media"


def test_leituras_vencidas_ficam_anexadas():
    """Um revisor precisa ver quem corroborou e quem divergiu, não só o vencedor."""
    out, _ = resolve([
        reading("docling:parser", ReaderKind.TEXT_LAYER, "PARTICIPAÇÕES S.A."),
        reading("docling:vlm", ReaderKind.OCR_VLM, "PARTICIPAÇÔES S.A."),
    ])
    assert out.alternatives
    assert any("PARTICIPAÇÔES" in text for _, text in out.alternatives)


# --------------------------------------------------------------------------
# granularidade: igualar antes de comparar
# --------------------------------------------------------------------------


def em(text, x0, x1, y0=336, y1=350) -> Block:
    return Block(
        text=text,
        bbox=BBox(page=1, x0=x0, y0=y0, x1=x1, y1=y1, page_width=595, page_height=842),
        reader_kind=ReaderKind.OCR_VLM,
    )


def test_a_linha_se_reconstroi_na_ordem_horizontal():
    """No doc 07 o VLM parte a linha da tabela em rótulo e valor e o OCR devolve
    a linha inteira. Comparar linha contra rótulo diverge por construção."""
    from asset_servicing.extraction.consensus import rows

    saida = rows([em("R$ 0,1124300000", 249, 373), em("Valor bruto por ação PN", 75, 297)])
    assert len(saida) == 1
    assert saida[0].text == "Valor bruto por ação PN R$ 0,1124300000"
    assert (saida[0].bbox.x0, saida[0].bbox.x1) == (75, 373)


def test_linhas_vizinhas_nao_se_juntam():
    from asset_servicing.extraction.consensus import rows

    saida = rows([em("Data-base (data com)", 75, 300, y0=379, y1=401),
                  em("Data ex-JCP", 75, 300, y0=397, y1=419)])
    assert len(saida) == 2


def test_a_granularidade_deixa_de_ser_desacordo():
    """O efeito medido: a linha do valor bruto passa a ter duas opiniões."""
    ocr = reading("docling:ocr", ReaderKind.OCR_DET,
                  "Valor bruto por ação PN ,,,,.,, R$ 0,1124300000")
    from asset_servicing.extraction.consensus import rows

    partido = rows([em("Valor bruto por ação PN", 75, 297),
                    em("R$ 0,112430000", 249, 373)])[0]
    out, _ = resolve([ocr, Reading("docling:vlm", ReaderKind.OCR_VLM, partido)])
    assert set(out.corroborating_kinds) == {ReaderKind.OCR_DET, ReaderKind.OCR_VLM}


def test_o_bloco_fino_sobrevive_ao_voto_por_linha():
    """Fundir de verdade apagaria as caixas exatas das células e desfaria os
    pares rótulo/valor de que o extrator de regras depende — foi o que quebrou
    a moeda do doc 01 quando a fusão era permanente."""
    import pymupdf

    from asset_servicing.extraction.consensus import default_readers, read_page_consensus

    with pymupdf.open("Case AI Dev - Envio/documents/01_energetica_vale_tiete_dividendo.pdf") as pdf:
        blocos, _, _, leituras = read_page_consensus(pdf[0], default_readers())
    # a célula do rótulo e a do valor continuam separadas
    rotulos = [b for b in blocos if b.text.strip() == "Data-base ('data com')"]
    valores = [b for b in blocos if b.text.strip() == "12/06/2026"]
    assert rotulos and valores
    assert rotulos[0].bbox.x1 <= valores[0].bbox.x0

    # e as três leituras inteiras seguem disponíveis, cada uma como o seu
    # mecanismo a entregou: é sobre elas que a varredura final relê o documento
    assert set(leituras) == {ReaderKind.TEXT_LAYER, ReaderKind.OCR_DET, ReaderKind.OCR_VLM}
    assert all(texto.strip() for texto in leituras.values())


# --------------------------------------------------------------------------
# tipar: o passo que estava na docstring e não no código
# --------------------------------------------------------------------------


def test_mesmo_decimal_com_outra_grafia_nao_e_desacordo():
    """O caso medido no doc 07: OCR leu `R$ 0,1124300000`, VLM leu
    `R$ 0,112430000`. Mesmo número. Comparados como texto viravam desacordo, o
    campo perdia a corroboração e um humano gastava tempo para concluir que os
    dois leitores tinham lido a mesma coisa."""
    from asset_servicing.extraction.consensus import vote_key

    assert vote_key("R$ 0,1124300000") == vote_key("R$ 0,112430000")
    assert vote_key("R$ 1.000,00") == vote_key("R$ 1000,00")


def test_tipar_nao_apaga_diferenca_de_identificador():
    """Canonizar inteiro solto faria `0001` e `001` casarem — concordância
    fabricada bem onde ela custa mais caro."""
    from asset_servicing.extraction.consensus import vote_key

    assert vote_key("ISIN BRTLNRACNPR2") != vote_key("ISIN BRTLNRACPR2")
    assert vote_key("0001") != vote_key("001")
    assert vote_key("22/06/2026") != vote_key("22/06/2027")
    # ponto como decimal engoliria o CNPJ; ele tem de sair intacto
    assert vote_key("33.222.111/0001-44") == vote_key("33.222.111/0001-44")


def test_o_voto_conta_a_corroboracao_recuperada_pela_tipagem():
    out, _ = resolve([
        reading("docling:ocr", ReaderKind.OCR_DET, "Valor bruto R$ 0,1124300000"),
        reading("docling:vlm", ReaderKind.OCR_VLM, "Valor bruto R$ 0,112430000"),
    ])
    assert set(out.corroborating_kinds) == {ReaderKind.OCR_DET, ReaderKind.OCR_VLM}
    assert out.level == "media"


# --------------------------------------------------------------------------
# corroboração: um fato, não um número
# --------------------------------------------------------------------------


def test_o_bloco_guarda_quem_concordou():
    """O degrau resume a corroboração, mas o operador precisa saber *quem*
    concordou — "média" não distingue OCR+VLM de VLM+OCR, e a lista sim."""
    out, _ = resolve([
        reading("docling:ocr", ReaderKind.OCR_DET, "22/06/2026"),
        reading("docling:vlm", ReaderKind.OCR_VLM, "22/06/2026"),
    ])
    assert set(out.corroborating_kinds) == {ReaderKind.OCR_DET, ReaderKind.OCR_VLM}

    sozinho, _ = resolve([reading("docling:ocr", ReaderKind.OCR_DET, "22/06/2026")])
    assert sozinho.corroborating_kinds == [ReaderKind.OCR_DET]
    assert sozinho.level == "baixa"


def test_quem_divergiu_nao_conta_como_corroboracao():
    """O leitor que perdeu o voto leu *outra coisa*; anexá-lo à trilha não o
    torna uma segunda opinião sobre o vencedor."""
    out, _ = resolve([
        reading("docling:ocr", ReaderKind.OCR_DET, "22/06/2026"),
        reading("docling:vlm", ReaderKind.OCR_VLM, "Data-base"),
    ])
    assert out.corroborating_kinds == [ReaderKind.OCR_DET]
    assert out.alternatives  # continua na trilha


def test_leitor_de_pixel_sozinho_vai_para_revisao_mesmo_passando_no_limiar():
    """O caso medido no doc 07: nove campos com 0,90, sete deles passando —
    ISIN, CNPJ e ticker exatamente na linha de corte de 0,90, num documento
    onde o VLM corrompeu justamente o ISIN."""
    from asset_servicing.models import (
        Classification,
        DocumentInfo,
        EventRecord,
        EventType,
        ExtractedField,
        Grounding,
        GroundingStatus,
        ReasonCode,
    )
    from asset_servicing.validation.evidence import load_thresholds
    from asset_servicing.validation.triage import triage

    def campo(name, kinds):
        return ExtractedField(
            name=name, value="BRTLNRACNPR2", value_raw="BRTLNRACNPR2",
            evidence_text="BRTLNRACNPR2", page=1, reading_level="baixa",
            reader_kind=ReaderKind.OCR_DET, corroborating_kinds=kinds,
            grounding=Grounding(status=GroundingStatus.EXACT, score=1.0),
        )

    rec = EventRecord(
        document=DocumentInfo(file_name="scan.pdf", n_pages=1),
        event_type=EventType.JCP,
        classification=Classification(event_type=EventType.JCP),
        fields={"isin": campo("isin", [ReaderKind.OCR_DET])},
    )
    out = triage(rec, load_thresholds())
    assert ReasonCode.NO_CORROBORATION in {r.code for r in out.reasons}
    assert "isin" in out.fields_for_review


def test_camada_de_texto_sozinha_nao_dispara():
    """Ela devolve a string do programa que gerou o PDF, não um palpite sobre
    tinta — exigir segunda opinião ali mandaria todo documento nativo a revisão.
    """
    from asset_servicing.models import (
        Classification,
        DocumentInfo,
        EventRecord,
        EventType,
        ExtractedField,
        Grounding,
        GroundingStatus,
        ReasonCode,
    )
    from asset_servicing.validation.evidence import load_thresholds
    from asset_servicing.validation.triage import triage

    rec = EventRecord(
        document=DocumentInfo(file_name="nativo.pdf", n_pages=1),
        event_type=EventType.JCP,
        classification=Classification(event_type=EventType.JCP),
        fields={"isin": ExtractedField(
            name="isin", value="BRTIETACNOR3", evidence_text="BRTIETACNOR3", page=1,
            reading_level="alta",
            reader_kind=ReaderKind.TEXT_LAYER,
            corroborating_kinds=[ReaderKind.TEXT_LAYER],
            grounding=Grounding(status=GroundingStatus.EXACT, score=1.0),
        )},
    )
    out = triage(rec, load_thresholds())
    assert ReasonCode.NO_CORROBORATION not in {r.code for r in out.reasons}


# --------------------------------------------------------------------------
# o relator: escreve prosa, não dado
# --------------------------------------------------------------------------


def test_relator_devolve_apenas_narrativa():
    """A garantia estrutural: o schema do relator não tem onde escrever dado.

    A disposição é calculada em código antes dele rodar, e o nó devolve só
    `justification` — nenhum campo, validação ou veredito passa por ele.
    """
    from asset_servicing.llm.schemas import VerificationOutput

    assert set(VerificationOutput.model_fields) == {"justification", "cited_reason_codes"}


def test_figura_inventada_e_recusada():
    """Ele não consegue alterar dado, mas conseguiria contar outra história
    sobre o mesmo dado — e é a história que o operador lê."""
    from pydantic_ai import ModelRetry

    from asset_servicing.llm.agents import _canon_figure, _FIGURE

    registro = {_canon_figure(v) for v in {"21/08/2026", "R$ 0,1124300000"}}
    citadas = _FIGURE.findall("O pagamento ocorre em 15/08/2026.")
    inventadas = [f for f in citadas if _canon_figure(f) not in registro]
    assert inventadas == ["15/08/2026"]
    assert ModelRetry  # o validador levanta isto, e o agente tenta de novo


def test_mesma_figura_em_grafia_diferente_passa():
    """`R$ 0,4275` e `0,4275000000` são a mesma afirmação; a grafia do relator
    não deveria decidir se ele passa."""
    from asset_servicing.llm.agents import _canon_figure

    assert _canon_figure("R$ 0,4275") == _canon_figure("0,4275000000")
    assert _canon_figure("22/06/2026") != _canon_figure("21/08/2026")
