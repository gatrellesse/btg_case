"""As duas verificações de cada campo, e o português que o operador lê.

Os casos usados aqui são os medidos no lote — estão fixos no teste de
propósito, para que a regressão que motivou o módulo tenha um nome.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asset_servicing.models import (  # noqa: E402
    REASON_ORIGIN,
    Classification,
    Disposition,
    DocumentInfo,
    EventRecord,
    EventType,
    ExtractedField,
    Grounding,
    GroundingStatus,
    Layer,
    ReaderKind,
    ReasonCode,
    Triage,
    TriageReason,
    ValidationResult,
    ValidationStatus,
    spec_for,
)
from asset_servicing.reporter import audit, labels, report  # noqa: E402
from asset_servicing.validation.evidence import (  # noqa: E402
    load_thresholds,
    note_reference_matches,
)
from asset_servicing.validation.tools import Validators  # noqa: E402

TH = load_thresholds()


def field(name, value, level="muito_alta", **kw) -> ExtractedField:
    return ExtractedField(
        name=name,
        value=value,
        value_raw=str(value),
        evidence_text=str(value),
        reading_level=level,
        page=1,
        reader_kind=ReaderKind.TEXT_LAYER,
        grounding=Grounding(status=GroundingStatus.EXACT, score=1.0, matched_text=str(value)),
        **kw,
    )


def record(fields, validations) -> EventRecord:
    rec = EventRecord(
        document=DocumentInfo(file_name="teste.pdf", n_pages=1),
        event_type=EventType.DIVIDENDO,
        classification=Classification(event_type=EventType.DIVIDENDO),
        fields={f.name: f for f in fields},
        validations=validations,
    )
    note_reference_matches(rec.fields, rec.validations)
    audit.annotate(rec, TH)
    return rec


# --------------------------------------------------------------------------
# as duas verificações, e o que cada uma responde
# --------------------------------------------------------------------------


def test_data_incoerente_derruba_a_validacao_nao_a_leitura():
    """A regressão que motivou o módulo.

    No doc 05 os quatro campos de data saíam com `confiança 0,29` e mais nada.
    Lidos, eles eram quase perfeitos: o 0,29 era a cronologia impossível se
    propagando por uma multiplicação. Mostrar só o produto manda o revisor abrir
    o PDF, que é exatamente a ação errada — o que ele precisa é do nome da
    ferramenta que reprovou, e da leitura intacta ao lado.
    """
    rec = record(
        [field("com_date", "15/07/2026")],
        [ValidationResult(
            tool="check_date_coherence",
            status=ValidationStatus.FAIL,
            reason_codes=[ReasonCode.DATE_INCOHERENCE],
            fields_affected=["com_date"],
        )],
    )
    leitura = rec.fields["com_date"].audit.reading
    assert leitura.level == "muito_alta"
    assert leitura.cut_by == ["check_date_coherence"]


def test_campo_que_ninguem_reprovou_nao_tem_culpado():
    """Sem ferramenta reprovando, o que sobra é o degrau — e nada mais."""
    rec = record([field("tax_rate", "0,1", "media")], [])
    leitura = rec.fields["tax_rate"].audit.reading
    assert leitura.level == "media"
    assert leitura.cut_by == []


def test_a_base_confirma_sem_promover_a_leitura():
    """As duas verificações são independentes, e é este o teste que trava isso.

    Doc 07: ISIN, ticker e CNPJ casaram na base de referência — quatro chaves
    independentes na mesma linha — e mesmo assim foram lidos por um mecanismo de
    pixel só. "Confere com a base" e "lido só pelo OCR" são respostas a
    perguntas diferentes, e o campo carrega as duas. O piso por corroboração
    tentava resolver uma na outra e não resolvia nem isso: era capado pelo
    próprio score da leitura.
    """
    rec = record(
        [field("isin", "BRTLNRACNPR2", "baixa",
               corroborating_kinds=[ReaderKind.OCR_DET])],
        [ValidationResult(
            tool="lookup_golden_record",
            status=ValidationStatus.PASS,
            detail={"status": "MATCH", "agreeing_keys": ["isin"],
                    "matched_row": {"isin": "BRTLNRACNPR2"}},
            fields_affected=["isin"],
        )],
    )
    bloco = rec.fields["isin"].audit
    assert bloco.golden.status == "confere"
    assert bloco.reading.level == "baixa"
    assert bloco.reading.corroborating_kinds == [ReaderKind.OCR_DET]
    assert "valor confirmado na base de referência" in rec.fields["isin"].notes


def test_a_base_distingue_nao_confere_de_nao_se_aplica():
    """Uma data não 'deixa de conferir' com a base: a base não opina sobre ela."""
    rec = record(
        [field("isin", "BRTIETACNOR3"),
         field("com_date", "15/07/2026")],
        [ValidationResult(
            tool="lookup_golden_record",
            status=ValidationStatus.PASS,
            detail={"status": "MATCH", "agreeing_keys": ["isin"], "matched_row": {}},
            fields_affected=["isin"],
        )],
    )
    assert rec.fields["isin"].audit.golden.status == "confere"
    assert rec.fields["com_date"].audit.golden.status == "nao_aplicavel"


# --------------------------------------------------------------------------
# as notas
# --------------------------------------------------------------------------


def test_o_paragrafo_repetido_desaparece():
    """Doc 08 `ratio_from` trazia o mesmo parágrafo de ~400 caracteres três
    vezes: os leitores devolveram a região inteira, não uma leitura divergente.
    Repetição não é evidência."""
    f = field("ratio_from", "20")
    f.evidence_text = "proporção de 1 (uma) ação nova para cada 20 (vinte) ações"
    f.notes = [
        "leituras alternativas da região: docling:ocr='A bonificação será atribuída na "
        "proporção de 1 (uma) ação nova para cada 20 (vinte) ações de que forem titulares "
        "os acionistas.'"
    ]
    assert audit.curate_notes(f) == []


def test_a_leitura_que_de_fato_diverge_permanece_e_sai_traduzida():
    f = field("com_date", "15/07/2026")
    f.evidence_text = "Data-base ('data com') 15/07/2026"
    f.notes = ["leituras alternativas da região: docling:vlm='Data-base (data corn) 15/07/2028'"]
    saida = audit.curate_notes(f)
    assert len(saida) == 1
    assert "VLM" in saida[0] and "docling" not in saida[0]


def test_as_leituras_dos_extratores_viram_prosa():
    f = field("com_date", "15/07/2026")
    f.notes = ["leituras: rule='15/07/2026'; text='15/07/2026'",
               "extratores independentes concordam"]
    saida = audit.curate_notes(f)
    assert saida == ["regras e modelo leram “15/07/2026”"]


# --------------------------------------------------------------------------
# a taxonomia e o vocabulário
# --------------------------------------------------------------------------


def test_todo_motivo_tem_camada_e_tipo():
    """Um código novo sem origem apareceria no relatório sem atribuição."""
    assert set(REASON_ORIGIN) == set(ReasonCode)


def test_toda_camada_e_achado_primario():
    """Havia uma quinta camada, `score`, que só hospedava consequência: o código
    de limiar era a aritmética de algo que outra camada já tinha reportado. As
    quatro que restam nomeiam quem descobriu o problema, e é isso que faz
    ordenar por camada colocar a causa antes do efeito sem heurística nenhuma."""
    assert {layer for layer, _ in REASON_ORIGIN.values()} == set(Layer)


def test_a_origem_segue_o_codigo_mesmo_em_registro_antigo():
    """Um registro escrito antes destes campos existirem não pode desserializar
    afirmando ser um achado de validação — mandaria o revisor ao lugar errado
    com toda a confiança."""
    antigo = TriageReason.model_validate(
        {"code": "NO_CORROBORATION", "severity": "review", "field": "isin"}
    )
    assert antigo.layer == Layer.CONSENSO


def test_todo_termo_do_sistema_tem_portugues():
    """Sem tradução, o termo vaza como código para o relatório. Falha aqui."""
    for code in ReasonCode:
        labels.motivo(code)
    for layer in Layer:
        labels.camada(layer)
    for kind in ReaderKind:
        labels.leitor(kind)
    for status in GroundingStatus:
        labels.evidencia(status)
    for disposition in Disposition:
        labels.disposicao(disposition)
    for event in EventType:
        labels.evento(event)
        for name in spec_for(event).fields:
            labels.campo(name)
    for tool in dir(Validators):
        if tool.startswith(("check_", "lookup_")):
            labels.ferramenta(tool)


def test_termo_sem_traducao_levanta_em_vez_de_cair_no_codigo():
    import pytest

    with pytest.raises(KeyError, match="glossario.yaml"):
        labels.campo("campo_que_nao_existe")


# --------------------------------------------------------------------------
# o relatório
# --------------------------------------------------------------------------


#: Códigos do sistema: CAIXA_ALTA_COM_UNDERSCORE, ou snake_case conhecido.
_GRITO = re.compile(r"\b[A-Z][A-Z_]{5,}\b")
_SNAKE = re.compile(r"\b[a-z]+_[a-z_]+\b")


def _relatorio(tmp_path) -> str:
    rec = record(
        [field("com_date", "15/07/2026"),
         field("payment_date", "10/07/2026", "baixa",
               corroborating_kinds=[ReaderKind.OCR_DET])],
        [ValidationResult(
            tool="check_date_coherence",
            status=ValidationStatus.FAIL,
            reason_codes=[ReasonCode.DATE_INCOHERENCE],
            fields_affected=["com_date", "payment_date"],
            message="cronologia impossível",
        )],
    )
    rec.triage = Triage(
        disposition=Disposition.BLOCKED,
        reasons=[TriageReason.of(ReasonCode.DATE_INCOHERENCE, "cronologia", "com_date"),
                 TriageReason.of(ReasonCode.NO_CORROBORATION, "um leitor só", "payment_date")],
        weakest_level="baixa",
        fields_for_review=["com_date", "payment_date"],
        justification="payment_date antecede com_date (DATE_INCOHERENCE).",
    )
    return report.write_exceptions_report([rec], tmp_path).read_text(encoding="utf-8")


def test_o_corpo_nao_fala_em_codigo(tmp_path):
    """A decisão que faz este relatório ser lido por quem não escreveu o
    pipeline. Sem um teste, o próximo campo novo volta a vazar."""
    texto = _relatorio(tmp_path)
    corpo = texto.split("<details><summary>Correspondência")[0]
    # o nome do arquivo é fato do documento, não identificador interno
    corpo = corpo.replace("teste.pdf", "")
    assert not _GRITO.findall(corpo), _GRITO.findall(corpo)
    assert not _SNAKE.findall(corpo), _SNAKE.findall(corpo)


def test_a_prosa_do_modelo_tambem_e_traduzida(tmp_path):
    """A justificativa vem do LLM, que é *instruído* a citar os códigos — o
    validador `only_assigned_codes` confere que ele só cite os atribuídos.
    Traduzir na saída preserva essa checagem; proibir os códigos a desmontaria.
    """
    texto = _relatorio(tmp_path)
    assert (
        "Data de pagamento antecede Data-base (com) "
        "(cronologia impossível entre as datas do evento)." in texto
    )


def test_a_causa_vem_antes_da_consequencia(tmp_path):
    """Consenso de leitura antes de Validação: quem não conseguiu ler vem antes
    de quem leu e achou o conteúdo contraditório."""
    texto = _relatorio(tmp_path)
    assert texto.index("Consenso de leitura") < texto.index("Validação")


def test_o_apendice_so_lista_o_que_o_relatorio_usa(tmp_path):
    """Um apêndice com o catálogo inteiro pesaria mais que o conteúdo."""
    texto = _relatorio(tmp_path)
    apendice = texto.split("<details><summary>Correspondência")[1]
    assert "`com_date`" in apendice
    assert "`ratio_from`" not in apendice


def test_a_ferramenta_que_reprovou_aparece_na_tabela(tmp_path):
    """O que o relatório antigo não dizia — e agora sem decimal nenhum: qual
    ferramenta reprovou é um fato, e um `0,30` ao lado não o esclarece."""
    texto = _relatorio(tmp_path)
    assert "reprovado em Coerência de datas" in texto
    assert "0,30" not in texto
