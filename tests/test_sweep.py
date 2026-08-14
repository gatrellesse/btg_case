"""A varredura final: as três guardas, sem chave de API.

A decisão inteira mora em `extraction/sweep.py` e é pura — o nó do grafo e o
motor linear só fazem a chamada. É por isso que dá para testar aqui o que
importa: que o modelo pode preencher buraco e não pode fazer mais nada.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asset_servicing.extraction import sweep  # noqa: E402
from asset_servicing.models import (  # noqa: E402
    AbsenceReason,
    EventType,
    ExtractedField,
    ReaderKind,
)

T, O, V = ReaderKind.TEXT_LAYER, ReaderKind.OCR_DET, ReaderKind.OCR_VLM

LEITURAS = {
    T: "AVISO AOS ACIONISTAS\nAlíquota de IR de 17,5%\nData de pagamento 21/08/2026",
    O: "AVISO AOS ACIONISTAS\nAliquota de IR de 17,5%\nData de pagamento 21/08/2026",
    V: "AVISO AOS ACIONISTAS\nData de pagamento 2/08/2026",
}


# --------------------------------------------------------------------------
# o que se pede ao modelo
# --------------------------------------------------------------------------


def test_ausencia_declarada_pelo_emissor_nao_e_procurada():
    """`a definir` é uma afirmação do aviso, não um buraco. Mandar o modelo
    procurar o que o documento diz não existir é convidá-lo a inventar."""
    fields = {
        "payment_date": ExtractedField(
            name="payment_date", absence_reason=AbsenceReason.STATED_UNDEFINED
        ),
        "tax_rate": ExtractedField(name="tax_rate", absence_reason=AbsenceReason.NOT_FOUND),
    }
    faltando = sweep.missing_fields(fields, EventType.JCP)
    assert "tax_rate" in faltando
    assert "payment_date" not in faltando


def test_campo_que_nao_se_aplica_ao_tipo_tambem_nao_e_procurado():
    fields = {
        "credit_date": ExtractedField(
            name="credit_date", absence_reason=AbsenceReason.NOT_APPLICABLE
        ),
    }
    assert "credit_date" not in sweep.missing_fields(fields, EventType.JCP)


def test_o_prompt_nomeia_cada_leitura():
    """Concatenar as três apaga a informação que decide: quando elas divergem,
    saber qual mecanismo disse o quê é o que permite ao modelo escolher."""
    prompt = sweep.build_prompt(LEITURAS, ["tax_rate"])
    assert "tax_rate" in prompt
    assert prompt.count("--- LEITURA ·") == 3
    assert prompt.index("CAMADA DE TEXTO") < prompt.index("OCR (") < prompt.index("VLM (")


# --------------------------------------------------------------------------
# as três guardas
# --------------------------------------------------------------------------


def test_so_preenche_buraco_nunca_reescreve():
    """A guarda que faz o passo ser seguro: o modelo devolve um campo que não
    estava ausente e ele é descartado, com a tentativa registrada."""
    aceitos, notas = sweep.accept(
        {"tax_rate": "17,5%", "issuer": "OUTRA COMPANHIA S.A."}, ["tax_rate"], LEITURAS
    )
    assert set(aceitos) == {"tax_rate"}
    assert any("issuer" in nota for _, nota in notas)


def test_valor_que_nenhuma_leitura_contem_e_fabricacao():
    """A guarda contra alucinação. O modelo tem liberdade para escrever valor
    neste passo; o que ele não tem é liberdade para escrever valor que o
    documento não traz."""
    aceitos, notas = sweep.accept({"tax_rate": "15%"}, ["tax_rate"], LEITURAS)
    assert aceitos == {}
    assert notas and notas[0][0] == "tax_rate"
    assert "não aparece em nenhuma" in notas[0][1]


def test_o_degrau_sai_de_quem_continha_o_valor():
    """Não é um degrau fixo por ter vindo da varredura: é a mesma medida do
    resto do pipeline — quantos mecanismos independentes sustentam o valor."""
    aceitos, _ = sweep.accept({"tax_rate": "17,5%"}, ["tax_rate"], LEITURAS)
    campo = aceitos["tax_rate"]
    assert campo.corroborating_kinds == [T, O]
    assert campo.reading_level == "muito_alta"
    assert campo.recovered is True


def test_valor_em_um_unico_leitor_de_pixel_nasce_sem_corroboracao():
    """O caminho que leva a triagem a levantar NO_CORROBORATION sozinha: a
    varredura não cria confiança, ela só encontra."""
    aceitos, _ = sweep.accept({"payment_date": "2/08/2026"}, ["payment_date"], LEITURAS)
    campo = aceitos["payment_date"]
    assert campo.corroborating_kinds == [V]
    assert campo.reading_level == "baixa"


def test_o_ocr_perder_espaco_nao_derruba_a_conferencia():
    """A comparação é achatada porque o OCR perde espaço o tempo todo, e uma
    conferência que dependesse deles rejeitaria leitura correta."""
    assert sweep.corroborating_reads("Alíquota  de   IR", LEITURAS) == [T, O]


def test_dicionario_vazio_e_resposta_correta():
    """O caso mais comum: os três mecanismos leram o documento inteiro e o dado
    não está lá. Nada acontece, e nada precisa acontecer."""
    aceitos, notas = sweep.accept({}, ["tax_rate"], LEITURAS)
    assert aceitos == {} and notas == []


# --------------------------------------------------------------------------
# o que a triagem faz com o que a varredura não achou
# --------------------------------------------------------------------------


def _registro(fields):
    from asset_servicing.models import (
        Classification,
        DocumentInfo,
        EventRecord,
    )

    return EventRecord(
        document=DocumentInfo(file_name="x.pdf", n_pages=1),
        event_type=EventType.JCP,
        classification=Classification(event_type=EventType.JCP),
        fields={f.name: f for f in fields},
    )


def test_opcional_que_ninguem_achou_deixou_de_ser_silencioso():
    """Antes da varredura, um opcional ausente não gerava achado: podia ser o
    extrator que não olhou direito. Depois dela, três mecanismos leram o
    documento inteiro e o dado não estava lá — isso é um fato sobre o aviso, e
    o operador tem de vê-lo."""
    from asset_servicing.models import Disposition, ReasonCode
    from asset_servicing.validation.evidence import load_thresholds
    from asset_servicing.validation.triage import triage

    rec = _registro([
        ExtractedField(name="tax_rate", absence_reason=AbsenceReason.NOT_FOUND),
    ])
    out = triage(rec, load_thresholds())
    assert ReasonCode.FIELD_NOT_FOUND in {r.code for r in out.reasons}
    assert "tax_rate" in out.fields_for_review
    assert out.disposition == Disposition.REVIEW


def test_obrigatorio_ausente_nao_e_reportado_duas_vezes():
    """`check_required_fields` já nomeia o campo e já bloqueia. Repetir aqui
    poria duas linhas dizendo a mesma coisa sobre a mesma célula."""
    from asset_servicing.models import ReasonCode
    from asset_servicing.validation.evidence import load_thresholds
    from asset_servicing.validation.triage import triage

    rec = _registro([
        ExtractedField(name="event_type_label", absence_reason=AbsenceReason.NOT_FOUND),
    ])
    out = triage(rec, load_thresholds())
    assert ReasonCode.FIELD_NOT_FOUND not in {r.code for r in out.reasons}
    assert "event_type_label" in out.fields_for_review


def test_o_que_nao_se_aplica_ao_tipo_continua_sem_achado():
    from asset_servicing.models import Disposition
    from asset_servicing.validation.evidence import load_thresholds
    from asset_servicing.validation.triage import triage

    rec = _registro([
        ExtractedField(name="credit_date", absence_reason=AbsenceReason.NOT_APPLICABLE),
    ])
    out = triage(rec, load_thresholds())
    assert out.reasons == []
    assert out.disposition == Disposition.AUTO


# --------------------------------------------------------------------------
# confirmação: o que um mecanismo só sustentava
# --------------------------------------------------------------------------


def _sozinho(name, value, kind=O):
    return ExtractedField(
        name=name, value=value, value_raw=value, evidence_text=value,
        reader_kind=kind, reading_level="baixa", corroborating_kinds=[kind],
    )


def test_so_entra_na_conferencia_quem_tem_um_mecanismo_de_pixel_so():
    fields = {
        "isin": _sozinho("isin", "BRTLNRACNPR2"),
        "com_date": ExtractedField(
            name="com_date", value="2026-06-22", reading_level="media",
            corroborating_kinds=[O, V],
        ),
        "issuer": ExtractedField(
            name="issuer", value="X S.A.", reading_level="alta",
            corroborating_kinds=[T],
        ),
    }
    assert sweep.uncorroborated_fields(fields) == ["isin"]


def test_confirmacao_promove_o_degrau_sem_tocar_no_valor():
    """A região não casou na votação; a leitura casou. Dois mecanismos passam a
    sustentar o valor, e a triagem deixa de levantar NO_CORROBORATION — não por
    perdão, mas porque a corroboração passou a existir."""
    fields = {"payment_date": _sozinho("payment_date", "21/08/2026")}
    notas = sweep.confirm(
        {"payment_date": "21/08/2026"}, fields, ["payment_date"], LEITURAS
    )
    campo = fields["payment_date"]
    assert campo.value == "21/08/2026"           # intocado
    assert campo.corroborating_kinds == [T, O]
    assert campo.reading_level == "muito_alta"
    assert notas == []


def test_ruido_de_ocr_nao_impede_a_confirmacao():
    """O que a checagem literal não resolve, e o motivo de haver um modelo aqui:
    `R$ 0,1124300000` e `0,112430000` são a mesma quantia."""
    assert sweep.same_value("R$ 0,1124300000", "0,112430000")
    assert sweep.same_value("Alíquota  de IR", "aliquota de ir")
    assert not sweep.same_value("17,5%", "15%")


def test_confirmacao_nunca_reescreve_valor_extraido():
    """O valor extraído passou por grounding, reparo e validação. Uma releitura
    de texto não desfaz isso — a divergência vira nota e o campo segue como
    está, já em revisão por não ter corroboração."""
    fields = {"payment_date": _sozinho("payment_date", "21/08/2026")}
    notas = sweep.confirm(
        {"payment_date": "23/08/2026"}, fields, ["payment_date"], LEITURAS
    )
    assert fields["payment_date"].value == "21/08/2026"
    assert fields["payment_date"].corroborating_kinds == [O]
    assert notas and "divergência registrada" in notas[0][1]


def test_confirmacao_que_nao_acha_segunda_leitura_nao_promove_nada():
    fields = {"tax_rate": _sozinho("tax_rate", "17,5%", kind=V)}
    leituras = {V: "Aliquota de IR de 17,5%"}
    notas = sweep.confirm({"tax_rate": "17,5%"}, fields, ["tax_rate"], leituras)
    assert fields["tax_rate"].corroborating_kinds == [V]
    assert fields["tax_rate"].reading_level == "baixa"
    assert notas and "segue sem corroboração" in notas[0][1]


def test_campo_perguntado_e_nao_devolvido_deixa_rastro():
    """"A varredura procurou e não achou" e "a varredura nem rodou" pedem ações
    diferentes de quem revisa. Sem a nota, o registro não distingue as duas."""
    fields = {"isin": _sozinho("isin", "BRTLNRACNPR2")}
    sweep.confirm({}, fields, ["isin"], LEITURAS)
    assert any("não localizou" in n for n in fields["isin"].notes)
