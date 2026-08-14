"""The orchestration layer: consensus, confrontation, bounded cycles.

These test the *decisions the graph makes*, not the business rules underneath —
those live in test_rules.py. Every case here is built in code, so none of it
needs a model, a key, or the sample batch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asset_servicing.validation.evidence import load_thresholds  # noqa: E402
from asset_servicing.graph import MAX_REPAIRS, MAX_REPROMPTS, build_graph  # noqa: E402
from asset_servicing.models import (  # noqa: E402
    Classification,
    EventType,
    ExtractedField,
    ReaderKind,
    Repair,
)
from asset_servicing.nodes import NodeContext, make_consensus, make_disambiguate, make_merge  # noqa: E402
from asset_servicing.state import AgentState, ClassifierVote, FieldCandidate  # noqa: E402
from asset_servicing.validation.tools import Validators  # noqa: E402

GOLDEN = """emissor,cnpj,isin,ticker,classe,segmento_listagem,status
Companhia Alfa S.A.,11.111.111/0001-11,BRALFAACNOR1,ALFA3,ON,Novo Mercado,ativo
"""


@pytest.fixture
def ctx(tmp_path: Path) -> NodeContext:
    golden = tmp_path / "golden.csv"
    golden.write_text(GOLDEN, encoding="utf-8")
    thresholds = load_thresholds()
    return NodeContext(
        validators=Validators(golden, thresholds),
        thresholds=thresholds,
        model="anthropic:claude-sonnet-5",
        use_model=False,
    )


class FakeDoc:
    """Enough of a SourceDocument for the nodes under test."""

    raw_text = "aviso aos acionistas — juros sobre o capital proprio"
    blocks: list = []


def state_with(**kwargs) -> AgentState:
    return AgentState(path="/tmp/x.pdf", document=FakeDoc(), **kwargs)


# --------------------------------------------------------------------------
# consensus
# --------------------------------------------------------------------------


def vote(modality, event_type, available=True) -> ClassifierVote:
    return ClassifierVote(
        modality=modality, event_type=event_type, available=available
    )


def test_unanimous_vote_is_recorded_as_unanimous(ctx):
    state = state_with(
        votes=[vote("rule", EventType.JCP), vote("text", EventType.JCP)]
    )
    out = make_consensus(ctx)(state)
    assert out["unanimous"] is True
    assert out["classification"].unanimous is True
    assert out["classification"].event_type == EventType.JCP


def test_any_dissent_routes_to_review(ctx):
    """Duas modalidades discordando é desacordo, e classificar errado o evento
    muda o tratamento tributário. O desempate era pela confiança que cada
    modalidade declarava de si — número que ninguém mediu decidindo o tipo do
    evento. Decide o prior determinístico, que leu os marcadores do aviso."""
    state = state_with(
        votes=[
            vote("rule", EventType.JCP),
            vote("text", EventType.DIVIDENDO),
        ]
    )
    out = make_consensus(ctx)(state)
    assert out["unanimous"] is False
    assert out["classification"].unanimous is False
    # o desempate é do prior: o texto do aviso fala em juros sobre capital próprio
    assert out["classification"].event_type == EventType.JCP


def test_no_modality_has_a_veto(ctx):
    """Peso igual: o voto determinístico pode perder, como qualquer outro. Ele
    não tem veto por ser determinístico — e quem desempata não é ele, é o que
    os marcadores do próprio aviso dizem."""
    state = state_with(
        votes=[
            vote("rule", EventType.DIVIDENDO),
            vote("text", EventType.JCP),
        ]
    )
    out = make_consensus(ctx)(state)
    assert out["classification"].event_type == EventType.JCP


def test_unavailable_modality_does_not_count_against_quorum(ctx):
    """Two readers agreeing is unanimity among those that ran; a modality that
    could not run is not a dissenting vote."""
    state = state_with(
        votes=[
            vote("rule", EventType.JCP),
            ClassifierVote(modality="text", available=False),
        ]
    )
    out = make_consensus(ctx)(state)
    assert out["unanimous"] is True


def test_no_usable_vote_degrades_instead_of_raising(ctx):
    state = state_with(votes=[ClassifierVote(modality="rule", available=False)])
    out = make_consensus(ctx)(state)
    assert out["classification"].event_type == EventType.OUTRO
    assert out["unanimous"] is False


# --------------------------------------------------------------------------
# extractor confrontation
# --------------------------------------------------------------------------


def candidate(extractor, value) -> FieldCandidate:
    return FieldCandidate(extractor=extractor, value_raw=value, evidence_text=value)


def merged(ctx, pairs):
    state = state_with(
        classification=Classification(event_type=EventType.JCP), candidates=pairs
    )
    return make_merge(ctx)(state)


def test_agreeing_extractors_are_recorded_as_corroboration(ctx):
    out = merged(
        ctx,
        [
            ("gross_per_share", candidate("rule", "R$ 0,17")),
            ("gross_per_share", candidate("text", "R$ 0,17")),
        ],
    )
    field = out["fields"]["gross_per_share"]
    assert out["disagreements"] == []
    assert any("concordam" in note for note in field.notes)


def test_disagreement_on_critical_field_keeps_the_anchored_reading(ctx):
    """Not a majority vote. On a financial field, voting is how a wrong value
    enters the base looking like consensus — so the record keeps both readings,
    escala o campo, e o valor que fica é o ancorado num rótulo do aviso, que é
    conferível no recorte. Era a leitura de menor confiança declarada, o que
    fazia o desempate depender do autorrelato do extrator."""
    out = merged(
        ctx,
        [
            ("gross_per_share", candidate("rule", "R$ 0,17")),
            ("gross_per_share", candidate("text", "R$ 0,71")),
        ],
    )
    field = out["fields"]["gross_per_share"]
    assert out["disagreements"] == ["gross_per_share"]
    assert field.value_raw == "R$ 0,17"
    assert any("0,17" in note and "0,71" in note for note in field.notes)


def test_disagreement_on_descriptive_field_is_not_escalated(ctx):
    """Confrontation is scoped to fields where being wrong costs something."""
    out = merged(
        ctx,
        [
            ("share_class", candidate("rule", "ON")),
            ("share_class", candidate("text", "PN")),
        ],
    )
    assert out["disagreements"] == []
    assert out["fields"]["share_class"].value_raw == "ON"


def test_reextraction_supersedes_the_earlier_reading(ctx):
    """The reducer only appends, so the later candidate from the same extractor
    has to win — otherwise a re-prompt could never correct anything."""
    out = merged(
        ctx,
        [
            ("com_date", candidate("text", "inventado")),
            ("com_date", candidate("text", "12/06/2026")),
        ],
    )
    assert out["fields"]["com_date"].value_raw == "12/06/2026"


# --------------------------------------------------------------------------
# constraint-based disambiguation
# --------------------------------------------------------------------------


def ambiguous_field(name, candidates) -> ExtractedField:
    field = ExtractedField(name=name, reader_kind=ReaderKind.OCR_DET, reading_level="baixa")
    field.repairs.append(Repair(field=name, rule="data_ambigua_ocr", candidates=candidates))
    return field


def test_chronology_resolves_a_reading_the_ocr_could_not(ctx):
    """Inference from evidence, not a guess: only one candidate is compatible
    with the rest of the event's dates."""
    state = state_with(
        fields={
            "com_date": ExtractedField(name="com_date", value="2026-08-15"),
            "ex_date": ExtractedField(name="ex_date", value="2026-08-18"),
            "payment_date": ambiguous_field(
                "payment_date", ["02/08/2026", "12/08/2026", "22/08/2026"]
            ),
        }
    )
    out = make_disambiguate(ctx)(state)
    assert out["fields"]["payment_date"].value == "22/08/2026"
    assert out["repair_count"] == 1


def test_ambiguity_survives_when_constraints_do_not_discriminate(ctx):
    """Two candidates still possible means it stays unresolved and goes to a
    human — the constraint narrowed nothing decisive."""
    state = state_with(
        fields={
            "com_date": ExtractedField(name="com_date", value="2026-06-22"),
            "ex_date": ExtractedField(name="ex_date", value="2026-06-23"),
            "payment_date": ambiguous_field(
                "payment_date", ["02/08/2026", "12/08/2026", "22/08/2026"]
            ),
        }
    )
    out = make_disambiguate(ctx)(state)
    assert out["fields"]["payment_date"].value is None


# --------------------------------------------------------------------------
# topology
# --------------------------------------------------------------------------


def test_graph_has_the_layers_it_claims(ctx):
    nodes = set(build_graph(ctx).get_graph().nodes)
    for expected in (
        "ingest",
        "rule_classifier",
        "text_classifier",
        "consensus",
        "rule_extractor",
        "text_extractor",
        "merge",
        "grounding",
        "repair",
        "validate",
        "triage",
        "reporter",
    ):
        assert expected in nodes


def test_every_cycle_is_bounded(ctx):
    """A state graph makes it easy to add a loop that resamples forever. Both
    cycles here are capped by a counter that lives in the state."""
    assert MAX_REPROMPTS == 1
    assert MAX_REPAIRS == 1


def test_graph_renders_its_own_topology(ctx):
    diagram = build_graph(ctx).get_graph().draw_mermaid()
    assert "consensus" in diagram and "merge" in diagram
    assert "reprompt_extract" in diagram  # o ciclo aparece no desenho


def test_mesma_quantia_com_outro_recorte_nao_e_desacordo(ctx):
    """O caso medido no doc 08: as regras citaram `R$ 7,820000` e o modelo
    `R$ 7,820000 por ação`. Mesma quantia, mesmo valor final — o que difere é o
    recorte da citação. Comparadas como texto viravam desacordo, e um humano
    gastava tempo para concluir que os dois leram o mesmo número."""
    out = merged(
        ctx,
        [
            ("cost_basis", candidate("rule", "R$ 7,820000")),
            ("cost_basis", candidate("text", "R$ 7,820000 por ação")),
        ],
    )
    assert out["disagreements"] == []
    assert any("concordam" in n for n in out["fields"]["cost_basis"].notes)


def test_proporcao_lida_ao_contrario_nao_e_desacordo(ctx):
    """`{1, 20}` e `{20, 1}` são a mesma proporção lida em sentidos opostos, e o
    sentido não é achado de extração: é regra do evento, aplicada em
    `repair.order_ratio`."""
    out = merged(
        ctx,
        [
            ("ratio_from", candidate("rule", "1")),
            ("ratio_to", candidate("rule", "20")),
            ("ratio_from", candidate("text", "20")),
            ("ratio_to", candidate("text", "1")),
        ],
    )
    assert out["disagreements"] == []


def test_proporcao_com_numero_diferente_continua_sendo_desacordo(ctx):
    """A guarda que impede o conserto de engolir desacordo de verdade: aqui os
    extratores discordam sobre o que o documento diz, não sobre o sentido."""
    out = merged(
        ctx,
        [
            ("ratio_from", candidate("rule", "1")),
            ("ratio_to", candidate("rule", "20")),
            ("ratio_from", candidate("text", "1")),
            ("ratio_to", candidate("text", "10")),
        ],
    )
    assert set(out["disagreements"]) == {"ratio_from", "ratio_to"}
