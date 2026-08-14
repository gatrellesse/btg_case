"""Captura entrada e saída reais de cada nó do grafo.

Mesmo princípio de `tool_examples.py`: nada é inventado. Cada nó é embrulhado
antes de o grafo ser construído, e o que fica registrado é o que ele de fato
recebeu e devolveu ao processar documentos do lote.

Roda mais de um documento porque dois nós são condicionais — `reprompt_extract`
só entra quando falta evidência e `disambiguate` só quando um valor tem mais de
um candidato compatível. Num documento limpo eles nunca aparecem, e um exemplo
ausente é mais honesto que um exemplo fabricado.

    PYTHONPATH=src python experiments/node_examples.py   # -> out/node_examples.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _source import essence

from asset_servicing import nodes as N
from asset_servicing.llm.agents import DEFAULT_MODEL, load_dotenv
from asset_servicing.validation.evidence import load_thresholds
from asset_servicing.validation.tools import Validators

DOCS = Path("Case AI Dev - Envio/documents")
GOLDEN = Path("Case AI Dev - Envio/golden_records/golden records.csv")
OUT = Path("out/node_examples.json")

#: Documentos escolhidos pelo que exercitam, não por serem representativos:
#: um limpo, um com incoerência de datas, um escaneado.
#: Três bastam para exercitar os doze nós que rodam. Já foi medido no lote
#: inteiro: `reprompt_extract` e `disambiguate` não disparam em nenhum dos oito
#: documentos, então varrer os oito só custa tempo e não acrescenta exemplo.
SAMPLE = [
    "01_energetica_vale_tiete_dividendo.pdf",
    "05_aurora_saneamento_dividendo_datas.pdf",
    "07_telecom_norte_jcp_SCAN.pdf",
]

FACTORIES = {
    "make_ingest": "ingest",
    "make_rule_classifier": "rule_classifier",
    "make_text_classifier": "text_classifier",
    "make_consensus": "consensus",
    "make_rule_extractor": "rule_extractor",
    "make_text_extractor": "text_extractor",
    "make_merge": "merge",
    "make_grounding": "grounding",
    "make_reprompt_extract": "reprompt_extract",
    "make_repair": "repair",
    "make_sweep": "sweep",
    "make_validate": "validate",
    "make_disambiguate": "disambiguate",
    "make_triage": "triage",
    "make_reporter": "reporter",
}

WHY = {
    "ingest": "lê a página pelas três modalidades e funde região a região",
    "rule_classifier": "opinião determinística sobre o tipo, sem modelo — prior fraco é resultado válido",
    "text_classifier": "a mesma pergunta ao LLM, sobre o texto extraído",
    "consensus": "unanimidade ou nada: qualquer divergência vai para revisão, porque errar o tipo muda o tratamento tributário",
    "rule_extractor": "âncoras de rótulo do config, offline — o segundo par de olhos que o LLM enfrenta",
    "text_extractor": "extração tipada por schema do tipo de evento",
    "merge": "reúne candidatos por campo; não vota, guarda — divergência entre extratores não se resolve por maioria",
    "grounding": "confere cada evidência contra o texto CRU, antes de qualquer normalização",
    "reprompt_extract": "muda a entrada e pede de novo; reexecutar com a mesma entrada devolveria a mesma resposta",
    "repair": "normaliza para a convenção pt-BR e deriva o que a regra do evento permite",
    "sweep": "relê as três leituras inteiras atrás do que faltou e do que um mecanismo só sustentava; três guardas em código decidem o que entra",
    "validate": "as seis tools determinísticas, sempre todas, com args e retorno no registro",
    "disambiguate": "escolhe entre candidatos o único compatível com o resto das datas do evento",
    "triage": "o veredito em código: severidade dos reason codes decide AUTO, REVIEW ou BLOCKED",
    "reporter": "escreve a narrativa; a disposição já está calculada e ele não a toca",
}

_captured: dict[str, dict] = {}


def _brief(value, depth: int = 0):
    """Resumo legível: coleção vira contagem + primeiro item."""
    if depth > 2:
        return "…"
    if hasattr(value, "model_dump"):
        return _brief(value.model_dump(mode="json"), depth + 1)
    if isinstance(value, dict):
        if len(value) > 4:
            first = next(iter(value))
            return {"n": len(value), "chaves": list(value)[:4],
                    first: _brief(value[first], depth + 1)}
        return {k: _brief(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        if not value:
            return []
        return {"n": len(value), "primeiro": _brief(value[0], depth + 1)}
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, str):
        return value[:110]
    if value is None or isinstance(value, (int, bool)):
        return value
    return type(value).__name__


def _state_in(state) -> dict:
    """O que o nó tinha à mão. Não o estado inteiro — o que muda de nó para nó."""
    doc = state.document
    out = {
        "documento": state.file_name or Path(state.path).name,
        "blocos": len(doc.blocks) if doc else 0,
    }
    if state.votes:
        out["votos"] = [
            f"{v.modality}={v.event_type.value if v.event_type else '—'}" for v in state.votes
        ]
    if state.classification:
        out["tipo"] = state.classification.event_type.value
    if state.candidates:
        out["candidatos"] = len(state.candidates)
    if state.fields:
        out["campos"] = len(state.fields)
    if state.validations:
        out["validações"] = len(state.validations)
    if state.triage:
        out["disposição"] = state.triage.disposition.value
    return out


def _wrap_all() -> None:
    for factory_name, node in FACTORIES.items():
        original = getattr(N, factory_name)

        def make(ctx, _original=original, _node=node):
            fn = _original(ctx)

            def wrapped(state):
                before = _state_in(state)
                out = fn(state)
                if _node not in _captured and out:
                    _captured[_node] = {
                        "layer": "Grafo",
                        "tool": _node,
                        "why": WHY.get(_node, ""),
                        "input": before,
                        # o painel do meio: o código que transformou um no outro
                        "logic": essence(_original),
                        "output": {k: _brief(v) for k, v in out.items()},
                    }
                return out

            return wrapped

        setattr(N, factory_name, make)


def main() -> None:
    load_dotenv()
    _wrap_all()
    # importado DEPOIS do wrap: o pipeline constrói o grafo a cada documento,
    # então ele já pega as fábricas embrulhadas
    from asset_servicing.pipeline import process_document_graph

    th = load_thresholds()
    validators = Validators(GOLDEN, th)
    for name in SAMPLE:
        process_document_graph(DOCS / name, validators, th,
                               model=DEFAULT_MODEL, use_model=True)
        print(f"  {name[:40]:40s} nós capturados: {len(_captured)}", flush=True)

    missing = [n for n in FACTORIES.values() if n not in _captured]
    ordered = [_captured[n] for n in FACTORIES.values() if n in _captured]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(ordered, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n{len(ordered)} nós → {OUT}")
    if missing:
        print(f"sem exemplo (não rodaram nestes documentos): {missing}")


if __name__ == "__main__":
    main()
