"""LangGraph wiring.

What the graph buys over a straight function chain, concretely:

* **fan-out/fan-in** — three classifiers and three extractors run as branches
  that rejoin at a node which waits for all of them, without hand-rolled
  concurrency;
* **conditional edges** — "send this back" is an edge in the topology, visible
  and inspectable, not an ``if`` buried three calls deep;
* **bounded cycles** — every loop is guarded by a counter in the state, so the
  bound is data you can print, not control flow you have to trace;
* **checkpointing** — the state after each node is inspectable, which is what
  makes the thing debuggable live.

What it deliberately does *not* buy here: durable execution and human
interrupts. Review is delivered through the exceptions report, so
``interrupt()`` would be machinery with no consumer.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from . import nodes
from .nodes import NodeContext
from .state import AgentState

MAX_REPROMPTS = 1
MAX_REPAIRS = 1


def _route_after_grounding(ctx: NodeContext):
    def route(state: AgentState) -> str:
        """Send extraction back only when a critical value cannot be found in
        the document at all — a fabrication, not a weak reading — and only once."""
        if not ctx.use_model or state.reprompt_count >= MAX_REPROMPTS:
            return "repair"
        return "reprompt" if nodes.ungrounded_critical(state.fields, ctx.thresholds) else "repair"

    return route


def _route_after_validate(ctx: NodeContext):
    def route(state: AgentState) -> str:
        """Try constraint-based disambiguation once, and only if there is an
        unresolved reading for the constraints to act on."""
        if state.repair_count >= MAX_REPAIRS:
            return "triage"
        unresolved = any(
            r.candidates for field in state.fields.values() for r in field.repairs
        )
        return "disambiguate" if unresolved else "triage"

    return route


def build_graph(ctx: NodeContext, checkpointer: MemorySaver | None = None):
    builder = StateGraph(AgentState)

    builder.add_node("ingest", nodes.make_ingest(ctx))

    builder.add_node("rule_classifier", nodes.make_rule_classifier(ctx))
    builder.add_node("text_classifier", nodes.make_text_classifier(ctx))
    builder.add_node("consensus", nodes.make_consensus(ctx))

    builder.add_node("rule_extractor", nodes.make_rule_extractor(ctx))
    builder.add_node("text_extractor", nodes.make_text_extractor(ctx))
    builder.add_node("merge", nodes.make_merge(ctx))

    builder.add_node("grounding", nodes.make_grounding(ctx))
    builder.add_node("reprompt_extract", nodes.make_reprompt_extract(ctx))
    builder.add_node("repair", nodes.make_repair(ctx))

    builder.add_node("sweep", nodes.make_sweep(ctx))
    builder.add_node("validate", nodes.make_validate(ctx))
    builder.add_node("disambiguate", nodes.make_disambiguate(ctx))
    builder.add_node("triage", nodes.make_triage(ctx))
    builder.add_node("reporter", nodes.make_reporter(ctx))

    builder.add_edge(START, "ingest")

    # Classification: the modalities that read the *extracted* text, in
    # parallel, rejoining at consensus. Vision was removed from here — it
    # re-rendered the page instead of reading the cascade's blocks, which put a
    # fourth reader inside the classification layer and made the "equal weight"
    # rule count pixel evidence twice on a scan.
    for node in ("rule_classifier", "text_classifier"):
        builder.add_edge("ingest", node)
        builder.add_edge(node, "consensus")

    # Extração: mesma forma. Um nó que não pode contribuir (sem modelo)
    # devolve nada em vez de ser contornado por aresta — topologia fixa é mais
    # fácil de ler e de depurar.
    #
    # A modalidade de imagem saiu daqui: ela re-renderizava a página para um VLM
    # depois de a ingestão já ter lido aquela página com um VLM, e a leitura
    # dela já está nos blocos e em `alternatives`. Medido no único scan do lote:
    # nenhum valor mudou, e ela levantava três EXTRACTOR_DISAGREEMENT sobre
    # campos cujo valor final era idêntico com ou sem ela.
    for node in ("rule_extractor", "text_extractor"):
        builder.add_edge("consensus", node)
        builder.add_edge(node, "merge")

    builder.add_edge("merge", "grounding")
    builder.add_conditional_edges(
        "grounding",
        _route_after_grounding(ctx),
        {"reprompt": "reprompt_extract", "repair": "repair"},
    )
    builder.add_edge("reprompt_extract", "grounding")

    # A varredura entra entre o reparo e a validação, e não depois da triagem:
    # quem conta ausência é o `check_required_fields`, e um campo recuperado
    # depois dele sairia com valor e com o achado dizendo que não veio.
    builder.add_edge("repair", "sweep")
    builder.add_edge("sweep", "validate")
    builder.add_conditional_edges(
        "validate",
        _route_after_validate(ctx),
        {"disambiguate": "disambiguate", "triage": "triage"},
    )
    builder.add_edge("disambiguate", "validate")

    builder.add_edge("triage", "reporter")
    builder.add_edge("reporter", END)

    return builder.compile(checkpointer=checkpointer or MemorySaver())


def mermaid(ctx: NodeContext) -> str:
    """The graph's own rendering of itself — useful in the README and as a
    check that the drawn diagram still matches the code."""
    return build_graph(ctx).get_graph().draw_mermaid()
