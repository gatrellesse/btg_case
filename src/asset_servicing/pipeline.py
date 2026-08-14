"""End-to-end orchestration for one document, and for a batch.

Each document is processed in isolation. A document that raises becomes a
``FAILED`` record carrying its traceback and the batch continues: one
malformed PDF must never cost an operator the other seven records.
"""

from __future__ import annotations

import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .classification import classify
from .extraction import extract, provenance
from .reporter import audit
from .validation import evidence, grounding, repair, triage, verify
from .extraction.consensus import IngestConfig
from .llm import trace
from .extraction.ingest import load
from .models import Disposition, DocumentInfo, EventRecord, EventType, Triage
from .validation.tools import Validators


def process_document(
    path: str | Path,
    validators: Validators,
    thresholds: dict,
    client=None,
    ingest_cfg: IngestConfig | None = None,
    snippet_dir: str | Path | None = None,
    config_dir: str | None = None,
) -> EventRecord:
    path = Path(path)
    try:
        doc = load(path, cfg=ingest_cfg)

        classification = classify.classify(doc, client=client, config_dir=config_dir)
        fields, extra = extract.extract(
            doc, classification.event_type, client=client, config_dir=config_dir
        )

        # Order matters: grounding checks the document as it reads, before
        # normalization rewrites the values it would be looking for.
        fuzzy_min = float((thresholds.get("grounding") or {}).get("fuzzy_min", 0.85))
        grounding.ground(fields, doc.raw_text, fuzzy_min)
        repair.repair(fields)
        repair.order_ratio(fields, classification.event_type)
        repair.derive(fields, classification.event_type)

        # Última chance de preencher, antes de qualquer validador contar
        # ausência: os campos que sobraram voltam ao modelo com as três leituras
        # inteiras. Mesmo passo do motor de grafo, mesmas guardas — o que muda
        # entre os dois é quem chama, não o que é decidido.
        _sweep(fields, doc, classification.event_type, thresholds, client)

        record = EventRecord(
            document=doc.to_info(),
            event_type=classification.event_type,
            classification=classification,
            fields=fields,
            extra_fields=extra,
        )
        record.validations = verify.run_validators(record, validators)
        evidence.note_reference_matches(record.fields, record.validations)
        audit.annotate(record, thresholds)
        record.triage = triage.triage(record, thresholds)
        record.triage.justification = verify.justify(record, validators, client=client)

        if snippet_dir:
            _attach_snippets(record, path, snippet_dir)
        return record

    except Exception as exc:  # noqa: BLE001
        return EventRecord(
            document=DocumentInfo(file_name=path.name, n_pages=0),
            event_type=EventType.OUTRO,
            classification=classify.Classification(event_type=EventType.OUTRO),
            triage=Triage(
                disposition=Disposition.FAILED,
                justification=f"Falha no processamento: {type(exc).__name__}: {exc}",
            ),
            error=traceback.format_exc(limit=5),
        )


def process_document_graph(
    path: str | Path,
    validators: Validators,
    thresholds: dict,
    model: str,
    use_model: bool,
    ingest_cfg: IngestConfig | None = None,
    snippet_dir: str | Path | None = None,
    config_dir: str | None = None,
) -> EventRecord:
    """Run one document through the LangGraph agent.

    Same contract as ``process_document`` — a document that raises becomes a
    FAILED record and the batch survives. The graph replaces the orchestration,
    not the guarantees.
    """
    trace.set_document(Path(path).name)
    from .graph import build_graph
    from .nodes import NodeContext, _as_record
    from .state import AgentState

    path = Path(path)
    try:
        ctx = NodeContext(
            validators=validators,
            thresholds=thresholds,
            model=model,
            use_model=use_model,
            config_dir=config_dir,
            ingest_cfg=ingest_cfg,
        )
        compiled = build_graph(ctx)
        final = compiled.invoke(
            AgentState(path=str(path)),
            config={"configurable": {"thread_id": path.stem}},
        )
        state = AgentState.model_validate(final)

        record = _as_record(state)
        record.triage.justification = state.justification or record.triage.justification
        if state.errors:
            record.error = "; ".join(f"{e.node}: {e.error}" for e in state.errors)
        if snippet_dir:
            _attach_snippets(record, path, snippet_dir)
        return record

    except Exception as exc:  # noqa: BLE001
        return EventRecord(
            document=DocumentInfo(file_name=path.name, n_pages=0),
            event_type=EventType.OUTRO,
            classification=classify.Classification(event_type=EventType.OUTRO),
            triage=Triage(
                disposition=Disposition.FAILED,
                justification=f"Falha no grafo: {type(exc).__name__}: {exc}",
            ),
            error=traceback.format_exc(limit=8),
        )


def _sweep(fields, doc, event_type, thresholds: dict, client) -> None:
    """A varredura final no motor linear. Mesmas guardas do nó do grafo.

    A decisão inteira vive em ``extraction/sweep.py``; o que difere aqui é
    apenas quem faz a chamada — este motor fala com o cliente direto, o grafo
    fala com um agente tipado. Uma varredura que não roda deixa o registro
    exatamente como estava.
    """
    from .extraction import sweep
    from .llm.agents import SWEEPER_INSTRUCTIONS
    from .llm.schemas import RecoveredFields

    faltando = sweep.missing_fields(fields, event_type)
    sozinhos = sweep.uncorroborated_fields(fields)
    conferir = {n: fields[n].value_raw or str(fields[n].value) for n in sozinhos}
    reads = doc.reads
    if (not faltando and not conferir) or not reads or client is None or not client.available():
        return

    try:
        response = client.structured(
            sweep.build_prompt(reads, faltando, conferir),
            RecoveredFields,
            system=SWEEPER_INSTRUCTIONS,
        )
        devolvidos = dict(getattr(response.parsed, "found", None) or {})
    except Exception:  # noqa: BLE001
        return

    notas = sweep.confirm(devolvidos, fields, sozinhos, reads)
    recuperados, descartes = sweep.accept(
        {k: v for k, v in devolvidos.items() if k not in set(sozinhos)}, faltando, reads
    )
    descartes = list(descartes) + [(a, n) for a, n in notas]
    fuzzy_min = float((thresholds.get("grounding") or {}).get("fuzzy_min", 0.85))
    grounding.ground(recuperados, doc.raw_text, fuzzy_min)
    repair.repair(recuperados)
    for name, field in recuperados.items():
        fields[name] = extract._attach_provenance(field, doc)
    for alvo, nota in descartes:
        if alvo and alvo in fields:
            fields[alvo].notes.append(nota)


def _attach_snippets(record: EventRecord, pdf_path: Path, snippet_dir: str | Path) -> None:
    """Cut the source pixels for anything a human will be asked to look at.

    Only for flagged fields: cropping all of them would bloat the report and
    bury the ones that matter.
    """
    directory = Path(snippet_dir) / pdf_path.stem
    for name in record.triage.fields_for_review:
        field = record.fields.get(name)
        if field is None or field.bbox is None:
            continue
        field.snippet_path = provenance.crop(pdf_path, field.bbox, directory / f"{name}.png")


def process_batch(
    paths: list[Path],
    validators: Validators,
    thresholds: dict,
    client=None,
    ingest_cfg: IngestConfig | None = None,
    snippet_dir: str | Path | None = None,
    config_dir: str | None = None,
    workers: int = 4,
    engine: str = "graph",
    model: str | None = None,
    use_model: bool = False,
) -> list[EventRecord]:
    def run(path: Path) -> EventRecord:
        if engine == "graph":
            return process_document_graph(
                path,
                validators,
                thresholds,
                model=model or "",
                use_model=use_model,
                ingest_cfg=ingest_cfg,
                snippet_dir=snippet_dir,
                config_dir=config_dir,
            )
        return process_document(
            path,
            validators,
            thresholds,
            client=client,
            ingest_cfg=ingest_cfg,
            snippet_dir=snippet_dir,
            config_dir=config_dir,
        )

    # Documents are independent, so the batch parallelises trivially. Kept
    # modest: the OCR path is CPU-bound and oversubscribing slows it down.
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(run, paths))
