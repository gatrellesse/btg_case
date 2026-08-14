"""Stage 8 — the outputs an operator actually works from.

Three artefacts: one JSON per document, a short exceptions report, and a run
summary. The report carries a cropped image of each flagged region, which is
what lets the enunciado's requirement be met literally — auditing a value
without reopening the document.

The report is written entirely in Portuguese, with no internal identifier in
the body: it is read by an Asset Servicing operator, and `com_date`,
`CRITICAL_FIELD_LOW_CONFIDENCE` and `text_layer` require knowing the
implementation to audit a value. The identifiers stay in the per-document JSON,
which is the machine layer, and the correspondence between the two is generated
from `config/glossario.yaml` into a single appendix.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from . import labels
from ..models import LAYER_ORDER, Disposition, EventRecord, Severity

_MARK = {Disposition.AUTO: "✅", Disposition.REVIEW: "⚠️", Disposition.BLOCKED: "🛑",
         Disposition.FAILED: "💥"}


def write_records(records: list[EventRecord], out_dir: str | Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for record in records:
        path = out_dir / f"{Path(record.document.file_name).stem}.json"
        path.write_text(
            json.dumps(record.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        written.append(path)
    return written


def _titulo(record: EventRecord) -> str:
    """O emissor, quando extraído; senão o arquivo. Nome de arquivo é fato, não
    identificador interno — fica como subtítulo em qualquer caso."""
    issuer = record.fields.get("issuer")
    if issuer is not None and issuer.value:
        return str(issuer.value)
    return Path(record.document.file_name).stem


def _leitura_cell(field) -> str:
    """Quem leu — e, se for o caso, quem reprovou.

    Sem esta distinção, um campo lido pelos três mecanismos e derrubado pela
    cronologia do documento lia-se como "não conseguimos ler", e a ação que isso
    sugere é abrir o PDF: exatamente a errada.
    """
    audit = field.audit
    if audit is None:
        return "—"
    if audit.reading.cut_by:
        quem = ", ".join(labels.ferramenta(t) for t in dict.fromkeys(audit.reading.cut_by))
        return f"reprovado em {quem}"
    return labels.nivel_curto(audit.reading.level)


def _rastreio_cell(field) -> str:
    if field.page is None:
        return "não localizado na página"
    leitor = labels.leitor(field.reader_kind) if field.reader_kind else "—"
    return (
        f"pág. {field.page} · {leitor} · "
        f"evidência {labels.evidencia(field.grounding.status)}"
    )


def _detalhe(record: EventRecord, name: str) -> list[str]:
    """O que não cabe em célula: a leitura, a base, a evidência, o recorte."""
    from .audit import curate_notes

    field = record.fields.get(name)
    if field is None:
        return []
    rotulo = labels.campo(name)
    itens: list[str] = []

    audit = field.audit
    if audit is not None:
        leitura = f"leitura: {labels.nivel_de_leitura(audit.reading.level)}"
        if audit.reading.cut_by:
            quem = ", ".join(labels.ferramenta(t) for t in dict.fromkeys(audit.reading.cut_by))
            leitura += f" — mas **{quem}** reprovou este campo"
        itens.append(leitura)
        itens.append(
            f"base de referência: {labels.base_de_referencia(audit.golden.status)}"
            + (f" (esperado “{audit.golden.expected}”)" if audit.golden.expected else "")
        )

    if field.page is not None:
        box = ""
        if field.bbox:
            box = (f" · região ({field.bbox.x0:.0f}, {field.bbox.y0:.0f})–"
                   f"({field.bbox.x1:.0f}, {field.bbox.y1:.0f})")
        leitor = labels.leitor(field.reader_kind) if field.reader_kind else "—"
        itens.append(
            f"pág. {field.page}{box} · {leitor} · evidência "
            f"{labels.evidencia(field.grounding.status)}"
        )
    if field.evidence_text:
        # a evidência guarda a quebra de linha do PDF; ela não diz nada aqui e
        # quebraria o item ao meio
        itens.append(f"evidência: “{' '.join(field.evidence_text.split())[:220]}”")

    for step in field.repairs:
        if step.candidates:
            itens.append(f"leituras possíveis: {', '.join(step.candidates)} — **não resolvido**")
        else:
            itens.append(f"padronizado: “{step.value_from}” → “{step.value_to}”")

    itens += curate_notes(field)

    lines = [f"<details><summary>{rotulo} — evidência e notas</summary>", ""]
    lines += [f"- {item}" for item in itens]
    if field.snippet_path:
        lines += ["", f"![{rotulo}]({Path(field.snippet_path).as_posix()})"]
    lines += ["", "</details>", ""]
    return lines


def _panorama(records: list[EventRecord]) -> list[str]:
    """Todos os documentos, inclusive os liberados — hoje invisíveis no relatório."""
    lines = [
        "| documento | evento | situação | pendências |",
        "|---|---|---|---|",
    ]
    order = {Disposition.BLOCKED: 0, Disposition.FAILED: 1, Disposition.REVIEW: 2}
    for record in sorted(records, key=lambda r: order.get(r.triage.disposition, 9)):
        por_camada = Counter(labels.camada(r.layer) for r in record.triage.reasons)
        pend = " · ".join(f"{n} de {c.lower()}" for c, n in por_camada.most_common()) or "—"
        lines.append(
            f"| {_titulo(record)} | {labels.evento(record.event_type)} | "
            f"{_MARK[record.triage.disposition]} {labels.disposicao(record.triage.disposition)} | {pend} |"
        )
    return lines + [""]


def _apendice(records: list[EventRecord]) -> list[str]:
    """O único mapeamento visível entre os termos e os códigos do sistema."""
    usados: set[str] = set()
    for record in records:
        if record.triage.disposition == Disposition.AUTO:
            continue
        for reason in record.triage.reasons:
            usados |= {reason.code.value, reason.layer.value, reason.error_type.value}
            if reason.field:
                usados.add(reason.field)
        for name in record.triage.fields_for_review:
            usados.add(name)
            field = record.fields.get(name)
            if field is not None:
                if field.reader_kind:
                    usados.add(field.reader_kind.value)
                if field.audit:
                    usados |= set(field.audit.reading.cut_by)

    lines = [
        "---",
        "",
        "<details><summary>Correspondência entre os termos deste relatório e os "
        "códigos do JSON</summary>",
        "",
        "| termo no relatório | código no JSON |",
        "|---|---|",
    ]
    lines += [f"| {pt} | `{code}` |" for pt, code in labels.correspondencia(usados)]
    return lines + ["", "</details>", ""]


def write_exceptions_report(records: list[EventRecord], out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    exceptions = [r for r in records if r.triage.disposition != Disposition.AUTO]

    lines = [
        "# Relatório de exceções — eventos corporativos",
        "",
        f"Gerado em {datetime.now(timezone.utc).strftime('%d/%m/%Y %H:%M UTC')} · "
        f"{len(exceptions)} de {len(records)} documentos exigem atuação humana.",
        "",
        "Cada campo abaixo mostra o valor lido, quanta confiança ele merece, **qual "
        "fator limitou essa confiança**, e onde na página ele foi encontrado. Onde há "
        "recorte, o pixel de origem está embutido aqui — não é preciso reabrir o PDF.",
        "",
    ]
    lines += _panorama(records)

    order = {Disposition.BLOCKED: 0, Disposition.FAILED: 1, Disposition.REVIEW: 2}
    for record in sorted(exceptions, key=lambda r: order.get(r.triage.disposition, 9)):
        lines += [
            "---",
            "",
            f"## {_titulo(record)} — {labels.evento(record.event_type)}",
            "",
            f"{_MARK[record.triage.disposition]} **{labels.disposicao(record.triage.disposition)}** · "
            f"`{record.document.file_name}`",
            "",
        ]
        if record.error:
            lines += ["```", record.error.strip(), "```", ""]
            continue

        lines += [
            labels.em_portugues(record.triage.justification.strip()),
            "",
            "**Motivos**",
            "",
        ]
        lines += ["| camada | tipo | campo | o que houve |", "|---|---|---|---|"]
        # por camada: a camada Confiança é sempre consequência, então ordenar
        # assim põe a causa antes do efeito sem heurística de agrupamento
        for reason in sorted(record.triage.reasons, key=lambda r: LAYER_ORDER[r.layer]):
            marker = "🛑 " if reason.severity == Severity.BLOCK else ""
            alvo = labels.campo(reason.field) if reason.field else "—"
            lines.append(
                f"| {marker}{labels.camada(reason.layer)} | {labels.tipo_de_erro(reason.error_type)} "
                f"| {alvo} | {labels.motivo(reason.code)} |"
            )
        lines.append("")

        if record.triage.fields_for_review:
            lines += [
                "**Campos a conferir**",
                "",
                "| campo | valor | confiança | limitado por | onde foi lido |",
                "|---|---|---|---|---|",
            ]
            for name in record.triage.fields_for_review:
                field = record.fields.get(name)
                if field is None:
                    continue
                valor = field.value if field.value is not None else "—"
                falha = " ⛔" if name in record.triage.fields_for_review else ""
                lines.append(
                    f"| {labels.campo(name)} | {valor} | "
                    f"{labels.nivel_curto(field.reading_level)}{falha} | "
                    f"{_leitura_cell(field)} | {_rastreio_cell(field)} |"
                )
            lines.append("")
            for name in record.triage.fields_for_review:
                lines += _detalhe(record, name)

    if not exceptions:
        lines.append("_Nenhuma exceção: todos os documentos foram liberados automaticamente._")

    lines += _apendice(records)
    path = out_dir / "exceptions_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_run_summary(
    records: list[EventRecord], out_dir: str | Path, elapsed: float, meta: dict | None = None
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dispositions = Counter(r.triage.disposition.value for r in records)
    reason_counts = Counter(
        reason.code.value for r in records for reason in r.triage.reasons
    )
    total = len(records) or 1

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "documents": len(records),
        "elapsed_seconds": round(elapsed, 2),
        # The metric the business actually argues about. Tightening thresholds
        # cuts STP and raises review cost; loosening raises STP and pushes
        # error into the downstream base.
        "stp_rate": round(dispositions.get("auto", 0) / total, 4),
        "exception_rate": round(1 - dispositions.get("auto", 0) / total, 4),
        "dispositions": dict(dispositions),
        "reason_codes": dict(reason_counts.most_common()),
        "by_document": [
            {
                "file": r.document.file_name,
                "event_type": r.event_type.value,
                "disposition": r.triage.disposition.value,
                "weakest_level": r.triage.weakest_level,
                "readers": [k.value for k in r.document.reader_kinds],
                "escalated": r.document.escalated,
                "reasons": [reason.code.value for reason in r.triage.reasons],
            }
            for r in records
        ],
        **(meta or {}),
    }

    path = out_dir / "run_summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
