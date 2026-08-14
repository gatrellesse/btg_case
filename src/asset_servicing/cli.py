"""Command line entry point.

    python -m asset_servicing.cli run --docs <dir> --golden <csv> --out out/

Runs without any API key: the deterministic path takes over and the record
says so. That is a real fallback, not a mock — it is also what the tests use.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .llm.agents import DEFAULT_MODEL as DEFAULT_AGENT_MODEL
from .llm.agents import load_dotenv, model_available
from .extraction.consensus import IngestConfig
from .validation.evidence import load_thresholds
from .llm.cache import ResponseCache
from .llm.gemini import DEFAULT_MODEL, GeminiClient
from .models import Disposition
from .pipeline import process_batch
from .reporter.pdf_report import write_exceptions_pdf
from .reporter.report import write_exceptions_report, write_records, write_run_summary
from .validation.tools import Validators
from .llm import trace
from .reporter.trace_viewer import build_trace_viewer
from .reporter.viewer import build_viewer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="asset_servicing", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="processa um lote de avisos")
    run.add_argument("--docs", required=True, help="diretório com os PDFs")
    run.add_argument("--golden", required=True, help="golden_records.csv")
    run.add_argument("--out", default="out", help="diretório de saída")
    run.add_argument("--config", default=None, help="diretório de configuração")
    run.add_argument("--workers", type=int, default=4)
    run.add_argument(
        "--offline",
        action="store_true",
        help="não chama o provider; usa extração determinística e o cache",
    )
    run.add_argument(
        "--engine",
        choices=["graph", "linear"],
        default="graph",
        help="grafo LangGraph (padrão) ou o pipeline linear da v1, para comparação",
    )
    run.add_argument(
        "--trace",
        action="store_true",
        help="registra o que foi enviado e recebido em cada chamada de agente "
             "(out/trace.json + trace.html); depuração, não produção",
    )
    run.add_argument(
        "--model",
        default=None,
        help=f"modelo no formato provider:nome (padrão {DEFAULT_AGENT_MODEL})",
    )
    return parser


def command_run(args: argparse.Namespace) -> int:
    docs_dir = Path(args.docs)
    paths = sorted(p for p in docs_dir.glob("*.pdf"))
    if not paths:
        print(f"nenhum PDF encontrado em {docs_dir}", file=sys.stderr)
        return 1

    out_dir = Path(args.out)
    thresholds = load_thresholds(args.config)
    validators = Validators(args.golden, thresholds)

    load_dotenv()
    # Antes de construir qualquer agente: o wrap acontece em build_*, e um
    # agente criado antes disto não seria registrado.
    if getattr(args, "trace", False):
        trace.enable()
    model = args.model or DEFAULT_AGENT_MODEL
    use_model = (not args.offline) and model_available(model)
    mode = model if use_model else "determinístico"
    if not use_model and not args.offline:
        provider = model.split(":", 1)[0]
        print(
            f"aviso: sem chave para o provider '{provider}' — extração determinística por "
            "âncoras de rótulo. Os registros indicam essa origem.",
            file=sys.stderr,
        )

    cache = ResponseCache(out_dir / "cache")
    client = GeminiClient(model=DEFAULT_MODEL, cache=cache, offline=args.offline)

    print(
        f"processando {len(paths)} documento(s) · motor: {args.engine} · extração: {mode}",
        file=sys.stderr,
    )

    started = time.time()
    records = process_batch(
        paths,
        validators,
        thresholds,
        client=client,
        ingest_cfg=IngestConfig(),
        snippet_dir=out_dir / "snippets",
        config_dir=args.config,
        workers=args.workers,
        engine=args.engine,
        model=model,
        use_model=use_model,
    )
    elapsed = time.time() - started

    write_records(records, out_dir)
    report_path = write_exceptions_report(records, out_dir)
    pdf_path = write_exceptions_pdf(records, docs_dir, out_dir)
    viewer_path = build_viewer(records, docs_dir, out_dir / "viewer.html")
    summary_path = write_run_summary(
        records,
        out_dir,
        elapsed,
        meta={
            "engine": args.engine,
            "extraction_mode": mode,
            "model": model if use_model else None,
        },
    )

    trace_paths = []
    if trace.enabled():
        n = trace.dump(out_dir / "trace.json")
        trace_paths = [out_dir / "trace.json",
                       build_trace_viewer(trace.records(), out_dir / "trace.html")]
        print(f"\n{n} chamada(s) de agente registradas", file=sys.stderr)

    counts: dict[str, int] = {}
    for record in records:
        counts[record.triage.disposition.value] = counts.get(record.triage.disposition.value, 0) + 1

    print(f"\nconcluído em {elapsed:.1f}s", file=sys.stderr)
    for disposition in (Disposition.AUTO, Disposition.REVIEW, Disposition.BLOCKED, Disposition.FAILED):
        if counts.get(disposition.value):
            print(f"  {disposition.value:8} {counts[disposition.value]}", file=sys.stderr)
    stp = counts.get("auto", 0) / max(len(records), 1)
    print(f"  STP rate {stp:.0%}", file=sys.stderr)
    outputs = [pdf_path, report_path, summary_path, viewer_path, *trace_paths]
    print("\n" + "\n".join(str(p) for p in outputs), file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return command_run(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
