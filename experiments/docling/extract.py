"""Três extrações isoladas do mesmo documento, uma por classe de mecanismo.

O ponto de isolar: uma modalidade só serve de segunda opinião se não tiver visto
a saída da outra. O pipeline padrão do Docling não serve para isso — ele elimina
os clusters que já têm texto programático antes de chamar o OCR, de modo que as
duas fontes nunca leem a mesma região. Aqui cada motor lê a página inteira.

    parser  do_ocr=False          camada de texto do PDF; abstém-se em scan
    ocr     OcrMode.FULL_PAGE     descarta as células do PDF, PP-OCRv6 em tudo
    vlm     VlmPipeline           granite-docling 258M, DocTags

Uso:
    python experiments/docling/extract.py --mode parser --out out/parser_only
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    OcrMode,
    PdfPipelineOptions,
    VlmPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline

#: Modo -> ReaderKind do projeto (src/asset_servicing/models.py).
READER_KIND = {
    "parser": "text_layer",
    "ocr": "ocr_deterministic",
    "vlm": "ocr_generative",
}


def build_converter(mode: str) -> DocumentConverter:
    if mode == "vlm":
        opts = VlmPipelineOptions()
        opts.generate_page_images = True
        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=VlmPipeline, pipeline_options=opts
                )
            }
        )

    opts = PdfPipelineOptions()
    if mode == "parser":
        # Sem OCR nenhum: num scan o resultado é vazio, e vazio é a resposta
        # correta — abstenção não é discordância.
        opts.do_ocr = False
    else:
        # FULL_PAGE descarta as células do PDF em post_process_cells, então a
        # leitura sai 100% do reconhecimento de pixels mesmo em PDF nativo.
        opts.ocr_options.mode = OcrMode.FULL_PAGE
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", required=True, choices=sorted(READER_KIND))
    ap.add_argument("--docs", default="Case AI Dev - Envio/documents")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = Path(args.out or f"out/consensus/{args.mode}")
    out.mkdir(parents=True, exist_ok=True)
    converter = build_converter(args.mode)

    manifest = []
    for pdf in sorted(Path(args.docs).glob("*.pdf")):
        t0 = time.perf_counter()
        result = converter.convert(pdf)
        elapsed = time.perf_counter() - t0
        doc = result.document

        md = doc.export_to_markdown()
        (out / f"{pdf.stem}.md").write_text(md, encoding="utf-8")

        c = result.confidence
        entry = {
            "doc": pdf.stem,
            "mode": args.mode,
            "reader_kind": READER_KIND[args.mode],
            "seconds": round(elapsed, 2),
            "pages": len(doc.pages),
            "tables": len(doc.tables),
            "texts": len(doc.texts),
            "chars": len(md),
            # nan não sobrevive a JSON estrito; None diz a mesma coisa
            "parse_score": None if c.parse_score != c.parse_score else c.parse_score,
            "ocr_score": None if c.ocr_score != c.ocr_score else c.ocr_score,
        }
        manifest.append(entry)
        print(
            f"{pdf.stem:52s} {elapsed:6.1f}s tables={entry['tables']} "
            f"chars={entry['chars']:5d} parse={entry['parse_score']} "
            f"ocr={entry['ocr_score']}",
            flush=True,
        )

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nwrote {out}/manifest.json")


if __name__ == "__main__":
    main()
