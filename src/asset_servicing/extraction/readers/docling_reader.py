"""Docling como leitor do projeto — os três modos no mesmo formato de bloco.

O ponto de existir: `parser`, `ocr` e `vlm` são pipelines diferentes do Docling,
com saídas diferentes (`DoclingDocument` com tabela, sem tabela, com DocTags),
e comparar três formatos é comparar os conversores, não as leituras. Aqui os
três atravessam a mesma tradução e saem como `Block` — mesmo bbox em pontos do
PDF, mesmo `role` — de modo que a única diferença entre eles passa a ser *o que
leram*.

Isolamento é requisito, não detalhe. O pipeline padrão do Docling elimina os
clusters que já têm texto programático antes de chamar o OCR, então as duas
fontes nunca leem a mesma região e não existe segunda opinião. `FULL_PAGE`
descarta as células do PDF em `post_process_cells`, e só assim o OCR vira
leitura independente de uma página nativa.

Conversão é por documento, não por página: o protocolo `Reader` é por página,
mas o Docling roda o pipeline inteiro de uma vez. O resultado fica em cache por
(arquivo, modo, mtime) — sem isso um PDF de 10 páginas dispararia 10 conversões
completas.
"""

from __future__ import annotations

import threading
from typing import Literal

import pymupdf

from ...models import BBox, ReaderKind
from .base import Block, Reader, ReaderCapabilities, ReadResult

Mode = Literal["parser", "ocr", "vlm"]

READER_KIND: dict[str, ReaderKind] = {
    "parser": ReaderKind.TEXT_LAYER,
    "ocr": ReaderKind.OCR_DET,
    "vlm": ReaderKind.OCR_VLM,
}

#: Rótulos do Docling → papéis do projeto. O que não está aqui vira parágrafo:
#: `role` governa agrupamento geométrico e não pode inventar categoria.
ROLE = {
    "title": "title",
    "section_header": "title",
    "page_header": "paragraph",
    "page_footer": "paragraph",
    "caption": "paragraph",
    "text": "paragraph",
    "list_item": "paragraph",
    "key_value_region": "paragraph",
    "form": "paragraph",
}

_CACHE: dict[tuple[str, str, float], object] = {}

#: Conversão é serializada de propósito. O lote roda em ThreadPoolExecutor e
#: cada documento dispara três pipelines; com 4 workers isso vira até doze
#: conversões simultâneas disputando a mesma GPU. O que acontece então não é
#: lentidão: o modelo estoura memória, o `read` devolve zero bloco, e o consenso
#: registra "abstém-se" — indistinguível de um parser que legitimamente não tem
#: camada de texto para ler. Foi assim que o doc 07 saiu com o ISIN corrompido
#: do VLM como valor final, sem ninguém para contradizê-lo.
#:
#: Paralelizar aqui não compraria nada de todo modo: o gargalo é a GPU, que é
#: uma só.
_CONVERT_LOCK = threading.Lock()


def _converter(mode: Mode):
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        OcrMode,
        PdfPipelineOptions,
        VlmPipelineOptions,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption

    if mode == "vlm":
        from docling.pipeline.vlm_pipeline import VlmPipeline

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
        opts.do_ocr = False  # num scan devolve nada, e nada é a resposta certa
    else:
        opts.ocr_options.mode = OcrMode.FULL_PAGE  # descarta as células do PDF
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )


def _convert(path: str, mode: Mode):
    from pathlib import Path as _Path

    key = (path, mode, _Path(path).stat().st_mtime)
    with _CONVERT_LOCK:
        if key not in _CACHE:
            _CACHE[key] = _converter(mode).convert(path)
        return _CACHE[key]


def _to_topleft(bbox, page_height: float) -> tuple[float, float, float, float]:
    """Docling entrega BOTTOMLEFT com `t` acima de `b`; o projeto usa topo-esquerda."""
    top, bottom = max(bbox.t, bbox.b), min(bbox.t, bbox.b)
    if str(bbox.coord_origin).endswith("BOTTOMLEFT"):
        return bbox.l, page_height - top, bbox.r, page_height - bottom
    return bbox.l, bottom, bbox.r, top


class DoclingReader:
    """Um modo do Docling, traduzido para o contrato de leitor do projeto."""

    def __init__(self, mode: Mode = "parser") -> None:
        self.mode: Mode = mode
        self.name = f"docling:{mode}"
        self.caps = ReaderCapabilities(
            returns_bbox=True,
            # nenhum dos três muda de resposta se chamado de novo na mesma
            # entrada — mudar a entrada é o que muda a leitura
            resampling_helps=False,
            returns_layout_roles=True,
            returns_table_structure=mode != "vlm",
            is_hosted=False,
        )

    def available(self) -> bool:
        try:
            import docling  # noqa: F401
        except Exception:
            return False
        return True

    def read(self, page: pymupdf.Page, clip: pymupdf.Rect | None = None) -> ReadResult:
        path = page.parent.name
        if not path:
            return ReadResult(reader_name=self.name, caps=self.caps, note="documento sem caminho")
        try:
            result = _convert(path, self.mode)
        except Exception as exc:  # noqa: BLE001
            # "falhou:" é o prefixo que o consenso procura. Um leitor que
            # quebrou não é um leitor que se absteve, e tratar os dois como a
            # mesma coisa deixa o campo com um leitor só sem dizer por quê.
            return ReadResult(reader_name=self.name, caps=self.caps,
                              note=f"falhou: {type(exc).__name__}: {exc}")

        page_no = page.number + 1
        height = float(page.rect.height)
        kind = READER_KIND[self.mode]

        blocks: list[Block] = []
        for item, _level in result.document.iterate_items():
            prov = getattr(item, "prov", None)
            if not prov or prov[0].page_no != page_no:
                continue
            blocks.extend(self._blocks_for(item, prov[0], page_no, height, kind))

        if clip is not None:
            blocks = [
                b for b in blocks
                if pymupdf.Rect(b.bbox.x0, b.bbox.y0, b.bbox.x1, b.bbox.y1).intersects(clip)
            ]
        for order, block in enumerate(blocks):
            block.order = order
        return ReadResult(blocks=blocks, reader_name=self.name, caps=self.caps)

    # ------------------------------------------------------------------
    def _blocks_for(self, item, prov, page_no, height, kind) -> list[Block]:
        label = str(getattr(item, "label", "")).lower()
        x0, y0, x1, y1 = _to_topleft(prov.bbox, height)
        box = BBox(
            page=page_no, x0=x0, y0=y0, x1=x1, y1=y1,
            page_width=float(getattr(prov, "page_width", 0.0)) or 0.0,
            page_height=height,
        )

        cells = self._table_cells(item, page_no, height, kind, box)
        if cells:
            return cells

        text = (getattr(item, "text", "") or "").strip()
        if not text:
            return []
        return [Block(text=text, bbox=box, reader_kind=kind,
                      role=ROLE.get(label, "paragraph"))]

    def _table_cells(self, item, page_no, height, kind, table_box: BBox) -> list[Block]:
        """Uma tabela vira uma célula por bloco, com linha e coluna.

        É a informação que o agrupamento geométrico do projeto tem de inferir e
        que aqui já vem pronta — e é também o que permite comparar região a
        região entre os três modos, já que o `vlm` frequentemente não devolve
        tabela nenhuma e cai em texto solto.

        A célula sem caixa **não é descartada**. Medido: nos modelos de doctags
        (granite-docling e SmolDocling) as 12 células de uma tabela vêm com zero
        coordenadas — o `<otsl>` traz as quatro locs da tabela e as células são
        `<fcel>` puros. Descartá-las tirava o VLM da votação exatamente na
        tabela, que é onde ficam todos os valores do evento. Ela sai com a caixa
        da tabela e `geometry_pending`, para o consenso pedir emprestada a
        geometria de quem mediu.
        """
        data = getattr(item, "data", None)
        grid = getattr(data, "grid", None) if data else None
        if not grid:
            return []
        out: list[Block] = []
        for r, row in enumerate(grid):
            for c, cell in enumerate(row):
                text = (getattr(cell, "text", "") or "").strip()
                if not text:
                    continue
                bbox = getattr(cell, "bbox", None)
                if bbox is None:
                    caixa, pendente = table_box, True
                else:
                    x0, y0, x1, y1 = _to_topleft(bbox, height)
                    caixa = BBox(page=page_no, x0=x0, y0=y0, x1=x1, y1=y1,
                                 page_width=0.0, page_height=height)
                    pendente = False
                out.append(
                    Block(
                        text=text,
                        bbox=caixa,
                        reader_kind=kind,
                        role="table_cell",
                        row=r,
                        col=c,
                        geometry_pending=pendente,
                    )
                )
        return out

