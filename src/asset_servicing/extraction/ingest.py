"""Stage 1 — turn a PDF into a `SourceDocument` with per-block provenance.

Cada página é lida pelas três modalidades e fundida região a região, de modo que
um arquivo híbrido funciona sem tratamento especial: onde há camada de texto os
três leem, onde não há o parser se abstém, e o bloco carrega quantos mecanismos
independentes o sustentam. O ``producer`` do PDF é registrado porque corrobora,
mas nunca decide — é trivial de forjar, enquanto a ausência de leitura é um fato
sobre os bytes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

from . import layout
from .consensus import IngestConfig, default_readers, read_page_consensus
from ..models import DocumentInfo, ReaderKind
from .readers.base import Block


@dataclass
class PageRead:
    page_number: int
    blocks: list[Block] = field(default_factory=list)
    reader_kinds: list[ReaderKind] = field(default_factory=list)
    escalated: bool = False
    notes: list[str] = field(default_factory=list)
    #: O que cada mecanismo leu nesta página, antes da fusão.
    by_reader: dict[ReaderKind, str] = field(default_factory=dict)


@dataclass
class SourceDocument:
    file_name: str
    path: str
    n_pages: int
    producer: str | None
    pages: list[PageRead] = field(default_factory=list)
    escalated: bool = False

    @property
    def blocks(self) -> list[Block]:
        return [b for page in self.pages for b in page.blocks]

    @property
    def raw_text(self) -> str:
        """Reading-order text, used for grounding and for the regex layer.

        Grounding is checked against *this*, before any normalization, so the
        evidence a field claims must exist as the document actually reads.
        """
        return "\n".join(b.text for page in self.pages for b in page.blocks)

    @property
    def notes(self) -> list[str]:
        return [n for page in self.pages for n in page.notes]

    @property
    def reads(self) -> dict[ReaderKind, str]:
        """As três leituras inteiras, uma por mecanismo, do documento todo.

        `raw_text` é o resultado da votação: uma versão da página, a que venceu
        região a região. Aqui estão as três como cada mecanismo as entregou —
        é o que permite reler o documento atrás de um campo que a votação, a
        geometria ou o extrator deixaram cair.
        """
        textos: dict[ReaderKind, list[str]] = {}
        for page in self.pages:
            for kind, texto in page.by_reader.items():
                if texto.strip():
                    textos.setdefault(kind, []).append(texto)
        return {kind: "\n".join(partes) for kind, partes in textos.items()}

    @property
    def reader_kinds(self) -> list[ReaderKind]:
        seen: list[ReaderKind] = []
        for page in self.pages:
            for kind in page.reader_kinds:
                if kind not in seen:
                    seen.append(kind)
        return seen

    def to_info(self) -> DocumentInfo:
        return DocumentInfo(
            file_name=self.file_name,
            n_pages=self.n_pages,
            producer=self.producer,
            reader_kinds=self.reader_kinds,
            escalated=self.escalated,
            escalation_note="; ".join(self.notes) or None,
        )


def load(path: str | Path, cfg: IngestConfig | None = None) -> SourceDocument:
    path = Path(path)
    cfg = cfg or IngestConfig()
    readers = default_readers(cfg.modes)

    with pymupdf.open(path) as pdf:
        doc = SourceDocument(
            file_name=path.name,
            path=str(path),
            n_pages=pdf.page_count,
            producer=(pdf.metadata or {}).get("producer") or None,
        )
        for page in pdf:
            blocks, notes, kinds, leituras = read_page_consensus(page, readers)
            page_read = PageRead(
                page_number=page.number + 1,
                blocks=blocks,
                reader_kinds=kinds,
                notes=notes,
                by_reader=leituras,
                # "escalado" agora significa: alguma região ficou sem
                # corroboração — um leitor só sustentando o valor
                escalated=any(b.level == "baixa" for b in blocks),
            )
            # One geometric pass serves both routes: rows are rows whether the
            # coordinates came from the PDF or from an OCR detector.
            page_read.blocks = layout.enrich(page_read.blocks)
            doc.pages.append(page_read)
            doc.escalated = doc.escalated or page_read.escalated

    return doc
