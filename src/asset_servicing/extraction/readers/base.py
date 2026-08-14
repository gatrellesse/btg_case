"""Reader protocol.

Swapping an OCR engine is an architecture decision, not a config change,
because the rest of the pipeline depends on a signal that not every engine
returns: a box, which is what makes grounding possible. Making each reader
*declare* what it provides turns "it's pluggable" into a precise statement of
what is gained, what is lost, and how the pipeline compensates.

Confiança declarada pelo motor **não** está entre os sinais. Estava: cada leitor
anunciava a granularidade da sua confiança e o bloco saía com o número dela. Só
que nenhum ramo do pipeline chegou a ler esse anúncio, e o número que ele
governava foi substituído por quantos mecanismos independentes concordaram —
uma medida que não depende de o motor ter opinião sobre si mesmo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import pymupdf

from ...models import BBox, ReaderKind


@dataclass(frozen=True)
class ReaderCapabilities:
    """What a reader gives back. The pipeline branches on these, not on names."""

    returns_bbox: bool
    #: False means re-issuing the same call on the same input is pointless —
    #: the only way to a different answer is to change the input.
    resampling_helps: bool
    returns_layout_roles: bool
    returns_table_structure: bool
    #: True means the document leaves the machine. Asset Servicing will ask.
    is_hosted: bool


@dataclass
class Block:
    """A unit of text with everything needed to trace it back to pixels."""

    text: str
    bbox: BBox
    reader_kind: ReaderKind
    #: A força da leitura, e a escala inteira em que ela é medida:
    #: muito_alta | alta | media | baixa. Deriva de quem concordou, não de
    #: quanto um motor se acha seguro. Havia um `score` ao lado, peso do degrau,
    #: para as contas de confiança — as contas saíram e o peso com elas.
    level: str = "baixa"
    role: str = "paragraph"  # title | paragraph | table_cell | signature
    row: int | None = None
    col: int | None = None
    order: int = 0
    #: Competing readings of the same region, as (reader_name, text). Populated
    #: pelo consenso: toda modalidade que leu a região fica anexada, tenha
    #: vencido ou não — um revisor precisa ver quem corroborou e quem divergiu,
    #: não só o vencedor.
    alternatives: list[tuple[str, str]] = field(default_factory=list)
    #: Os mecanismos que leram esta região *e concordaram*. O degrau resume a
    #: lista; a lista responde qual mecanismo, que é o que decide se vale a pena
    #: abrir o recorte — um VLM sozinho e a camada de texto sozinha são degraus
    #: diferentes por motivos diferentes.
    corroborating_kinds: list[ReaderKind] = field(default_factory=list)
    #: A caixa é a da tabela inteira, não a da célula: o leitor leu o conteúdo
    #: mas não soube situá-lo. Acontece com todo modelo de doctags — o `<otsl>`
    #: carrega as quatro coordenadas da tabela e as células vêm como `<fcel>`
    #: puros. Marcar em vez de descartar é o que permite a outro leitor, que
    #: mediu as células, emprestar a geometria.
    geometry_pending: bool = False

    @property
    def is_cell(self) -> bool:
        return self.role == "table_cell"


@dataclass
class ReadResult:
    blocks: list[Block] = field(default_factory=list)
    reader_name: str = ""
    caps: ReaderCapabilities | None = None
    note: str | None = None

    @property
    def text(self) -> str:
        return "\n".join(b.text for b in self.blocks)


@runtime_checkable
class Reader(Protocol):
    name: str
    caps: ReaderCapabilities

    def available(self) -> bool:
        """False when an optional dependency or device is missing.

        Readers degrade rather than raise: an unavailable tier is skipped and
        the cascade continues to the next one.
        """
        ...

    def read(self, page: pymupdf.Page, clip: pymupdf.Rect | None = None) -> ReadResult:
        ...

