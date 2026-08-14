"""Constrói um PDF híbrido de verdade a partir de um nativo.

O lote do case só tem dois extremos — sete nativos e um scan. Híbrido é o caso
que separa "decide por documento" de "decide por região": prosa na camada de
texto, tabela em pixels, na MESMA página. Um pipeline que sonda a página inteira
classifica errado; um que sonda por região acerta.

Recorta a região da tabela, rasteriza, apaga o texto por baixo e cola a imagem
no lugar. O resultado tem camada de texto na prosa e nenhuma na tabela.

    python experiments/docling/make_hybrid.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pymupdf

RASTER_DPI = 150


def find_table_rect(page: pymupdf.Page) -> pymupdf.Rect:
    """Retângulo que cobre as linhas rótulo→valor.

    Âncora nos extremos verticais da tabela: a primeira linha que contém o
    rótulo de abertura e a última que contém o código de negociação.
    """
    top, bottom = None, None
    for label in ("Tipo de evento", "Natureza do provento", "Valor bruto"):
        hits = page.search_for(label)
        if hits:
            top = hits[0].y0
            break
    hits = page.search_for("Código de negociação") or page.search_for("Codigo de negociacao")
    if hits:
        bottom = hits[-1].y1
    if top is None or bottom is None:
        raise SystemExit("não localizei a tabela na página")
    r = page.rect
    return pymupdf.Rect(r.x0 + 20, top - 8, r.x1 - 20, bottom + 8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--src", default="Case AI Dev - Envio/documents/01_energetica_vale_tiete_dividendo.pdf"
    )
    ap.add_argument("--out", default="out/consensus/hybrid/09_hibrido_tabela_rasterizada.pdf")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(args.src)
    page = doc[0]
    rect = find_table_rect(page)

    # rasteriza a região antes de apagar o texto
    pix = page.get_pixmap(clip=rect, dpi=RASTER_DPI)
    img = pix.tobytes("png")

    # remove o texto sob o retângulo — sem isso a camada continua legível
    page.add_redact_annot(rect)
    page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE)

    page.insert_image(rect, stream=img)
    doc.save(out, garbage=3, deflate=True)

    check = pymupdf.open(out)[0]
    words_in_rect = [w for w in check.get_text("words") if pymupdf.Rect(w[:4]).intersects(rect)]
    total_words = len(check.get_text("words"))
    print(f"escrito {out}")
    print(f"  palavras na página : {total_words}")
    print(f"  palavras na tabela : {len(words_in_rect)}  (esperado 0)")
    print(f"  região rasterizada : {rect}")


if __name__ == "__main__":
    main()
