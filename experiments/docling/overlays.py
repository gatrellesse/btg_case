"""Renderiza o que cada motor viu, sobre a própria página.

Contagem de caracteres não mostra onde o leitor esteve. Isto mostra: a página
renderizada uma vez, e por cima dela a caixa de cada bloco que aquele motor
devolveu. Faixa vazia é motor que não leu nada ali.

O quarto painel é a fusão: cada região da página colorida por QUANTOS motores
a cobriram. É o mapa de corroboração — verde onde três leram a mesma região,
âmbar onde dois, cinza onde um só.

    python experiments/docling/overlays.py
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pymupdf

from extract import build_converter

DOCS = Path("out/consensus/docs")
OUT = Path("out/consensus/overlays.json")
MODES = ("parser", "ocr", "vlm")
THUMB_DPI = 96

#: sobreposição mínima para dizer que dois motores cobriram a mesma região
OVERLAP_MIN = 0.15


def page_image(pdf: Path) -> tuple[str, float, float]:
    """JPEG da primeira página como data URI, com o tamanho em pontos."""
    with pymupdf.open(pdf) as doc:
        page = doc[0]
        pix = page.get_pixmap(dpi=THUMB_DPI)
        data = pix.tobytes("jpeg", jpg_quality=78)
        b64 = base64.b64encode(data).decode("ascii")
        return f"data:image/jpeg;base64,{b64}", page.rect.width, page.rect.height


def item_text(item) -> str:
    """Texto do item — para tabela, o conteúdo das células.

    Um TableItem sem nenhuma célula com texto é uma caixa que o modelo de layout
    desenhou sobre pixels que ninguém leu. No híbrido é exatamente o que o parser
    devolve: enxerga a tabela, não lê uma palavra dela.
    """
    text = getattr(item, "text", "") or ""
    if text.strip():
        return text
    data = getattr(item, "data", None)
    grid = getattr(data, "grid", None) if data else None
    if grid:
        return " ".join(c.text for row in grid for c in row if getattr(c, "text", ""))
    return ""


def boxes_of(document, page_height: float) -> list[dict]:
    """Caixas dos itens, convertidas de BOTTOMLEFT para topo-esquerda."""
    out = []
    for item, _level in document.iterate_items():
        prov = getattr(item, "prov", None)
        if not prov:
            continue
        b = prov[0].bbox
        top, bottom = max(b.t, b.b), min(b.t, b.b)
        if str(b.coord_origin).endswith("BOTTOMLEFT"):
            y = page_height - top
        else:
            y = bottom
        text = item_text(item)
        out.append(
            {
                "label": str(getattr(item, "label", "")),
                "x": round(b.l, 1),
                "y": round(y, 1),
                "w": round(b.r - b.l, 1),
                "h": round(top - bottom, 1),
                "text": text[:90],
                # caixa sem conteúdo não corrobora nada: entra no desenho como
                # contorno vazio e fica de fora da contagem de cobertura
                "filled": bool(text.strip()),
            }
        )
    return out


def overlap_ratio(a: dict, b: dict) -> float:
    """Interseção sobre a MENOR das duas áreas — um motor pode devolver a
    tabela como um bloco só e outro como oito linhas; IoU puniria isso."""
    x0 = max(a["x"], b["x"])
    y0 = max(a["y"], b["y"])
    x1 = min(a["x"] + a["w"], b["x"] + b["w"])
    y1 = min(a["y"] + a["h"], b["y"] + b["h"])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    smallest = min(a["w"] * a["h"], b["w"] * b["h"]) or 1.0
    return inter / smallest


def merge_map(per_mode: dict[str, list[dict]]) -> list[dict]:
    """Grade de referência = as caixas do OCR (é o único que lê toda página nos
    três casos). Cada uma recebe quantos motores a cobriram."""
    reference = [b for b in per_mode.get("ocr") or [] if b["filled"]]
    merged = []
    for box in reference:
        readers = ["ocr"]
        for mode in ("parser", "vlm"):
            if any(
                other["filled"] and overlap_ratio(box, other) >= OVERLAP_MIN
                for other in per_mode.get(mode, [])
            ):
                readers.append(mode)
        merged.append({**box, "readers": sorted(readers), "n": len(readers)})
    return merged


def main() -> None:
    converters = {m: build_converter(m) for m in MODES}
    report: dict[str, dict] = {}

    for pdf in sorted(DOCS.glob("*.pdf")):
        uri, width, height = page_image(pdf)
        per_mode: dict[str, list[dict]] = {}
        stats: dict[str, dict] = {}

        for mode in MODES:
            result = converters[mode].convert(pdf)
            doc = result.document
            per_mode[mode] = boxes_of(doc, height)
            stats[mode] = {
                "blocks": len(per_mode[mode]),
                "tables": len(doc.tables),
                "chars": len(doc.export_to_markdown()),
            }
            print(f"{pdf.stem:36s} {mode:7s} blocos={stats[mode]['blocks']:3d} "
                  f"tabelas={stats[mode]['tables']}", flush=True)

        merged = merge_map(per_mode)
        report[pdf.stem] = {
            "page": uri,
            "width": width,
            "height": height,
            "modes": per_mode,
            "stats": stats,
            "merged": merged,
            "coverage": {
                str(n): sum(1 for m in merged if m["n"] == n) for n in (1, 2, 3)
            },
        }
        print(f"{'':36s} fusão   {report[pdf.stem]['coverage']}\n", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report), encoding="utf-8")
    size_mb = OUT.stat().st_size / 1e6
    print(f"wrote {OUT} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
