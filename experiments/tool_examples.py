"""Captura entrada e saída reais de cada ferramenta do pipeline.

Exemplo inventado documenta o que se pretendia; este arquivo documenta o que
acontece. Tudo aqui sai de uma chamada de verdade sobre os documentos do lote —
inclusive os casos que falham, que são os que ensinam alguma coisa.

    PYTHONPATH=src python experiments/tool_examples.py   # -> out/tool_examples.json
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _source import essence  # noqa: E402

import pymupdf

from asset_servicing.classification import markers
from asset_servicing.extraction import layout, provenance
from asset_servicing.extraction.consensus import default_readers, read_page_consensus, resolve
from asset_servicing.extraction.identifiers import find_identifiers
from asset_servicing.extraction.readers.docling_reader import DoclingReader
from asset_servicing.validation.evidence import load_thresholds
from asset_servicing.validation.tools import Validators

DOCS = Path("Case AI Dev - Envio/documents")
GOLDEN = Path("Case AI Dev - Envio/golden_records/golden records.csv")
OUT = Path("out/tool_examples.json")

NATIVE = DOCS / "01_energetica_vale_tiete_dividendo.pdf"
SCAN = DOCS / "07_telecom_norte_jcp_SCAN.pdf"


def show(value, limit: int = 3):
    """Forma legível: dataclass vira dict, listas longas são cortadas."""
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    elif hasattr(value, "model_dump"):  # Pydantic: ClassificationPrior & cia.
        value = value.model_dump(mode="json")
    if isinstance(value, dict):
        return {k: show(v, limit) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        head = [show(v, limit) for v in list(value)[:limit]]
        if len(value) > limit:
            head.append(f"… mais {len(value) - limit}")
        return head
    if hasattr(value, "value"):  # StrEnum
        return value.value
    if isinstance(value, float):
        return round(value, 4)
    return value


def block_view(block) -> dict:
    return {
        "text": block.text[:70],
        "bbox": [round(block.bbox.x0), round(block.bbox.y0),
                 round(block.bbox.x1), round(block.bbox.y1)],
        "level": block.level,
        "reader_kind": block.reader_kind.value,
        "role": block.role,
        "row": block.row,
        "col": block.col,
        "alternatives": [(n, t[:40]) for n, t in block.alternatives],
    }


def main() -> None:
    examples: list[dict] = []

    def add(layer, tool, why, entrada, saida, fn=None):
        entry = {"layer": layer, "tool": tool, "why": why,
                 "input": entrada, "output": saida}
        if fn is not None:
            entry["logic"] = essence(fn)
        examples.append(entry)

    # ---------------- ingestão -------------------------------------------
    native = pymupdf.open(NATIVE)[0]
    scan = pymupdf.open(SCAN)[0]

    for mode, page, note in (
        ("parser", native, "PDF nativo"),
        ("ocr", native, "mesma página nativa, lida por pixel"),
        ("vlm", native, "mesma página nativa, gerada por tokens"),
    ):
        reader = DoclingReader(mode)  # type: ignore[arg-type]
        result = reader.read(page)
        cells = [b for b in result.blocks if b.is_cell]
        add(
            "Ingestão", f"docling:{mode}",
            f"{note} — os três saem no mesmo formato de Block",
            {"page": NATIVE.name, "modo": mode},
            {"blocos": len(result.blocks), "células": len(cells),
             "exemplo": block_view(cells[0] if cells else result.blocks[0])},
            DoclingReader.read,
        )

    parser_scan = DoclingReader("parser").read(scan)
    add(
        "Ingestão", "docling:parser (scan)",
        "abstenção: sem camada de texto não há o que ler, e zero bloco é a "
        "resposta certa — não rebaixa o degrau de ninguém",
        {"page": SCAN.name, "modo": "parser"},
        {"blocos": len(parser_scan.blocks), "nota": parser_scan.note},
        DoclingReader.read,
    )

    blocks, notes, kinds, _ = read_page_consensus(native, default_readers())
    corroborated = next((b for b in blocks if len(b.alternatives) >= 2), blocks[0])
    add(
        "Ingestão", "consensus.read_page",
        "funde região a região por sobreposição sobre a MENOR área — um motor "
        "devolve a tabela inteira, outro devolve oito células",
        {"page": NATIVE.name, "leitores": ["docling:parser", "docling:ocr", "docling:vlm"]},
        {"regiões": len(blocks), "kinds": [k.value for k in kinds],
         "notas": notes, "exemplo": block_view(corroborated)},
        read_page_consensus,
    )

    scan_blocks, scan_notes, _, _ = read_page_consensus(scan, default_readers())
    add(
        "Ingestão", "consensus.resolve",
        "o degrau vem de QUEM concordou: a camada de texto não compartilha modo de "
        "falha com pixel; dois leitores dos mesmos pixels erram junto e valem menos",
        {"leituras da região": ["docling:ocr", "docling:vlm"], "parser": "abstido"},
        {"exemplo": block_view(scan_blocks[0]),
         "por que media": "OCR e VLM concordando — dois leitores dos mesmos pixels",
         "notas": scan_notes},
        resolve,
    )

    enriched = layout.enrich(list(blocks))
    cell = next((b for b in enriched if b.is_cell), enriched[0])
    add(
        "Extração", "layout.enrich",
        "agrupamento geométrico por linha: estes avisos usam tabela sem borda, "
        "então linha é geometria, não marcação",
        {"blocos": len(blocks)},
        {"exemplo": block_view(cell)},
        layout.enrich,
    )

    # ---------------- classificação --------------------------------------
    raw_text = "\n".join(b.text for b in enriched)
    prior = markers.pre_classify(raw_text)
    add(
        "Classificação", "markers.pre_classify",
        "opinião determinística antes do modelo — prior fraco é resultado "
        "válido, não erro",
        {"text": raw_text[:120] + "…"},
        show(prior),
        markers.pre_classify,
    )

    ids = find_identifiers(raw_text)
    add(
        "Extração", "identifiers.find_identifiers",
        "camada 1 varre por estrutura (ISO 6166, ticker B3, CNPJ) sem depender "
        "de rótulo; camada 2 usa âncora como reforço",
        {"text": raw_text[:120] + "…"},
        show(ids),
        find_identifiers,
    )

    block, quality = provenance.locate("R$ 0,4275000000", enriched)
    add(
        "Extração", "provenance.locate",
        "o valor volta para a região da página de onde foi lido — sem isto o "
        "campo não é auditável",
        {"evidence": "R$ 0,4275000000"},
        {"qualidade": round(quality, 4),
         "bloco": block_view(block) if block else None},
        provenance.locate,
    )

    # ---------------- validação -------------------------------------------
    validators = Validators(GOLDEN, load_thresholds())

    add("Validação", "lookup_golden_record",
        "emissor conhecido: identidade confirmada contra a base de referência",
        {"issuer": "Energética Vale do Tietê S.A.", "isin": "BRTIETACNOR3"},
        show(validators.lookup_golden_record(
            issuer="Energética Vale do Tietê S.A.", isin="BRTIETACNOR3")),
        Validators._lookup)

    add("Validação", "lookup_golden_record (ausente)",
        "ler certo e não existir no registro são coisas diferentes — o doc 08 "
        "foi lido corretamente e o emissor não está na base",
        {"issuer": "Construtora Horizonte S.A.", "isin": "BRCNHZACNOR5"},
        show(validators.lookup_golden_record(
            issuer="Construtora Horizonte S.A.", isin="BRCNHZACNOR5")), Validators._lookup)

    # O valor padronizado é pt-BR: `dd/mm/aaaa` e `1.000,00`. Os parsers são
    # liberais (ISO ainda entra, para registro antigo), os formatadores não.
    add("Validação", "check_date_coherence",
        "cronologia que qualquer evento respeita: direito fixa na data com, a "
        "ação negocia ex no pregão seguinte, ninguém recebe antes",
        {"approval_date": "28/05/2026", "com_date": "12/06/2026",
         "ex_date": "15/06/2026", "payment_date": "03/07/2026",
         "event_type": "dividendo"},
        show(validators.check_date_coherence(
            approval_date="28/05/2026", com_date="12/06/2026",
            ex_date="15/06/2026", payment_date="03/07/2026",
            event_type="dividendo")),
        Validators.check_date_coherence)

    add("Validação", "check_date_coherence (violação)",
        "o caso que bloqueia o doc 05: pagamento antes da data com",
        {"approval_date": "01/06/2026", "com_date": "15/07/2026",
         "ex_date": "16/07/2026", "payment_date": "10/07/2026",
         "event_type": "dividendo"},
        show(validators.check_date_coherence(
            approval_date="01/06/2026", com_date="15/07/2026",
            ex_date="16/07/2026", payment_date="10/07/2026",
            event_type="dividendo")),
        Validators.check_date_coherence)

    add("Validação", "check_date_coherence (legado ISO)",
        "MESMAS datas em ISO: liberal ao ler, estrito ao escrever. Registro "
        "antigo continua sendo lido e a violação continua disparando — o que "
        "sai padronizado é sempre pt-BR",
        {"com_date": "2026-07-15", "ex_date": "2026-07-16",
         "payment_date": "2026-07-10", "event_type": "dividendo"},
        show(validators.check_date_coherence(
            com_date="2026-07-15", ex_date="2026-07-16",
            payment_date="2026-07-10", event_type="dividendo")),
        Validators.check_date_coherence)

    add("Validação", "check_amount_coherence",
        "bruto × (1 − alíquota) = líquido — três campos que se confirmam entre "
        "si valem mais que três leitores concordando sobre um",
        {"gross_per_share": "0,1124300000", "tax_rate": "0,175",
         "net_per_share": "0,0927547500"},
        show(validators.check_amount_coherence(
            gross_per_share="0,1124300000", tax_rate="0,175",
            net_per_share="0,0927547500")),
        Validators.check_amount_coherence)

    add("Validação", "check_event_type_consistency",
        "o doc 03 é titulado 'proventos' e é JCP na substância — o conflito "
        "vira reason code, não decisão silenciosa",
        {"declared_type": "jcp", "title_prior": "dividendo", "body_prior": "jcp"},
        show(validators.check_event_type_consistency(
            declared_type="jcp", title_prior="dividendo", body_prior="jcp")),
        Validators.check_event_type_consistency)

    add("Validação", "check_identifier_format",
        "SEMPRE INFO: os identificadores do lote são sintéticos e quase nenhum "
        "fecha o dígito verificador; vetar por checksum reprovaria a base toda",
        {"isin": "BRTIETACNOR3", "cnpj": "12.345.678/0001-90", "ticker": "TIET3"},
        show(validators.check_identifier_format(
            isin="BRTIETACNOR3", cnpj="12.345.678/0001-90", ticker="TIET3")),
        Validators.check_identifier_format)

    add("Validação", "check_required_fields",
        "distingue TIPOS de ausência: adiada pelo emissor, inaplicável ao "
        "evento, ou faltando de verdade",
        {"event_type": "dividendo",
         "present": "issuer,isin,gross_per_share,com_date,ex_date",
         "absent": "payment_date"},
        show(validators.check_required_fields(
            event_type="dividendo",
            present="issuer,isin,gross_per_share,com_date,ex_date",
            absent="payment_date")),
        Validators.check_required_fields)

    # ---------------- reporter --------------------------------------------
    if block is not None:
        snippet = provenance.crop(str(NATIVE), block.bbox, "out/tool_examples_snippet.png")
        add("Reporter", "provenance.crop",
            "o recorte do pixel de origem entra no relatório — o operador não "
            "precisa reabrir o PDF para conferir",
            {"bbox": block_view(block)["bbox"], "pdf": NATIVE.name},
            {"arquivo": snippet, "dpi": 200}, provenance.crop)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(examples, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(examples)} exemplos → {OUT}")
    for e in examples:
        print(f"  {e['layer']:14s} {e['tool']}")


if __name__ == "__main__":
    main()
