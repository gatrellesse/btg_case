"""A data em que o aviso foi redigido.

Não é campo do schema — nenhuma regra de evento depende dela. Mas é o que
distingue dois avisos do mesmo emissor e do mesmo tipo no índice, e é por isso
que ela entra: `jcp_02-06-2026` identifica um documento, `jcp` identifica uma
pilha.

Os avisos brasileiros fecham com `Cidade (UF), DD de mês de AAAA.` logo antes da
assinatura do DRI. Lida da camada de texto, no fim do documento. Num documento
escaneado não há camada de texto e a resposta é honestamente "não localizada" —
inventar uma data para compor um nome de índice seria o pior resultado possível.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pymupdf

MESES = {
    "janeiro": 1, "fevereiro": 2, "março": 3, "marco": 3, "abril": 4,
    "maio": 5, "junho": 6, "julho": 7, "agosto": 8, "setembro": 9,
    "outubro": 10, "novembro": 11, "dezembro": 12,
}

#: `São Paulo (SP), 28 de maio de 2026.` — exige a cidade antes da data para não
#: capturar as datas do corpo do aviso, que são o evento e não a redação.
_LOCAL_E_DATA = re.compile(
    r"[A-ZÀ-Ú][\w\s.'-]{2,40}\s*\([A-Z]{2}\)\s*,\s*(\d{1,2})\s+de\s+([a-zç]+)\s+de\s+(\d{4})",
    re.IGNORECASE,
)


def from_pdf(path: str | Path) -> date | None:
    """A última data de redação do documento, ou `None` se não houver texto."""
    try:
        with pymupdf.open(path) as pdf:
            text = " ".join(pdf[page].get_text() for page in range(pdf.page_count))
    except Exception:  # noqa: BLE001
        return None

    text = " ".join(text.split())
    achadas = _LOCAL_E_DATA.findall(text)
    if not achadas:
        return None
    # a última: a assinatura fecha o documento
    dia, mes, ano = achadas[-1]
    numero = MESES.get(mes.lower())
    if numero is None:
        return None
    try:
        return date(int(ano), numero, int(dia))
    except ValueError:
        return None


def slug(event_type, redigido: date | None) -> str:
    """`jcp_02-06-2026` — o nome do documento no índice."""
    tipo = getattr(event_type, "value", str(event_type))
    return f"{tipo}_{redigido.strftime('%d-%m-%Y')}" if redigido else f"{tipo}_sem-data"
