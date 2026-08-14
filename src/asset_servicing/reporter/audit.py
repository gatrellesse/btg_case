"""Stage 8a — as duas verificações de cada campo, explicitadas.

O caso que motivou este módulo: em `05_aurora` os quatro campos de data saíam com
`confiança 0,29` e o relatório mostrava só isso. As parcelas por trás — leitura
0,998, modelo 0,98, validação 0,30 — diziam que a leitura tinha sido quase
perfeita e que o 0,29 era inteiramente a cronologia impossível se propagando.
São diagnósticos opostos: um manda abrir o PDF, o outro manda ligar para o
emissor.

O módulo continua, os números não: o que o campo carrega agora é *quem leu* e
*o que a base de referência disse*, e o nome da ferramenta que porventura o
reprovou. Cada fato de sua fonte, nenhum multiplicado por outro.

Puro, sem I/O, porque markdown, PDF e JSON precisam exatamente do mesmo cálculo.
"""

from __future__ import annotations

import re

from ..models import (
    EventRecord,
    ExtractedField,
    FieldAudit,
    GoldenCheck,
    Reading,
    ValidationResult,
)
from ..text import _squash
from ..validation.evidence import (
    cutting_tools,
    field_class,
    is_critical,
)

#: Chaves que a base de referência pode confirmar. Para o resto — datas,
#: valores, classe — ela não tem opinião, e "não se aplica" é uma resposta
#: diferente de "não confere".
IDENTITY_KEYS = ("issuer", "cnpj", "isin", "ticker")


def _reading(field: ExtractedField, name: str, culprits: dict) -> Reading:
    """Quem leu, quem concordou, e quem reprovou.

    `cut_by` continua aqui e continua separado do resto porque responde a
    pergunta que o operador faz primeiro: *abro o recorte ou ligo para o
    emissor?*. Um campo lido pelos três mecanismos e reprovado no
    `check_date_coherence` pede a segunda coisa, e o número que os
    multiplicava pedia a primeira.
    """
    return Reading(
        level=field.reading_level,
        corroborating_kinds=list(field.corroborating_kinds),
        cut_by=culprits.get(name, []),
    )


def _golden(name: str, value, validations: list[ValidationResult]) -> GoldenCheck:
    """O resultado contra a base de referência, campo a campo."""
    if name not in IDENTITY_KEYS:
        return GoldenCheck(status="nao_aplicavel")

    for validation in validations:
        if validation.tool != "lookup_golden_record":
            continue
        detail = validation.detail
        if name in (detail.get("agreeing_keys") or []):
            return GoldenCheck(status="confere")
        row = detail.get("matched_row") or {}
        # a linha da base usa nomes em português; só interessa quando existe
        expected = row.get({"issuer": "emissor"}.get(name, name))
        if expected and value and _squash(str(expected)) != _squash(str(value)):
            return GoldenCheck(status="diverge", expected=str(expected))
        return GoldenCheck(status="ausente" if not row else "confere")
    return GoldenCheck(status="ausente")


# --------------------------------------------------------------------------
# notas: curar e traduzir
# --------------------------------------------------------------------------

_READINGS = re.compile(r"^leituras: (.+)$")
_ONE_READING = re.compile(r"(\S+)='(.*?)'")
_ALTERNATIVES = re.compile(r"^leituras alternativas da região: (.+)$")
_ONE_ALT = re.compile(r"([\w:]+)='(.*?)'(?:; |$)")
_QUALITY = re.compile(r"^evidência localizada com qualidade ([\d.]+)$")


def _translate_readings(body: str) -> str:
    """`rule='15/07/2026'; text='15/07/2026'` → prosa.

    Quando os dois leram a mesma coisa, dizer isso uma vez vale mais que repetir
    o valor duas vezes.
    """
    from . import labels

    pairs = _ONE_READING.findall(body)
    if not pairs:
        return body
    valores = {v for _, v in pairs}
    quem = [labels.extrator(n) if n in ("rule", "text") else n for n, _ in pairs]
    if len(valores) == 1:
        return f"{' e '.join(quem)} leram “{pairs[0][1]}”"
    return "leituras divergentes: " + "; ".join(
        f"{labels.extrator(n) if n in ('rule', 'text') else n} leu “{v}”" for n, v in pairs
    )


def _translate_alternatives(body: str, evidence: str | None) -> str | None:
    """Descarta o parágrafo repetido; traduz e encurta o que sobrar.

    Um leitor que devolveu a região inteira como um bloco não está divergindo do
    valor — está com outra granularidade. Repetir 400 caracteres idênticos três
    vezes por campo é o que fazia o relatório dobrar de tamanho sem dizer nada.
    """
    from . import labels

    kept = []
    fold = _squash(evidence or "")
    for reader, text in _ONE_ALT.findall(body):
        if fold and fold in _squash(text):
            continue  # contém a própria evidência: mesma leitura, outro recorte
        nome = labels.extrator(reader) if reader in labels._table("extratores") else reader
        trecho = text if len(text) <= 90 else text[:90].rstrip() + "…"
        kept.append(f"{nome} leu “{trecho}”")
    return "outras leituras da região: " + "; ".join(kept) if kept else None


#: Já dito pela nota de leituras, que nomeia os dois extratores e o valor.
_IMPLIED = "extratores independentes concordam"


def curate_notes(field: ExtractedField) -> list[str]:
    """As notas que sobrevivem, em português, sem repetição."""
    out: list[str] = []
    said_readings = False
    for note in field.notes:
        readings = _READINGS.match(note)
        alternatives = _ALTERNATIVES.match(note)
        quality = _QUALITY.match(note)
        if readings:
            out.append(_translate_readings(readings.group(1)))
            said_readings = True
        elif alternatives:
            translated = _translate_alternatives(alternatives.group(1), field.evidence_text)
            if translated:
                out.append(translated)
        elif quality:
            # a qualidade do casamento já aparece na coluna de rastreio
            continue
        elif note == _IMPLIED and said_readings:
            continue
        else:
            out.append(note)

    seen: set[str] = set()
    unique = []
    for note in out:
        if note not in seen:
            seen.add(note)
            unique.append(note)
    return unique


# --------------------------------------------------------------------------


def annotate(record: EventRecord, thresholds: dict) -> None:
    """Preenche `field.audit` para cada campo, nos dois motores."""
    culprits = cutting_tools(record.validations)

    for name, field in record.fields.items():
        field.audit = FieldAudit(
            field_class=field_class(name, thresholds),
            critical=is_critical(name, thresholds),
            reading=_reading(field, name, culprits),
            golden=_golden(name, field.value, record.validations),
        )
