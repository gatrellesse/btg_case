"""A convenção pt-BR do registro: `dd/mm/aaaa` e `1.000,00`.

Por que existe um módulo só para isto: o valor padronizado é lido por camadas
diferentes — validação compara datas, reparo deriva umas das outras, o relatório
mostra ao operador — e basta uma delas continuar esperando ISO para que a
comparação falhe *em silêncio*. `date.fromisoformat("15/07/2026")` não levanta
erro no lugar errado: devolve `None` lá na frente, a violação some, e o registro
sai PASS. Um documento com pagamento antes da data com passaria batido.

Daí a regra: **liberal ao ler, estrito ao escrever.** O parser aceita as duas
convenções, porque o valor pode vir do documento, do golden record ou de um
registro antigo; o formatador emite só pt-BR, porque é isso que a base de
custódia e o operador brasileiro esperam ver.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from decimal import Decimal, InvalidOperation

ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")
BR_DATE = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{4})")


def format_date(value: date) -> str:
    return value.strftime("%d/%m/%Y")


def parse_date(value) -> date | None:
    """Aceita `dd/mm/aaaa` e ISO; devolve `None` em vez de levantar."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    match = BR_DATE.match(text)
    if match:
        day, month, year = (int(g) for g in match.groups())
        try:
            return date(year, month, day)
        except ValueError:
            return None
    if ISO_DATE.match(text):
        try:
            return date.fromisoformat(text[:10])
        except ValueError:
            return None
    return None


def format_decimal(value: Decimal, min_decimals: int = 2) -> str:
    """`1.000,00` — ponto agrupa milhar, vírgula é a casa decimal.

    Não corta casas: um provento por ação com dez decimais é o valor que a
    companhia publicou, e arredondar aqui mudaria o que o registro afirma que
    o documento diz.
    """
    sign = "-" if value < 0 else ""
    digits = format(abs(value), "f")
    whole, _, frac = digits.partition(".")
    frac = frac.ljust(min_decimals, "0")
    groups = []
    while len(whole) > 3:
        groups.insert(0, whole[-3:])
        whole = whole[:-3]
    groups.insert(0, whole)
    return f"{sign}{'.'.join(groups)}" + (f",{frac}" if frac else "")


def parse_decimal(value) -> Decimal | None:
    """Aceita `1.000,00`, `1000.00` e `0,175`; devolve `None` em vez de levantar.

    A desambiguação é pela vírgula: se ela aparece, o ponto só pode ser
    separador de milhar. Sem vírgula, o ponto é decimal — é assim que
    `0.4275000000` vindo de um registro antigo continua sendo lido certo.
    """
    if value is None or value == "":
        return None
    if isinstance(value, Decimal):
        return value
    if isinstance(value, (int, float)):
        return Decimal(str(value))
    text = re.sub(r"[^\d,.\-]", "", str(value))
    if not text:
        return None
    if "," in text:
        text = text.replace(".", "").replace(",", ".")
    try:
        return Decimal(text)
    except InvalidOperation:
        return None
