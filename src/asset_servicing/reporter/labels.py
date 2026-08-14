"""Código → português, para tudo que o operador lê.

Acesso fino sobre `config/glossario.yaml`, que é o único lugar onde a
correspondência existe. Levanta em vez de cair no identificador: um termo sem
tradução tem de quebrar o build, senão a falta aparece como `com_date` no meio
do relatório e ninguém percebe até um operador perguntar o que aquilo quer
dizer.

O JSON por documento continua em inglês — é a camada de máquina. Só a
apresentação passa por aqui.
"""

from __future__ import annotations

import re
from decimal import Decimal

from ..config import load_config
from ..formats import format_decimal


def _table(section: str, config_dir: str | None = None) -> dict[str, str]:
    return load_config("glossario", config_dir).get(section) or {}


def term(section: str, key, config_dir: str | None = None) -> str:
    """A tradução, ou um erro dizendo exatamente o que falta traduzir."""
    key = getattr(key, "value", key)
    table = _table(section, config_dir)
    if key not in table:
        raise KeyError(
            f"'{key}' não tem termo em português: acrescente em "
            f"config/glossario.yaml, seção '{section}'"
        )
    return table[key]


def campo(name: str) -> str:
    return term("campos", name)


def camada(layer) -> str:
    return term("camadas", layer)


def tipo_de_erro(kind) -> str:
    return term("tipos_de_erro", kind)


def ferramenta(tool: str) -> str:
    return term("ferramentas", tool)


def leitor(kind) -> str:
    return term("leitores", kind)


def extrator(name: str) -> str:
    return term("extratores", name)


def evidencia(status) -> str:
    return term("evidencia", status)


def disposicao(value) -> str:
    return term("disposicoes", value)


def evento(value) -> str:
    return term("eventos", value)


def motivo(code) -> str:
    return term("motivos", code)


def base_de_referencia(status: str) -> str:
    return term("base_de_referencia", status)


def fator(name: str) -> str:
    return term("fatores", name)


def nivel_de_leitura(name: str) -> str:
    return term("niveis_de_leitura", name)


def nivel_curto(name: str) -> str:
    """Só o nível, sem a explicação — para caber em célula de tabela."""
    return nivel_de_leitura(name).split(" — ")[0]


def numero(value: float, casas: int = 2) -> str:
    """`0,29` — a convenção pt-BR já vale para o dado, e vale para o score."""
    return format_decimal(Decimal(str(round(value, casas))), min_decimals=casas)


#: Os identificadores escapam na prosa que o modelo escreve: a justificativa do
#: doc 05 diz "payment_date (10/07/2026) antecede ex_date", e a do doc 01 cita
#: "(FIELD_LOW_CONFIDENCE)" — ele é *instruído* a citar os códigos, e o
#: validador `only_assigned_codes` confere que só cite os atribuídos. Traduzir
#: na saída é determinístico; pedir ao modelo que não os escreva desmontaria a
#: checagem que impede ele de inventar motivo.
def _prose_pattern() -> re.Pattern:
    glossario = load_config("glossario")
    termos = list(glossario["campos"]) + list(glossario["motivos"])
    return re.compile(r"\b(" + "|".join(sorted(termos, key=len, reverse=True)) + r")\b")


_CODE_IN_PROSE = _prose_pattern()


#: O modelo escreve `DATE_INCOHERENCE: cronologia impossível` — o código como
#: rótulo e a glosa dele logo em seguida. Traduzir o rótulo aí produz
#: "cronologia impossível: cronologia impossível"; quando ele já explicou, o
#: rótulo é que sobra.
_LABELLED = re.compile(r"\b[A-Z][A-Z_]{5,}: ")


def em_portugues(text: str) -> str:
    """Troca identificadores pelos termos, dentro de texto corrido."""

    def troca(match: re.Match) -> str:
        key = match.group(1)
        section = "campos" if key in _table("campos") else "motivos"
        return term(section, key)

    return _CODE_IN_PROSE.sub(troca, _LABELLED.sub("", text))


def correspondencia(usados: set[str]) -> list[tuple[str, str]]:
    """As duas colunas do apêndice, geradas do glossário.

    Gerada e não escrita à mão, porque um apêndice que possa divergir do
    vocabulário que ele explica é pior que nenhum apêndice. Só os termos que
    aparecem neste relatório: listar o catálogo inteiro faria o apêndice pesar
    mais que o conteúdo.
    """
    linhas: list[tuple[str, str]] = []
    for section in ("campos", "motivos", "ferramentas", "leitores", "camadas", "tipos_de_erro"):
        for code, pt in _table(section).items():
            if code in usados:
                linhas.append((pt, code))
    return sorted(linhas, key=lambda pair: pair[0].lower())
