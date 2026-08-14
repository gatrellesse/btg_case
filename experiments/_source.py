"""Extrai o miolo do código de uma função, para o painel do meio.

Entrada e saída dizem o que mudou; não dizem por quê. Entre as duas falta a
regra — e descrevê-la em prosa recria o problema que estes exemplos existem para
evitar: a descrição envelhece e o código não. Então o painel do meio mostra o
código de verdade, lido da fonte na hora da captura.

Corta o invólucro (a fábrica, o `return`) e a docstring, porque a docstring já
vira o texto do "por quê" — o que sobra é a decisão.
"""

from __future__ import annotations

import inspect
import re
import textwrap

MAX_LINES = 30


def essence(obj, max_lines: int = MAX_LINES) -> str:
    """O código que decide, sem o andaime em volta."""
    try:
        src = textwrap.dedent(inspect.getsource(obj))
    except (OSError, TypeError):
        return ""

    lines = src.splitlines()

    # fábrica de nó: `def make_x(ctx):` ... `return x_node` — o que interessa é
    # a função interna, não o fecho
    if lines and re.match(r"\s*def make_\w+\(", lines[0]):
        lines = lines[1:]
        while lines and (not lines[-1].strip() or re.match(r"\s*return \w+_node\s*$", lines[-1])):
            lines.pop()
        src = textwrap.dedent("\n".join(lines))
        lines = src.splitlines()

    # docstring fora: ela já é o "por quê" do painel
    out, i = [], 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith(('"""', "'''")):
            quote = stripped[:3]
            if stripped.count(quote) < 2 or len(stripped) == 3:
                i += 1
                while i < len(lines) and quote not in lines[i]:
                    i += 1
            i += 1
            continue
        out.append(line)
        i += 1

    while out and not out[0].strip():
        out.pop(0)
    while out and not out[-1].strip():
        out.pop()

    if len(out) > max_lines:
        out = out[:max_lines] + ["", f"# … mais {len(out) - max_lines} linhas"]
    return "\n".join(out)
