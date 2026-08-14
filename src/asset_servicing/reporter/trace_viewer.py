"""Auditor da conversa: o que foi enviado ao modelo e o que ele devolveu.

O `viewer.html` responde "de onde veio este valor" — página, caixa, recorte.
Este responde a pergunta anterior: *o que o modelo recebeu para produzir isso*.
São perguntas diferentes e falham de formas diferentes — um valor errado pode
vir de prompt truncado, de resposta ignorada ou de schema que obrigou o modelo
a preencher campo ausente, e nenhuma das três se distingue olhando o registro.

Autocontido: as miniaturas das páginas vão embutidas como data URI, então o
arquivo abre em qualquer lugar sem servidor e sem rede.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

#: Papéis das partes que o pydantic-ai devolve, no vocabulário do painel.
PART_LABEL = {
    "system-prompt": "instruções",
    "user-prompt": "prompt",
    "text": "resposta",
    "tool-call": "chamada de ferramenta",
    "tool-return": "retorno da ferramenta",
    "retry-prompt": "retentativa exigida",
    "thinking": "raciocínio",
}


def _esc(value: Any) -> str:
    return html.escape(str(value))


def _render_part(part: dict[str, Any]) -> str:
    kind = part.get("kind", "")
    label = PART_LABEL.get(kind, kind)
    cls = "part " + kind.replace("-", "_")

    if kind == "tool-call":
        body = f"<pre>{_esc(json.dumps(part.get('args'), ensure_ascii=False, indent=2))}</pre>"
        return f'<div class="{cls}"><span class="plabel">{_esc(label)} · {_esc(part.get("tool"))}</span>{body}</div>'

    if "pieces" in part:
        chunks = []
        for piece in part["pieces"]:
            if piece.get("kind") == "image":
                kb = piece.get("bytes", 0) / 1024
                img = (
                    f'<img src="{piece["data_uri"]}" alt="página enviada ao modelo">'
                    if piece.get("data_uri")
                    else '<div class="noimg">imagem sem miniatura</div>'
                )
                chunks.append(
                    f'<figure class="sent-img">{img}'
                    f'<figcaption>{_esc(piece.get("media_type"))} · {kb:.0f} KB '
                    f"enviados ao modelo</figcaption></figure>"
                )
            else:
                clip = ' <span class="clip">truncado no viewer</span>' if piece.get("clipped") else ""
                chunks.append(f'<pre>{_esc(piece.get("text", ""))}</pre>{clip}')
        return f'<div class="{cls}"><span class="plabel">{_esc(label)}</span>{"".join(chunks)}</div>'

    clip = ' <span class="clip">truncado no viewer</span>' if part.get("clipped") else ""
    return (
        f'<div class="{cls}"><span class="plabel">{_esc(label)}</span>{clip}'
        f'<pre>{_esc(part.get("text", ""))}</pre></div>'
    )


def _render_call(call: dict[str, Any], index: int) -> str:
    sent, received = [], []
    for message in call.get("exchange", []):
        target = sent if message.get("role") == "request" else received
        target.extend(_render_part(p) for p in message.get("parts", []))

    usage = call.get("usage") or {}
    tokens = (
        f'<span class="meta">{usage.get("input_tokens", "?")} in · '
        f'{usage.get("output_tokens", "?")} out</span>'
        if usage
        else ""
    )
    error = (
        f'<div class="error"><span class="plabel">falhou</span><pre>{_esc(call["error"])}</pre></div>'
        if call.get("error")
        else ""
    )
    output = (
        f'<details class="out"><summary>saída tipada (o que virou registro)</summary>'
        f"<pre>{_esc(json.dumps(call['output'], ensure_ascii=False, indent=2))}</pre></details>"
        if call.get("output") is not None
        else ""
    )

    return f"""
<section class="call" id="call-{index}">
  <header class="call-head">
    <span class="agent">{_esc(call.get("agent"))}</span>
    <span class="doc">{_esc(call.get("document") or "—")}</span>
    <span class="meta">{call.get("seconds", 0)}s</span>
    {tokens}
  </header>
  <div class="cols">
    <div class="col sent"><h3>enviado</h3>{"".join(sent) or '<p class="empty">nada registrado</p>'}</div>
    <div class="col recv"><h3>recebido</h3>{"".join(received) or '<p class="empty">nada registrado</p>'}{error}</div>
  </div>
  {output}
</section>"""


def build_trace_viewer(calls: list[dict[str, Any]], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    by_agent: dict[str, int] = {}
    failures = 0
    for call in calls:
        by_agent[call.get("agent", "?")] = by_agent.get(call.get("agent", "?"), 0) + 1
        if call.get("error"):
            failures += 1

    chips = "".join(
        f'<span class="chip"><b>{n}</b> {_esc(agent)}</span>'
        for agent, n in sorted(by_agent.items())
    )
    if failures:
        chips += f'<span class="chip bad"><b>{failures}</b> com falha</span>'

    body = "".join(_render_call(c, i) for i, c in enumerate(calls))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_PAGE.replace("<!--CHIPS-->", chips).replace("<!--CALLS-->", body),
                        encoding="utf-8")
    return out_path


_PAGE = """<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Conversa com o modelo — asset servicing</title>
<style>
:root {
  --paper:#F6F7F9; --card:#FFF; --ink:#101722; --muted:#5A6472;
  --rule:#DCE0E7; --soft:#EDF0F4;
  --sent:#1C5C86; --recv:#2C6A50; --bad:#A23A2B;
  --mono: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  --sans: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --paper:#0E1319; --card:#151C25; --ink:#E4E8EE; --muted:#939DAB;
  --rule:#263040; --soft:#1B222D;
  --sent:#6BAEDC; --recv:#63BE91; --bad:#E2806F;
} }
:root[data-theme="dark"] {
  --paper:#0E1319; --card:#151C25; --ink:#E4E8EE; --muted:#939DAB;
  --rule:#263040; --soft:#1B222D;
  --sent:#6BAEDC; --recv:#63BE91; --bad:#E2806F;
}
body { background:var(--paper); color:var(--ink); font-family:var(--sans);
       margin:0; padding:2.5rem 1.5rem 4rem; line-height:1.6; }
.wrap { max-width:80rem; margin:0 auto; }
h1 { font-family:var(--mono); font-size:1.5rem; margin:0 0 .5rem; letter-spacing:-.02em; }
.lede { color:var(--muted); max-width:62ch; margin:0 0 1.25rem; }
.chips { display:flex; flex-wrap:wrap; gap:.5rem; margin-bottom:2rem; }
.chip { font-family:var(--mono); font-size:.75rem; border:1px solid var(--rule);
        background:var(--card); border-radius:999px; padding:.25rem .75rem; }
.chip b { font-variant-numeric:tabular-nums; }
.chip.bad { color:var(--bad); border-color:var(--bad); }
.call { background:var(--card); border:1px solid var(--rule); border-radius:6px;
        margin-bottom:1.25rem; overflow:hidden; }
.call-head { display:flex; gap:.9rem; align-items:baseline; flex-wrap:wrap;
             padding:.6rem .9rem; border-bottom:1px solid var(--rule);
             background:var(--soft); font-family:var(--mono); font-size:.78rem; }
.agent { font-weight:650; }
.doc { color:var(--muted); }
.meta { color:var(--muted); margin-left:auto; font-variant-numeric:tabular-nums; }
.cols { display:grid; grid-template-columns:1fr 1fr; gap:1px; background:var(--rule); }
@media (max-width:64rem) { .cols { grid-template-columns:1fr; } }
.col { background:var(--card); padding:.9rem; min-width:0; }
.col h3 { font-family:var(--mono); font-size:.72rem; text-transform:uppercase;
          letter-spacing:.1em; margin:0 0 .6rem; }
.sent h3 { color:var(--sent); }
.recv h3 { color:var(--recv); }
.part { margin-bottom:.8rem; }
.plabel { font-family:var(--mono); font-size:.68rem; text-transform:uppercase;
          letter-spacing:.08em; color:var(--muted); }
pre { font-family:var(--mono); font-size:.76rem; line-height:1.5; background:var(--soft);
      border:1px solid var(--rule); border-radius:4px; padding:.6rem .7rem;
      margin:.3rem 0 0; white-space:pre-wrap; word-break:break-word;
      max-height:26rem; overflow:auto; }
.clip { font-family:var(--mono); font-size:.68rem; color:var(--muted); }
.sent-img { margin:.4rem 0 0; }
.sent-img img { max-width:100%; height:auto; border:1px solid var(--rule); border-radius:4px; }
.sent-img figcaption { font-family:var(--mono); font-size:.68rem; color:var(--muted);
                       margin-top:.25rem; }
.noimg { font-family:var(--mono); font-size:.72rem; color:var(--muted); }
.error pre { border-color:var(--bad); color:var(--bad); }
.out { border-top:1px solid var(--rule); padding:.6rem .9rem; }
.out summary { font-family:var(--mono); font-size:.74rem; cursor:pointer; color:var(--muted); }
.out pre { margin-top:.5rem; }
.empty { font-family:var(--mono); font-size:.75rem; color:var(--muted); }
</style></head><body><div class="wrap">
<h1>Conversa com o modelo</h1>
<p class="lede">Cada bloco é uma chamada de agente: à esquerda o que foi enviado
(instruções, prompt, imagem da página), à direita o que voltou. A saída tipada é
o que efetivamente virou registro — quando ela diverge do texto da resposta, o
schema é que decidiu.</p>
<div class="chips"><!--CHIPS--></div>
<!--CALLS-->
</div></body></html>
"""
