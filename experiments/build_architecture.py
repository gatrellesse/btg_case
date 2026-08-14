"""Injeta os exemplos reais na página de arquitetura.

Acrescenta três coisas ao HTML: os dados de `out/tool_examples.json` embutidos,
um nó de saída clicável depois de cada estágio da ingestão, e um modal que abre
a entrada e a saída daquela chamada. A tabela de ferramentas ganha o mesmo
botão em cada linha.

    PYTHONPATH=src python experiments/tool_examples.py
    python experiments/build_architecture.py <arquivo.html>
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

EXAMPLES = Path("out/tool_examples.json")
NODE_EXAMPLES = Path("out/node_examples.json")

# --- nova Figura 1: cada estágio ganha um nó de saída clicável --------------
FIGURE = '''      <svg viewBox="0 0 900 540" role="img"
           aria-label="Ingestão por consenso: o mesmo PDF passa por três leituras isoladas — camada de texto, OCR de página inteira e VLM. Cada uma tem um nó de saída clicável que abre o formato real que produz. As três são comparadas região a região por tipagem, validação e voto, produzindo blocos que carregam quais leitores concordaram.">
        <defs>
          <marker id="c3" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M0 0 L10 5 L0 10 z" fill="currentColor"/>
          </marker>
        </defs>

        <rect x="400" y="12" width="100" height="28" rx="14" fill="var(--accent-soft)" stroke="var(--accent)" stroke-width="1.2"/>
        <text x="450" y="30" font-family="ui-monospace,monospace" font-size="11" font-weight="600" fill="var(--accent)" text-anchor="middle">PDF</text>

        <rect x="40"  y="96" width="200" height="60" rx="6" fill="var(--surface)" stroke="var(--m-rule)"  stroke-width="1.6"/>
        <rect x="350" y="96" width="200" height="60" rx="6" fill="var(--surface)" stroke="var(--m-text)"  stroke-width="1.6"/>
        <rect x="660" y="96" width="200" height="60" rx="6" fill="var(--surface)" stroke="var(--m-image)" stroke-width="1.6"/>

        <g font-family="ui-monospace,monospace" text-anchor="middle">
          <g font-size="12" font-weight="600" fill="var(--ink)">
            <text x="140" y="120">parser</text>
            <text x="450" y="120">ocr</text>
            <text x="760" y="120">vlm</text>
          </g>
          <g font-size="9.5" fill="var(--ink-faint)">
            <text x="140" y="135">do_ocr=False</text>
            <text x="450" y="135">OcrMode.FULL_PAGE</text>
            <text x="760" y="135">granite-docling 258M</text>
            <text x="140" y="148">abstém-se em scan</text>
            <text x="450" y="148">descarta as células do PDF</text>
            <text x="760" y="148">gera tokens da imagem</text>
          </g>
        </g>

        <g stroke="currentColor" stroke-width="1.4" fill="none" marker-end="url(#c3)" color="var(--ink-soft)">
          <path d="M140 76 V 96"/><path d="M450 76 V 96"/><path d="M760 76 V 96"/>
          <path d="M140 156 V 176"/><path d="M450 156 V 176"/><path d="M760 156 V 176"/>
        </g>
        <g stroke="var(--ink-soft)" stroke-width="1.4" fill="none">
          <path d="M450 40 V 62 M140 62 H 760 M140 62 V 76 M760 62 V 76"/>
        </g>

        <g class="outnode" data-ex="docling:parser" tabindex="0" role="button"
           aria-label="Ver a saída real de docling:parser">
          <rect x="40" y="176" width="200" height="30" rx="5"/>
          <text x="140" y="195" text-anchor="middle">▣ 24 blocos · 14 células</text>
        </g>
        <g class="outnode" data-ex="docling:ocr" tabindex="0" role="button"
           aria-label="Ver a saída real de docling:ocr">
          <rect x="350" y="176" width="200" height="30" rx="5"/>
          <text x="450" y="195" text-anchor="middle">▣ 21 blocos · 13 células</text>
        </g>
        <g class="outnode" data-ex="docling:vlm" tabindex="0" role="button"
           aria-label="Ver a saída real de docling:vlm">
          <rect x="660" y="176" width="200" height="30" rx="5"/>
          <text x="760" y="195" text-anchor="middle">▣ 10 blocos · 0 células</text>
        </g>

        <g font-family="ui-monospace,monospace" font-size="9.5" font-weight="600" text-anchor="middle">
          <text x="140" y="224" fill="var(--m-rule)">erro 0,02</text>
          <text x="450" y="224" fill="var(--m-text)">erro 0,10</text>
          <text x="760" y="224" fill="var(--m-image)">erro 0,25</text>
        </g>

        <g stroke="currentColor" stroke-width="1.4" fill="none" marker-end="url(#c3)" color="var(--ink-soft)">
          <path d="M140 234 V 258 M450 234 V 258 M760 234 V 258"/>
        </g>
        <path d="M140 258 H 760" stroke="var(--ink-soft)" stroke-width="1.4" fill="none"/>
        <path d="M450 258 V 286" stroke="currentColor" stroke-width="1.4" fill="none" marker-end="url(#c3)" color="var(--ink-soft)"/>

        <rect x="250" y="286" width="400" height="86" rx="8" fill="var(--surface)" stroke="var(--accent)" stroke-width="1.6"/>
        <text x="450" y="310" font-family="ui-monospace,monospace" font-size="12" font-weight="600" fill="var(--ink)" text-anchor="middle">tipar → validar → votar</text>
        <g font-family="ui-monospace,monospace" font-size="9.5" fill="var(--ink-faint)" text-anchor="middle">
          <text x="450" y="328">estrito, não conserta · estrutura elimina · sobreviventes votam</text>
          <text x="450" y="344">mecanismos disjuntos multiplicam · derivados de pixel, não</text>
          <text x="450" y="360">abstenção não é discordância · eliminado sai do voto, não do placar</text>
        </g>

        <g font-family="ui-monospace,monospace" font-size="9.5" fill="var(--accent)">
          <text x="662" y="308">0,1124300000 = 0,112430000</text>
          <text x="662" y="319">mesmo Decimal, nunca foi conflito</text>
          <text x="662" y="342">BRTLNRACPR2 → 11 ≠ 12</text>
          <text x="662" y="353">ISO 6166 decide sem humano</text>
        </g>

        <path d="M450 372 V 396" stroke="currentColor" stroke-width="1.4" fill="none" marker-end="url(#c3)" color="var(--ink-soft)"/>

        <g class="outnode accent" data-ex="consensus.resolve" tabindex="0" role="button"
           aria-label="Ver a saída real do consenso">
          <rect x="250" y="396" width="400" height="30" rx="5"/>
          <text x="450" y="415" text-anchor="middle">▣ Block fundido · alternatives[] com quem divergiu</text>
        </g>

        <path d="M450 426 V 448" stroke="currentColor" stroke-width="1.4" fill="none" marker-end="url(#c3)" color="var(--ink-soft)"/>
        <rect x="250" y="448" width="400" height="46" rx="8" fill="var(--accent-soft)" stroke="var(--accent)" stroke-width="1.4"/>
        <text x="450" y="468" font-family="ui-monospace,monospace" font-size="11.5" font-weight="600" fill="var(--accent)" text-anchor="middle">blocks[] · readers[] · grau</text>
        <text x="450" y="484" font-family="ui-monospace,monospace" font-size="9.5" fill="var(--accent)" text-anchor="middle">quantos mecanismos independentes concordaram</text>

        <g font-family="ui-monospace,monospace" font-size="10" text-anchor="middle" font-weight="600">
          <text x="300" y="518" fill="var(--auto)">10 alta</text>
          <text x="400" y="518" fill="var(--review)">6 média</text>
          <text x="520" y="518" fill="var(--ink-faint)">11 leitor único</text>
          <text x="640" y="518" fill="var(--auto)">0 revisões</text>
        </g>
      </svg>'''


# --------------------------------------------------------------------------
# Figura 2 gerada a partir do grafo compilado.
#
# O layout é escolhido à mão porque legibilidade não se deriva; a TOPOLOGIA não
# é. Nós e arestas saem de `build_graph(...).get_graph()`, de modo que uma seta
# que não existe no código não pode ser desenhada, e um nó novo sem posição
# quebra a geração em vez de sumir da figura em silêncio.
# --------------------------------------------------------------------------

W, H = 168, 46          # caixa
LANE = {                # nó -> (x, y, cor da borda, legenda)
    "rule_classifier":  (216, 96,  "var(--m-rule)", "regex · marcadores da lei"),
    "text_classifier":  (516, 96,  "var(--m-text)", "LLM · texto extraído"),
    "consensus":        (366, 200, None, "2/2 para AUTO · peso igual"),
    "rule_extractor":   (76,  330, None, "âncoras · offline"),
    "text_extractor":   (366, 330, "var(--m-text)", "LLM · schema por tipo"),
    "merge":            (366, 418, None, "candidatos por extrator"),
    "grounding":        (366, 540, None, "evidência no texto CRU"),
    "reprompt_extract": (656, 540, None, "muda a entrada, não repete"),
    "repair":           (366, 618, None, "datas · moeda · IDs · derivações"),
    "sweep":            (366, 696, "var(--m-text)", "LLM · o que faltou, nas 3 leituras"),
    "validate":         (366, 794, None, "6 tools determinísticas"),
    "disambiguate":     (656, 794, None, "candidato compatível com o resto"),
    "triage":           (366, 868, None, "veredito em código"),
    "reporter":         (366, 956, None, "reason codes → texto do operador"),
}
BANDS = [
    (52, 214, "CLASSIFICAÇÃO · fan-out sobre o texto extraído"),
    (286, 198, "EXTRAÇÃO · fan-out, fan-in no merge"),
    (504, 156, "GROUNDING E REPARO · ciclo lateral, limitado"),
    (676, 92, "VARREDURA FINAL · o que faltou e o que ninguém corroborou"),
    (776, 148, "VALIDAÇÃO · ciclo lateral, limitado"),
    (942, 72, "REPORTER"),
]
#: Rótulo de aresta condicional, por (origem, destino).
COND = {
    ("grounding", "reprompt_extract"): "evidência ausente",
    ("grounding", "repair"): "evidência ok",
    ("validate", "disambiguate"): "valor ambíguo",
    ("validate", "triage"): "sem ambiguidade",
}


def _captured_nodes() -> set[str]:
    if not NODE_EXAMPLES.exists():
        return set()
    return {e["tool"] for e in json.loads(NODE_EXAMPLES.read_text(encoding="utf-8"))}


def _graph():
    from asset_servicing.graph import build_graph
    from asset_servicing.nodes import NodeContext
    from asset_servicing.validation.evidence import load_thresholds
    from asset_servicing.validation.tools import Validators

    th = load_thresholds()
    ctx = NodeContext(
        validators=Validators("Case AI Dev - Envio/golden_records/golden records.csv", th),
        thresholds=th, model="none", use_model=False,
    )
    return build_graph(ctx).get_graph()


def _edge_path(a: str, b: str) -> str:
    """Caminho de A para B. Lateral quando estão na mesma faixa; senão vertical
    com desvio em L, que é o que mantém as setas do fan-out separadas em vez de
    virarem um barramento horizontal — barra compartilhada lê como aresta entre
    os vizinhos, que é justamente o que não existe."""
    ax, ay = LANE[a][0], LANE[a][1]
    bx, by = LANE[b][0], LANE[b][1]
    acx, bcx = ax + W / 2, bx + W / 2

    if abs(ay - by) < 8:                      # mesma linha: ida e volta lateral
        if bx > ax:
            return f"M{ax + W} {ay + 16} H {bx}"
        return f"M{ax} {ay + 30} H {bx + W}"
    if abs(acx - bcx) < 8:                    # mesma coluna: reta
        return f"M{acx} {ay + H} V {by}"
    mid = ay + H + (by - ay - H) / 2          # colunas diferentes: L
    return f"M{acx} {ay + H} V {mid} H {bcx} V {by}"


def figure_two() -> str:
    g = _graph()
    nodes = [n for n in g.nodes if not n.startswith("__")]
    missing = [n for n in nodes if n not in LANE and n != "ingest"]
    if missing:
        raise SystemExit(f"nó sem posição no layout: {missing} — atualize LANE")
    # a guarda simétrica: posição sem nó desenharia uma caixa que não existe
    # mais no grafo, que é a mesma classe de erro na direção contrária
    stale = [n for n in LANE if n not in nodes]
    if stale:
        raise SystemExit(f"posição sem nó no grafo: {stale} — remova de LANE")

    altura = LANE["reporter"][1] + H + 80
    out = [f'      <svg viewBox="0 0 900 {altura}" role="img"',
           '           aria-label="Grafo do agente, gerado a partir das arestas reais: '
           'classificação em duas modalidades, extração em três nós paralelos reunidos no merge, '
           'uma varredura final que relê as três leituras atrás do que faltou, '
           'e dois ciclos laterais — grounding com reprompt_extract, validate com disambiguate.">',
           '        <defs><marker id="c2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
           'markerHeight="6" orient="auto-start-reverse">'
           '<path d="M0 0 L10 5 L0 10 z" fill="currentColor"/></marker></defs>']

    out.append('        <g fill="var(--band)">')
    for y, h, _ in BANDS:
        out.append(f'          <rect x="16" y="{y}" width="868" height="{h}" rx="8"/>')
    out.append("        </g>")
    out.append('        <g font-family="ui-monospace,monospace" font-size="10.5" '
               'letter-spacing="1.4" fill="var(--ink-faint)">')
    for y, _, label in BANDS:
        out.append(f'          <text x="30" y="{y + 19}">{label}</text>')
    out.append("        </g>")

    out.append('        <rect x="300" y="10" width="300" height="30" rx="8" '
               'fill="var(--accent-soft)" stroke="var(--accent)" stroke-width="1.3"/>')
    out.append('        <text x="450" y="30" font-family="ui-monospace,monospace" font-size="11" '
               'font-weight="600" fill="var(--accent)" text-anchor="middle">'
               'blocks[] · bbox · degrau · readers[]</text>')

    captured = _captured_nodes()
    for name, (x, y, stroke, _) in LANE.items():
        color = stroke or "var(--rule)"
        width = "1.6" if stroke else "1.2"
        rect = (f'<rect x="{x}" y="{y}" width="{W}" height="{H}" rx="6" '
                f'fill="var(--surface)" stroke="{color}" stroke-width="{width}"/>')
        if name in captured:
            # o ▣ só aparece onde existe captura real: um nó condicional que não
            # rodou no lote fica sem botão em vez de abrir um exemplo vazio
            out.append(f'        <g class="nodebtn" data-ex="{name}" tabindex="0" role="button" '
                       f'aria-label="Ver entrada e saída reais de {name}">{rect}'
                       f'<text x="{x + W - 9}" y="{y + 14}" text-anchor="end">▣</text></g>')
        else:
            out.append("        " + rect)

    out.append('        <g font-family="ui-monospace,monospace" text-anchor="middle">')
    out.append('          <g font-size="12" font-weight="600" fill="var(--ink)">')
    for name, (x, y, _, _) in LANE.items():
        out.append(f'            <text x="{x + W // 2}" y="{y + 22}">{name}</text>')
    out.append('          </g>')
    out.append('          <g font-size="9.5" fill="var(--ink-faint)">')
    for name, (x, y, _, sub) in LANE.items():
        out.append(f'            <text x="{x + W // 2}" y="{y + 37}">{sub}</text>')
    out.append("          </g>\n        </g>")

    # entrada: blocks[] alimenta os dois classificadores
    out.append('        <g stroke="var(--ink-soft)" stroke-width="1.4" fill="none">'
               '<path d="M450 40 V 60 M300 60 H 600 M300 60 V 74 M600 60 V 74"/></g>')
    out.append('        <g stroke="currentColor" stroke-width="1.4" fill="none" '
               'marker-end="url(#c2)" color="var(--ink-soft)">'
               '<path d="M300 74 V 96"/><path d="M600 74 V 96"/></g>')

    normal, conditional, labels = [], [], []
    for edge in g.edges:
        a, b = edge.source, edge.target
        if a not in LANE or b not in LANE:
            continue                      # __start__, __end__ e ingest (é a Figura 1)
        path = f'<path d="{_edge_path(a, b)}"/>'
        (conditional if (a, b) in COND else normal).append(path)
        if (a, b) in COND:
            ax, ay = LANE[a][0], LANE[a][1]
            bx, by = LANE[b][0], LANE[b][1]
            if abs(ay - by) < 8:
                labels.append((ax + W + 60, ay + 8, COND[(a, b)], "middle"))
            else:
                labels.append((ax + W // 2 + 12, ay + H + 22, COND[(a, b)], "start"))

    out.append('        <g stroke="currentColor" stroke-width="1.4" fill="none" '
               'marker-end="url(#c2)" color="var(--ink-soft)">')
    out.extend("          " + p for p in normal)
    out.append("        </g>")
    out.append('        <g stroke="currentColor" stroke-width="1.3" fill="none" '
               'marker-end="url(#c2)" color="var(--accent)">')
    out.extend("          " + p for p in conditional)
    out.append("        </g>")
    out.append('        <g font-family="ui-monospace,monospace" font-size="9.5" fill="var(--accent)">')
    for x, y, text, anchor in labels:
        out.append(f'          <text x="{x:.0f}" y="{y:.0f}" text-anchor="{anchor}">{text}</text>')
    out.append("        </g>")

    # A caixa do veredito segue o `reporter`, e não uma coordenada fixa: um nó
    # novo empurra a coluna inteira para baixo, e uma constante aqui desenharia
    # o veredito por cima do último nó.
    fim = LANE["reporter"][1] + H
    out.append(f'        <rect x="300" y="{fim + 26}" width="300" height="34" rx="8" '
               'fill="var(--surface)" stroke="var(--rule)" stroke-width="1.2"/>')
    out.append(f'        <path d="M450 {fim} V {fim + 26}" stroke="currentColor" '
               'stroke-width="1.4" fill="none" marker-end="url(#c2)" color="var(--ink-soft)"/>')
    out.append('        <g font-family="ui-monospace,monospace" font-size="10.5" '
               'text-anchor="middle" font-weight="600">'
               f'<text x="360" y="{fim + 48}" fill="var(--auto)">AUTO</text>'
               f'<text x="450" y="{fim + 48}" fill="var(--review)">REVIEW</text>'
               f'<text x="548" y="{fim + 48}" fill="var(--block)">BLOCKED</text></g>')
    out.append("      </svg>")
    print(f"figura 2: {len(nodes)} nós, {len(normal) + len(conditional)} arestas do grafo real")
    return "\n".join(out)


STYLE = """
  /* nós de saída clicáveis ------------------------------------------- */
  .outnode { cursor: zoom-in; }
  .outnode rect { fill: var(--sunk); stroke: var(--rule); stroke-width: 1.1; }
  .outnode text { font-family: var(--mono); font-size: 10.5px; fill: var(--ink-soft); }
  .outnode:hover rect, .outnode:focus rect { stroke: var(--accent); fill: var(--accent-soft); }
  .outnode:hover text, .outnode:focus text { fill: var(--accent); }
  .outnode:focus { outline: none; }
  .outnode.accent rect { border-color: var(--accent); }

  .nodebtn { cursor: zoom-in; }
  .nodebtn text { font-family: var(--mono); font-size: 11px; fill: var(--ink-faint); }
  .nodebtn:hover rect, .nodebtn:focus rect { stroke: var(--accent); stroke-width: 2; }
  .nodebtn:hover text, .nodebtn:focus text { fill: var(--accent); }
  .nodebtn:focus { outline: none; }

  .exbtn { font-family: var(--mono); font-size: 10.5px; cursor: zoom-in;
           background: var(--sunk); color: var(--ink-soft); border: 1px solid var(--rule);
           border-radius: 4px; padding: 2px 8px; white-space: nowrap; }
  .exbtn:hover, .exbtn:focus { border-color: var(--accent); color: var(--accent);
                               background: var(--accent-soft); outline: none; }

  /* modal ------------------------------------------------------------- */
  .exdim { position: fixed; inset: 0; background: rgba(8,12,18,.62);
           display: none; align-items: center; justify-content: center;
           padding: 24px; z-index: 50; }
  .exdim[open] { display: flex; }
  .exwin { background: var(--surface); border: 1px solid var(--rule); border-radius: 12px;
           max-width: 900px; width: 100%; max-height: 86vh; overflow: auto;
           padding: 22px 24px 26px; box-shadow: 0 24px 60px rgba(0,0,0,.32); }
  .exwin h3 { font-family: var(--mono); font-size: 15px; margin: 0 0 2px; }
  .exwin .lay { font-family: var(--mono); font-size: 10.5px; letter-spacing: .1em;
                text-transform: uppercase; color: var(--accent); }
  .exwin .why { font-size: 13.5px; color: var(--ink-soft); margin: 10px 0 18px;
                max-width: 72ch; }
  .exwin h4 { font-family: var(--mono); font-size: 10.5px; letter-spacing: .1em;
              text-transform: uppercase; color: var(--ink-faint); margin: 0 0 6px;
              font-weight: 500; }
  .exio { display: grid; gap: 16px; grid-template-columns: 1fr 1fr; }
  .exio.tri { grid-template-columns: 1fr 1.25fr 1fr; }
  @media (max-width: 72rem) { .exio.tri { grid-template-columns: 1fr; } }
  @media (max-width: 60rem) { .exio { grid-template-columns: 1fr; } }
  /* o painel do meio é código, não dado: fundo próprio para não ser lido como
     mais um JSON de entrada ou saída */
  .exlogic pre { background: var(--band); border-color: var(--accent);
                 font-size: 11.5px; }
  .exlogic h4 { color: var(--accent); }
  .exclose { float: right; font-family: var(--mono); font-size: 12px; cursor: pointer;
             background: none; border: 1px solid var(--rule); border-radius: 4px;
             color: var(--ink-soft); padding: 3px 9px; }
  .exclose:hover { border-color: var(--accent); color: var(--accent); }
"""

SCRIPT = """
<div class="exdim" id="exdim" role="dialog" aria-modal="true" aria-labelledby="extitle">
  <div class="exwin">
    <button class="exclose" id="exclose" aria-label="Fechar">esc ✕</button>
    <p class="lay" id="exlayer"></p>
    <h3 id="extitle"></h3>
    <p class="why" id="exwhy"></p>
    <div class="exio" id="exio">
      <div><h4>entrada</h4><pre id="exin"></pre></div>
      <div class="exlogic" id="exmid"><h4>o que o código faz</h4><pre id="exlogic"></pre></div>
      <div><h4>saída</h4><pre id="exout"></pre></div>
    </div>
  </div>
</div>
<script>
(function () {
  var data = JSON.parse(document.getElementById('exdata').textContent);
  var byTool = {};
  data.forEach(function (e) { byTool[e.tool] = e; });

  var dim = document.getElementById('exdim');
  var last = null;

  function open(tool) {
    var e = byTool[tool];
    if (!e) return;
    document.getElementById('exlayer').textContent = e.layer;
    document.getElementById('extitle').textContent = e.tool;
    document.getElementById('exwhy').textContent = e.why;
    document.getElementById('exin').textContent = JSON.stringify(e.input, null, 2);
    document.getElementById('exout').textContent = JSON.stringify(e.output, null, 2);
    // sem código capturado o painel do meio some, em vez de abrir vazio
    var mid = document.getElementById('exmid');
    var io = document.getElementById('exio');
    if (e.logic) {
      document.getElementById('exlogic').textContent = e.logic;
      mid.style.display = '';
      io.classList.add('tri');
    } else {
      mid.style.display = 'none';
      io.classList.remove('tri');
    }
    dim.setAttribute('open', '');
    document.getElementById('exclose').focus();
  }
  function close() {
    dim.removeAttribute('open');
    if (last) { last.focus(); last = null; }
  }

  document.addEventListener('click', function (ev) {
    var t = ev.target.closest('[data-ex]');
    if (t) { last = t; open(t.getAttribute('data-ex')); return; }
    if (ev.target.id === 'exclose' || ev.target === dim) close();
  });
  document.addEventListener('keydown', function (ev) {
    if (ev.key === 'Escape') close();
    if ((ev.key === 'Enter' || ev.key === ' ') && document.activeElement &&
        document.activeElement.hasAttribute('data-ex')) {
      ev.preventDefault();
      last = document.activeElement;
      open(document.activeElement.getAttribute('data-ex'));
    }
  });
})();
</script>
"""


def main() -> None:
    target = Path(sys.argv[1])
    html = target.read_text(encoding="utf-8")
    examples = json.loads(EXAMPLES.read_text(encoding="utf-8"))
    if NODE_EXAMPLES.exists():
        examples += json.loads(NODE_EXAMPLES.read_text(encoding="utf-8"))
    payload = json.dumps(examples, ensure_ascii=False).replace("</", "<\\/")

    # a figura do grafo é regerada em toda execução: é o que impede a topologia
    # de divergir do código entre um build e outro
    start = html.index('      <svg viewBox="0 0 900 10')
    end = html.index("</svg>", start) + len("</svg>")
    html = html[:start] + figure_two() + html[end:]

    # estilo entra sempre que faltar, e a guarda olha o seletor MAIS NOVO: o
    # caminho idempotente abaixo retorna cedo, então guardar por um seletor
    # antigo faz um estilo novo nunca chegar a uma página já construída — foi
    # assim que os nós da figura 2 saíram clicáveis mas sem cursor nem hover
    if ".exlogic pre {" not in html:
        html = html.replace("  .removed .n{color:var(--block)}",
                            "  .removed .n{color:var(--block)}" + STYLE, 1)

    # modal desatualizado: troca o bloco inteiro. Mesma razão do estilo — ele
    # só era injetado na primeira construção, então um painel novo nunca
    # chegava a uma página já feita.
    if 'id="exmid"' not in html and 'class="exdim"' in html:
        html = (html[:html.index('<div class="exdim"')]
                + SCRIPT.strip() + "\n"
                + html[html.rindex("</script>") + len("</script>"):])

    # já injetado: figura e modal não se duplicam, só os dados são atualizados
    if 'id="exdata"' in html:
        html = re.sub(r'(<script type="application/json" id="exdata">).*?(</script>)',
                      lambda m: m.group(1) + payload + m.group(2), html, flags=re.S)
        target.write_text(html, encoding="utf-8")
        print(f"dados atualizados ({len(examples)} exemplos) · {target}")
        return

    # 1. figura da ingestão com nós de saída
    start = html.index('      <svg viewBox="0 0 900 470"')
    end = html.index("</svg>", start) + len("</svg>")
    html = html[:start] + FIGURE + html[end:]

    # 2. estilos do nó e do modal
    html = html.replace("  .removed .n{color:var(--block)}", "  .removed .n{color:var(--block)}" + STYLE, 1)

    # 4. botão de exemplo em cada linha da tabela de ferramentas
    have = {e["tool"] for e in examples}
    html = html.replace("<th>Se falhar</th></tr>", "<th>Se falhar</th><th>Exemplo</th></tr>", 1)
    for tool in sorted(have, key=len, reverse=True):
        needle = f"<td><code>{tool}</code></td>"
        if needle not in html:
            continue
        row_end = html.index("</tr>", html.index(needle))
        button = (f'<td><button class="exbtn" data-ex="{tool}">▣ entrada · saída</button></td>')
        html = html[:row_end] + button + html[row_end:]

    # linhas sem exemplo capturado ficam com a coluna vazia, não com botão morto
    html = html.replace("<td>—</td></tr>", "<td>—</td><td></td></tr>")

    # 5. dados e modal
    html = html.replace(
        "</div>\n",
        f'</div>\n<script type="application/json" id="exdata">{payload}</script>\n{SCRIPT}\n',
        1,
    ) if "</div>\n" in html else html

    target.write_text(html, encoding="utf-8")
    n = sum(1 for e in examples if f'data-ex="{e["tool"]}"' in html)
    print(f"{n}/{len(examples)} exemplos ligados a um nó ou botão · {target}")


if __name__ == "__main__":
    main()
