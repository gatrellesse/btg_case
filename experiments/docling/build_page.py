"""Gera out/consensus/pipeline.html a partir dos dados medidos.

A página mostra, para cada tipo de documento, a mesma página renderizada quatro
vezes: uma por motor, com as caixas do que aquele motor leu, e uma quarta com a
fusão — cada região colorida por quantos motores a cobriram.

    PYTHONPATH=experiments/docling python experiments/docling/build_page.py
"""

from __future__ import annotations

import html
import json
from pathlib import Path

OVERLAYS = Path("out/consensus/overlays.json")
CONSENSUS = Path("out/consensus/consensus.json")
OUT = Path("out/consensus/pipeline.html")

LANES = [("parser", "parser"), ("ocr", "ocr"), ("vlm", "vlm")]

CASES = [
    {
        "doc": "01_energetica_vale_tiete_dividendo",
        "eyebrow": "Documento nativo",
        "title": "01_energetica_vale_tiete_dividendo.pdf",
        "prose": [
            "Camada de texto íntegra na página inteira. Os três motores atravessam e "
            "cobrem as mesmas nove regiões — a fusão sai verde de ponta a ponta.",
            "O voto combina mecanismos disjuntos: a camada de texto devolve a string do "
            "próprio programa que gerou o PDF, o OCR classifica tinta renderizada. Eles "
            "não compartilham modo de falha, então as taxas de erro se multiplicam.",
        ],
        "caption": "Nenhum candidato morre no portão, então o score é o produto das três "
                   "taxas de erro: 1 − (0,02 × 0,10 × 0,25) = 0,9995.",
    },
    {
        "doc": "07_telecom_norte_jcp_SCAN",
        "eyebrow": "Documento escaneado",
        "title": "07_telecom_norte_jcp_SCAN.pdf",
        "prose": [
            "Sem camada de texto, o painel do parser fica vazio: zero bloco. Abstenção "
            "não é discordância — não derruba score, apenas deixa cada campo com dois "
            "leitores.",
            "E os dois que sobram leem os mesmos pixels, então erram junto. Por isso a "
            "concordância entre eles vale menos que a camada de texto sozinha: "
            "0,95 contra 0,98.",
        ],
        "caption": "As dezesseis regiões saem em âmbar — dois leitores em todas elas, "
                   "nenhuma com os três.",
    },
    {
        "doc": "09_hibrido_tabela_rasterizada",
        "eyebrow": "Documento híbrido",
        "title": "09_hibrido_tabela_rasterizada.pdf",
        "prose": [
            "Construído para o experimento: a tabela do doc 01 rasterizada e o texto "
            "por baixo apagado — 181 palavras na página, zero na região da tabela.",
            "O painel do parser mostra o efeito em uma imagem: a prosa vem cheia, e "
            "sobre a tabela sobra um contorno tracejado — o modelo de layout enxerga a "
            "tabela, o leitor não tira uma palavra dela. A fusão separa a página em "
            "dois regimes na mesma folha.",
        ],
        "caption": "Oito regiões com três leitores, uma com um só. A faixa cinza é a "
                   "tabela: apenas o OCR chegou lá.",
    },
]

FIELD_ROWS = {
    "01_energetica_vale_tiete_dividendo": [
        ("gross_per_share", "0.4275", "0.9995", "parser · ocr · vlm", "alta", "g-high"),
        ("tax_rate", "0.1", "0.9995", "parser · ocr · vlm", "alta", "g-high"),
        ("com_date", "2026-06-12", "0.9995", "parser · ocr · vlm", "alta", "g-high"),
        ("ex_date", "2026-06-15", "0.9995", "parser · ocr · vlm", "alta", "g-high"),
        ("payment_date", "2026-07-03", "0.9995", "parser · ocr · vlm", "alta", "g-high"),
        ("isin", "BRTIETACNOR3", "0.9995", "parser · ocr · vlm", "alta", "g-high"),
    ],
    "07_telecom_norte_jcp_SCAN": [
        ("isin", "BRTLNRACNPR2", "0.8100", "ocr — vlm eliminado", "um leitor", "g-single"),
        ("gross_per_share", "0.11243", "0.9500", "ocr · vlm", "média", "g-medium"),
        ("net_per_share", "0.09275475", "0.9500", "ocr · vlm", "média", "g-medium"),
        ("tax_rate", "0.175", "0.9500", "ocr · vlm", "média", "g-medium"),
        ("com_date", "2026-06-22", "0.9500", "ocr · vlm", "média", "g-medium"),
        ("payment_date", "2026-08-21", "0.7500", "vlm — ocr não tipou", "um leitor", "g-single"),
    ],
    "09_hibrido_tabela_rasterizada": [
        ("tax_rate", "0.1", "0.9995", "prosa · três motores", "alta", "g-high"),
        ("gross_per_share", "0.4275", "0.9000", "tabela · só ocr", "um leitor", "g-single"),
        ("com_date", "2026-06-12", "0.7290", "tabela · só ocr", "um leitor", "g-single"),
        ("ex_date", "2026-06-15", "0.9000", "tabela · só ocr", "um leitor", "g-single"),
        ("payment_date", "2026-07-03", "0.9000", "tabela · só ocr", "um leitor", "g-single"),
        ("isin", "BRTIETACNOR3", "0.9000", "tabela · só ocr", "um leitor", "g-single"),
    ],
}

NOTES = {
    "07_telecom_norte_jcp_SCAN": (
        "<strong>A aritmética fecha por cima.</strong> Com bruto, alíquota e líquido "
        "presentes, <span class=\"m\">0,11243 × (1 − 0,175) − 0,09275475 = 0</span> — "
        "delta exato. Três campos que se confirmam entre si valem mais que três "
        "leitores concordando sobre um deles."
    ),
    "09_hibrido_tabela_rasterizada": (
        "<strong>Por que com_date cai para 0,729.</strong> O parser acha a âncora "
        "<span class=\"m\">data-base</span> na prosa mas não consegue tipar data "
        "nenhuma ali — o texto escreve o dia por extenso. Rótulo encontrado sem valor "
        "tipável é leitor que sofreu na região: sai do voto e leva "
        "<span class=\"m\">0,90</span> do score junto."
    ),
}


def mini_page(entry: dict, mode: str | None) -> str:
    """SVG da página com as caixas de um motor — ou da fusão, se mode é None."""
    w, h = entry["width"], entry["height"]
    boxes = entry["merged"] if mode is None else entry["modes"][mode]
    label = "fusão" if mode is None else mode

    parts = [
        f'<svg class="mini" viewBox="0 0 {w:.0f} {h:.0f}" role="img" '
        f'aria-label="Página com as regiões lidas por {label}">',
        f'<image href="{entry["page"]}" x="0" y="0" width="{w:.0f}" height="{h:.0f}"/>',
    ]
    for b in boxes:
        if mode is None:
            cls = {3: "mg3", 2: "mg2", 1: "mg1"}[b["n"]]
            tip = f'{b["n"]} leitores: {", ".join(b["readers"])}'
        elif b["filled"]:
            cls = f"bx {mode[0]}x"
            tip = b["text"] or b["label"]
        else:
            cls = "bx empty"
            tip = f'{b["label"]} — caixa sem conteúdo'
        parts.append(
            f'<rect class="{cls}" x="{b["x"]:.0f}" y="{b["y"]:.0f}" '
            f'width="{b["w"]:.0f}" height="{b["h"]:.0f}" rx="1">'
            f"<title>{html.escape(tip)}</title></rect>"
        )
    parts.append("</svg>")
    return "".join(parts)


def panel(entry: dict, mode: str | None) -> str:
    stats = entry["stats"]
    if mode is None:
        cov = entry["coverage"]
        meta = " · ".join(
            f"{cov[k]}×{k}" for k in ("3", "2", "1") if int(cov.get(k, 0)) > 0
        )
        head = '<span class="pl merge">fusão</span>'
        sub = f"{meta} leitores"
    else:
        s = stats[mode]
        head = f'<span class="pl {mode}">{mode}</span>'
        blocks = s["blocks"]
        sub = (
            "nenhum bloco" if blocks == 0
            else f'{blocks} blocos · {s["chars"]} car'
        )
    return (
        f'<figure class="panel">{mini_page(entry, mode)}'
        f'<figcaption class="panel-cap">{head}<span>{html.escape(sub)}</span>'
        f"</figcaption></figure>"
    )


def case_section(case: dict, data: dict) -> str:
    entry = data[case["doc"]]
    panels = "".join(panel(entry, m) for m in ("parser", "ocr", "vlm", None))
    prose = "".join(f"<p>{p}</p>" for p in case["prose"])
    rows = "".join(
        f'<tr><td class="field">{f}</td><td class="val">{v}</td>'
        f'<td class="num">{s}</td><td class="val">{who}</td>'
        f'<td><span class="grade {cls}">{g}</span></td></tr>'
        for f, v, s, who, g, cls in FIELD_ROWS[case["doc"]]
    )
    note = NOTES.get(case["doc"])
    note_html = f'<p class="note">{note}</p>' if note else ""
    return f"""
<section class="case">
  <div class="case-head">
    <p class="eyebrow">{case["eyebrow"]}</p>
    <h2>{case["title"]}</h2>
    <div class="prose">{prose}</div>
  </div>
  <div class="panels">{panels}</div>
  <p class="figcap">{case["caption"]}</p>
  <div class="tablescroll">
    <table>
      <thead><tr><th>campo</th><th>valor</th><th>score</th><th>faixas</th><th>grau</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
  {note_html}
</section>"""


def main() -> None:
    data = json.loads(OVERLAYS.read_text(encoding="utf-8"))
    cases = "".join(case_section(c, data) for c in CASES)
    OUT.write_text(TEMPLATE.replace("<!--CASES-->", cases), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


TEMPLATE = r"""<title>Consenso entre modalidades — nativo, escaneado, híbrido</title>

<style>
  :root {
    --paper:#F6F7F9; --card:#FFFFFF; --ink:#101722; --muted:#5A6472;
    --rule:#DCE0E7; --rule-soft:#EAEDF1;
    --parser:#1C5C86; --ocr:#8F6220; --vlm:#6B4B8E;
    --pass:#2C6A50; --reject:#A23A2B;
    --parser-bg:#EAF1F7; --ocr-bg:#F7F0E4; --vlm-bg:#F2EDF7;
    --pass-bg:#E7F1EC; --reject-bg:#F8EAE7;
    --mono: ui-monospace, "SF Mono", "JetBrains Mono", Menlo, Consolas, monospace;
    --sans: system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --paper:#0E1319; --card:#151C25; --ink:#E4E8EE; --muted:#939DAB;
      --rule:#263040; --rule-soft:#1C2430;
      --parser:#6BAEDC; --ocr:#D4A45E; --vlm:#AE90D0;
      --pass:#63BE91; --reject:#E2806F;
      --parser-bg:#16283A; --ocr-bg:#302516; --vlm-bg:#271E35;
      --pass-bg:#14291F; --reject-bg:#2E1A16;
    }
  }
  :root[data-theme="dark"] {
    --paper:#0E1319; --card:#151C25; --ink:#E4E8EE; --muted:#939DAB;
    --rule:#263040; --rule-soft:#1C2430;
    --parser:#6BAEDC; --ocr:#D4A45E; --vlm:#AE90D0;
    --pass:#63BE91; --reject:#E2806F;
    --parser-bg:#16283A; --ocr-bg:#302516; --vlm-bg:#271E35;
    --pass-bg:#14291F; --reject-bg:#2E1A16;
  }

  body {
    background: var(--paper); color: var(--ink); font-family: var(--sans);
    font-size:16px; line-height:1.65; margin:0; padding:3rem 1.5rem 5rem;
  }
  .wrap { max-width: 72rem; margin: 0 auto; }
  .prose { max-width: 62ch; }

  h1 { font-family:var(--mono); font-size:clamp(1.5rem,3.4vw,2.1rem); font-weight:600;
       letter-spacing:-.02em; line-height:1.25; text-wrap:balance; margin:0 0 1rem; }
  h2 { font-family:var(--mono); font-size:1.1rem; font-weight:600;
       letter-spacing:-.01em; margin:0 0 .35rem; }
  h3 { font-family:var(--mono); font-size:.8rem; font-weight:600; letter-spacing:.08em;
       text-transform:uppercase; color:var(--muted); margin:0 0 .75rem; }
  p { margin:0 0 1rem; }
  strong { font-weight:650; }
  code, .m { font-family:var(--mono); font-size:.89em; }
  code { background:var(--rule-soft); padding:.1em .34em; border-radius:3px; }

  .eyebrow { font-family:var(--mono); font-size:.72rem; letter-spacing:.14em;
             text-transform:uppercase; color:var(--muted); margin:0 0 .4rem; }
  .lede { font-size:1.05rem; color:var(--muted); }
  header.top { border-bottom:1px solid var(--rule); padding-bottom:2rem; margin-bottom:2.5rem; }

  .chips { display:flex; flex-wrap:wrap; gap:.6rem; margin-top:1.5rem; }
  .chip { font-family:var(--mono); font-size:.78rem; border:1px solid var(--rule);
          background:var(--card); border-radius:999px; padding:.3rem .8rem;
          display:flex; gap:.45rem; align-items:baseline; }
  .chip b { font-weight:650; font-variant-numeric:tabular-nums; }
  .chip span { color:var(--muted); }

  .legend { display:grid; grid-template-columns:repeat(auto-fit,minmax(14rem,1fr));
            gap:1px; background:var(--rule); border:1px solid var(--rule);
            border-radius:6px; overflow:hidden; margin:0 0 3.5rem; }
  .legend > div { background:var(--card); padding:1rem 1.1rem; }
  .legend .key { font-family:var(--mono); font-size:.8rem; font-weight:650;
                 display:flex; align-items:center; gap:.5rem; margin-bottom:.3rem; }
  .legend .swatch { width:1.6rem; height:3px; border-radius:2px; flex:none; }
  .legend p { font-size:.86rem; color:var(--muted); margin:0; line-height:1.5; }

  .case { margin:0 0 4.5rem; }
  .case-head { margin-bottom:1.5rem; }

  /* mini pages -------------------------------------------------------- */
  .panels { display:grid; grid-template-columns:repeat(4,1fr); gap:1rem;
            align-items:start; }
  @media (max-width: 60rem) { .panels { grid-template-columns:repeat(2,1fr); } }
  .panel { margin:0; }
  svg.mini { display:block; width:100%; height:auto; background:var(--card);
             border:1px solid var(--rule); border-radius:4px; }
  .mini image { opacity:.34; }
  .mini rect { fill:none; stroke-width:1; vector-effect:non-scaling-stroke; }
  .mini .px { stroke:var(--parser); fill:var(--parser); fill-opacity:.14; }
  .mini .ox { stroke:var(--ocr);    fill:var(--ocr);    fill-opacity:.14; }
  .mini .vx { stroke:var(--vlm);    fill:var(--vlm);    fill-opacity:.14; }
  .mini .empty { stroke:var(--muted); stroke-dasharray:3 2; fill:none; opacity:.65; }
  .mini .mg3 { stroke:var(--pass);   fill:var(--pass);   fill-opacity:.2; }
  .mini .mg2 { stroke:var(--ocr);    fill:var(--ocr);    fill-opacity:.2; }
  .mini .mg1 { stroke:var(--reject); fill:var(--reject); fill-opacity:.2; }

  .panel-cap { display:flex; flex-direction:column; gap:.15rem; margin-top:.5rem;
               font-family:var(--mono); font-size:.72rem; color:var(--muted); }
  .pl { font-weight:650; letter-spacing:.04em; }
  .pl.parser { color:var(--parser); }
  .pl.ocr { color:var(--ocr); }
  .pl.vlm { color:var(--vlm); }
  .pl.merge { color:var(--ink); }

  .figcap { font-size:.87rem; color:var(--muted); margin:1rem 0 1.5rem; max-width:62ch; }

  .tablescroll { overflow-x:auto; border:1px solid var(--rule); border-radius:6px; }
  table { border-collapse:collapse; width:100%; font-size:.85rem; background:var(--card); }
  th, td { text-align:left; padding:.5rem .8rem; border-bottom:1px solid var(--rule-soft);
           white-space:nowrap; }
  thead th { font-family:var(--mono); font-size:.7rem; letter-spacing:.07em;
             text-transform:uppercase; color:var(--muted); font-weight:600;
             border-bottom:1px solid var(--rule); }
  tbody tr:last-child td { border-bottom:none; }
  td.num, td.val { font-family:var(--mono); font-variant-numeric:tabular-nums; }
  td.field { font-family:var(--mono); }

  .grade { font-family:var(--mono); font-size:.72rem; padding:.12rem .5rem;
           border-radius:3px; border:1px solid; }
  .g-high   { color:var(--pass);   border-color:var(--pass);   background:var(--pass-bg); }
  .g-medium { color:var(--ocr);    border-color:var(--ocr);    background:var(--ocr-bg); }
  .g-single { color:var(--muted);  border-color:var(--rule);   background:var(--rule-soft); }
  .g-rej    { color:var(--reject); border-color:var(--reject); background:var(--reject-bg); }

  .note { border-left:2px solid var(--rule); padding-left:1rem; margin:1.5rem 0 0;
          font-size:.92rem; color:var(--muted); max-width:62ch; }
  .note strong { color:var(--ink); }

  figure.gatefig { margin:0 0 1.25rem; }
  .figscroll { overflow-x:auto; }
  svg.flow { display:block; width:100%; min-width:40rem; height:auto; }
  .flow text { font-family:var(--mono); fill:var(--ink); }
  .flow .lbl { font-size:12px; }
  .flow .lbl-xs { font-size:9.5px; fill:var(--muted); }
  .flow .box { fill:var(--card); stroke:var(--rule); stroke-width:1; }
  .flow .edge { stroke:var(--muted); stroke-width:1.2; fill:none; }
  .flow .out-box { fill:var(--pass-bg); stroke:var(--pass); stroke-width:1.4; }
  .flow .out-txt { fill:var(--pass); }
  .flow .rej { stroke:var(--reject); stroke-width:1.6; }
  .flow .rej-txt { fill:var(--reject); }
  figcaption.fc { font-size:.87rem; color:var(--muted); margin-top:.75rem; max-width:62ch; }

  footer { border-top:1px solid var(--rule); margin-top:4rem; padding-top:2rem;
           font-size:.88rem; color:var(--muted); }
</style>

<div class="wrap">

<header class="top">
  <p class="eyebrow">Asset servicing · leitura de avisos</p>
  <h1>Três leitores, uma página, um voto</h1>
  <div class="prose">
    <p class="lede">
      A mesma página lida por três mecanismos independentes — camada de texto, OCR
      determinístico e VLM generativo. Cada painel mostra onde aquele motor esteve;
      o quarto mostra a fusão, região por região.
    </p>
    <p>
      O pipeline padrão do Docling não produz isso: ele elimina os clusters que já
      têm texto programático antes de chamar o OCR, então as duas fontes nunca leem
      a mesma região — por construção não existe segunda opinião. Aqui cada motor lê
      a página inteira, e a sobreposição vira o score.
    </p>
  </div>
  <div class="chips">
    <div class="chip"><b>3</b><span>documentos</span></div>
    <div class="chip"><b>27</b><span>campos avaliados</span></div>
    <div class="chip"><b>10</b><span>alta corroboração</span></div>
    <div class="chip"><b>1</b><span>candidato eliminado pela estrutura</span></div>
    <div class="chip"><b>0</b><span>revisões humanas necessárias</span></div>
  </div>
</header>

<h3>Como ler os painéis</h3>
<div class="legend">
  <div>
    <div class="key"><span class="swatch" style="background:var(--parser)"></span>parser</div>
    <p><code>do_ocr=False</code> — a string do próprio programa que gerou o PDF.
    Erra pouco e <em>abstém-se</em> quando não há camada de texto.</p>
  </div>
  <div>
    <div class="key"><span class="swatch" style="background:var(--ocr)"></span>ocr</div>
    <p><code>OcrMode.FULL_PAGE</code> — descarta as células do PDF e classifica
    tinta renderizada. Lê qualquer página, inclusive nativa.</p>
  </div>
  <div>
    <div class="key"><span class="swatch" style="background:var(--vlm)"></span>vlm</div>
    <p><code>granite-docling 258M</code> — gera tokens a partir da imagem.
    Estrutura melhor, transcreve pior: corrompeu 2 dos 8 ISIN do lote.</p>
  </div>
  <div>
    <div class="key"><span class="swatch" style="background:var(--pass)"></span>fusão</div>
    <p>Verde: três motores na mesma região. Âmbar: dois. Vermelho: um só.
    Contorno tracejado é caixa <em>sem</em> conteúdo — o layout viu, o leitor não leu.</p>
  </div>
</div>

<!--CASES-->

<section class="case">
  <div class="case-head">
    <p class="eyebrow">O portão</p>
    <h2>Ler → tipar → validar → votar</h2>
    <div class="prose">
      <p>
        A ordem não é estilística. Cada passo existe por um caso medido neste lote,
        e votar cedo demais custa revisão humana em conflito que a estrutura
        resolvia sozinha.
      </p>
    </div>
  </div>

  <figure class="gatefig">
    <div class="figscroll">
      <svg class="flow" viewBox="0 0 1000 244" role="img"
           aria-label="Quatro candidatos reais atravessam tipagem e validação: dois sobrevivem e viram voto, dois são eliminados antes de qualquer comparação.">
        <defs>
          <marker id="a-g" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7"
                  markerHeight="7" orient="auto-start-reverse">
            <polygon points="0,1 8,4 0,7" fill="context-stroke"/>
          </marker>
        </defs>
        <text class="lbl-xs" x="120" y="26" text-anchor="middle">candidato lido</text>
        <text class="lbl-xs" x="420" y="26" text-anchor="middle">tipar (estrito, não conserta)</text>
        <text class="lbl-xs" x="700" y="26" text-anchor="middle">validar (estrutura)</text>
        <text class="lbl-xs" x="915" y="26" text-anchor="middle">votar</text>

        <rect class="box" x="16" y="44" width="208" height="38" rx="3"/>
        <text class="lbl" x="28" y="68">0,1124300000 · 0,112430000</text>
        <line class="edge" x1="224" y1="63" x2="300" y2="63" marker-end="url(#a-g)"/>
        <rect class="box" x="304" y="44" width="232" height="38" rx="3"/>
        <text class="lbl" x="316" y="68">Decimal("0.11243") ×2</text>
        <line class="edge" x1="536" y1="63" x2="600" y2="63" marker-end="url(#a-g)"/>
        <rect class="box" x="604" y="44" width="192" height="38" rx="3"/>
        <text class="lbl" x="616" y="68">passa</text>
        <line class="edge" x1="796" y1="63" x2="836" y2="63" marker-end="url(#a-g)"/>
        <rect class="box out-box" x="840" y="44" width="150" height="38" rx="3"/>
        <text class="lbl out-txt" x="915" y="68" text-anchor="middle">nunca foi conflito</text>

        <rect class="box" x="16" y="96" width="208" height="38" rx="3"/>
        <text class="lbl rej-txt" x="28" y="120">BRTLNRACPR2</text>
        <line class="edge" x1="224" y1="115" x2="300" y2="115" marker-end="url(#a-g)"/>
        <rect class="box" x="304" y="96" width="232" height="38" rx="3"/>
        <text class="lbl" x="316" y="120">tipa como identificador</text>
        <line class="edge" x1="536" y1="115" x2="600" y2="115" marker-end="url(#a-g)"/>
        <rect class="box" x="604" y="96" width="192" height="38" rx="3"/>
        <text class="lbl rej-txt" x="616" y="120">11 ≠ 12 · ISO 6166</text>
        <line class="rej" x1="812" y1="105" x2="832" y2="125"/>
        <line class="rej" x1="812" y1="125" x2="832" y2="105"/>
        <text class="lbl-xs rej-txt" x="915" y="120" text-anchor="middle">eliminado antes do voto</text>

        <rect class="box" x="16" y="148" width="208" height="38" rx="3"/>
        <text class="lbl rej-txt" x="28" y="172">.../08/22026</text>
        <line class="edge" x1="224" y1="167" x2="300" y2="167" marker-end="url(#a-g)"/>
        <rect class="box" x="304" y="148" width="232" height="38" rx="3"/>
        <text class="lbl rej-txt" x="316" y="172">não é data — recusa</text>
        <line class="rej" x1="546" y1="157" x2="566" y2="177"/>
        <line class="rej" x1="546" y1="177" x2="566" y2="157"/>
        <text class="lbl-xs rej-txt" x="700" y="172" text-anchor="middle">nunca chega a validar</text>
        <text class="lbl-xs" x="915" y="172" text-anchor="middle">score −10%</text>

        <rect class="box" x="16" y="200" width="208" height="38" rx="3"/>
        <text class="lbl" x="28" y="224">BRTIETACNOR3</text>
        <line class="edge" x1="224" y1="219" x2="300" y2="219" marker-end="url(#a-g)"/>
        <rect class="box" x="304" y="200" width="232" height="38" rx="3"/>
        <text class="lbl" x="316" y="224">12 caracteres · passa</text>
        <line class="edge" x1="536" y1="219" x2="600" y2="219" marker-end="url(#a-g)"/>
        <rect class="box" x="604" y="200" width="192" height="38" rx="3"/>
        <text class="lbl" x="616" y="224">mod 10 falha · INFO</text>
        <line class="edge" x1="796" y1="219" x2="836" y2="219" marker-end="url(#a-g)"/>
        <rect class="box out-box" x="840" y="200" width="150" height="38" rx="3"/>
        <text class="lbl out-txt" x="915" y="224" text-anchor="middle">vota mesmo assim</text>
      </svg>
    </div>
    <figcaption class="fc">
      A última linha separa checagem de veto: os ISIN deste lote são sintéticos e
      <em>nenhum</em> fecha o mod 10 — o ISIN real <span class="m">US0378331005</span>
      fecha, então a implementação está certa e o dado é que é fabricado. Reprovar
      por checksum reprovaria o lote inteiro.
    </figcaption>
  </figure>
</section>

<footer>
  <div class="prose">
    <p>
      <strong>O que este score não diz.</strong> Ele mede transcrição — se os
      leitores concordam sobre o que está escrito na região. Não diz se a região é a
      do campo certo, nem se o valor existe no registro. No doc 01 os três leitores
      leem <span class="m">0,4275</span> na mesma célula e o documento não tem valor
      líquido nenhum; no doc 08 os três leem <span class="m">BRCNHZACNOR5</span>
      corretamente e o emissor não está nos golden records. Unânime e errado são
      compatíveis, e por isso os três eixos — transcrição, identificação, validade —
      não se somam num número só.
    </p>
    <p>
      <strong>Custo.</strong> O VLM roda a 12–23 s por página contra ~1 s do parser.
      Três passadas completas em todo documento não se pagam: acionar a segunda e a
      terceira modalidade só em campo crítico, ou quando a primeira devolve
      <span class="m">parse_score</span> baixo ou nenhuma âncora casada.
    </p>
    <p>
      Gerado por <span class="m">experiments/docling/build_page.py</span> a partir de
      <span class="m">overlays.json</span> e <span class="m">consensus.json</span>.
    </p>
  </div>
</footer>

</div>
"""

if __name__ == "__main__":
    main()
