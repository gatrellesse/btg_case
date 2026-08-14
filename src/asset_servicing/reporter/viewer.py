"""Builds a self-contained HTML auditor for a run.

The exceptions report answers "what needs attention". This answers the other
question an operator has: *show me the page, and show me where every value came
from.* Each document is rendered once and every extracted field is drawn as a
box over it, so a wrong reading is visible rather than inferred from a score.

Self-contained by design — page images are embedded as data URIs and the boxes
are SVG in PDF-point space, so the file opens anywhere with no assets, no
server and no network.
"""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import pymupdf

from ..models import Disposition, EventRecord, GroundingStatus

PAGE_DPI = 110


def _page_images(pdf_path: Path, dpi: int = PAGE_DPI) -> list[tuple[str, float, float]]:
    """One data-URI JPEG per page, with its size in PDF points."""
    out = []
    try:
        with pymupdf.open(pdf_path) as pdf:
            for page in pdf:
                pix = page.get_pixmap(dpi=dpi)
                data = pix.tobytes("jpeg", jpg_quality=72)
                encoded = base64.b64encode(data).decode("ascii")
                out.append(
                    (f"data:image/jpeg;base64,{encoded}", page.rect.width, page.rect.height)
                )
    except Exception:
        pass
    return out


def _status(record: EventRecord, name: str) -> str:
    """How a field should read at a glance, from strongest signal down."""
    field = record.fields.get(name)
    if field is None or field.value is None:
        return "absent"
    if field.grounding.status == GroundingStatus.ABSENT:
        return "ungrounded"
    if name in record.triage.fields_for_review:
        return "review"
    return "ok"


def _field_rows(record: EventRecord) -> list[dict]:
    rows = []
    for name, field in record.fields.items():
        rows.append(
            {
                "name": name,
                "value": "—" if field.value is None else str(field.value),
                "raw": field.value_raw or "",
                "level": field.reading_level,
                "golden": field.audit.golden.status if field.audit else "nao_aplicavel",
                "grounding": field.grounding.status.value,
                "reader": field.reader_kind.value if field.reader_kind else "—",
                "page": field.bbox.page if field.bbox else None,
                "box": (
                    [field.bbox.x0, field.bbox.y0, field.bbox.x1, field.bbox.y1]
                    if field.bbox
                    else None
                ),
                "evidence": (field.evidence_text or "")[:220],
                "absence": field.absence_reason.value if field.absence_reason else "",
                "repairs": [
                    {
                        "rule": r.rule,
                        "from": r.value_from,
                        "to": r.value_to,
                        "candidates": r.candidates,
                    }
                    for r in field.repairs
                ],
                "notes": field.notes,
                "status": _status(record, name),
            }
        )
    rows.sort(key=lambda r: (r["box"] is None, r["box"][1] if r["box"] else 0))
    return rows


def _document_payload(record: EventRecord, pdf_dir: Path) -> dict:
    pdf_path = pdf_dir / record.document.file_name
    pages = _page_images(pdf_path)
    return {
        "file": record.document.file_name,
        "event_type": record.event_type.value,
        "disposition": record.triage.disposition.value,
        "weakest_level": record.triage.weakest_level,
        "justification": record.triage.justification,
        "readers": [k.value for k in record.document.reader_kinds],
        "classification": record.classification.rationale,
        "pages": [{"src": src, "w": w, "h": h} for src, w, h in pages],
        "fields": _field_rows(record),
        "reasons": [
            {
                "code": r.code.value,
                "severity": r.severity.value,
                "field": r.field or "",
                "message": r.message,
            }
            for r in record.triage.reasons
        ],
        "validations": [
            {
                "tool": v.tool,
                "status": v.status.value,
                "message": v.message,
                "detail": v.detail,
            }
            for v in record.validations
        ],
    }


def build_viewer(records: list[EventRecord], pdf_dir: str | Path, out_path: str | Path) -> Path:
    payload = [_document_payload(r, Path(pdf_dir)) for r in records]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render(payload), encoding="utf-8")
    return out_path


def _render(payload: list[dict]) -> str:
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    counts = {}
    for doc in payload:
        counts[doc["disposition"]] = counts.get(doc["disposition"], 0) + 1
    stp = counts.get("auto", 0) / max(len(payload), 1)
    summary = " · ".join(f"{k} {v}" for k, v in sorted(counts.items()))
    return _TEMPLATE.replace("__DATA__", data).replace(
        "__SUMMARY__", html.escape(f"{len(payload)} documentos · {summary} · STP {stp:.0%}")
    )


_TEMPLATE = """<title>Auditor de extração — eventos corporativos</title>
<style>
  :root{
    --ground:#F7F8FA;--surface:#FFFFFF;--sunk:#EDEFF3;--rule:#D8DDE5;
    --ink:#11151B;--ink-soft:#454D59;--ink-faint:#7C8593;
    --accent:#1B4B73;--accent-soft:#E4EBF2;
    --ok:#2C6A45;--ok-bg:rgba(44,106,69,.13);
    --review:#8C6114;--review-bg:rgba(140,97,20,.15);
    --bad:#94382B;--bad-bg:rgba(148,56,43,.15);
    --mono:ui-monospace,'SF Mono','Cascadia Mono','Roboto Mono',Menlo,Consolas,monospace;
    --sans:ui-sans-serif,-apple-system,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif;
  }
  @media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
    --ground:#0E1218;--surface:#161C24;--sunk:#1C232D;--rule:#2C3542;
    --ink:#E7EBF1;--ink-soft:#A8B2BF;--ink-faint:#77828F;
    --accent:#7FB2DC;--accent-soft:#1B2836;
    --ok:#6FBF8E;--ok-bg:rgba(111,191,142,.16);
    --review:#D9A94A;--review-bg:rgba(217,169,74,.18);
    --bad:#E08272;--bad-bg:rgba(224,130,114,.18);
  }}
  :root[data-theme="dark"]{
    --ground:#0E1218;--surface:#161C24;--sunk:#1C232D;--rule:#2C3542;
    --ink:#E7EBF1;--ink-soft:#A8B2BF;--ink-faint:#77828F;
    --accent:#7FB2DC;--accent-soft:#1B2836;
    --ok:#6FBF8E;--ok-bg:rgba(111,191,142,.16);
    --review:#D9A94A;--review-bg:rgba(217,169,74,.18);
    --bad:#E08272;--bad-bg:rgba(224,130,114,.18);
  }
  *{box-sizing:border-box}
  body{background:var(--ground);color:var(--ink);font-family:var(--sans);
       margin:0;padding:28px 20px 60px;line-height:1.55;-webkit-font-smoothing:antialiased}
  .wrap{max-width:1280px;margin:0 auto}
  h1{font-size:22px;letter-spacing:-.015em;font-weight:650;margin:0 0 4px}
  .sub{font-family:var(--mono);font-size:11.5px;color:var(--ink-faint);margin:0 0 20px}
  .tabs{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:18px}
  .tab{font-family:var(--mono);font-size:11.5px;padding:7px 11px;border-radius:6px;
       border:1px solid var(--rule);background:var(--surface);color:var(--ink-soft);
       cursor:pointer;display:flex;align-items:center;gap:7px}
  .tab:hover{border-color:var(--accent)}
  .tab[aria-selected="true"]{background:var(--accent-soft);border-color:var(--accent);color:var(--ink)}
  .dot{width:8px;height:8px;border-radius:50%;flex:none}
  .dot.auto{background:var(--ok)}.dot.review{background:var(--review)}
  .dot.blocked,.dot.failed{background:var(--bad)}
  .grid{display:grid;grid-template-columns:minmax(320px,1fr) minmax(340px,1fr);gap:18px;align-items:start}
  @media(max-width:900px){.grid{grid-template-columns:1fr}}
  .card{background:var(--surface);border:1px solid var(--rule);border-radius:10px;padding:14px}
  .pagebox{position:sticky;top:14px}
  svg.page{width:100%;height:auto;display:block;border-radius:6px;background:#fff}
  svg.page rect.f{fill:transparent;stroke-width:1.4;cursor:pointer}
  svg.page rect.f.ok{stroke:var(--ok)}
  svg.page rect.f.review{stroke:var(--review)}
  svg.page rect.f.ungrounded{stroke:var(--bad)}
  svg.page rect.f.on{stroke-width:2.6;fill:var(--accent-soft);fill-opacity:.42}
  .hd{display:flex;flex-wrap:wrap;gap:10px;align-items:baseline;margin-bottom:10px}
  .pill{font-family:var(--mono);font-size:10.5px;padding:3px 8px;border-radius:999px;
        text-transform:uppercase;letter-spacing:.06em}
  .pill.auto{background:var(--ok-bg);color:var(--ok)}
  .pill.review{background:var(--review-bg);color:var(--review)}
  .pill.blocked,.pill.failed{background:var(--bad-bg);color:var(--bad)}
  .just{font-size:13px;color:var(--ink-soft);margin:0 0 12px}
  table{width:100%;border-collapse:collapse;font-size:12.5px}
  th{text-align:left;font-family:var(--mono);font-size:10px;letter-spacing:.09em;
     text-transform:uppercase;color:var(--ink-faint);font-weight:500;padding:7px 8px;
     border-bottom:1px solid var(--rule)}
  td{padding:7px 8px;border-bottom:1px solid var(--rule);vertical-align:top}
  tr.row{cursor:pointer}
  tr.row:hover td{background:var(--sunk)}
  tr.row.on td{background:var(--accent-soft)}
  .fname{font-family:var(--mono);font-size:11.5px}
  .val{font-family:var(--mono);font-weight:600}
  .val.absent{color:var(--ink-faint);font-weight:400}
  .meta{font-family:var(--mono);font-size:10.5px;color:var(--ink-faint)}
  .bar{width:44px;height:5px;border-radius:3px;background:var(--sunk);overflow:hidden;margin-top:4px}
  .bar i{display:block;height:100%}
  .bar i.ok{background:var(--ok)}.bar i.review{background:var(--review)}
  .bar i.ungrounded,.bar i.absent{background:var(--bad)}
  .detail{font-size:12px;color:var(--ink-soft);padding:2px 8px 10px}
  .detail div{margin-top:3px}
  .tag{font-family:var(--mono);font-size:10px;padding:1px 5px;border-radius:4px;background:var(--sunk)}
  h3{font-size:12px;font-family:var(--mono);letter-spacing:.08em;text-transform:uppercase;
     color:var(--ink-faint);font-weight:500;margin:16px 0 7px}
  .reason{display:flex;gap:8px;font-size:12.5px;padding:5px 0;border-bottom:1px solid var(--rule)}
  .reason code{font-family:var(--mono);font-size:11px}
  .sev{flex:none;width:4px;border-radius:2px}
  .sev.block{background:var(--bad)}.sev.review{background:var(--review)}.sev.info{background:var(--ink-faint)}
  .vd{display:flex;justify-content:space-between;gap:10px;font-size:12px;padding:5px 0;
      border-bottom:1px solid var(--rule)}
  .vs{font-family:var(--mono);font-size:10.5px}
  .vs.pass{color:var(--ok)}.vs.fail{color:var(--bad)}.vs.warn{color:var(--review)}.vs.info{color:var(--ink-faint)}
  .legend{display:flex;gap:14px;flex-wrap:wrap;font-family:var(--mono);font-size:10.5px;
          color:var(--ink-faint);margin-top:10px}
  .legend span{display:flex;align-items:center;gap:5px}
  .sw{width:14px;height:0;border-top:2px solid}
</style>
<div class="wrap">
  <h1>Auditor de extração</h1>
  <p class="sub">__SUMMARY__ — clique num campo para ver onde ele foi lido na página</p>
  <div class="tabs" id="tabs" role="tablist"></div>
  <div class="grid">
    <div class="card pagebox" id="pagebox"></div>
    <div><div class="card" id="panel"></div></div>
  </div>
</div>
<script>
const DOCS = __DATA__;
// A escada, do mais forte ao mais fraco: a barra tem quatro degraus e não
// cem por cento, porque a evidência tem quatro estados e não cem.
const LEVELS = ["baixa","media","alta","muito_alta"];
let cur = 0, sel = null;
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));

function tabs(){
  document.getElementById("tabs").innerHTML = DOCS.map((d,i)=>
    `<button class="tab" role="tab" aria-selected="${i===cur}" onclick="pick(${i})">
       <span class="dot ${d.disposition}"></span>${esc(d.file.replace(/\\.pdf$/i,""))}</button>`).join("");
}

function page(){
  const d = DOCS[cur], p = d.pages[0];
  if(!p){ document.getElementById("pagebox").innerHTML = "<p class='meta'>página não renderizada</p>"; return; }
  // Boxes are drawn in PDF-point space, so the viewBox maps them 1:1 onto the
  // page image at any display size.
  const boxes = d.fields.filter(f=>f.box && f.page===1).map(f=>
    `<rect class="f ${f.status}" data-f="${esc(f.name)}" x="${f.box[0]}" y="${f.box[1]}"
       width="${Math.max(f.box[2]-f.box[0],1)}" height="${Math.max(f.box[3]-f.box[1],1)}"
       rx="2" onclick="sel='${esc(f.name)}';draw()"><title>${esc(f.name)}: ${esc(f.value)}</title></rect>`).join("");
  document.getElementById("pagebox").innerHTML =
    `<svg class="page" viewBox="0 0 ${p.w} ${p.h}" xmlns="http://www.w3.org/2000/svg">
       <image href="${p.src}" x="0" y="0" width="${p.w}" height="${p.h}"/>${boxes}</svg>
     <div class="legend">
       <span><i class="sw" style="border-color:var(--ok)"></i>conferido</span>
       <span><i class="sw" style="border-color:var(--review)"></i>revisar</span>
       <span><i class="sw" style="border-color:var(--bad)"></i>sem evidência</span>
     </div>`;
}

function panel(){
  const d = DOCS[cur];
  const rows = d.fields.map(f=>{
    const degrau = LEVELS.indexOf(f.level) + 1;
    const det = sel===f.name ? `<tr><td colspan="3" class="detail">
        ${f.raw && f.raw!==f.value ? `<div>lido como <span class="tag">${esc(f.raw)}</span></div>`:""}
        ${f.evidence ? `<div>evidência: “${esc(f.evidence)}”</div>`:""}
        ${f.absence ? `<div>ausência: <span class="tag">${esc(f.absence)}</span></div>`:""}
        ${f.repairs.map(r=>`<div>reparo <span class="tag">${esc(r.rule)}</span>
            ${r.candidates.length ? "candidatos: "+r.candidates.map(esc).join(", ")
              : (r.from!=null? esc(r.from)+" → "+esc(r.to) : "")}</div>`).join("")}
        ${f.notes.map(n=>`<div class="meta">${esc(n)}</div>`).join("")}
      </td></tr>` : "";
    return `<tr class="row ${sel===f.name?"on":""}" onclick="sel=${sel===f.name?"null":`'${esc(f.name)}'`};draw()">
        <td class="fname">${esc(f.name)}</td>
        <td><span class="val ${f.value==="—"?"absent":""}">${esc(f.value)}</span>
            <div class="bar"><i class="${f.status}" style="width:${degrau*25}%"></i></div></td>
        <td class="meta">${esc(f.level)}<br>${esc(f.grounding)}<br>${esc(f.golden)}<br>${esc(f.reader)}</td>
      </tr>${det}`;
  }).join("");

  document.getElementById("panel").innerHTML = `
    <div class="hd">
      <span class="pill ${d.disposition}">${esc(d.disposition)}</span>
      <strong>${esc(d.event_type)}</strong>
      <span class="meta">leitura crítica mais fraca: ${esc(d.weakest_level)} · ${esc(d.readers.join(", "))}</span>
    </div>
    <p class="just">${esc(d.justification)}</p>
    <table><thead><tr><th>campo</th><th>valor</th><th>leitura / evidência / base / leitor</th></tr></thead>
      <tbody>${rows}</tbody></table>
    ${d.reasons.length? `<h3>Motivos</h3>${d.reasons.map(r=>
      `<div class="reason"><i class="sev ${r.severity}"></i><div>
         <code>${esc(r.code)}</code>${r.field?` <span class="meta">(${esc(r.field)})</span>`:""}
         <div class="meta">${esc(r.message)}</div></div></div>`).join("")}`:""}
    <h3>Validações</h3>
    ${d.validations.map(v=>`<div class="vd"><span>${esc(v.tool)}</span>
        <span class="vs ${v.status}">${esc(v.status)}</span></div>`).join("")}
    <h3>Classificação</h3><p class="meta">${esc(d.classification)}</p>`;
}

function draw(){ page(); panel(); tabs(); }
function pick(i){ cur=i; sel=null; draw(); }
draw();
</script>
"""
