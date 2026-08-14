"""Consenso por campo entre as três extrações isoladas.

Ordem: ler -> tipar -> validar -> votar entre os sobreviventes.

Votar antes de tipar produz conflito onde não há: `0,1124300000` e `0,112430000`
são strings diferentes e o mesmo Decimal. Votar antes de validar gasta revisão
humana com candidato que a estrutura já reprova: os dois ISIN corrompidos pelo
VLM têm 11 caracteres onde ISO 6166 exige 12 — isso se decide sem segunda
opinião.

Três eixos, conjuntivos, reportados separados:

    transcrição     os leitores concordam sobre o que está escrito na região
    identificação   a região é a do campo (âncora de rótulo / varredura ISO)
    validade        o valor tipa, passa checksum e fecha a aritmética

Um campo pode ser unânime no primeiro e errado no segundo: no doc 01 os três
leitores leem `0,4275` na mesma célula, e o documento não tem valor líquido
nenhum. Média entre eixos esconderia isso; por isso eles não se somam.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import yaml

# --------------------------------------------------------------------------
# taxa de erro por classe de mecanismo
#
# Não são votos equivalentes. A camada de texto devolve a string do próprio
# programa que gerou o PDF; o OCR classifica tinta renderizada; o VLM gera
# tokens. Medido nestes 8 documentos: o VLM corrompeu 2 dos 8 ISIN (um deles em
# PDF nativo, onde a string correta estava na camada de texto), o OCR nenhum.
# Amostra pequena — a ordem importa mais que os valores.
# --------------------------------------------------------------------------
ERROR_RATE = {
    "text_layer": 0.02,
    "ocr_deterministic": 0.10,
    "ocr_generative": 0.25,
}

#: Mecanismos que leem os mesmos pixels erram junto; concordância entre eles
#: vale menos que entre camada de texto e pixel, que não compartilham falha.
PIXEL_DERIVED = {"ocr_deterministic", "ocr_generative"}

READERS = {
    "parser": "text_layer",
    "ocr": "ocr_deterministic",
    "vlm": "ocr_generative",
}

FIELD_TYPES = {
    "event_type_label": "text",
    "gross_per_share": "money",
    "net_per_share": "money",
    "cost_basis": "money",
    "tax_rate": "rate",
    "ratio": "text",
    "approval_date": "date",
    "com_date": "date",
    "ex_date": "date",
    "payment_date": "date",
    "trading_start": "date",
    "credit_date": "date",
    "fraction_period": "text",
}

#: Identificadores não dependem de rótulo — varredura estrutural no documento
#: inteiro, como manda a camada 1 do config/anchors.yaml.
STRUCTURAL = {
    # 11 a 13 caracteres terminando em dígito: largo o bastante para capturar o
    # ISIN corrompido de 11 (que precisa aparecer para ser reprovado), estreito
    # o bastante para não casar com "ACIONISTAS" nem "PARTICIPACOES"
    "isin": re.compile(r"\b([A-Z]{2}[A-Z0-9]{8,10}[0-9])\b"),
    "ticker": re.compile(r"\b([A-Z]{4}[0-9]{1,2})\b"),
}

MONEY_RE = re.compile(r"R\$\s*([0-9]+(?:\.[0-9]{3})*,[0-9]+)")
# alíquota sai com e sem decimal: "17,5%" no JCP, "10%" no dividendo
RATE_RE = re.compile(r"([0-9]{1,2}(?:,[0-9]+)?)\s*%")
DATE_RE = re.compile(r"\b([0-3][0-9])/([0-1][0-9])/([0-9]{4})\b")

#: Sequências de pontos/vírgulas/reticências que separam rótulo de valor. O
#: mesmo leader impresso sai `....`, `,,,,.,,` ou `……` conforme o motor.
LEADER = re.compile(r"[.,…·]{2,}|\s{3,}")


# --------------------------------------------------------------------------
# normalização e tipagem
# --------------------------------------------------------------------------
def deaccent(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)
    )


def norm(s: str) -> str:
    """Minúsculas, sem acento, pontuação colapsada — o formato das âncoras.

    Aspas somem dos dois lados do casamento: o mesmo rótulo impresso sai
    `Data 'ex-dividendos'`, `Data "ex-dividendos"` ou `Data ex-dividendos`
    conforme o motor, e nenhuma dessas variações é informação.
    """
    s = deaccent(unicodedata.normalize("NFKC", s)).lower()
    s = re.sub(r"[\"'“”’‘]", "", s)
    s = re.sub(r"[^a-z0-9\s()/-]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def isin_structure_ok(code: str) -> bool:
    """ISO 6166, forma: 12 caracteres, prefixo de país, verificador numérico.

    É esta a checagem que elimina candidato. Os dois ISIN corrompidos pelo VLM
    têm 11 caracteres — decidido sem segunda opinião.
    """
    return len(code) == 12 and code[:2].isalpha() and code[-1].isdigit()


def isin_checksum_ok(code: str) -> bool:
    """Dígito verificador mod 10. INFO, nunca veto — os ISIN deste lote são
    sintéticos e nenhum fecha (o de verdade, US0378331005, fecha)."""
    if not isin_structure_ok(code):
        return False
    digits = "".join(str(int(c, 36)) for c in code)
    total, double = 0, True
    for ch in reversed(digits[:-1]):
        d = int(ch) * (2 if double else 1)
        total += d - 9 if d > 9 else d
        double = not double
    return (10 - total % 10) % 10 == int(digits[-1])


#: compat: o eixo de validade reporta os dois separados
def isin_valid(code: str) -> bool:
    return isin_checksum_ok(code)


def type_value(raw: str, kind: str) -> tuple[Any | None, str | None, str]:
    """(valor tipado, forma canônica, status). Estrito: não conserta, recusa."""
    if kind == "money":
        m = MONEY_RE.search(raw)
        if not m:
            return None, None, "untyped"
        try:
            v = Decimal(m.group(1).replace(".", "").replace(",", "."))
        except InvalidOperation:
            return None, None, "untyped"
        # normalize() apaga a diferença entre 0,1124300000 e 0,112430000
        return v, format(v.normalize(), "f"), "ok"

    if kind == "rate":
        m = RATE_RE.search(raw)
        if not m:
            return None, None, "untyped"
        v = Decimal(m.group(1).replace(",", ".")) / 100
        return v, format(v.normalize(), "f"), "ok"

    if kind == "date":
        m = DATE_RE.search(raw)
        if not m:
            return None, None, "untyped"
        d, mo, y = (int(x) for x in m.groups())
        try:
            v = date(y, mo, d)
        except ValueError:
            return None, None, "invalid"  # 31/02: tipa como data, não existe
        return v, v.isoformat(), "ok"

    if kind == "isin":
        code = raw.strip().upper()
        if not isin_structure_ok(code):
            return code, code, "invalid"
        return code, code, "ok"

    v = norm(raw)
    return (v, v, "ok") if v else (None, None, "untyped")


# --------------------------------------------------------------------------
# leitura dos markdown e localização dos campos
# --------------------------------------------------------------------------
def pairs_from(md: str) -> list[tuple[str, str]]:
    """(rótulo, valor) de linhas de tabela e de prosa com leader."""
    out: list[tuple[str, str]] = []
    lines = [ln.strip() for ln in md.splitlines()]
    for i, line in enumerate(lines):
        if not line:
            continue
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if all(set(c) <= set("-: ") for c in cells):
                continue
            if len(cells) >= 2:
                out.append((cells[0], " ".join(cells[1:])))
            continue
        line = re.sub(r"^#+\s*", "", line)
        parts = LEADER.split(line, maxsplit=1)
        if len(parts) > 1:
            out.append((parts[0], parts[1]))
        else:
            # rótulo e valor em parágrafos separados (o VLM faz isso): olha a
            # próxima linha não vazia como valor candidato
            nxt = next((x for x in lines[i + 1 : i + 3] if x), "")
            out.append((line, nxt))
    return out


def load_anchors(path: Path) -> dict[str, list[str]]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {f: sorted(a, key=len, reverse=True) for f, a in cfg["fields"].items()}


def match_field(label: str, anchors: dict[str, list[str]]) -> tuple[str | None, str]:
    """Âncora MAIS LONGA vence — sem isso 'valor bruto por acao' e 'valor
    liquido por acao' colidiriam em 'valor' (config/anchors.yaml)."""
    nl = norm(label)
    best_field, best_anchor = None, ""
    for fname, alist in anchors.items():
        for a in alist:
            if a in nl and len(a) > len(best_anchor):
                best_field, best_anchor = fname, a
    return best_field, best_anchor


@dataclass
class Candidate:
    reader: str
    kind: str
    raw: str
    canon: str | None
    status: str  # ok | untyped | invalid
    anchor: str = ""
    note: str = ""


def read_document(md: str, anchors: dict[str, list[str]], reader: str) -> dict[str, Candidate]:
    kind = READERS[reader]
    found: dict[str, Candidate] = {}

    for label, value in pairs_from(md):
        fname, anchor = match_field(label, anchors)
        if not fname:
            continue
        ftype = FIELD_TYPES.get(fname, "text")
        # o valor pode ter sido fundido no rótulo (o OCR faz isso quando a
        # célula perde a borda): tenta os dois lados
        for source in (value, label):
            typed, canon, status = type_value(source, ftype)
            if status == "ok" or typed is not None:
                break
        if canon is None and status == "untyped":
            # leitor achou o rótulo e não produziu valor tipável: isso é
            # informação (leitor sofreu na região), não silêncio
            found.setdefault(
                fname,
                Candidate(reader, kind, value or label, None, "untyped", anchor),
            )
            continue
        prev = found.get(fname)
        if prev is None or (prev.status != "ok" and status == "ok"):
            found[fname] = Candidate(reader, kind, source, canon, status, anchor)

    # identificadores: varredura estrutural, independente de rótulo
    for fname, pattern in STRUCTURAL.items():
        for m in pattern.finditer(md):
            code = m.group(1)
            typed, canon, status = type_value(code, fname if fname == "isin" else "text")
            if fname == "isin" and status != "ok":
                # guarda o inválido: ele precisa constar para penalizar o campo
                found.setdefault(
                    fname, Candidate(reader, kind, code, canon, "invalid", "ISO 6166")
                )
                continue
            found[fname] = Candidate(reader, kind, code, canon, "ok", "ISO 6166")
            break
    return found


# --------------------------------------------------------------------------
# os três eixos
# --------------------------------------------------------------------------
def transcription(cands: dict[str, Candidate]) -> dict[str, Any]:
    survivors = {r: c for r, c in cands.items() if c.status == "ok"}
    eliminated = {r: c for r, c in cands.items() if c.status != "ok"}

    if not survivors:
        return {
            "grade": "absent" if not eliminated else "no_valid_reading",
            "score": None,
            "value": None,
            "supporting": [],
            "eliminated": {r: c.status for r, c in eliminated.items()},
        }

    values = {c.canon for c in survivors.values()}
    if len(values) > 1:
        return {
            "grade": "conflict",
            "score": None,
            "value": None,
            "supporting": sorted(survivors),
            "readings": {r: c.canon for r, c in survivors.items()},
            "eliminated": {r: c.status for r, c in eliminated.items()},
        }

    kinds = {c.kind for c in survivors.values()}
    errs = [ERROR_RATE[k] for k in kinds]
    if len(kinds) == 1:
        combined = errs[0]
    elif kinds <= PIXEL_DERIVED:
        # mesma entrada, falhas correlacionadas: não multiplica
        combined = min(errs) * 0.5
    else:
        combined = 1.0
        for e in errs:
            combined *= e

    # candidato eliminado não vota, mas sinaliza que aquele leitor sofreu ali
    score = (1 - combined) * (0.9 ** len(eliminated))

    if "text_layer" in kinds and len(kinds) > 1:
        grade = "high"
    elif "text_layer" in kinds:
        grade = "medium_high"
    elif len(kinds) > 1:
        grade = "medium"
    else:
        grade = "single_reader"
    if eliminated:
        grade = "resolved" if grade in ("high", "medium") else grade

    return {
        "grade": grade,
        "score": round(score, 4),
        "value": next(iter(values)),
        "supporting": sorted(survivors),
        "kinds": sorted(kinds),
        "eliminated": {r: c.status for r, c in eliminated.items()},
    }


def identification(cands: dict[str, Candidate]) -> dict[str, Any]:
    anchored = {r: c.anchor for r, c in cands.items() if c.anchor}
    return {
        "anchored_by": sorted(anchored),
        "anchors": sorted(set(anchored.values())),
        "grade": "anchored" if anchored else "unanchored",
    }


def validity(fname: str, cands: dict[str, Candidate]) -> dict[str, Any]:
    checks = []
    for r, c in cands.items():
        if c.status == "invalid":
            checks.append({"reader": r, "check": "structure", "status": "fail", "value": c.raw})
        elif c.status == "untyped":
            checks.append({"reader": r, "check": "typing", "status": "fail", "value": c.raw[:40]})
        else:
            checks.append({"reader": r, "check": "typing", "status": "pass", "value": c.canon})
        # checksum é INFO: registra, nunca reprova (models.ValidationStatus.INFO)
        if fname == "isin" and c.canon:
            checks.append(
                {
                    "reader": r,
                    "check": "checksum mod 10",
                    "status": "info",
                    "value": c.canon,
                    "passed": isin_checksum_ok(c.canon),
                }
            )
    gating = [c for c in checks if c["status"] != "info"]
    failed = [c for c in gating if c["status"] == "fail"]
    return {
        "grade": "fail" if failed and len(failed) == len(gating) else
                 ("partial" if failed else "pass"),
        "checks": checks,
    }


def arithmetic_check(fields: dict[str, dict]) -> dict[str, Any] | None:
    """gross x (1 - rate) = net, quando os três estão presentes e concordes."""
    def val(name):
        f = fields.get(name, {}).get("transcription", {})
        return f.get("value") if f.get("grade") not in (None, "conflict", "absent") else None

    g, n, r = val("gross_per_share"), val("net_per_share"), val("tax_rate")
    if not (g and n and r):
        return None
    try:
        gd, nd, rd = Decimal(g), Decimal(n), Decimal(r)
    except InvalidOperation:
        return None
    delta = gd * (1 - rd) - nd
    return {
        "check": "gross x (1 - rate) = net",
        "gross": g, "rate": r, "net": n,
        "delta": format(delta, "f"),
        "status": "pass" if delta == 0 else "fail",
    }


# --------------------------------------------------------------------------
def analyse(doc: str, runs: dict[str, Path], anchors) -> dict[str, Any]:
    per_reader = {}
    for reader, base in runs.items():
        p = base / f"{doc}.md"
        md = p.read_text(encoding="utf-8") if p.exists() else ""
        per_reader[reader] = read_document(md, anchors, reader) if md.strip() else {}

    names = sorted({f for r in per_reader.values() for f in r})
    fields = {}
    for fname in names:
        cands = {r: c for r, cs in per_reader.items() if (c := cs.get(fname))}
        fields[fname] = {
            "candidates": {r: asdict(c) for r, c in cands.items()},
            "transcription": transcription(cands),
            "identification": identification(cands),
            "validity": validity(fname, cands),
        }

    return {
        "doc": doc,
        "abstained": [r for r, cs in per_reader.items() if not cs],
        "fields": fields,
        "record_checks": [c for c in [arithmetic_check(fields)] if c],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="out/consensus")
    ap.add_argument("--anchors", default="config/anchors.yaml")
    ap.add_argument("--out", default="out/consensus/consensus.json")
    args = ap.parse_args()

    base = Path(args.runs)
    runs = {r: base / r for r in READERS}
    anchors = load_anchors(Path(args.anchors))

    docs = sorted({p.stem for r in runs.values() if r.exists() for p in r.glob("*.md")})
    report = [analyse(d, runs, anchors) for d in docs]

    tally: dict[str, int] = {}
    for entry in report:
        print(f"\n=== {entry['doc']} ===")
        if entry["abstained"]:
            print(f"  abstenção: {', '.join(entry['abstained'])}")
        for fname, f in entry["fields"].items():
            t = f["transcription"]
            tally[t["grade"]] = tally.get(t["grade"], 0) + 1
            score = f"{t['score']:.4f}" if t["score"] is not None else "  --  "
            extra = ""
            if t["grade"] == "conflict":
                extra = "  " + " vs ".join(f"{k}={v}" for k, v in t["readings"].items())
            elif t["eliminated"]:
                extra = "  eliminado: " + ", ".join(
                    f"{k}({v})" for k, v in t["eliminated"].items()
                )
            print(
                f"  {fname:18s} {str(t['value'])[:22]:>22s} {score:>8s} "
                f"{t['grade']:14s} {','.join(t['supporting']) or '-':16s}{extra}"
            )
        for c in entry["record_checks"]:
            print(f"  [record] {c['check']}: delta={c['delta']} {c['status']}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== tally ===")
    for k in sorted(tally):
        print(f"  {k}: {tally[k]}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
