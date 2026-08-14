"""Deterministic classification prior — no model call.

Runs before the model so the pipeline has a *second, independent opinion* to
weigh the LLM against: agreement is what releases the record, disagreement
becomes a reason code. It is also the half you can exercise offline, with no API key.

The markers come from ``config/event_types.yaml``; nothing sample-specific
lives in this file.
"""

from __future__ import annotations

from ..config import compile_marker, load_config
from ..models import ClassificationPrior, EventType, MarkerHit
from ..text import normalize

# --------------------------------------------------------------------------
# classification prior
# --------------------------------------------------------------------------


def split_title_body(text: str) -> tuple[str, str]:
    """Separate the heading from the body.

    Worth the trouble because a notice can be *titled* one thing and *be*
    another — the heading reflects how the issuer's IR team labelled it, the
    body reflects the act itself. Detecting that split deterministically is
    what catches a mislabelled notice before any model runs.
    """
    lines = text.splitlines()
    idx = next((i for i, ln in enumerate(lines) if "aviso aos acionistas" in normalize(ln)), None)
    if idx is None:
        cut = min(4, len(lines))
    else:
        cut = idx + 1
        # A short following line is a title continuation ("AVISO AOS
        # ACIONISTAS" / "Juros sobre o Capital Próprio"); a long one is prose.
        if cut < len(lines) and len(lines[cut].strip()) < 60:
            cut += 1
    return "\n".join(lines[:cut]), "\n".join(lines[cut:])


def _score_text(text: str, config: dict) -> tuple[dict[EventType, float], list[MarkerHit]]:
    normalized = normalize(text)
    scores: dict[EventType, float] = {}
    hits: list[MarkerHit] = []
    for type_name, spec in config.items():
        try:
            event_type = EventType(type_name)
        except ValueError:
            continue
        total = 0.0
        for group, definitional in (("definitional", True), ("corroborating", False)):
            for marker in spec.get(group) or []:
                match = compile_marker(marker["pattern"]).search(normalized)
                if not match:
                    continue
                weight = float(marker["weight"])
                total += weight
                hits.append(
                    MarkerHit(
                        event_type=event_type,
                        marker=marker["pattern"],
                        weight=weight,
                        definitional=definitional,
                        matched_text=match.group(0)[:80],
                    )
                )
        if total > 0:
            scores[event_type] = total
    return scores, hits


def pre_classify(text: str, config_dir: str | None = None) -> ClassificationPrior:
    """Form a deterministic opinion about the event type.

    A weak or tied prior is a legitimate outcome — it means "you decide, LLM",
    not an error.
    """
    config = load_config("event_types", config_dir)
    title, body = split_title_body(text)

    whole_scores, hits = _score_text(text, config)
    title_scores, _ = _score_text(title, config)
    body_scores, _ = _score_text(body, config)

    ranked = sorted(whole_scores.items(), key=lambda kv: kv[1], reverse=True)
    return ClassificationPrior(
        ranked=[(t, round(s, 2)) for t, s in ranked],
        hits=hits,
        title_prior=max(title_scores, key=title_scores.get) if title_scores else None,
        body_prior=max(body_scores, key=body_scores.get) if body_scores else None,
    )

