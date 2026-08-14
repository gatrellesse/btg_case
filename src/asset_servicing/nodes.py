"""The graph's nodes.

Each returns only the keys it changed. Business logic lives in the modules the
linear pipeline already used — this layer is about *orchestration*: who runs,
in what order, what happens on disagreement, and where work legitimately gets
sent back.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .classification import markers
from .extraction import ingest, provenance, rules, sweep
from .llm import agents
from .reporter import audit
from .validation import evidence, grounding, repair, verify
from .extraction.consensus import IngestConfig
from .validation.evidence import is_critical
from .llm.schemas import LLMField
from .formats import format_date
from .models import (
    AbsenceReason,
    Classification,
    EventType,
    ExtractedField,
    ReaderKind,
    ReasonCode,
    Repair,
    spec_for,
)
from .text import _squash
from .state import AgentState, ClassifierVote, FieldCandidate, NodeError
from .validation.tools import Validators
from .validation.triage import triage as run_triage


@dataclass
class NodeContext:
    """Runtime dependencies, closed over by the node factories."""

    validators: Validators
    thresholds: dict
    model: str
    use_model: bool
    config_dir: str | None = None
    ingest_cfg: IngestConfig | None = None
    snippet_dir: str | None = None


# --------------------------------------------------------------------------
# 1 · ingestion
# --------------------------------------------------------------------------


def make_ingest(ctx: NodeContext):
    def ingest_node(state: AgentState) -> dict:
        doc = ingest.load(state.path, cfg=ctx.ingest_cfg)
        return {"document": doc, "file_name": doc.file_name}

    return ingest_node


# --------------------------------------------------------------------------
# 2 · classification — three modalities, one vote each
# --------------------------------------------------------------------------


def make_rule_classifier(ctx: NodeContext):
    def rule_classifier_node(state: AgentState) -> dict:
        prior = markers.pre_classify(state.document.raw_text, ctx.config_dir)
        vote = ClassifierVote(
            modality="rule",
            event_type=prior.top,
            rationale=(
                "marcadores definitórios: "
                + ", ".join(h.matched_text for h in prior.hits if h.definitional)[:200]
            )
            if prior.hits
            else "nenhum marcador definitório casou",
            evidence=[h.matched_text for h in prior.hits if h.definitional][:8],
        )
        return {"votes": [vote]}

    return rule_classifier_node


def _vote_from_output(modality: str, output) -> ClassifierVote:
    try:
        event_type = EventType(output.event_type.strip().lower())
    except (ValueError, AttributeError):
        event_type = EventType.OUTRO
    return ClassifierVote(
        modality=modality,
        event_type=event_type,
        rationale=output.rationale or "",
    )


def make_text_classifier(ctx: NodeContext):
    def text_classifier_node(state: AgentState) -> dict:
        if not ctx.use_model:
            return {"votes": [ClassifierVote(modality="text", available=False)]}
        try:
            agent = agents.build_text_classifier(ctx.model)
            result = agent.run_sync(
                f"Classifique o evento corporativo do aviso abaixo.\n\n--- AVISO ---\n"
                f"{state.document.raw_text}"
            )
            return {"votes": [_vote_from_output("text", result.output)]}
        except Exception as exc:  # noqa: BLE001
            return {
                "votes": [ClassifierVote(modality="text", available=False, rationale=str(exc)[:200])],
                "errors": [NodeError(node="text_classifier", error=f"{type(exc).__name__}: {exc}")],
            }

    return text_classifier_node


def make_consensus(ctx: NodeContext):
    def consensus_node(state: AgentState) -> dict:
        """Unanimity or nothing.

        Every available modality has one vote and none has a veto — including
        the deterministic one, which can also be wrong. Anything short of
        unanimity routes to a human, because misclassifying the event type
        changes the tax treatment: a false AUTO costs more than an extra review.

        A modality that could not run does not count against the quorum. The
        record says how many actually voted, so "unanimous" is never read as
        stronger evidence than it is.
        """
        usable = [v for v in state.votes if v.available and v.event_type is not None]
        if not usable:
            return {
                "classification": Classification(event_type=EventType.OUTRO, unanimous=False),
                "unanimous": False,
            }

        prior = markers.pre_classify(state.document.raw_text, ctx.config_dir)
        counts = Counter(v.event_type for v in usable)
        winner, _ = counts.most_common(1)[0]
        empatados = [t for t, n in counts.items() if n == counts[winner]]
        if len(empatados) > 1:
            # Empate em contagem. Era resolvido pela confiança que a modalidade
            # declarava de si mesma — um número que ninguém mediu decidindo o
            # tipo do evento. Decide o prior determinístico, que é auditável:
            # ele leu os marcadores definitórios do aviso. Se nem ele estiver
            # entre os empatados, vale a ordem declarada das modalidades.
            ordem = {"rule": 0, "text": 1}
            winner = (
                prior.top
                if prior.top in empatados
                else min(
                    (v for v in usable if v.event_type in empatados),
                    key=lambda v: ordem.get(v.modality, 9),
                ).event_type
            )

        unanimous = len(counts) == 1
        detail = ", ".join(
            f"{v.modality}={v.event_type.value if v.event_type else '—'}" for v in usable
        )
        return {
            "classification": Classification(
                event_type=winner,
                unanimous=unanimous,
                rationale=(
                    f"{len(usable)} modalidade(s) votaram: {detail}. "
                    + ("Unânime." if unanimous else "SEM unanimidade — enviado para revisão.")
                ),
                prior=prior,
                agrees_with_prior=unanimous or prior.top in (None, winner),
            ),
            "unanimous": unanimous,
        }

    return consensus_node


# --------------------------------------------------------------------------
# 3 · extraction — several readers, confronted on the fields that matter
# --------------------------------------------------------------------------


def _candidate_from_llm(name: str, raw: LLMField, modality: str) -> tuple[str, FieldCandidate]:
    absence = None
    if raw.value is None and raw.absence_reason:
        try:
            absence = AbsenceReason(raw.absence_reason.strip().lower())
        except ValueError:
            absence = AbsenceReason.NOT_FOUND
    return name, FieldCandidate(
        extractor=modality,
        value_raw=raw.value,
        evidence_text=raw.evidence,
        rationale=raw.rationale,
        absence_reason=absence,
    )


def make_rule_extractor(ctx: NodeContext):
    def rule_extractor_node(state: AgentState) -> dict:
        event_type = state.classification.event_type if state.classification else EventType.OUTRO
        found = rules.extract(state.document, event_type, ctx.config_dir)
        return {
            "candidates": [
                (
                    name,
                    FieldCandidate(
                        extractor="rule",
                        value_raw=raw.value_raw,
                        evidence_text=raw.evidence_text,
                        rationale=raw.rationale,
                    ),
                )
                for name, raw in found.items()
            ]
        }

    return rule_extractor_node


def make_text_extractor(ctx: NodeContext):
    def text_extractor_node(state: AgentState) -> dict:
        if not ctx.use_model:
            return {}
        event_type = state.classification.event_type if state.classification else EventType.OUTRO
        prompt = (
            f"Tipo de evento já classificado: {event_type.value}.\n"
            f"Campos obrigatórios: {', '.join(spec_for(event_type).required)}.\n\n"
            f"--- AVISO ---\n{state.document.raw_text}"
        )
        if state.reprompt_count:
            # The retry is only justified because the prompt gained information
            # the first attempt did not have.
            prompt += (
                "\n\nATENÇÃO: na tentativa anterior algum trecho citado como evidência não "
                "foi encontrado no documento. Cite APENAS trechos copiados literalmente."
            )
        try:
            agent = agents.build_extractor(event_type, ctx.model)
            parsed = agent.run_sync(prompt).output
        except Exception as exc:  # noqa: BLE001
            return {"errors": [NodeError(node="text_extractor", error=f"{type(exc).__name__}: {exc}")]}

        out = []
        for name in parsed.model_dump():
            if name == "extra_fields":
                continue
            raw = getattr(parsed, name)
            if isinstance(raw, LLMField):
                out.append(_candidate_from_llm(name, raw, "text"))
        return {"candidates": out}

    return text_extractor_node


#: Ordem de desempate entre extratores: `rule` casa o valor com um rótulo
#: literal do aviso, e essa âncora é conferível no recorte; `text` interpreta.
_EXTRACTOR_ORDER = {"rule": 0, "text": 1}


#: A proporção é lida como par: `ratio_from` e `ratio_to` sozinhos não dizem
#: nada, e qual é "de" e qual é "para" quem decide é o evento — ver
#: `repair.order_ratio`. Comparados campo a campo, dois extratores que leram a
#: mesma proporção em ordens diferentes contam como dois desacordos.
_RATIO_PAIR = ("ratio_from", "ratio_to")


def typed(name: str, raw: str | None):
    """A leitura tipada, para comparar valor com valor e não texto com texto.

    É o passo *tipar* da ordem `ler → tipar → validar → votar`, que a ingestão
    já aplica entre mecanismos de leitura (`consensus.vote_key`) e que a
    confrontação entre extratores nunca teve. O custo apareceu medido no doc 08:

        regras  'R$ 7,820000'
        modelo  'R$ 7,820000 por ação'

    É a mesma quantia — o `Decimal` é o mesmo e o valor final do campo é o
    mesmo. Comparadas como texto viram desacordo, e o campo vai para revisão
    com um humano gastando tempo para concluir que os dois extratores leram o
    mesmo número. A diferença ali não é de valor, é de recorte da citação.

    Tipar não é reparar: só se pergunta *que número (ou que data) é este*, e o
    que não tipa cai no texto achatado, que é a comparação de antes.
    """
    if raw is None:
        return None
    if name in repair.DATE_FIELDS:
        data, _ = repair.parse_date(raw)
        if data is not None:
            return data
    if name in repair.MONEY_FIELDS | repair.RATE_FIELDS | repair.INT_FIELDS:
        numero = repair.parse_ptbr_number(raw)
        if numero is not None:
            return numero
    return _squash(raw)


def _winning_candidate(name: str, candidates: list[FieldCandidate]) -> FieldCandidate:
    """A leitura que mais extratores produziram; no empate, a mais ancorada."""
    votos = Counter(typed(name, c.value_raw) for c in candidates)
    return min(
        candidates,
        key=lambda c: (
            -votos[typed(name, c.value_raw)],
            _EXTRACTOR_ORDER.get(c.extractor, 9),
        ),
    )


def _ratio_pair_agrees(state: AgentState) -> bool | None:
    """Os dois extratores leram a mesma proporção, em qualquer ordem?

    Comparação por conjunto: `{1, 20}` e `{20, 1}` são a mesma proporção lida em
    sentidos opostos, e o sentido não é achado de extração — é regra do evento,
    aplicada em `repair.order_ratio`. Já `{1, 20}` contra `{1, 10}` é desacordo
    de verdade sobre o que o documento diz: a proporção é uma leitura só, e com
    metade contestada as duas metades vão para conferência.

    Devolve `None` quando algum extrator não produziu o par inteiro — comparar
    meia proporção com uma inteira acusaria desacordo onde há leitura parcial.
    Aí vale a comparação campo a campo, como em qualquer outro campo.
    """
    pares: dict[str, dict] = {}
    for papel in _RATIO_PAIR:
        for candidate in state.candidates_for(papel):
            if candidate.value_raw is not None:
                pares.setdefault(candidate.extractor, {})[papel] = typed(
                    papel, candidate.value_raw
                )
    inteiros = [set(par.values()) for par in pares.values() if len(par) == len(_RATIO_PAIR)]
    if len(inteiros) != len(pares) or len(inteiros) < 2:
        return None
    return all(par == inteiros[0] for par in inteiros[1:])


def make_merge(ctx: NodeContext):
    def merge_node(state: AgentState) -> dict:
        """Collapse candidates into fields — without hiding disagreement.

        Where two independent extractors disagree on a *critical* field, the
        record gets ``EXTRACTOR_DISAGREEMENT``. Deliberately not a majority
        vote: on a financial field, voting is how a wrong value enters the base
        wearing the appearance of consensus.

        Qual candidato vence quando eles divergem era decidido pela confiança
        que cada extrator declarava de si — as regras traziam constantes de
        calibração (0,97 para valor ancorado em tabela, 0,80 para padrão em
        prosa) e o modelo trazia seu próprio palpite, comparados como se
        fossem a mesma grandeza. Agora vence a leitura que mais extratores
        produziram e, no empate, o extrator de regras: ele é ancorado num
        rótulo do documento, então a escolha é auditável linha a linha.
        """
        names = {name for name, _ in state.candidates}
        event_type = state.classification.event_type if state.classification else EventType.OUTRO
        fields: dict[str, ExtractedField] = {}
        disagreements: list[str] = []

        for name in names:
            candidates = [c for c in state.candidates_for(name) if c.value_raw is not None]
            absent = [c for c in state.candidates_for(name) if c.value_raw is None]

            if not candidates:
                stated = next((c for c in absent if c.absence_reason), None)
                fields[name] = ExtractedField(
                    name=name,
                    absence_reason=stated.absence_reason if stated else AbsenceReason.NOT_FOUND,
                    rationale=(stated.rationale if stated else "nenhum extrator encontrou o campo"),
                )
                continue

            best = _winning_candidate(name, candidates)
            field = ExtractedField(
                name=name,
                value_raw=best.value_raw,
                evidence_text=best.evidence_text,
                rationale=best.rationale,
            )

            if len(candidates) > 1:
                readings = {typed(name, c.value_raw) for c in candidates}
                field.notes.append(
                    "leituras: "
                    + "; ".join(f"{c.extractor}={c.value_raw!r}" for c in candidates)
                )
                # A proporção é decidida como par; o resto, campo a campo.
                par = _ratio_pair_agrees(state) if name in _RATIO_PAIR else None
                concordam = len(readings) == 1 if par is None else par
                if not concordam and is_critical(name, ctx.thresholds):
                    disagreements.append(name)
                    field.notes.append(
                        "extratores divergem neste campo crítico — vale a leitura ancorada em rótulo"
                    )
                elif concordam:
                    field.notes.append("extratores independentes concordam")

            fields[name] = _attach_provenance(field, state.document)

        for name in spec_for(event_type).fields:
            fields.setdefault(
                name,
                ExtractedField(
                    name=name,
                    absence_reason=AbsenceReason.NOT_FOUND,
                    rationale="nenhum extrator retornou este campo",
                ),
            )
        return {"fields": fields, "disagreements": disagreements}

    return merge_node


def _attach_provenance(field: ExtractedField, doc) -> ExtractedField:
    if not field.evidence_text:
        return field
    block, quality = provenance.locate(field.evidence_text, doc.blocks, field.value_raw)
    if block is None:
        return field
    field.page = block.bbox.page
    field.bbox = block.bbox
    field.reader_kind = block.reader_kind
    field.reading_level = block.level
    field.corroborating_kinds = list(block.corroborating_kinds)
    field.notes.append(f"evidência localizada com qualidade {quality:.2f}")
    if block.alternatives:
        field.notes.append(
            "leituras alternativas da região: "
            + "; ".join(f"{n}={t!r}" for n, t in block.alternatives)
        )
    return field


# --------------------------------------------------------------------------
# 4 · grounding, repair, disambiguation
# --------------------------------------------------------------------------


def make_grounding(ctx: NodeContext):
    def grounding_node(state: AgentState) -> dict:
        fuzzy_min = float((ctx.thresholds.get("grounding") or {}).get("fuzzy_min", 0.85))
        grounding.ground(state.fields, state.document.raw_text, fuzzy_min)
        return {"fields": state.fields}

    return grounding_node


def ungrounded_critical(fields: dict[str, ExtractedField], thresholds: dict) -> list[str]:
    from .models import GroundingStatus

    return [
        name
        for name, field in fields.items()
        if field.value_raw is not None
        and field.grounding.status == GroundingStatus.ABSENT
        and is_critical(name, thresholds)
    ]


def make_reprompt_extract(ctx: NodeContext):
    def reprompt_extract_node(state: AgentState) -> dict:
        """One more attempt, with the failure fed back into the prompt.

        The retry is legitimate only because the *input changed*: the model is
        now told that a span it cited does not exist in the document. Re-issuing
        the identical prompt would return the identical answer.

        Kept as its own node rather than looping back into the extraction
        fan-out, so the cycle is a straight line the graph can bound with one
        counter.
        """
        stale = ungrounded_critical(state.fields, ctx.thresholds)
        if not stale or not ctx.use_model:
            return {"reprompt_count": state.reprompt_count + 1}

        event_type = state.classification.event_type if state.classification else EventType.OUTRO
        prompt = (
            f"Tipo de evento: {event_type.value}.\n"
            f"Os campos {stale} foram rejeitados: o trecho citado como evidência NÃO existe "
            f"no documento. Extraia novamente citando APENAS texto copiado literalmente.\n\n"
            f"--- AVISO ---\n{state.document.raw_text}"
        )
        try:
            parsed = agents.build_extractor(event_type, ctx.model).run_sync(prompt).output
        except Exception as exc:  # noqa: BLE001
            return {
                "reprompt_count": state.reprompt_count + 1,
                "errors": [NodeError(node="reprompt_extract", error=f"{type(exc).__name__}: {exc}")],
            }

        fields = dict(state.fields)
        for name in stale:
            raw = getattr(parsed, name, None)
            if not isinstance(raw, LLMField):
                continue
            _, candidate = _candidate_from_llm(name, raw, "text")
            replacement = ExtractedField(
                name=name,
                value_raw=candidate.value_raw,
                evidence_text=candidate.evidence_text,
                rationale=candidate.rationale,
                absence_reason=candidate.absence_reason,
            )
            replacement.notes.append("reextraído após falha de grounding")
            fields[name] = _attach_provenance(replacement, state.document)

        return {"fields": fields, "reprompt_count": state.reprompt_count + 1}

    return reprompt_extract_node


def make_repair(ctx: NodeContext):
    def repair_node(state: AgentState) -> dict:
        event_type = state.classification.event_type if state.classification else EventType.OUTRO
        repair.repair(state.fields)
        repair.order_ratio(state.fields, event_type)
        repair.derive(state.fields, event_type)
        return {"fields": state.fields}

    return repair_node


def make_disambiguate(ctx: NodeContext):
    def disambiguate_node(state: AgentState) -> dict:
        """Let the chronology settle a reading the OCR could not.

        A date left ambiguous carries its candidates. If exactly one of them
        satisfies every coherence rule, the constraint has resolved what the
        pixels could not — and that is inference from evidence, not a guess. If
        two or more survive, it stays unresolved and goes to a human, which is
        the same rule as before: ambiguity is never broken by picking the
        likeliest.
        """
        resolved: list[str] = []
        for name, field in state.fields.items():
            candidates = next((r.candidates for r in field.repairs if r.candidates), None)
            if not candidates:
                continue

            surviving = []
            for candidate in candidates:
                parsed, _ = repair.parse_date(candidate)
                if parsed is None:
                    continue
                trial = {
                    key: (state.fields[key].value if key in state.fields else None)
                    for key in ("approval_date", "com_date", "ex_date", "payment_date")
                }
                trial[name] = format_date(parsed)
                if ctx.validators.check_date_coherence(**trial)["status"] == "PASS":
                    surviving.append(format_date(parsed))

            if len(surviving) == 1:
                field.value = surviving[0]
                field.repairs.append(
                    Repair(
                        field=name,
                        rule="desambiguada_por_coerencia_de_datas",
                        value_from=", ".join(candidates),
                        value_to=surviving[0],
                    )
                )
                field.notes.append(
                    f"{len(candidates)} leituras possíveis; apenas uma satisfaz a cronologia do evento"
                )
                resolved.append(name)
            elif surviving:
                field.notes.append(
                    f"cronologia não desempata: {len(surviving)} candidatos ainda possíveis"
                )

        return {"fields": state.fields, "repair_count": state.repair_count + 1}

    return disambiguate_node


# --------------------------------------------------------------------------
# 5 · validation, triage, reporting
# --------------------------------------------------------------------------


def make_sweep(ctx: NodeContext):
    """A varredura final, antes de qualquer validador contar ausências.

    Roda aqui, e não depois da triagem, por uma razão de ordem: quem decide se
    um campo faltou é o `check_required_fields`, e um campo recuperado depois
    dele sairia no registro com valor *e* com o achado dizendo que não veio.
    A varredura é a última chance de preencher; o veredito vem depois dela.

    O valor recuperado passa pela mesma esteira dos outros — grounding e
    normalização — porque ter chegado por um caminho diferente não lhe dá
    passe livre: se o modelo devolveu uma data que não vira data, ela vai para
    revisão como qualquer outra leitura ilegível.
    """

    def sweep_node(state: AgentState) -> dict:
        event_type = state.classification.event_type if state.classification else EventType.OUTRO
        faltando = sweep.missing_fields(state.fields, event_type)
        # Duas perguntas sobre o mesmo material: onde está o que falta, e o que
        # um mecanismo só sustenta aparece em mais alguma leitura.
        sozinhos = sweep.uncorroborated_fields(state.fields)
        conferir = {
            name: state.fields[name].value_raw or str(state.fields[name].value)
            for name in sozinhos
        }
        if (not faltando and not conferir) or not ctx.use_model:
            return {}

        reads = state.document.reads
        if not reads:
            return {}

        try:
            agent = agents.build_field_sweeper(ctx.model)
            result = agent.run_sync(sweep.build_prompt(reads, faltando, conferir))
            devolvidos = dict(result.output.found or {})
        except Exception as exc:  # noqa: BLE001
            # Uma varredura que não roda deixa o registro exatamente como estava:
            # ela só acrescenta, nunca é pré-requisito de nada.
            return {"errors": [NodeError(node="sweep", error=f"{type(exc).__name__}: {exc}")]}

        fields = dict(state.fields)
        # Confirmação primeiro: ela mexe em campo que já existe, e o que sobra
        # do dicionário é candidato a preencher ausência.
        notas = sweep.confirm(devolvidos, fields, sozinhos, reads)
        recuperados, descartes = sweep.accept(
            {k: v for k, v in devolvidos.items() if k not in set(sozinhos)}, faltando, reads
        )
        if not recuperados and not descartes and not notas:
            return {"fields": fields}

        fuzzy_min = float((ctx.thresholds.get("grounding") or {}).get("fuzzy_min", 0.85))
        grounding.ground(recuperados, state.document.raw_text, fuzzy_min)
        repair.repair(recuperados)

        for name, field in recuperados.items():
            fields[name] = _attach_provenance(field, state.document)

        erros = []
        for alvo, nota in list(descartes) + [(a, n) for a, n in notas]:
            if alvo and alvo in fields:
                fields[alvo].notes.append(nota)
            else:
                erros.append(NodeError(node="sweep", error=nota))
        return {"fields": fields, "errors": erros} if erros else {"fields": fields}

    return sweep_node


def make_validate(ctx: NodeContext):
    def validate_node(state: AgentState) -> dict:
        record = _as_record(state)
        return {"validations": verify.run_validators(record, ctx.validators)}

    return validate_node


def make_triage(ctx: NodeContext):
    def triage_node(state: AgentState) -> dict:
        record = _as_record(state)
        evidence.note_reference_matches(record.fields, record.validations)
        audit.annotate(record, ctx.thresholds)
        result = run_triage(record, ctx.thresholds)

        # Findings the graph produced that no single validator owns.
        from .models import TriageReason

        extra = []
        if not state.unanimous:
            extra.append(
                TriageReason.of(
                    ReasonCode.CLASSIFIER_DISAGREEMENT,
                    field="event_type",
                    message=state.classification.rationale if state.classification else "",
                )
            )
        for name in state.disagreements:
            extra.append(
                TriageReason.of(
                    ReasonCode.EXTRACTOR_DISAGREEMENT,
                    field=name,
                    message=f"extratores independentes leram '{name}' de formas diferentes",
                )
            )
        if extra:
            existing = {(r.code, r.field) for r in result.reasons}
            result.reasons.extend(r for r in extra if (r.code, r.field) not in existing)
            result.fields_for_review = sorted(
                set(result.fields_for_review) | set(state.disagreements)
            )
            from .models import Disposition, Severity

            if result.disposition == Disposition.AUTO and any(
                r.severity in (Severity.REVIEW, Severity.BLOCK) for r in result.reasons
            ):
                result.disposition = Disposition.REVIEW

        return {"fields": state.fields, "triage": result}

    return triage_node


def make_reporter(ctx: NodeContext):
    def reporter_node(state: AgentState) -> dict:
        record = _as_record(state)
        if not ctx.use_model:
            return {"justification": verify._fallback_justification(record)}
        allowed = {r.code.value for r in state.triage.reasons} if state.triage else set()
        # tudo que o registro realmente afirma, nas duas grafias: o relator pode
        # citar o valor lido ou o padronizado, mas não um terceiro
        figures = {
            str(v)
            for f in state.fields.values()
            for v in (f.value, f.value_raw)
            if v not in (None, "")
        }
        try:
            agent = agents.build_reporter(ctx.model)
            result = agent.run_sync(
                _reporter_prompt(state),
                deps=agents.ReporterDeps(allowed_codes=allowed, allowed_figures=figures),
            )
            return {"justification": result.output.justification.strip()}
        except Exception as exc:  # noqa: BLE001
            return {
                "justification": verify._fallback_justification(record),
                "errors": [NodeError(node="reporter", error=f"{type(exc).__name__}: {exc}")],
            }

    return reporter_node


def _reporter_prompt(state: AgentState) -> str:
    lines = [
        f"Documento: {state.file_name}",
        f"Tipo: {state.classification.event_type.value if state.classification else '—'}",
        f"Disposição calculada: {state.triage.disposition.value if state.triage else '—'}",
        "",
        "Validações executadas:",
    ]
    for validation in state.validations:
        lines.append(f"  - {validation.tool}: {validation.status.value} — {validation.message}")
    lines += ["", "Reason codes atribuídos (use apenas estes):"]
    for reason in state.triage.reasons if state.triage else []:
        lines.append(f"  - {reason.code.value} [{reason.severity.value}] {reason.message}")
    if state.triage and state.triage.fields_for_review:
        lines.append(f"Campos a revisar: {', '.join(state.triage.fields_for_review)}")
    return "\n".join(lines)


def _as_record(state: AgentState):
    """Project the graph state onto the record shape the shared modules expect."""
    from .models import EventRecord, Triage

    return EventRecord(
        document=state.document.to_info(),
        event_type=state.classification.event_type if state.classification else EventType.OUTRO,
        classification=state.classification or Classification(event_type=EventType.OUTRO),
        fields=state.fields,
        extra_fields=state.extra_fields,
        validations=state.validations,
        triage=state.triage or Triage(),
    )
