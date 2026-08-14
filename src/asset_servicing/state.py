"""The graph's state contract.

In LangGraph the state *is* the interface between nodes: each one receives the
whole thing and returns only what it changed. Making it a Pydantic model means
schema validation happens at every node boundary rather than once at the end —
a node that returns a malformed field fails where it was produced, not three
stages later where it is hard to attribute.

``votes`` and ``candidates`` carry reducers because the classification and
extraction layers fan out: several nodes write to the same key concurrently and
their results must accumulate rather than overwrite.
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from .models import (
    AbsenceReason,
    Classification,
    EventType,
    ExtractedField,
    Triage,
    ValidationResult,
)

#: As modalidades que restam. Não há mais leitura de imagem *dentro* do
#: grafo: a página já foi lida por três mecanismos na ingestão, e o que
#: chega aqui são blocos, não pixels.
Modality = Literal["rule", "text"]


class ClassifierVote(BaseModel):
    """One reader's opinion about the event type.

    ``available=False`` records that a modality could not run (no API key, no
    GPU). That is different from a modality that ran and abstained, and the
    consensus rule has to tell them apart: a quorum of three cannot be demanded
    from two readers.
    """

    modality: Modality
    event_type: EventType | None = None
    rationale: str = ""
    available: bool = True
    evidence: list[str] = Field(default_factory=list)


class FieldCandidate(BaseModel):
    """One extractor's reading of one field, before any merge.

    Kept separately per extractor so that disagreement stays visible. Once
    candidates are collapsed into a single value, the fact that two independent
    readers disagreed is unrecoverable — and on a financial field that fact
    matters more than either reading.
    """

    extractor: Modality
    value_raw: str | None = None
    evidence_text: str | None = None
    rationale: str | None = None
    absence_reason: AbsenceReason | None = None


class NodeError(BaseModel):
    node: str
    error: str


class AgentState(BaseModel):
    """Everything that flows through the graph."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # --- input ---
    path: str
    file_name: str = ""

    # --- ingestion ---
    document: object | None = None  # SourceDocument (dataclass, not a model)

    # --- classification (fan-out: accumulates) ---
    votes: Annotated[list[ClassifierVote], operator.add] = Field(default_factory=list)
    classification: Classification | None = None
    unanimous: bool = True

    # --- extraction (fan-out: accumulates per field) ---
    candidates: Annotated[list[tuple[str, FieldCandidate]], operator.add] = Field(
        default_factory=list
    )
    fields: dict[str, ExtractedField] = Field(default_factory=dict)
    extra_fields: dict[str, ExtractedField] = Field(default_factory=dict)
    disagreements: list[str] = Field(default_factory=list)

    # --- validation ---
    validations: list[ValidationResult] = Field(default_factory=list)
    triage: Triage | None = None
    justification: str = ""

    # --- cycle control -------------------------------------------------
    # The only defence against a state graph looping forever. Every cycle in
    # this graph is bounded by one of these, and each is incremented by the
    # node that *changes the input* — never by a node that merely retries.
    reprompt_count: int = 0
    repair_count: int = 0

    # Same reducer, same reason as ``votes``: the fan-out nodes that write here
    # run in one superstep, so two failing in the same tick must accumulate. A
    # plain field raises InvalidUpdateError instead — and it only shows up when
    # a second modality fails, which the deterministic path never reaches.
    errors: Annotated[list[NodeError], operator.add] = Field(default_factory=list)

    def candidates_for(self, field: str) -> list[FieldCandidate]:
        """One candidate per extractor — the most recent one it produced.

        A re-prompted extractor appends rather than replaces (the reducer only
        knows how to concatenate), so the later reading of the same field by
        the same extractor supersedes the earlier one.
        """
        latest: dict[str, FieldCandidate] = {}
        for name, candidate in self.candidates:
            if name == field:
                latest[candidate.extractor] = candidate
        return list(latest.values())

    @property
    def modalities_available(self) -> int:
        return sum(1 for v in self.votes if v.available)
