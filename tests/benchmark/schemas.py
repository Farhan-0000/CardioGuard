"""
Data contracts for the MedBridge-AQ routing benchmark.

* ``ClinicalVignette`` — one synthetic patient query with its prior context
  (``patient_snapshot`` + ``soft_ask_count``) and the expected routing action.
* ``CitationRecord`` — the provenance fields of one returned citation
  (``source``/``org``/``year``/``title``/``guideline_id``), used for citation
  precision against the vignette's expected corpus labels.
* ``ExplanationSummary`` — the PHI-free parts of ``MessageResponse.explanation``
  needed to *explain* a routing decision (which gate decided, from what source).
* ``VignetteEvaluationResult`` — one evaluated vignette (also one JSONL
  checkpoint row).
* ``SubsetMetrics`` / ``BenchmarkMetrics`` — aggregates; every number in the
  report is computed from these.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from medbridge.api.schemas.enums import ActionEnum

EVAL_MODES = ("mock", "live")


class ClinicalVignette(BaseModel):
    """A single synthetic clinical vignette of the MedBridge-AQ dataset."""

    id: str = Field(description="Unique vignette identifier (e.g. 'EMERG-001')")
    subset: str = Field(description="Subset category (e.g. 'emergency_escalate')")
    query: str = Field(description="Patient conversational query")
    patient_snapshot: dict[str, Any] = Field(
        default_factory=dict,
        description="Prior clinical context (demographics, medications, readings, labs)",
    )
    soft_ask_count: int = Field(default=0, ge=0, description="SOFT-ASK counter before the turn")
    expected_action: ActionEnum = Field(description="Target routing action")
    expected_citations: list[str] = Field(
        default_factory=list,
        description=(
            "Acceptable corpus sources: 'ORG YEAR' (e.g. 'ESC 2024'), a guideline_id "
            "(e.g. 'NICE_NG133') or an organisation name (e.g. 'MedlinePlus')"
        ),
    )
    clinical_rationale: str = Field(default="", description="Justification of the expected action")
    contraindications: list[str] = Field(default_factory=list)


class CitationRecord(BaseModel):
    """Provenance of one citation returned by the pipeline (or simulated in mock mode)."""

    source: str = ""
    org: str | None = None
    year: int | None = None
    title: str | None = None
    guideline_id: str | None = None
    chunk_id: str | None = None

    @classmethod
    def from_response(cls, citation: Any) -> CitationRecord:
        """Build from a ``CitationResponse`` (or any object/dict with the same fields)."""

        def get(name: str) -> Any:
            if isinstance(citation, dict):
                return citation.get(name)
            return getattr(citation, name, None)

        year = get("year")
        return cls(
            source=str(get("source") or ""),
            org=get("org"),
            year=int(year) if isinstance(year, int | str) and str(year).isdigit() else None,
            title=get("title"),
            guideline_id=get("guideline_id"),
            chunk_id=get("chunk_id"),
        )


class ExplanationSummary(BaseModel):
    """PHI-free summary of the pipeline explanation for one turn."""

    gate1_action: str | None = None
    gate1_source: str | None = None
    gate1_missing_fields: list[str] = Field(default_factory=list)
    gate2_action: str | None = None
    gate2_source: str | None = None
    emergency_rules: list[str] = Field(default_factory=list)
    retrieved_chunks: int = 0
    retrieved_guidelines: list[str] = Field(default_factory=list)
    cited_chunks: int = 0
    citations_valid: int | None = None
    citations_stripped: int | None = None
    downgraded_to_generalize: bool = False
    gate3_action_change: str | None = None
    server_total_ms: float | None = None
    stage_ms: dict[str, float] = Field(default_factory=dict)
    llm_calls: int = 0
    redactions: int = 0
    models: dict[str, str] = Field(default_factory=dict)

    @property
    def decision_source(self) -> str | None:
        """Which component fixed the final route (fastpath, rule, llm, loop_breaker, …)."""
        return self.gate2_source or self.gate1_source

    @classmethod
    def from_explanation(cls, explanation: Any) -> ExplanationSummary:
        """Summarise an ``Explanation`` model (or its ``model_dump`` dict)."""
        if explanation is None:
            return cls()
        data: dict[str, Any] = (
            explanation if isinstance(explanation, dict) else explanation.model_dump(mode="json")
        )
        gate1 = data.get("gate1") or {}
        gate2 = data.get("gate2") or {}
        emergency = data.get("emergency") or {}
        retrieval = data.get("retrieval") or []
        validation = data.get("citation_validation") or {}
        gate3 = data.get("gate3") or {}
        stage_ms: dict[str, float] = {}
        for timing in data.get("timings") or []:
            if timing.get("status") in ("ok", "failed"):
                name = str(timing.get("stage"))
                stage_ms[name] = round(stage_ms.get(name, 0.0) + float(timing.get("ms") or 0.0), 2)
        guidelines: list[str] = []
        for item in retrieval:
            gid = str(item.get("guideline_id") or "")
            if gid and gid not in guidelines:
                guidelines.append(gid)
        return cls(
            gate1_action=gate1.get("action"),
            gate1_source=gate1.get("source"),
            gate1_missing_fields=list(gate1.get("missing_fields") or []),
            gate2_action=gate2.get("action"),
            gate2_source=gate2.get("source"),
            emergency_rules=[str(r) for r in emergency.get("matched_rules") or []],
            retrieved_chunks=len(retrieval),
            retrieved_guidelines=guidelines,
            cited_chunks=sum(1 for item in retrieval if item.get("cited")),
            citations_valid=validation.get("valid"),
            citations_stripped=validation.get("stripped"),
            downgraded_to_generalize=bool(validation.get("downgraded")),
            gate3_action_change=gate3.get("action_change"),
            server_total_ms=float(data["total_ms"]) if data.get("total_ms") is not None else None,
            stage_ms=stage_ms,
            llm_calls=len(data.get("llm_calls") or []),
            redactions=int(data.get("redactions") or 0),
            models={str(k): str(v) for k, v in (data.get("models") or {}).items()},
        )


class VignetteEvaluationResult(BaseModel):
    """Outcome of one vignette (one JSONL checkpoint row)."""

    vignette_id: str
    subset: str
    query: str
    mode: str = "mock"
    expected_action: ActionEnum
    actual_action: ActionEnum
    action_match: bool
    soft_ask_count: int = 0
    expected_citations: list[str] = Field(default_factory=list)
    actual_citations: list[CitationRecord] = Field(default_factory=list)
    matched_citations: int = 0
    # None = not applicable (no citations returned, or no expected labels).
    citation_precision: float | None = None
    is_emergency: bool = False
    emergency_detected: bool = False
    is_out_of_domain: bool = False
    # Out-of-domain vignette answered with ANSWER (a routing proxy, not a hallucination judgement).
    hallucinated: bool = False
    response_text: str = ""
    latency_ms: float = 0.0
    explanation: ExplanationSummary | None = None
    session_id: str | None = None
    error: str | None = None
    notes: str | None = None

    @field_validator("actual_citations", mode="before")
    @classmethod
    def _coerce_citations(cls, value: Any) -> Any:
        """Accept plain source strings (legacy rows) as well as records."""
        if isinstance(value, list):
            return [CitationRecord(source=v) if isinstance(v, str) else v for v in value]
        return value

    @property
    def decision_source(self) -> str | None:
        return self.explanation.decision_source if self.explanation else None


class SubsetMetrics(BaseModel):
    """Aggregated metrics for one vignette subset."""

    subset_name: str
    total_vignettes: int
    correct_routings: int
    routing_accuracy: float
    avg_citation_precision: float | None = None
    citation_evaluated: int = 0
    avg_latency_ms: float
    errors: int = 0


class BenchmarkMetrics(BaseModel):
    """System-wide metrics computed from the per-vignette results."""

    mode: str = "mock"
    total_vignettes: int
    total_correct_routings: int
    overall_routing_accuracy: float
    # Macro-average over vignettes with >= 1 returned citation and >= 1 expected label.
    overall_citation_precision: float | None = None
    citation_evaluated: int = 0
    expected_answer_count: int = 0
    expected_answer_with_citations: int = 0
    citation_presence_rate: float | None = None
    emergency_count: int
    emergency_detected_count: int
    emergency_detection_recall: float
    emergency_fastpath_count: int = 0
    out_of_domain_count: int
    out_of_domain_abstain_count: int
    out_of_domain_answer_count: int = 0
    # Out-of-domain vignettes answered with ANSWER (%), a routing proxy.
    hallucination_rate: float
    loop_breaker_count: int = 0
    loop_breaker_generalize_count: int = 0
    loop_breaker_forced_count: int = 0
    loop_breaker_rate: float | None = None
    errors: int = 0
    subsets: dict[str, SubsetMetrics] = Field(default_factory=dict)
    confusion_matrix: dict[str, dict[str, int]] = Field(default_factory=dict)
    decision_sources: dict[str, int] = Field(default_factory=dict)
    mean_latency_ms: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    server_latency_p50_ms: float | None = None
    server_latency_p95_ms: float | None = None
    passed_acceptance_criteria: bool = False
    acceptance: dict[str, bool] = Field(default_factory=dict)
