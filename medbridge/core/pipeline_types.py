"""
Shared pipeline types (Module M-08 support).

Typed contracts shared by the orchestrator, the streaming endpoint, the
evaluation harness and the UI:

* ``StageName`` / ``STAGE_LABELS`` — the 8-step workflow stages and the
  patient-facing progress labels streamed over SSE.
* ``PipelineEvent`` / ``EventCallback`` — stage progress events.
* ``AblationConfig`` — *internal* switches used only by the evaluation
  harness (``evaluation/``) to run baselines and gate ablations. It is never
  exposed through the public API; production always runs ``AblationConfig.full()``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Awaitable, Callable, Literal


class StageName(StrEnum):
    PRIVACY = "privacy"
    EMERGENCY = "emergency"
    LOAD_STATE = "load_state"
    SMALL_TALK = "small_talk"
    EXTRACTION = "extraction"
    CONTEXT_GATE = "context_gate"
    RETRIEVAL = "retrieval"
    RERANK = "rerank"
    EVIDENCE_GATE = "evidence_gate"
    GENERATION = "generation"
    VERIFICATION = "verification"
    PERSISTENCE = "persistence"


# Patient-facing progress labels (streamed to the UI as stage events).
STAGE_LABELS: dict[StageName, str] = {
    StageName.PRIVACY: "Protecting your privacy…",
    StageName.EMERGENCY: "Checking for emergencies…",
    StageName.LOAD_STATE: "Loading your conversation…",
    StageName.SMALL_TALK: "Reading your message…",
    StageName.EXTRACTION: "Understanding your context…",
    StageName.CONTEXT_GATE: "Checking what I know about you…",
    StageName.RETRIEVAL: "Searching guidelines…",
    StageName.RERANK: "Ranking the evidence…",
    StageName.EVIDENCE_GATE: "Verifying evidence…",
    StageName.GENERATION: "Writing answer…",
    StageName.VERIFICATION: "Double-checking claims…",
    StageName.PERSISTENCE: "Saving securely…",
}


EventType = Literal["stage", "meta", "token", "done", "error"]
StageStatus = Literal["start", "end", "skipped"]


@dataclass
class PipelineEvent:
    """A progress event emitted by the orchestrator while processing a turn."""

    type: EventType
    stage: StageName | None = None
    status: StageStatus | None = None
    label: str | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def to_sse_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = dict(self.data)
        if self.stage is not None:
            payload["stage"] = self.stage.value
        if self.status is not None:
            payload["status"] = self.status
        if self.label is not None:
            payload["label"] = self.label
        return payload


EventCallback = Callable[[PipelineEvent], Awaitable[None]]

GeneratorMode = Literal["grounded", "direct", "prompt_safety"]


@dataclass(frozen=True)
class AblationConfig:
    """Pipeline switches for the hallucination study (evaluation only).

    Conditions (CLAUDE_AGENT_PROMPT §6.2 / §6.7):
      C0 Direct LLM          — no retrieval, no gates, generator answers from parametric knowledge
      C1 Vanilla RAG         — retrieval + grounded generation, no gates
      C2 RAG + prompt safety — retrieval + generator with safety instructions, no gates
      C3 Context Gate only   — Gate 1 active, Gate 2 bypassed (always ANSWER)
      C4 Evidence Gate only  — Gate 1 bypassed (always PROCEED), Gate 2 active
      C5 Full MedBridge      — Dual Answerability Gate
      V1 Calibrated Gate 2   — C5 + quantitative evidence signal + permutation votes
      V2 Triple Gate         — C5 + post-generation Grounding Verifier (Gate 3)
      V12                    — C5 + V1 + V2
    """

    name: str = "C5"
    use_emergency_fastpath: bool = True
    use_small_talk: bool = True
    use_context_gate: bool = True
    use_retrieval: bool = True
    use_evidence_gate: bool = True
    generator_mode: GeneratorMode = "grounded"
    # ``None`` means "follow Settings"; True/False force the variant on/off.
    calibrated_evidence_gate: bool | None = None
    grounding_verifier: bool | None = None

    @classmethod
    def full(cls) -> "AblationConfig":
        return cls(name="C5")

    @classmethod
    def preset(cls, name: str) -> "AblationConfig":
        presets: dict[str, AblationConfig] = {
            "C0": cls(name="C0", use_emergency_fastpath=False, use_small_talk=False,
                      use_context_gate=False, use_retrieval=False, use_evidence_gate=False,
                      generator_mode="direct", calibrated_evidence_gate=False,
                      grounding_verifier=False),
            "C1": cls(name="C1", use_context_gate=False, use_evidence_gate=False,
                      generator_mode="grounded", calibrated_evidence_gate=False,
                      grounding_verifier=False),
            "C2": cls(name="C2", use_context_gate=False, use_evidence_gate=False,
                      generator_mode="prompt_safety", calibrated_evidence_gate=False,
                      grounding_verifier=False),
            "C3": cls(name="C3", use_evidence_gate=False, calibrated_evidence_gate=False,
                      grounding_verifier=False),
            "C4": cls(name="C4", use_context_gate=False, calibrated_evidence_gate=False,
                      grounding_verifier=False),
            "C5": cls(name="C5", calibrated_evidence_gate=False, grounding_verifier=False),
            "V1": cls(name="V1", calibrated_evidence_gate=True, grounding_verifier=False),
            "V2": cls(name="V2", calibrated_evidence_gate=False, grounding_verifier=True),
            "V12": cls(name="V12", calibrated_evidence_gate=True, grounding_verifier=True),
        }
        key = name.upper()
        if key not in presets:
            raise ValueError(f"Unknown ablation preset '{name}'. Known: {sorted(presets)}")
        return presets[key]

    @classmethod
    def all_names(cls) -> list[str]:
        return ["C0", "C1", "C2", "C3", "C4", "C5", "V1", "V2", "V12"]
