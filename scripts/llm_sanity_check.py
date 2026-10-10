"""Live sanity check of the four LLM calls against one local Ollama model.

Runs the Context Extractor, Context Gate, Evidence Gate (with two synthetic
evidence chunks) and Response Generator once each and prints, per call,
whether the structured output parsed and how long it took. The model is
warmed up first so the timings exclude the one-off model load.

Usage:
    python scripts/llm_sanity_check.py                  # settings.LLM_MODEL
    python scripts/llm_sanity_check.py --model qwen3.5:4b --out reports/sanity_qwen35.json
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from any cwd

import argparse
import asyncio
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from medbridge.ai.context_extractor import ContextExtractor
from medbridge.ai.context_gate import ContextGate
from medbridge.ai.evidence_gate import EvidenceGate
from medbridge.ai.llm_wrapper import ResilientLLMWrapper, llm_call_log
from medbridge.ai.response_generator import ResponseGenerator
from medbridge.config import get_settings

MESSAGE = (
    "I'm 62 and my bp was 152 over 96 this morning. I take Telma 40 once a day. "
    "Should I be worried?"
)
# Synthetic, clearly-labelled test evidence (not quoted from any guideline).
CHUNKS: list[dict[str, Any]] = [
    {
        "chunk_id": "synthetic_1",
        "guideline_id": "SYNTHETIC_TEST",
        "section_title": "Blood pressure targets",
        "chunk_text": (
            "Synthetic test passage: for most adults treated for hypertension, home readings "
            "persistently above the agreed target should be reviewed with a clinician, who may "
            "adjust lifestyle measures or medicines."
        ),
    },
    {
        "chunk_id": "synthetic_2",
        "guideline_id": "SYNTHETIC_TEST",
        "section_title": "Drug classes",
        "chunk_text": (
            "Synthetic test passage: angiotensin receptor blockers such as telmisartan are one of "
            "the drug classes used to treat high blood pressure."
        ),
    },
]
SNAPSHOT: dict[str, Any] = {
    "demographics": {"age": 62},
    "recent_bp_readings": [{"systolic": 152, "diastolic": 96}],
    "current_medications": [{"drug": "telmisartan", "dosage": "40 mg", "frequency": "once daily"}],
}


async def run(model: str) -> dict[str, Any]:
    settings = get_settings().model_copy(
        update={"LLM_MODEL": model, "LLM_MODEL_GATES": "", "LLM_MODEL_GENERATOR": ""}
    )
    llm = ResilientLLMWrapper(settings)
    t0 = time.perf_counter()
    warm = await llm.warmup([model])
    warmup_ms = round((time.perf_counter() - t0) * 1000.0, 1)

    outputs: dict[str, Any] = {}
    with llm_call_log() as stats:
        extraction = await ContextExtractor(llm, settings).extract(MESSAGE, {})
        outputs["context_extractor"] = extraction.model_dump(mode="json") if extraction else None

        intent = extraction.raw_intent if extraction else MESSAGE
        gate1 = await ContextGate(llm, settings).evaluate(SNAPSHOT, MESSAGE, intent)
        outputs["context_gate"] = gate1.model_dump(mode="json") if gate1 else None

        gate2 = await EvidenceGate(llm, settings).evaluate(CHUNKS, SNAPSHOT, message=MESSAGE)
        outputs["evidence_gate"] = gate2.model_dump(mode="json") if gate2 else None

        action = gate2.action if gate2 and gate2.action in ("ANSWER", "GENERALIZE") else "GENERALIZE"
        reply = await ResponseGenerator(llm, settings).generate(action, CHUNKS, SNAPSHOT,
                                                                 message=MESSAGE)
        outputs["response_generator"] = reply.model_dump(mode="json") if reply else None
    await llm.close()
    return {
        "model": model,
        "warmup": {"ok": warm.get(model, False), "ms": warmup_ms},
        "calls": [asdict(s) for s in stats],
        "outputs": outputs,
    }


def _print(report: dict[str, Any]) -> None:
    print(f"\nModel: {report['model']}  (warm-up ok={report['warmup']['ok']}, "
          f"{report['warmup']['ms']} ms)")
    print(f"{'call':<20}{'parsed':<8}{'latency_ms':>11}{'prompt_tok':>12}{'compl_tok':>11}"
          f"{'attempts':>10}")
    for call in report["calls"]:
        print(f"{call['call_name']:<20}{call['success']!s:<8}{call['latency_ms']:>11.0f}"
              f"{call['prompt_tokens']:>12}{call['completion_tokens']:>11}{call['attempts']:>10}")
    for name, output in report["outputs"].items():
        print(f"\n[{name}]\n{json.dumps(output, indent=2, ensure_ascii=False)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=get_settings().LLM_MODEL, help="Ollama model tag")
    parser.add_argument("--out", type=Path, help="optional JSON file for the full report")
    args = parser.parse_args()
    report = asyncio.run(run(args.model))
    _print(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
