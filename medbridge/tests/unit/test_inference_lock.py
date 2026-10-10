"""Gate 3 verifier and reranker never run PyTorch inference concurrently (MPS segfault guard)."""
import asyncio
import threading
import time

from medbridge.ai.grounding_verifier import GroundingVerifier
from medbridge.core.inference_lock import TORCH_INFERENCE_LOCK
from medbridge.retrieval.hybrid_retriever import RetrievedChunk


class _ExclusiveModel:
    """Fake NLI model that records whether two predict calls ever overlapped."""

    config = type("C", (), {"id2label": {0: "contradiction", 1: "entailment", 2: "neutral"}})()

    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self._guard = threading.Lock()

    def predict(self, pairs, apply_softmax=True, show_progress_bar=False):
        with self._guard:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.01)
        with self._guard:
            self.active -= 1
        return [[0.1, 0.8, 0.1] for _ in pairs]


def test_concurrent_verifications_are_serialised():
    model = _ExclusiveModel()
    verifier = GroundingVerifier(model=model)
    chunk = RetrievedChunk(chunk_id="c1", chunk_text="Stage 1 hypertension is 130-139/80-89 mm Hg.",
                           guideline_id="X", section_title="s")

    async def run() -> None:
        await asyncio.gather(*[
            verifier.verify(f"Stage 1 hypertension is 130-139/80-89 mm Hg [1]. Item {i} [1].", [chunk])
            for i in range(8)
        ])

    asyncio.run(run())
    assert model.max_active == 1


def test_lock_is_reentrant():
    with TORCH_INFERENCE_LOCK:
        with TORCH_INFERENCE_LOCK:
            pass
