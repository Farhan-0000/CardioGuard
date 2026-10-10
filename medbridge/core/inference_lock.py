"""
Process-wide lock for PyTorch model loading and inference.

The cross-encoder reranker and the Gate 3 NLI verifier both run PyTorch models in
worker threads (``asyncio.to_thread``). Concurrent ``predict`` calls on a shared
accelerator are not safe on every backend: on Apple MPS two threads running
inference at the same time (same or different models) can crash the process with a
segmentation fault. Every PyTorch load and inference in the application therefore
goes through this single re-entrant lock. The models are small, so serialising them
costs little compared with the LLM calls they sit between.
"""
import threading

TORCH_INFERENCE_LOCK = threading.RLock()
