# Application model decision (2026-10-10)

Experiment: `python -m evaluation.model_selection --models qwen3.5:4b,gemma3:4b,qwen3:8b --split dev`
on the final (post-review) prompts — results in `model_selection.md` / `summary.json`.
Round 1 (pre-review prompts) is archived in `../model_selection_round1_prefix/`.

| Model | Routing strict | Routing lenient | Macro-F1 | Turn p50 | Size (Q4_K_M) |
|---|---|---|---|---|---|
| qwen3.5:4b | 69.5 % | 78.0 % | 0.71 | 10.3 s | 3.3 GB |
| gemma3:4b | 40.7 % | 50.8 % | 0.47 | 9.3 s | 3.4 GB |
| qwen3:8b | 71.2 % | 84.7 % | 0.74 | 16.0 s | 5.2 GB |

Paired comparison qwen3.5:4b vs qwen3:8b on the same 59 dev turns (McNemar exact):
strict 8 vs 9 discordant turns, p = 1.000; lenient 4 vs 8, p = 0.388 — no significant difference.

**Decision: qwen3.5:4b** for all roles (extractor, both gates, generator). Reasons: routing quality
statistically indistinguishable from qwen3:8b on the dev split, ~1.55x lower latency, and it fits
entirely in the 6 GB VRAM of the target demo laptop (RTX 4050) together with its KV cache, whereas
qwen3:8b (5.2 GB weights) would spill to CPU there. gemma3:4b is clearly worse on the final prompts.
All three models produced schema-valid JSON on the first attempt for every gate call (Ollama
structured outputs). Judge model for the study: gemma3:12b (different model family from the
application model, to avoid self-preference bias).
