# AI layer (`medbridge.ai`)

The four LLM calls of the pipeline, the two optional robustness variants (V1,
V2) and the shared prompt utilities. AI modules never touch PostgreSQL or
Qdrant: the orchestrator passes everything in and receives typed results back
(`None` on permanent failure, which the orchestrator turns into the spec
fallback).

## Modules

| Module | Call | Model / sampling | Output |
|---|---|---|---|
| `llm_wrapper.py` | gateway for every call | — | `ResilientLLMWrapper.call(call_name, system_prompt, user_prompt, output_schema, temperature, max_tokens, *, model, seed)` → validated Pydantic model or `None` |
| `context_extractor.py` | LLM 1 | gates model, T = 0, seed 42 | `ExtractorOutput(delta_events, search_query, raw_intent)` |
| `context_gate.py` | LLM 2 — Gate 1 | gates model, T = 0, seed 42 | `ContextGateOutput(action: SOFT-ASK \| PROCEED, missing_fields, rationale)` |
| `evidence_gate.py` | LLM 3 — Gate 2 | gates model, T = 0, seed 42 | `EvidenceGateOutput(action: ANSWER \| GENERALIZE \| ABSTAIN \| ESCALATE, evidence_sufficient, evidence_used, rationale)` |
| `response_generator.py` | LLM 4 | generator model, T = 0.3 | `ResponseGeneratorOutput(response_text)` with inline `[n]` markers |
| `evidence_calibration.py` | V1 calibrated Gate 2 | Gate 2 run on deterministic evidence permutations (still T = 0, seed 42) | `CalibratedDecision(action, rationale, votes, features, overridden)` |
| `grounding_verifier.py` | V2 "Gate 3" | local NLI cross-encoder (`GROUNDING_NLI_MODEL`), no LLM | `GroundingResult(sentences, n_claims, n_unsupported, unsupported_fraction, cleaned_text, available)` |
| `prompt_utils.py` | shared | — | XML blocks, tag neutralisation, snapshot/history/evidence rendering |
| `schemas/` | — | — | Pydantic output schemas (also sent to Ollama as the structured-output JSON schema) |

### `llm_wrapper.py`

* **Ollama native API** (default): `POST /api/chat` with `format` = the output
  schema (grammar-constrained decoding), `options` (temperature, seed,
  `num_ctx`, `num_predict`), `keep_alive`, and `think` only for models that
  advertise it.
* Optional Groq/OpenAI providers are never contacted while
  `LOCAL_ONLY_MODE=true` (default), and the Ollama endpoint itself must be
  loopback (`localhost`, `127.0.0.0/8`, `::1`) serving a non-cloud model tag —
  otherwise every call, warm-up and probe is refused (fail closed; the
  orchestrator applies its fallbacks) and `health()` reports `blocked`. No
  patient text leaves the machine. Set `LOCAL_ONLY_MODE=false` to use a LAN host.
* Exponential-backoff retries, think-block/markdown JSON extraction and repair,
  validation; `warmup(models)` loads models into Ollama; `health()` reports
  reachability and presence of the configured models.
* Truncation is a failure: a reply stopped by the token limit
  (`done_reason`/`finish_reason == "length"`) is discarded, retried once with a
  doubled `num_predict`/`max_tokens`, and never cached; structurally incomplete
  JSON (open string or bracket) is never "repaired" into a valid result, and
  such entries in an old request cache are ignored.
* `llm_call_log()` (context-variable scoped) records an `LLMCallStat` per call
  (`call_name, model, provider, latency_ms, prompt_tokens, completion_tokens,
  attempts, success, cached, truncated`) — the orchestrator stores these in the
  turn trace.
* `set_request_cache(cache)` memoises identical requests (evaluation only).

### Extractor post-processing (deterministic)

Payload keys are normalised and sanity-checked; BP readings found by
`nlp.bp_parse` are merged in only when they are the patient's own measured
values, and LLM "readings" that are really targets or hypothetical values are
dropped (a number the patient only asks about is never recorded); medication names are normalised brand → generic with `brand_name` and
`drug_class` kept (`nlp.drugs`); the search query is expanded with clinical
synonyms (`nlp.query_expansion`).

### Generator modes (ablation)

| Mode | Condition | Prompt |
|---|---|---|
| `grounded` | C3–C5, V1, V2 (production) | strict grounding + safety rules, routing action |
| `direct` | C0 | no evidence; answers from general knowledge |
| `vanilla` | C1 | plain RAG: answer from the evidence with `[n]` citations, nothing else |
| `prompt_safety` | C2 | `vanilla` + a prose safety block asking for what the gates do |

All modes keep the `<untrusted_user_input>` isolation rule.

### Few-shot decontamination

Few-shot examples use topics and an invented drug name (`Quilvetan`) that do
not occur in HalluBench. `tests/unit/test_prompts_contamination.py` compares
every example string with every HalluBench turn (dev and test) and fails on a
fabricated HalluBench entity name, high token similarity, shared rare words or
high distinctive-word overlap.

### Generator output format

The citation modes (`grounded`, `vanilla`, `prompt_safety`) end the system prompt with a
JSON example whose `response_text` itself carries `[n]` markers. With
grammar-constrained JSON output local models mirror that example; with a
marker-free placeholder, qwen3:8b dropped every marker in the live smoke test,
so each ANSWER was downgraded to GENERALIZE by citation validation.

### Which text the gates see

`EvidenceGate.evaluate(chunks, snapshot, query=None, message=None)`,
`ResponseGenerator.generate(..., query=None, message=None, ...)` and
`calibrated_evidence_decision(gate, chunks, snapshot, query, *, votes,
settings, message=None)` prefer `message` — the patient's own (redacted) words —
over `query` (the extractor's search query). The orchestrator passes both, so
false premises and unknown drug names are judged on what the patient wrote.

### V1 — calibrated Evidence Gate

Majority vote over up to `EVIDENCE_GATE_VOTES` evidence orderings (ties go to
the more conservative action), then a threshold veto from
`evidence_features(question, chunks, snapshot)` → `max_rerank`,
`mean_top3_rerank`, `entity_coverage`, `n_chunks` (redaction placeholders such
as `[NAME]`/`[PHONE]` and contact words are not counted as question entities):
ANSWER → GENERALIZE below `EVIDENCE_MIN_RERANK_ANSWER` or
`EVIDENCE_MIN_ENTITY_COVERAGE`; ANSWER/GENERALIZE → ABSTAIN below
`EVIDENCE_MIN_RERANK_ANY`. ESCALATE is never overridden, ABSTAIN never
upgraded. The plain Gate 2 path records the same features in
`explanation.gate2.features`, so thresholds can be fitted from C5 traces with
exactly the production feature function.

### V2 — grounding verifier (Gate 3)

Each claim sentence of an ANSWER/GENERALIZE draft is checked by NLI against
~3-sentence windows of the chunks it cites (all chunks when uncited); claims
below `GROUNDING_ENTAILMENT_THRESHOLD` are removed from `cleaned_text`. If the
model cannot be loaded, `available=False` and the text is returned unchanged.
`warmup()` loads the model and runs one inference (returns `False` when
unusable); `loaded` / `load_error` expose its state to `/health`.

## Prompt isolation (Non-Negotiable #5)

Every builder renders separate blocks (`<system_instructions>`,
`<patient_context>`, `<conversation_history>`, `<question_intent>`,
`<clinical_evidence>` with `<chunk index="n" source=… section=… page=…>`,
`<missing_information>`, `<user_query>`) and puts patient text inside
`<untrusted_user_input>`. `neutralize_tags()` is applied to every untrusted
slot — the message, earlier patient and assistant turns, snapshot JSON values,
the extractor's intent and evidence text — and escapes:

* complete tags (`</untrusted_user_input>`, `< / x >`, `<x attr="…">`), also
  with zero-width characters after `<`;
* unterminated openers (`</untrusted_user_input` at the end of the text,
  `<!--`, `<?xml`);
* Unicode look-alike brackets (`＜ ＞`, `‹ ›`, `⟨ ⟩`, …), folded to ASCII first.

Plain comparisons (`BP < 140`, `> 90`) are untouched. Chunk attributes are
HTML-attribute escaped. HTML sources (`source_format == "html"`) are rendered
without a `page` attribute and with `section="<title> (section N)"`.
`medbridge/tests/unit/test_prompts_contract.py` checks, for every builder and
every untrusted slot, that injected tags add no structural tag to the prompt.
