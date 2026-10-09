# Patient state (`medbridge.state`)

Event-sourced patient context. The snapshot is a pure, deterministic fold over
the append-only `clinical_events` stream; LLM output never edits it directly —
the extractor only proposes typed delta events.

## Modules

| Module | Responsibility |
|---|---|
| `projector.py` | `StateProjector.apply_events(snapshot, events)` → new snapshot (deep copy, input untouched); `reconstruct_from_events(events)`; legacy `persist_and_project()`. |
| `turn_store.py` | The persistence API the orchestrator and routes use: `persist_turn()` (one flush per turn), `append_manual_events()`, `save_feedback()`, `erase_session()`, and decrypting readers (`load_session_state`, `get_recent_turns`, `get_history`, `get_last_user_message`, `list_events`, `list_audit_entries`). Trace splitting (`split_trace` / `merge_trace`). |
| `session_manager.py` | Session creation (with its per-session data key), `SessionNotFoundError` / `SessionErasedError` / `MessageNotFoundError`, decryption helpers and legacy per-call helpers used by the Streamlit path and older suites. |

## Projection rules

| Event | Snapshot effect |
|---|---|
| `BP_READING` | Append to `recent_bp_readings` (last 5 kept; the full series stays in the events). |
| `MEDICATION_ADDED` | New drug (case-insensitive name) → `current_medications` entry `{drug, dosage, frequency, brand_name?, drug_class?}`. A known drug is a **dose change**: a non-empty `dosage`/`frequency` replaces the current value and the previous `{dosage, frequency[, changed_at]}` is appended to the entry's `history`. `brand_name`/`drug_class` (from the drug normaliser) are kept and only filled in or refreshed. |
| `MEDICATION_STOPPED` | Move the entry (with its history) to `discontinued_medications` with `reason` and `stopped_at` (from the event, so the fold stays pure). |
| `SYMPTOM_REPORTED` | Add to `symptoms` (case-insensitive de-duplication). |
| `DEMOGRAPHIC` | Merge into `demographics`: `age`, `sex`, `pregnancy_status` overwrite; `conditions` is an order-preserving, case-insensitive union. |
| `LAB_RESULT` | Append to `lab_results`, keeping the last 3 per lab type. |

## Persistence and privacy

* `persist_turn(db, TurnRecord)` adds the clinical events, the snapshot upsert
  and soft-ask counter, the user and assistant messages and one audit row, then
  flushes once; the caller commits. A regenerated turn stores no user message
  and no events.
* All free text and clinical payloads are encrypted with the session's data key
  (`privacy.crypto`); readers decrypt transparently and never raise for an
  erased session (`SessionState(erased=True)`, empty lists).
* `audit_logs.trace` is plaintext JSONB for metrics and calibration and must be
  PHI-free. `split_trace` keeps public: timings, totals, models, LLM call
  statistics, retrieved guideline chunks, gate actions/sources/votes, the
  numeric Gate 2 `features`, the numeric/status members of `gate3`,
  `citation_validation`, `speculative_retrieval` and the emergency `action`.
  Gate rationales, the emergency rule lists and BP reading, any non-numeric
  member and any unknown key go into an encrypted `trace["private"]` envelope.
  `merge_trace` restores the full explanation for the history view.
* `erase_session` crypto-shreds: the data key and the mutable rows (messages,
  snapshot, feedback) are deleted and `sessions.erased_at` is set; append-only
  rows stay but can never be decrypted again.
* Nothing here issues UPDATE/DELETE against `clinical_events` or `audit_logs`.

## Tests

`medbridge/tests/unit/test_state_projector.py`,
`medbridge/tests/integration/test_turn_store.py`, `test_projector_db.py`,
`test_session_manager.py` (PostgreSQL tests auto-skip when the database is down).
