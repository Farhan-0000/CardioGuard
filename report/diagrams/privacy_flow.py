"""
Privacy flow — where PII is removed, what is encrypted with which key, what stays
plaintext, and the crypto-shredding erasure path.

Sources:

* ``medbridge/core/orchestrator.py`` — ``_stage_privacy`` (redaction first; fail
  closed), ``_stage_emergency`` (raw text, record keeps rule ids, never spans);
* ``medbridge/privacy/redaction.py`` + ``privacy/README.md`` — categories redacted
  and values preserved; ``privacy/log_scrub.py`` — log scrubbing;
* ``medbridge/privacy/crypto.py`` — KEK from ``ENCRYPTION_MASTER_KEY`` wraps one
  Fernet DEK per session (``session_keys.wrapped_key``); text ``enc:v1:…`` and
  JSONB envelopes;
* ``medbridge/state/turn_store.py`` — ``persist_turn`` / ``split_trace``
  (public vs private trace keys), ``erase_session``; ``privacy/retention.py``;
* ``medbridge/ai/llm_wrapper.py`` / ``config.py`` — ``LOCAL_ONLY_MODE``.

Run: ``python report/diagrams/privacy_flow.py`` → ``report/figures/privacy_flow.{png,svg}``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import PALETTE, REPO_ROOT, save_matplotlib
from _mpl import Canvas

NAME = "privacy_flow"
RED = PALETTE["emergency"].line
GREEN = PALETTE["safe"].line
BLUE = PALETTE["llm"].line
PURPLE = PALETTE["storage"].line


def build() -> list[Path]:
    c = Canvas(9.2, 7.25)
    lab = c.label

    # ---- row 1: raw text → emergency / redaction -------------------------------------
    raw = c.box(
        1.0, 1.35, 1.55, 1.1, "Patient message", ["raw text,", "may contain PII"], kind="actor"
    )
    emer = c.box(
        3.55,
        0.85,
        2.5,
        0.52,
        "Emergency fast-path (no LLM)",
        ["reads the RAW text in memory only;", "trace keeps rule ids, never text spans"],
        kind="emergency",
        sub_size=6.5,
    )
    red = c.box(
        3.65,
        1.88,
        2.7,
        0.98,
        "PII / PHI redaction (deterministic)",
        [
            "names after cues, phones, e-mails, addresses,",
            "PIN/ZIP, Aadhaar, PAN, SSN, MRN/UHID/ABHA,",
            "insurance IDs, URLs/IPs; birth date → [AGE: N]",
            "kept: BP, labs, doses, drug names, ages",
        ],
        kind="safe",
        sub_size=6.4,
    )
    c.arrow([(raw.right, emer.cy), (emer.left, emer.cy)], color=RED)
    lab((raw.right + emer.left) / 2, emer.cy - 0.1, "raw", size=6.3, color=RED)
    c.arrow([(raw.right, red.cy), (red.left, red.cy)], color=GREEN)
    lab((raw.right + red.left) / 2, red.cy - 0.1, "raw", size=6.3, color=GREEN)
    lab(
        3.65,
        2.46,
        "fails closed: on error nothing reaches an LLM or the database",
        size=6.2,
        color=GREEN,
        italic=True,
        bg=False,
    )

    # ---- right column: consumers of the redacted text ------------------------------------
    xr, wr = 7.3, 3.25
    llm = c.box(
        xr,
        0.85,
        wr,
        0.6,
        "Local LLM prompts (Ollama, 127.0.0.1)",
        [
            "redacted text inside <untrusted_user_input>",
            "LOCAL_ONLY_MODE: remote providers refused",
        ],
        kind="llm",
        sub_size=6.5,
    )
    logs = c.box(
        xr,
        1.75,
        wr,
        0.52,
        "Application logs",
        ["PII scrubber on every value; no request bodies"],
        kind="det",
        sub_size=6.5,
    )
    ts = c.box(
        xr,
        2.72,
        wr,
        0.6,
        "persist_turn · one flush per turn",
        ["split_trace(): PHI-bearing parts → encrypted,", "PHI-free metrics → plaintext"],
        kind="det",
        sub_size=6.5,
    )
    bus = 5.33
    c.arrow([(red.right, red.cy), (bus, red.cy)], color=GREEN, head=False)
    c.arrow([(bus, llm.cy), (bus, ts.cy)], color=GREEN, head=False)
    for box in (llm, logs, ts):
        c.arrow([(bus, box.cy), (box.left, box.cy)], color=GREEN)
    c.junction(bus, red.cy, GREEN)
    lab(bus + 0.05, 1.31, "redacted\ntext", size=6.3, color=GREEN, ha="left", bg=False)

    # ---- access note ------------------------------------------------------------------------
    c.box(
        1.05,
        2.5,
        1.85,
        0.64,
        "Access control",
        ["HMAC-signed session tokens,", "admin token, rate limits, CSP;", "loopback HTTP only"],
        kind="note",
        title_size=7.4,
        sub_size=6.3,
    )

    # ---- row 3: keys, ciphertext, plaintext ------------------------------------------------
    kek = c.box(
        1.3,
        3.3,
        2.3,
        0.56,
        "ENCRYPTION_MASTER_KEY (KEK)",
        ["environment only, never in the database"],
        kind="det",
        sub_size=6.5,
        lw=1.8,
    )
    dek = c.box(
        1.3,
        4.12,
        2.3,
        0.72,
        "Per-session data key (DEK)",
        ["random Fernet key per session,", "stored wrapped: session_keys.wrapped_key"],
        kind="storage",
        sub_size=6.3,
    )
    c.arrow([(kek.cx, kek.bottom), (dek.cx, dek.top)], both=True)
    lab(kek.cx + 0.06, (kek.bottom + dek.top) / 2, "wraps / unwraps", size=6.3, ha="left")
    enc = c.box(
        4.3,
        4.02,
        2.7,
        1.2,
        "Ciphertext at rest (PostgreSQL)",
        [
            "Fernet = AES-128-CBC + HMAC-SHA256",
            "message_history.content",
            "audit_logs: request, response,",
            "Gate 1 / Gate 2 rationales, trace.private",
            "clinical_events.payload, context_snapshots",
            ".snapshot, feedback.comment",
        ],
        kind="storage",
        sub_size=6.3,
    )
    plain = c.box(
        7.45,
        4.02,
        3.0,
        1.2,
        "Stays plaintext (PHI-free)",
        [
            "gate_1 / gate_2 / final actions, latency_ms,",
            "evidence_chunk_ids, timestamps; trace:",
            "stage timings, models, LLM call stats,",
            "redaction counts, retrieval scores, gate features",
            "→ /api/metrics and calibration need no key",
        ],
        kind="det",
        sub_size=6.3,
    )
    c.arrow([(dek.right, dek.cy), (enc.left, dek.cy)], color=PURPLE)
    lab((dek.right + enc.left) / 2, dek.cy - 0.1, "encrypts", size=6.3, color=PURPLE)
    c.arrow([(ts.left, 2.95), (enc.cx, 2.95), (enc.cx, enc.top)], color=PURPLE)
    lab(enc.cx + 0.06, 3.18, "private part", size=6.3, color=PURPLE, ha="left")
    c.arrow([(plain.cx, ts.bottom), (plain.cx, plain.top)])
    lab(plain.cx + 0.06, (ts.bottom + plain.top) / 2, "public part", size=6.3, ha="left")

    # ---- row 4: erasure ------------------------------------------------------------------------
    c.group(
        0.12, 5.12, 9.08, 6.72, "Right to erasure = crypto-shredding", kind="safe", title_ha="left"
    )
    trig = c.box(
        1.4,
        5.98,
        2.3,
        0.72,
        "Erasure request",
        ["DELETE /api/sessions/{id}", "or retention job: inactive >", "DATA_RETENTION_DAYS (30)"],
        kind="actor",
        sub_size=6.4,
    )
    step = c.box(
        4.2,
        5.98,
        2.6,
        0.72,
        "erase_session()",
        [
            "delete the session_keys row (DEK gone)",
            "delete messages, snapshot, feedback",
            "set sessions.erased_at",
        ],
        kind="safe",
        sub_size=6.4,
    )
    res = c.box(
        7.45,
        5.98,
        3.05,
        0.72,
        "Append-only rows stay, unreadable",
        [
            "clinical_events / audit_logs ciphertext",
            "can never be decrypted again;",
            "routes answer 410 SESSION_ERASED",
        ],
        kind="det",
        sub_size=6.4,
    )
    c.arrow([(trig.right, trig.cy), (step.left, step.cy)])
    c.arrow([(step.right, step.cy), (res.left, res.cy)])
    c.arrow(
        [(3.25, step.top), (3.25, 4.86), (dek.cx, 4.86), (dek.cx, dek.bottom)],
        color=RED,
        dashed=True,
    )
    lab(2.28, 4.97, "deletes the DEK", size=6.3, color=RED)

    c.legend(
        6.98,
        [
            ("actor", "Input / request"),
            ("emergency", "Emergency check"),
            ("safe", "Privacy control"),
            ("llm", "Local LLM"),
            ("det", "Code / plaintext data"),
            ("storage", "Key material / ciphertext"),
        ],
        per_row=6,
        x0=0.2,
        col_w=1.5,
        size=6.6,
    )
    c.check()
    c.trim(top=0.38)
    return save_matplotlib(c.fig, NAME)


if __name__ == "__main__":
    for p in build():
        print(p.relative_to(REPO_ROOT))
