"""feedback_unique_legacy_marker

Concurrency and erasure hardening (finalization review, stream F2):

* ``feedback``: ``UNIQUE (session_id, message_id)`` named
  ``uq_feedback_session_message`` — "one rating per message" is enforced by
  the database and ``save_feedback`` upserts with ``ON CONFLICT … DO UPDATE``
  (the old DELETE-then-INSERT raced under READ COMMITTED). Existing duplicates
  are removed first, keeping the newest row per message (``feedback`` is an
  ordinary mutable table; it has no immutability rules).
* ``sessions.legacy_plaintext_rows`` (nullable INTEGER): recorded by
  ``erase_session`` — the number of append-only rows (``audit_logs``,
  ``clinical_events``) of the session written before encryption existed.
  Crypto-shredding cannot cover them and the immutability rules forbid
  rewriting them, so they are counted, logged and hidden instead.

The append-only tables and their RULEs are untouched.

Revision ID: h2b8d4f6a1c3
Revises: g1a7c3e9d2b4
Create Date: 2026-10-09 18:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "h2b8d4f6a1c3"
down_revision: str | Sequence[str] | None = "g1a7c3e9d2b4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute(
        "DELETE FROM feedback f USING feedback g "
        "WHERE f.session_id = g.session_id AND f.message_id = g.message_id "
        "AND (f.created_at, f.feedback_id) < (g.created_at, g.feedback_id)"
    )
    op.create_unique_constraint(
        "uq_feedback_session_message", "feedback", ["session_id", "message_id"]
    )
    op.add_column("sessions", sa.Column("legacy_plaintext_rows", sa.Integer(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("sessions", "legacy_plaintext_rows")
    op.drop_constraint("uq_feedback_session_message", "feedback", type_="unique")
