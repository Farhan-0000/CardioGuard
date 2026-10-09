"""privacy_trace_feedback

Privacy & persistence additions for the final release (ADL-031/032):

* ``audit_logs.trace`` JSONB (PHI-free pipeline trace) + index on ``created_at``
* ``sessions.erased_at`` (crypto-shredding marker)
* ``session_keys`` — wrapped per-session data keys (deletable → crypto-shred)
* ``feedback`` — 👍/👎 ratings with an optional encrypted comment

No UPDATE/DELETE rules are added to the new tables; the existing immutability
rules on ``clinical_events`` and ``audit_logs`` are untouched (adding a column
is DDL and is not affected by DML rules).

Revision ID: g1a7c3e9d2b4
Revises: f7737fab1c53
Create Date: 2026-10-09 12:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "g1a7c3e9d2b4"
down_revision: Union[str, Sequence[str], None] = "f7737fab1c53"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "audit_logs",
        sa.Column("trace", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.create_index("ix_audit_logs_created_at", "audit_logs", ["created_at"], unique=False)

    op.add_column("sessions", sa.Column("erased_at", sa.DateTime(timezone=True), nullable=True))

    op.create_table(
        "session_keys",
        sa.Column("session_id", sa.UUID(), nullable=False),
        sa.Column("wrapped_key", sa.Text(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.session_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("session_id"),
    )

    op.create_table(
        "feedback",
        sa.Column("feedback_id", sa.UUID(), nullable=False),
        sa.Column("session_id", sa.UUID(), nullable=False),
        sa.Column("message_id", sa.UUID(), nullable=False),
        sa.Column("rating", sa.SmallInteger(), nullable=False),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("rating IN (-1, 1)", name="ck_feedback_rating"),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.session_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("feedback_id"),
    )
    op.create_index("ix_feedback_session_id", "feedback", ["session_id"], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_feedback_session_id", table_name="feedback")
    op.drop_table("feedback")
    op.drop_table("session_keys")
    op.drop_column("sessions", "erased_at")
    op.drop_index("ix_audit_logs_created_at", table_name="audit_logs")
    op.drop_column("audit_logs", "trace")
