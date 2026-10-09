"""
Unit tests for TASK-11: Core Enumerations & Shared Types.
Verifies all enum members, string representations, and ADL-008 casing compliance.
"""
from enum import StrEnum

import pytest

from medbridge.api.schemas.enums import ActionEnum, EventTypeEnum

# ---------------------------------------------------------------------------
# ActionEnum Tests
# ---------------------------------------------------------------------------

class TestActionEnum:
    """Verify ActionEnum members, values, and string behavior (ADL-008)."""

    def test_inherits_from_str_enum(self) -> None:
        assert issubclass(ActionEnum, StrEnum)
        assert issubclass(ActionEnum, str)

    def test_has_exactly_five_members(self) -> None:
        assert len(ActionEnum) == 5

    def test_member_names(self) -> None:
        expected_names = {"SOFT_ASK", "ANSWER", "GENERALIZE", "ABSTAIN", "ESCALATE"}
        actual_names = {member.name for member in ActionEnum}
        assert actual_names == expected_names

    def test_member_values_match_adl_008_casing(self) -> None:
        """ADL-008 mandates these exact canonical string values."""
        assert ActionEnum.SOFT_ASK.value == "SOFT-ASK"
        assert ActionEnum.ANSWER.value == "ANSWER"
        assert ActionEnum.GENERALIZE.value == "GENERALIZE"
        assert ActionEnum.ABSTAIN.value == "ABSTAIN"
        assert ActionEnum.ESCALATE.value == "ESCALATE"

    def test_string_representation_equals_value(self) -> None:
        """StrEnum members must behave as their string value in comparisons."""
        assert ActionEnum.SOFT_ASK == "SOFT-ASK"
        assert ActionEnum.ANSWER == "ANSWER"
        assert ActionEnum.GENERALIZE == "GENERALIZE"
        assert ActionEnum.ABSTAIN == "ABSTAIN"
        assert ActionEnum.ESCALATE == "ESCALATE"

    def test_string_serialization(self) -> None:
        """Verify str() and f-string produce the enum value, not 'ActionEnum.X'."""
        assert str(ActionEnum.SOFT_ASK) == "SOFT-ASK"
        assert f"{ActionEnum.ESCALATE}" == "ESCALATE"

    def test_lookup_by_value(self) -> None:
        """Ensure enums can be reconstructed from their string values."""
        assert ActionEnum("SOFT-ASK") is ActionEnum.SOFT_ASK
        assert ActionEnum("ANSWER") is ActionEnum.ANSWER
        assert ActionEnum("GENERALIZE") is ActionEnum.GENERALIZE
        assert ActionEnum("ABSTAIN") is ActionEnum.ABSTAIN
        assert ActionEnum("ESCALATE") is ActionEnum.ESCALATE

    def test_invalid_value_raises(self) -> None:
        with pytest.raises(ValueError):
            ActionEnum("INVALID")
        with pytest.raises(ValueError):
            ActionEnum("soft-ask")  # case-sensitive


# ---------------------------------------------------------------------------
# EventTypeEnum Tests
# ---------------------------------------------------------------------------

class TestEventTypeEnum:
    """Verify EventTypeEnum members, values, and string behavior."""

    def test_inherits_from_str_enum(self) -> None:
        assert issubclass(EventTypeEnum, StrEnum)
        assert issubclass(EventTypeEnum, str)

    def test_has_exactly_six_members(self) -> None:
        assert len(EventTypeEnum) == 6

    def test_member_names(self) -> None:
        expected_names = {
            "BP_READING",
            "MEDICATION_ADDED",
            "MEDICATION_STOPPED",
            "SYMPTOM_REPORTED",
            "DEMOGRAPHIC",
            "LAB_RESULT",
        }
        actual_names = {member.name for member in EventTypeEnum}
        assert actual_names == expected_names

    def test_member_values(self) -> None:
        assert EventTypeEnum.BP_READING.value == "BP_READING"
        assert EventTypeEnum.MEDICATION_ADDED.value == "MEDICATION_ADDED"
        assert EventTypeEnum.MEDICATION_STOPPED.value == "MEDICATION_STOPPED"
        assert EventTypeEnum.SYMPTOM_REPORTED.value == "SYMPTOM_REPORTED"
        assert EventTypeEnum.DEMOGRAPHIC.value == "DEMOGRAPHIC"
        assert EventTypeEnum.LAB_RESULT.value == "LAB_RESULT"

    def test_string_representation_equals_value(self) -> None:
        assert EventTypeEnum.BP_READING == "BP_READING"
        assert EventTypeEnum.MEDICATION_STOPPED == "MEDICATION_STOPPED"

    def test_string_serialization(self) -> None:
        assert str(EventTypeEnum.DEMOGRAPHIC) == "DEMOGRAPHIC"
        assert f"{EventTypeEnum.LAB_RESULT}" == "LAB_RESULT"

    def test_lookup_by_value(self) -> None:
        for member in EventTypeEnum:
            assert EventTypeEnum(member.value) is member

    def test_invalid_value_raises(self) -> None:
        with pytest.raises(ValueError):
            EventTypeEnum("UNKNOWN")
        with pytest.raises(ValueError):
            EventTypeEnum("bp_reading")  # case-sensitive
