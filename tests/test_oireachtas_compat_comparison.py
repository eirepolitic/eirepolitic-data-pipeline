from __future__ import annotations

import pandas as pd
import pytest

from extract.oireachtas.compat_comparison import _scope_comparison_frames


def test_legacy_date_range_ignores_newer_compat_history() -> None:
    legacy = pd.DataFrame(
        {
            "date": ["2025-01-23", "2025-12-17"],
            "memberCode": ["m1", "m2"],
        }
    )
    compat = pd.DataFrame(
        {
            "date": ["2024-12-18", "2025-01-23", "2025-12-17", "2026-09-30"],
            "memberCode": ["m0", "m1", "m2", "m3"],
        }
    )
    config = {
        "comparison_name": "member_votes_compat",
        "comparison_scope": "legacy_date_range",
        "legacy_date_column": "date",
        "compat_date_column": "date",
    }

    legacy_scoped, compat_scoped, metadata = _scope_comparison_frames(legacy, compat, config)

    assert len(legacy_scoped) == 2
    assert len(compat_scoped) == 2
    assert compat_scoped["memberCode"].tolist() == ["m1", "m2"]
    assert metadata == {
        "comparison_scope": "legacy_date_range",
        "comparison_date_start": "2025-01-23",
        "comparison_date_end": "2025-12-17",
        "compat_rows_before_comparison_window": 1,
        "compat_rows_after_comparison_window": 1,
    }


def test_all_rows_scope_is_unchanged() -> None:
    legacy = pd.DataFrame({"member_code": ["m1"]})
    compat = pd.DataFrame({"member_code": ["m1", "m2"]})
    config = {"comparison_name": "members_roster_compat", "comparison_scope": "all_rows"}

    legacy_scoped, compat_scoped, metadata = _scope_comparison_frames(legacy, compat, config)

    assert legacy_scoped.equals(legacy)
    assert compat_scoped.equals(compat)
    assert metadata["comparison_scope"] == "all_rows"
    assert metadata["comparison_date_start"] == ""
    assert metadata["comparison_date_end"] == ""
    assert metadata["compat_rows_before_comparison_window"] == 0
    assert metadata["compat_rows_after_comparison_window"] == 0


def test_legacy_date_range_requires_valid_legacy_dates() -> None:
    legacy = pd.DataFrame({"date": ["not-a-date"], "memberCode": ["m1"]})
    compat = pd.DataFrame({"date": ["2025-01-23"], "memberCode": ["m1"]})
    config = {
        "comparison_name": "member_votes_compat",
        "comparison_scope": "legacy_date_range",
        "legacy_date_column": "date",
        "compat_date_column": "date",
    }

    with pytest.raises(ValueError, match="no valid dates"):
        _scope_comparison_frames(legacy, compat, config)
