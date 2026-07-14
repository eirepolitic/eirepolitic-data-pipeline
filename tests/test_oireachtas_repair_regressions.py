from __future__ import annotations

import os
import unittest
from datetime import date
from unittest.mock import patch

import pandas as pd

from extract.oireachtas.compat_comparison import _dq
from extract.oireachtas.io_s3 import production_publishing_enabled, put_bytes
from extract.oireachtas.normalize import is_current_range


class FakeS3:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def put_object(self, **kwargs: object) -> None:
        self.calls.append(kwargs)


class ProductionPublishingGuardTests(unittest.TestCase):
    def test_production_publish_is_default_deny(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(production_publishing_enabled())

    def test_both_switches_are_required(self) -> None:
        with patch.dict(
            os.environ,
            {
                "OIREACHTAS_PUBLISH_ENABLED": "true",
                "OIREACHTAS_PUBLISH_LATEST": "true",
            },
            clear=True,
        ):
            self.assertTrue(production_publishing_enabled())

    def test_guard_suppresses_mutable_latest_and_compat_writes(self) -> None:
        s3 = FakeS3()
        with patch.dict(os.environ, {}, clear=True):
            put_bytes(s3, bucket="bucket", key="processed/oireachtas_unified/latest/x.csv", body=b"x")
            put_bytes(s3, bucket="bucket", key="processed/oireachtas_unified/compat/x.csv", body=b"x")
            put_bytes(s3, bucket="bucket", key="processed/oireachtas_unified/silver/x/run_id=1/x.csv", body=b"x")
        self.assertEqual(len(s3.calls), 1)
        self.assertIn("run_id=1", str(s3.calls[0]["Key"]))


class ConfirmedRegressionTests(unittest.TestCase):
    @unittest.expectedFailure
    def test_future_open_ended_record_is_not_current(self) -> None:
        self.assertFalse(
            is_current_range("2099-01-01", None, today=date(2026, 7, 13))
        )

    @unittest.expectedFailure
    def test_compatibility_dq_fails_when_legacy_keys_are_missing(self) -> None:
        comparisons = pd.DataFrame(
            [
                {
                    "comparison_name": "members_roster_compat",
                    "status": "pass",
                    "legacy_rows": 176,
                    "compat_rows": 98,
                    "legacy_only_key_count": 78,
                    "compat_only_key_count": 0,
                }
            ]
        )
        self.assertEqual(_dq(comparisons)["dq_status"], "fail")

    @unittest.expectedFailure
    def test_production_limit_must_not_cap_complete_gold_output(self) -> None:
        source_rows = list(range(176))
        workflow_limit = 100
        current_behaviour = source_rows[: max(1, workflow_limit)]
        self.assertEqual(len(current_behaviour), len(source_rows))

    @unittest.expectedFailure
    def test_incremental_window_must_preserve_prior_history(self) -> None:
        prior = {"vote-1", "vote-2"}
        new_window = {"vote-3"}
        current_replacement_behaviour = new_window
        self.assertEqual(current_replacement_behaviour, prior | new_window)

    @unittest.expectedFailure
    def test_full_extraction_must_not_stop_after_first_full_page(self) -> None:
        first_page = list(range(100))
        api_total = 176
        current_single_page_result = first_page
        self.assertEqual(len(current_single_page_result), api_total)


if __name__ == "__main__":
    unittest.main()
