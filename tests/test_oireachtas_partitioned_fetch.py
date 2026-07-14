from __future__ import annotations

import unittest
from datetime import datetime

from extract.oireachtas.client import ApiResponseSummary
from extract.oireachtas.partitioned_fetch import get_date_partitioned_json_summary


class FakePartitionClient:
    def __init__(self, *, fail_spans_over_days: int = 1) -> None:
        self.fail_spans_over_days = fail_spans_over_days
        self.calls: list[dict[str, object]] = []

    def get_json_summary(self, endpoint: str, *, params: dict):
        self.calls.append(dict(params))
        start = datetime.strptime(params["date_start"], "%Y-%m-%d").date()
        end = datetime.strptime(params["date_end"], "%Y-%m-%d").date()
        span = (end - start).days + 1
        if span > self.fail_spans_over_days:
            return ApiResponseSummary(
                endpoint=endpoint,
                url="https://api.test/questions",
                params=params,
                status_code=422,
                ok=False,
                elapsed_seconds=0.1,
                error="Pagination failed on page 51: HTTPError: 422 Client Error",
                payload=None,
                pagination={"complete": False, "stop_reason": "page_error", "fetched_count": 10000},
            )
        results = [{"id": params["date_start"]}]
        return ApiResponseSummary(
            endpoint=endpoint,
            url="https://api.test/questions",
            params=params,
            status_code=200,
            ok=True,
            elapsed_seconds=0.1,
            payload={"results": results},
            pagination={"complete": True, "page_count": 1, "fetched_count": len(results), "stop_reason": "short_page"},
        )


class PartitionedFetchTests(unittest.TestCase):
    def test_splits_into_non_overlapping_daily_partitions(self) -> None:
        client = FakePartitionClient(fail_spans_over_days=1)
        summary = get_date_partitioned_json_summary(
            client,
            "/questions",
            params={"date_start": "2026-07-01", "date_end": "2026-07-04", "limit": 200},
        )
        self.assertTrue(summary.ok)
        self.assertEqual([row["id"] for row in summary.payload["results"]], [
            "2026-07-01", "2026-07-02", "2026-07-03", "2026-07-04"
        ])
        self.assertEqual(summary.pagination["partition_count"], 4)
        leaves = [(p["date_start"], p["date_end"]) for p in summary.pagination["partitions"]]
        self.assertEqual(leaves, [
            ("2026-07-01", "2026-07-01"),
            ("2026-07-02", "2026-07-02"),
            ("2026-07-03", "2026-07-03"),
            ("2026-07-04", "2026-07-04"),
        ])

    def test_preserves_original_params_in_merged_summary(self) -> None:
        client = FakePartitionClient(fail_spans_over_days=2)
        params = {
            "chamber": "dail",
            "house_no": "34",
            "date_start": "2026-07-01",
            "date_end": "2026-07-04",
            "limit": 200,
        }
        summary = get_date_partitioned_json_summary(client, "/questions", params=params)
        self.assertEqual(dict(summary.params), params)
        self.assertTrue(summary.pagination["complete"])
        self.assertEqual(summary.pagination["fetched_count"], 2)

    def test_single_day_offset_failure_is_not_hidden(self) -> None:
        client = FakePartitionClient(fail_spans_over_days=0)
        summary = get_date_partitioned_json_summary(
            client,
            "/questions",
            params={"date_start": "2026-07-01", "date_end": "2026-07-01", "limit": 200},
        )
        self.assertFalse(summary.ok)
        self.assertIn("422", summary.error)


if __name__ == "__main__":
    unittest.main()
