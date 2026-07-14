from __future__ import annotations

import unittest

from extract.oireachtas.client import OireachtasClient


class FakeResponse:
    def __init__(self, *, payload: dict, url: str = "https://api.test/v1/members", status_code: int = 200) -> None:
        self._payload = payload
        self.url = url
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict:
        return self._payload


class FakeSession:
    def __init__(self, pages: list[dict]) -> None:
        self.pages = list(pages)
        self.calls: list[dict] = []
        self.headers: dict[str, str] = {}

    def get(self, url: str, *, params: dict, timeout: int) -> FakeResponse:
        self.calls.append({"url": url, "params": dict(params), "timeout": timeout})
        if not self.pages:
            raise AssertionError("unexpected extra page request")
        return FakeResponse(payload=self.pages.pop(0), url=url)


class OireachtasPaginationTests(unittest.TestCase):
    def test_merges_pages_until_reported_total(self) -> None:
        session = FakeSession(
            [
                {"head": {"counts": {"totalCount": 3}}, "results": [{"id": 1}, {"id": 2}]},
                {"head": {"counts": {"totalCount": 3}}, "results": [{"id": 3}]},
            ]
        )
        client = OireachtasClient(session=session, sleep_seconds=0, retries=1)
        summary = client.get_json_summary("/members", params={"limit": 2})

        self.assertTrue(summary.ok)
        self.assertEqual(summary.payload["results"], [{"id": 1}, {"id": 2}, {"id": 3}])
        self.assertEqual([call["params"]["skip"] for call in session.calls], [0, 2])
        self.assertTrue(summary.pagination["complete"])
        self.assertEqual(summary.pagination["fetched_count"], 3)

    def test_short_page_completes_when_total_is_unavailable(self) -> None:
        session = FakeSession(
            [
                {"results": [{"id": 1}, {"id": 2}]},
                {"results": [{"id": 3}]},
            ]
        )
        client = OireachtasClient(session=session, sleep_seconds=0, retries=1)
        summary = client.get_json_summary("/members", params={"limit": 2})

        self.assertTrue(summary.ok)
        self.assertEqual(summary.pagination["stop_reason"], "short_page")
        self.assertEqual(len(summary.payload["results"]), 3)

    def test_repeated_page_fails_instead_of_looping(self) -> None:
        repeated = {"results": [{"id": 1}, {"id": 2}]}
        session = FakeSession([repeated, repeated])
        client = OireachtasClient(session=session, sleep_seconds=0, retries=1)
        summary = client.get_json_summary("/members", params={"limit": 2}, max_pages=5)

        self.assertFalse(summary.ok)
        self.assertEqual(summary.pagination["stop_reason"], "repeated_page")
        self.assertIn("made no progress", summary.error)

    def test_discovery_can_request_one_page_only(self) -> None:
        session = FakeSession([{"results": [{"id": 1}, {"id": 2}]}])
        client = OireachtasClient(session=session, sleep_seconds=0, retries=1)
        summary = client.get_json_summary("/members", params={"limit": 2}, paginate=False)

        self.assertTrue(summary.ok)
        self.assertEqual(len(session.calls), 1)
        self.assertIsNone(summary.pagination)


if __name__ == "__main__":
    unittest.main()
