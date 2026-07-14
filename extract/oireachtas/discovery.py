"""Endpoint discovery helpers for the Oireachtas API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .client import OireachtasClient


DEFAULT_CANDIDATES: tuple[str, ...] = (
    "/members",
    "/memberships",
    "/houses",
    "/constituencies",
    "/parties",
    "/offices",
    "/debates",
    "/debates/records",
    "/debates/sections",
    "/divisions",
    "/votes",
    "/questions",
    "/legislation",
    "/bills",
)


@dataclass(frozen=True)
class DiscoveryResult:
    """One endpoint discovery result."""

    endpoint: str
    status_code: int | None
    ok: bool
    result_count: int | None
    top_level_keys: tuple[str, ...]
    error: str | None


class EndpointDiscovery:
    """Probe candidate endpoints without extracting complete datasets."""

    def __init__(self, client: OireachtasClient) -> None:
        self.client = client

    def probe(
        self,
        endpoints: Sequence[str] = DEFAULT_CANDIDATES,
        *,
        common_params: Mapping[str, Any] | None = None,
        endpoint_params: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> list[DiscoveryResult]:
        """Probe endpoints and return compact summaries.

        Discovery intentionally requests one small page. Complete pagination is
        reserved for table extraction and is explicitly disabled here.
        """
        output: list[DiscoveryResult] = []
        common = dict(common_params or {})
        endpoint_specific = dict(endpoint_params or {})

        for endpoint in endpoints:
            params = dict(common)
            params.update(endpoint_specific.get(endpoint, {}))
            params.setdefault("limit", 1)
            summary = self.client.get_json_summary(endpoint, params=params, paginate=False)
            payload = summary.payload or {}
            results = payload.get("results") if isinstance(payload, Mapping) else None
            result_count = len(results) if isinstance(results, list) else None
            top_level_keys = tuple(sorted(str(key) for key in payload.keys())) if isinstance(payload, Mapping) else tuple()
            output.append(
                DiscoveryResult(
                    endpoint=endpoint,
                    status_code=summary.status_code,
                    ok=summary.ok,
                    result_count=result_count,
                    top_level_keys=top_level_keys,
                    error=summary.error,
                )
            )
        return output


def successful_endpoints(results: Iterable[DiscoveryResult]) -> list[str]:
    """Return endpoint names that responded successfully."""
    return [result.endpoint for result in results if result.ok]
