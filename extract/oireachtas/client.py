"""Shared Oireachtas API client for unified extraction and discovery."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional
from urllib.parse import urljoin

import requests


DEFAULT_API_BASE_URL = "https://api.oireachtas.ie/v1"
DEFAULT_DATA_BASE_URL = "https://data.oireachtas.ie"
DEFAULT_PAGE_SIZE = 200
DEFAULT_MAX_PAGES = 1000


@dataclass(frozen=True)
class ApiResponseSummary:
    """Response summary used by builders, discovery, and manifests."""

    endpoint: str
    url: str
    params: Mapping[str, Any]
    status_code: Optional[int]
    ok: bool
    elapsed_seconds: Optional[float]
    error: Optional[str] = None
    payload: Optional[Mapping[str, Any]] = field(default=None, repr=False)
    pagination: Optional[Mapping[str, Any]] = field(default=None, repr=False)


class OireachtasClient:
    """Resilient Oireachtas API client with complete offset pagination.

    Calls that contain a ``limit`` parameter are paginated automatically unless
    ``paginate=False`` is supplied. Discovery explicitly opts out. Production
    builders therefore receive one merged payload while retaining page telemetry.
    """

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_API_BASE_URL,
        data_base_url: str = DEFAULT_DATA_BASE_URL,
        timeout_seconds: int = 30,
        retries: int = 4,
        backoff_seconds: float = 1.5,
        sleep_seconds: float = 0.1,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.data_base_url = data_base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self.backoff_seconds = backoff_seconds
        self.sleep_seconds = sleep_seconds
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "eirepolitic-data-pipeline/oireachtas-unified",
            }
        )

    def endpoint_url(self, endpoint: str) -> str:
        """Return an absolute API URL for an endpoint or URL."""
        clean = endpoint.strip()
        if clean.startswith("http://") or clean.startswith("https://"):
            return clean
        return urljoin(f"{self.base_url}/", clean.lstrip("/"))

    def get_json_summary(
        self,
        endpoint: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        paginate: Optional[bool] = None,
        max_pages: Optional[int] = None,
        max_rows: Optional[int] = None,
    ) -> ApiResponseSummary:
        """GET JSON and return a complete merged result payload when paginated.

        ``limit`` is treated as page size, not as a production dataset cap.
        ``max_rows`` exists only for explicit tests and is recorded as an
        intentionally incomplete extraction.
        """
        params_dict = _clean_params(params)
        should_paginate = bool("limit" in params_dict) if paginate is None else bool(paginate)
        if not should_paginate:
            return self._get_one_page(endpoint, params=params_dict)

        page_size = _positive_int(params_dict.get("limit"), DEFAULT_PAGE_SIZE)
        page_size = min(page_size, DEFAULT_PAGE_SIZE)
        initial_skip = max(0, _nonnegative_int(params_dict.get("skip"), 0))
        configured_max_pages = _positive_int(
            max_pages if max_pages is not None else os.getenv("OIREACHTAS_MAX_PAGES"),
            DEFAULT_MAX_PAGES,
        )
        explicit_max_rows = None if max_rows is None else max(1, int(max_rows))

        combined_results: list[Any] = []
        first_payload: Optional[dict[str, Any]] = None
        first_url = self.endpoint_url(endpoint)
        last_status: Optional[int] = None
        total_elapsed = 0.0
        page_count = 0
        reported_total: Optional[int] = None
        seen_page_signatures: set[str] = set()
        stop_reason: Optional[str] = None

        while page_count < configured_max_pages:
            page_params = dict(params_dict)
            page_params["limit"] = page_size
            page_params["skip"] = initial_skip + len(combined_results)
            page = self._get_one_page(endpoint, params=page_params)
            page_count += 1
            last_status = page.status_code
            total_elapsed += float(page.elapsed_seconds or 0.0)
            if page_count == 1:
                first_url = page.url

            if not page.ok or page.payload is None:
                pagination = _pagination_metadata(
                    complete=False,
                    page_count=page_count,
                    page_size=page_size,
                    initial_skip=initial_skip,
                    fetched_count=len(combined_results),
                    reported_total=reported_total,
                    stop_reason="page_error",
                    intentionally_limited=False,
                )
                return ApiResponseSummary(
                    endpoint=endpoint,
                    url=first_url,
                    params=params_dict,
                    status_code=last_status,
                    ok=False,
                    elapsed_seconds=round(total_elapsed, 3),
                    error=f"Pagination failed on page {page_count}: {page.error or page.status_code}",
                    payload=None,
                    pagination=pagination,
                )

            payload = dict(page.payload)
            results = payload.get("results")
            if not isinstance(results, list):
                if page_count == 1:
                    return page
                pagination = _pagination_metadata(
                    complete=False,
                    page_count=page_count,
                    page_size=page_size,
                    initial_skip=initial_skip,
                    fetched_count=len(combined_results),
                    reported_total=reported_total,
                    stop_reason="invalid_results_shape",
                    intentionally_limited=False,
                )
                return ApiResponseSummary(
                    endpoint=endpoint,
                    url=first_url,
                    params=params_dict,
                    status_code=last_status,
                    ok=False,
                    elapsed_seconds=round(total_elapsed, 3),
                    error=f"Expected list payload['results'] on page {page_count}, got {type(results).__name__}",
                    payload=None,
                    pagination=pagination,
                )

            if first_payload is None:
                first_payload = payload
            reported_total = _reported_total(payload, current=reported_total)

            signature = _page_signature(results)
            if results and signature in seen_page_signatures:
                pagination = _pagination_metadata(
                    complete=False,
                    page_count=page_count,
                    page_size=page_size,
                    initial_skip=initial_skip,
                    fetched_count=len(combined_results),
                    reported_total=reported_total,
                    stop_reason="repeated_page",
                    intentionally_limited=False,
                )
                return ApiResponseSummary(
                    endpoint=endpoint,
                    url=first_url,
                    params=params_dict,
                    status_code=last_status,
                    ok=False,
                    elapsed_seconds=round(total_elapsed, 3),
                    error=f"Pagination made no progress: page {page_count} repeated a prior result page",
                    payload=None,
                    pagination=pagination,
                )
            seen_page_signatures.add(signature)
            combined_results.extend(results)

            if explicit_max_rows is not None and len(combined_results) >= explicit_max_rows:
                combined_results = combined_results[:explicit_max_rows]
                stop_reason = "max_rows"
                break
            if not results:
                stop_reason = "empty_page"
                break
            if reported_total is not None and initial_skip + len(combined_results) >= reported_total:
                stop_reason = "reported_total_reached"
                break
            if len(results) < page_size:
                stop_reason = "short_page"
                break

        intentionally_limited = stop_reason == "max_rows"
        complete = stop_reason in {"empty_page", "reported_total_reached", "short_page"}
        if stop_reason is None:
            stop_reason = "max_pages"

        pagination = _pagination_metadata(
            complete=complete,
            page_count=page_count,
            page_size=page_size,
            initial_skip=initial_skip,
            fetched_count=len(combined_results),
            reported_total=reported_total,
            stop_reason=stop_reason,
            intentionally_limited=intentionally_limited,
        )

        if not complete and not intentionally_limited:
            return ApiResponseSummary(
                endpoint=endpoint,
                url=first_url,
                params=params_dict,
                status_code=last_status,
                ok=False,
                elapsed_seconds=round(total_elapsed, 3),
                error=f"Pagination did not complete: {stop_reason}",
                payload=None,
                pagination=pagination,
            )

        merged_payload = dict(first_payload or {})
        merged_payload["results"] = combined_results
        merged_payload["_pagination"] = pagination
        return ApiResponseSummary(
            endpoint=endpoint,
            url=first_url,
            params=params_dict,
            status_code=last_status,
            ok=True,
            elapsed_seconds=round(total_elapsed, 3),
            payload=merged_payload,
            pagination=pagination,
        )

    def _get_one_page(self, endpoint: str, *, params: Mapping[str, Any]) -> ApiResponseSummary:
        """GET exactly one JSON page with retry and backoff."""
        url = self.endpoint_url(endpoint)
        params_dict = _clean_params(params)
        last_error: Optional[str] = None
        last_status: Optional[int] = None
        elapsed: Optional[float] = None

        for attempt in range(1, self.retries + 1):
            started = time.monotonic()
            try:
                response = self.session.get(url, params=params_dict, timeout=self.timeout_seconds)
                elapsed = round(time.monotonic() - started, 3)
                last_status = response.status_code
                if response.status_code == 429 or 500 <= response.status_code <= 599:
                    last_error = f"HTTP {response.status_code}: retryable response"
                    if attempt < self.retries:
                        time.sleep(self.backoff_seconds * attempt)
                        continue

                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise ValueError(f"Expected JSON object, got {type(payload).__name__}")
                if self.sleep_seconds:
                    time.sleep(self.sleep_seconds)
                return ApiResponseSummary(
                    endpoint=endpoint,
                    url=response.url,
                    params=params_dict,
                    status_code=response.status_code,
                    ok=True,
                    elapsed_seconds=elapsed,
                    payload=payload,
                )
            except Exception as exc:
                elapsed = round(time.monotonic() - started, 3)
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < self.retries:
                    time.sleep(self.backoff_seconds * attempt)

        return ApiResponseSummary(
            endpoint=endpoint,
            url=url,
            params=params_dict,
            status_code=last_status,
            ok=False,
            elapsed_seconds=elapsed,
            error=last_error,
            payload=None,
        )


def _clean_params(params: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    return {key: value for key, value in dict(params or {}).items() if value is not None and value != ""}


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _nonnegative_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 0 else default


def _reported_total(payload: Mapping[str, Any], *, current: Optional[int]) -> Optional[int]:
    candidates: list[Any] = []
    head = payload.get("head")
    if isinstance(head, Mapping):
        counts = head.get("counts")
        if isinstance(counts, Mapping):
            candidates.extend(counts.get(key) for key in ("totalCount", "total_count", "totalRecords"))
    pagination = payload.get("pagination")
    if isinstance(pagination, Mapping):
        candidates.extend(pagination.get(key) for key in ("totalCount", "total_count", "total", "totalRecords"))
    candidates.extend(payload.get(key) for key in ("totalCount", "total_count", "totalRecords"))

    parsed_values: list[int] = []
    for candidate in candidates:
        try:
            value = int(candidate)
        except (TypeError, ValueError):
            continue
        if value >= 0:
            parsed_values.append(value)
    if not parsed_values:
        return current
    discovered = max(parsed_values)
    return discovered if current is None else max(current, discovered)


def _page_signature(results: list[Any]) -> str:
    if not results:
        return "empty"
    sample = {"count": len(results), "first": results[0], "last": results[-1]}
    return json.dumps(sample, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))


def _pagination_metadata(
    *,
    complete: bool,
    page_count: int,
    page_size: int,
    initial_skip: int,
    fetched_count: int,
    reported_total: Optional[int],
    stop_reason: str,
    intentionally_limited: bool,
) -> dict[str, Any]:
    return {
        "complete": complete,
        "page_count": page_count,
        "page_size": page_size,
        "initial_skip": initial_skip,
        "fetched_count": fetched_count,
        "reported_total": reported_total,
        "stop_reason": stop_reason,
        "intentionally_limited": intentionally_limited,
    }
