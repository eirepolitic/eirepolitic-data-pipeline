"""Shared Oireachtas API client for unified extraction/discovery."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional
from urllib.parse import urljoin

import requests


DEFAULT_API_BASE_URL = "https://api.oireachtas.ie/v1"
DEFAULT_DATA_BASE_URL = "https://data.oireachtas.ie"


@dataclass(frozen=True)
class ApiResponseSummary:
    """Small response summary used by discovery and manifests."""

    endpoint: str
    url: str
    params: Mapping[str, Any]
    status_code: Optional[int]
    ok: bool
    elapsed_seconds: Optional[float]
    error: Optional[str] = None
    payload: Optional[Mapping[str, Any]] = field(default=None, repr=False)


class OireachtasClient:
    """Small resilient client for Oireachtas API calls.

    Full pagination/backfill behaviour is added in later packets. F03 needs
    reliable single-page discovery calls with retry/backoff and clear errors.
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
                "User-Agent": "eirepolitic-data-pipeline/oireachtas-unified-discovery",
            }
        )

    def endpoint_url(self, endpoint: str) -> str:
        """Return absolute API URL for `/endpoint` or `endpoint`."""
        clean = endpoint.strip()
        if clean.startswith("http://") or clean.startswith("https://"):
            return clean
        clean = clean.lstrip("/")
        return urljoin(f"{self.base_url}/", clean)

    def get_json_summary(self, endpoint: str, *, params: Optional[Mapping[str, Any]] = None) -> ApiResponseSummary:
        """GET one JSON page and return status, payload, and error details."""
        url = self.endpoint_url(endpoint)
        params_dict = {k: v for k, v in dict(params or {}).items() if v is not None and v != ""}
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
            except Exception as exc:  # intentionally broad for discovery reporting
                elapsed = round(time.monotonic() - started, 3)
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < self.retries:
                    time.sleep(self.backoff_seconds * attempt)
                    continue

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
