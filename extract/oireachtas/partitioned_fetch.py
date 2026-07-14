"""Adaptive date-window fetching for API endpoints with offset ceilings."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Mapping

from .client import ApiResponseSummary, OireachtasClient


def get_date_partitioned_json_summary(
    client: OireachtasClient,
    endpoint: str,
    *,
    params: Mapping[str, Any],
    max_depth: int = 12,
) -> ApiResponseSummary:
    """Fetch a date-bounded endpoint, splitting only when its offset cap is hit.

    Partitions are inclusive and non-overlapping: the left partition ends at the
    midpoint and the right partition begins the following day.
    """
    original = dict(params)
    partitions: list[dict[str, Any]] = []

    def fetch(window_params: dict[str, Any], depth: int) -> ApiResponseSummary:
        summary = client.get_json_summary(endpoint, params=window_params)
        if summary.ok:
            partitions.append(
                {
                    "date_start": window_params.get("date_start"),
                    "date_end": window_params.get("date_end"),
                    "fetched_count": len((summary.payload or {}).get("results") or []),
                    "page_count": (summary.pagination or {}).get("page_count"),
                    "stop_reason": (summary.pagination or {}).get("stop_reason"),
                }
            )
            return summary

        if not _is_offset_ceiling(summary) or depth >= max_depth:
            return summary
        start = _date(window_params.get("date_start"))
        end = _date(window_params.get("date_end"))
        if start is None or end is None or start >= end:
            return summary

        midpoint = start + timedelta(days=(end - start).days // 2)
        left_params = dict(window_params)
        left_params["date_start"] = start.isoformat()
        left_params["date_end"] = midpoint.isoformat()
        right_params = dict(window_params)
        right_params["date_start"] = (midpoint + timedelta(days=1)).isoformat()
        right_params["date_end"] = end.isoformat()

        left = fetch(left_params, depth + 1)
        if not left.ok:
            return left
        right = fetch(right_params, depth + 1)
        if not right.ok:
            return right
        return _merge(left, right, endpoint=endpoint, original_params=original, partitions=partitions)

    result = fetch(original, 0)
    if not result.ok:
        return result
    payload = dict(result.payload or {})
    pagination = dict(result.pagination or {})
    pagination.update(
        {
            "partitioned": len(partitions) > 1,
            "partition_count": len(partitions),
            "partitions": list(partitions),
            "complete": True,
            "fetched_count": len(payload.get("results") or []),
        }
    )
    payload["_pagination"] = pagination
    return ApiResponseSummary(
        endpoint=endpoint,
        url=result.url,
        params=original,
        status_code=result.status_code,
        ok=True,
        elapsed_seconds=result.elapsed_seconds,
        payload=payload,
        pagination=pagination,
    )


def _is_offset_ceiling(summary: ApiResponseSummary) -> bool:
    error = str(summary.error or "")
    stop_reason = str((summary.pagination or {}).get("stop_reason") or "")
    return stop_reason == "page_error" and "422" in error


def _date(value: Any):
    try:
        return datetime.strptime(str(value), "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None


def _merge(
    left: ApiResponseSummary,
    right: ApiResponseSummary,
    *,
    endpoint: str,
    original_params: Mapping[str, Any],
    partitions: list[dict[str, Any]],
) -> ApiResponseSummary:
    left_payload = dict(left.payload or {})
    right_payload = dict(right.payload or {})
    results = list(left_payload.get("results") or []) + list(right_payload.get("results") or [])
    payload = dict(left_payload)
    payload["results"] = results
    pagination = {
        "complete": True,
        "partitioned": True,
        "partition_count": len(partitions),
        "partitions": list(partitions),
        "page_count": int((left.pagination or {}).get("page_count") or 0)
        + int((right.pagination or {}).get("page_count") or 0),
        "fetched_count": len(results),
        "stop_reason": "date_partitions_complete",
    }
    payload["_pagination"] = pagination
    return ApiResponseSummary(
        endpoint=endpoint,
        url=left.url,
        params=dict(original_params),
        status_code=right.status_code or left.status_code,
        ok=True,
        elapsed_seconds=round(float(left.elapsed_seconds or 0) + float(right.elapsed_seconds or 0), 3),
        payload=payload,
        pagination=pagination,
    )
