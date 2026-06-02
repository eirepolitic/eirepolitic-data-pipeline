"""Endpoint discovery for the unified Oireachtas pipeline."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Optional

from .client import OireachtasClient
from .normalize import stable_hash, utc_now_iso


DISCOVERY_TABLE = "_discovery"


DISCOVERY_ENDPOINTS = [
    {
        "name": "houses",
        "endpoint": "/houses",
        "params": {"limit": 5},
    },
    {
        "name": "members",
        "endpoint": "/members",
        "params": {"chamber": "dail", "house_no": "34", "limit": 5},
    },
    {
        "name": "debates",
        "endpoint": "/debates",
        "params": {"chamber_id": "/ie/oireachtas/house/dail/34", "lang": "en", "limit": 5},
    },
    {
        "name": "divisions",
        "endpoint": "/divisions",
        "params": {"chamber": "dail", "house_no": "34", "date_start": "2025-01-01", "date_end": "2025-01-31", "limit": 5},
    },
    {
        "name": "votes_fallback_probe",
        "endpoint": "/votes",
        "params": {"chamber": "dail", "house_no": "34", "date_start": "2025-01-01", "date_end": "2025-01-31", "limit": 5},
    },
    {
        "name": "questions",
        "endpoint": "/questions",
        "params": {"chamber": "dail", "house_no": "34", "date_start": "2025-01-01", "date_end": "2025-01-31", "limit": 5},
    },
    {
        "name": "legislation",
        "endpoint": "/legislation",
        "params": {"chamber": "dail", "house_no": "34", "date_start": "2025-01-01", "date_end": "2025-01-31", "limit": 5},
    },
    {
        "name": "parties",
        "endpoint": "/parties",
        "params": {"limit": 5},
    },
    {
        "name": "constituencies",
        "endpoint": "/constituencies",
        "params": {"limit": 5},
    },
]


def run_endpoint_discovery(*, limit: int = 5, client: Optional[OireachtasClient] = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run one-page discovery across known endpoints."""
    client = client or OireachtasClient()
    rows: list[dict[str, Any]] = []
    payload_shapes: dict[str, Any] = {}
    started_at = utc_now_iso()

    for spec in DISCOVERY_ENDPOINTS:
        params = dict(spec["params"])
        params["limit"] = min(int(params.get("limit", limit)), limit)
        summary = client.get_json_summary(spec["endpoint"], params=params)
        payload = summary.payload or {}
        results = payload.get("results") if isinstance(payload, dict) else None
        results_list = results if isinstance(results, list) else []
        first_item = results_list[0] if results_list and isinstance(results_list[0], dict) else {}
        item_key_paths = sorted(_key_paths(first_item, max_depth=4)) if first_item else []
        top_keys = sorted(payload.keys()) if isinstance(payload, dict) else []
        result_wrapper_keys = sorted(first_item.keys()) if isinstance(first_item, dict) else []

        row = {
            "endpoint_name": spec["name"],
            "endpoint": spec["endpoint"],
            "ok": bool(summary.ok),
            "status_code": summary.status_code,
            "elapsed_seconds": summary.elapsed_seconds,
            "result_count": len(results_list),
            "top_keys": ",".join(top_keys),
            "result_wrapper_keys": ",".join(result_wrapper_keys),
            "schema_hash": stable_hash(item_key_paths, length=16) if item_key_paths else None,
            "error": summary.error,
            "url": summary.url,
        }
        rows.append(row)
        payload_shapes[spec["name"]] = {
            "endpoint": spec["endpoint"],
            "params": dict(summary.params),
            "ok": bool(summary.ok),
            "status_code": summary.status_code,
            "url": summary.url,
            "result_count": len(results_list),
            "top_keys": top_keys,
            "result_wrapper_keys": result_wrapper_keys,
            "item_key_paths": item_key_paths,
            "error": summary.error,
        }

    manifest = {
        "table": DISCOVERY_TABLE,
        "mode": "discover",
        "status": "success" if any(row["ok"] for row in rows) else "failed",
        "started_at_utc": started_at,
        "finished_at_utc": utc_now_iso(),
        "endpoint_count": len(rows),
        "ok_count": sum(1 for row in rows if row["ok"]),
        "failed_count": sum(1 for row in rows if not row["ok"]),
        "payload_shapes": payload_shapes,
        "notes": [
            "This is one-page endpoint discovery only, not a table build.",
            "Use the votes_fallback_probe row to compare /votes against documented /divisions.",
        ],
    }
    return rows, manifest


def discovery_schema(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Return schema metadata for discovery review output."""
    columns = list(rows[0].keys()) if rows else []
    return {
        "table": DISCOVERY_TABLE,
        "primary_key": ["endpoint_name"],
        "columns": columns,
    }


def discovery_dq(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Return lightweight DQ for discovery."""
    total = len(rows)
    ok_count = sum(1 for row in rows if row.get("ok"))
    status = "pass" if ok_count >= 1 else "fail"
    return {
        "table": DISCOVERY_TABLE,
        "dq_status": status,
        "checks": [
            {"check_name": "endpoint_rows", "status": "pass" if total else "fail", "metric_value": total, "threshold": 1},
            {"check_name": "at_least_one_endpoint_ok", "status": status, "metric_value": ok_count, "threshold": 1},
        ],
    }


def _key_paths(value: Any, *, prefix: str = "", depth: int = 0, max_depth: int = 4) -> set[str]:
    """Collect nested key paths from a JSON-like object."""
    if depth >= max_depth:
        return set()
    paths: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{prefix}.{key}" if prefix else str(key)
            paths.add(child_path)
            paths.update(_key_paths(child, prefix=child_path, depth=depth + 1, max_depth=max_depth))
    elif isinstance(value, list):
        paths.add(f"{prefix}[]" if prefix else "[]")
        if value:
            paths.update(_key_paths(value[0], prefix=f"{prefix}[]" if prefix else "[]", depth=depth + 1, max_depth=max_depth))
    return paths
