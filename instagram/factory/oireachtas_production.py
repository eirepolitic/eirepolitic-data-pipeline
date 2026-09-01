from __future__ import annotations

import json
import re
from typing import Any

BATCH_ROOT = "processed/oireachtas_unified/batches"
PRODUCTION_POINTER_KEY = "processed/oireachtas_unified/pointers/production.json"
_LATEST_PATTERN = re.compile(
    r"^processed/oireachtas_unified/latest/(?P<format>csv|parquet)/(?P<table>[^/]+)\.(?P<extension>csv|parquet)$"
)


def resolve_production_key(s3: Any, *, bucket: str, production_key: str) -> tuple[str, dict[str, str]]:
    """Resolve a unified logical production key through the promoted batch pointer."""
    obj = s3.get_object(Bucket=bucket, Key=PRODUCTION_POINTER_KEY)
    pointer = json.loads(obj["Body"].read().decode("utf-8"))
    mode = str(pointer.get("mode") or "batch")
    if mode == "legacy_direct":
        return production_key, {"mode": mode, "batch_id": ""}
    if mode != "batch":
        raise RuntimeError(f"Unsupported Oireachtas production pointer mode: {mode}")

    batch_id = str(pointer.get("batch_id") or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", batch_id):
        raise RuntimeError("Oireachtas production pointer contains an invalid batch_id")

    match = _LATEST_PATTERN.fullmatch(production_key)
    if not match:
        raise RuntimeError(f"Unsupported unified logical key: {production_key}")

    resolved = (
        f"{BATCH_ROOT}/{batch_id}/tables/{match.group('table')}/"
        f"{match.group('format')}/{match.group('table')}.{match.group('extension')}"
    )
    return resolved, {"mode": mode, "batch_id": batch_id}
