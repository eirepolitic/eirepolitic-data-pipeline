from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from typing import Any


def _normalize(value: Any) -> Any:
    if is_dataclass(value):
        return _normalize(asdict(value))
    if isinstance(value, dict):
        return {str(k): _normalize(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_normalize(v) for v in value]
    return value


def canonical_json(value: Any) -> str:
    """Return stable UTF-8 JSON used for approval fingerprints."""
    return json.dumps(
        _normalize(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_fingerprint(value: Any) -> str:
    payload = canonical_json(value).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def publication_request_fingerprint(request: Any, asset_hashes: list[str]) -> str:
    """Fingerprint a complete approved request plus ordered immutable asset hashes."""
    material = {
        "request": request,
        "ordered_asset_hashes": asset_hashes,
    }
    return sha256_fingerprint(material)
