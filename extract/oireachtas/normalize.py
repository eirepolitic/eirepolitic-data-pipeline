"""Shared normalization helpers for the unified Oireachtas pipeline.

These helpers are deterministic and side-effect free. They are safe to use in
table builders, tests, and review/reporting code.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from datetime import date, datetime, timezone
from typing import Any, Iterable, Mapping, Optional


_DATA_BASE_URL = "https://data.oireachtas.ie"


def safe_text(value: Any, *, default: str = "") -> str:
    """Return a stripped string for a potentially missing value."""
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def snake_case(value: Any, *, default: str = "col") -> str:
    """Convert a value to a stable lowercase snake_case identifier."""
    text = safe_text(value).lower()
    if not text:
        return default
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or default


def normalize_name(value: Any) -> str:
    """Normalize a display name for fallback matching only."""
    text = safe_text(value).lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_iso_date(value: Any) -> Optional[str]:
    """Parse common API date/datetime values into YYYY-MM-DD.

    Returns None when the value cannot be parsed. Keeps date handling simple at
    this layer so CSV and Parquet writers can choose their own dtypes later.
    """
    text = safe_text(value)
    if not text:
        return None

    match = re.match(r"^(\d{4}-\d{2}-\d{2})", text)
    if match:
        return match.group(1)

    for fmt in ("%d/%m/%Y", "%Y/%m/%d", "%d-%m-%Y"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue

    return None


def utc_now_iso() -> str:
    """Return a timezone-aware UTC timestamp string."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def stable_json_dumps(value: Any) -> str:
    """Serialize JSON-like data deterministically for hashing/manifests."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def stable_hash(parts: Iterable[Any], *, length: int = 16) -> str:
    """Create a deterministic short SHA-256 hash from stable fields."""
    joined = "|".join(safe_text(part).lower() for part in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:length]


def stable_record_hash(record: Mapping[str, Any], *, length: int = 16) -> str:
    """Hash a JSON-like mapping deterministically."""
    return hashlib.sha256(stable_json_dumps(record).encode("utf-8")).hexdigest()[:length]


def normalize_format_url(uri: Any, *, data_base_url: str = _DATA_BASE_URL) -> Optional[str]:
    """Normalize an Oireachtas formats URI into an absolute data.oireachtas.ie URL."""
    text = safe_text(uri)
    if not text:
        return None
    if text.startswith(("http://", "https://")):
        return text
    if not text.startswith("/"):
        text = f"/{text}"
    return f"{data_base_url.rstrip('/')}{text}"


def is_current_range(start: Any = None, end: Any = None, *, today: Optional[date] = None) -> bool:
    """Return whether a date range appears current today.

    Missing end date means current/open-ended. Missing or unparsable start does
    not prevent current=true if end is open/future.
    """
    current_day = today or date.today()
    end_iso = parse_iso_date(end)
    if not end_iso:
        return True
    try:
        return datetime.strptime(end_iso, "%Y-%m-%d").date() >= current_day
    except ValueError:
        return False
