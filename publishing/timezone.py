from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo


class TimeResolutionError(ValueError):
    pass


@dataclass(frozen=True)
class ResolvedScheduleTime:
    scheduled_local: str
    timezone: str
    scheduled_at_utc: str
    utc_offset: str


def _roundtrip_matches(local_naive: datetime, zone: ZoneInfo, fold: int) -> tuple[bool, datetime]:
    aware = local_naive.replace(tzinfo=zone, fold=fold)
    utc_value = aware.astimezone(timezone.utc)
    roundtrip = utc_value.astimezone(zone).replace(tzinfo=None)
    return roundtrip == local_naive, aware


def resolve_local_time(local_iso: str, timezone_name: str = "Europe/Dublin", *, fold: int | None = None) -> ResolvedScheduleTime:
    """Resolve one local wall-clock time to an unambiguous UTC instant.

    Raises for nonexistent DST times and for ambiguous times unless fold is explicitly 0 or 1.
    """
    local_naive = datetime.fromisoformat(local_iso)
    if local_naive.tzinfo is not None:
        raise TimeResolutionError("scheduled_local must be a timezone-free local datetime")

    zone = ZoneInfo(timezone_name)
    valid0, aware0 = _roundtrip_matches(local_naive, zone, 0)
    valid1, aware1 = _roundtrip_matches(local_naive, zone, 1)

    if not valid0 and not valid1:
        raise TimeResolutionError(f"{local_iso} does not exist in {timezone_name}")

    utc0 = aware0.astimezone(timezone.utc)
    utc1 = aware1.astimezone(timezone.utc)
    ambiguous = valid0 and valid1 and utc0 != utc1

    if ambiguous:
        if fold not in (0, 1):
            raise TimeResolutionError(f"{local_iso} occurs twice in {timezone_name}; explicit fold 0 or 1 is required")
        aware = aware0 if fold == 0 else aware1
    else:
        aware = aware0 if valid0 else aware1

    utc_value = aware.astimezone(timezone.utc)
    offset = aware.strftime("%z")
    offset = f"{offset[:3]}:{offset[3:]}" if offset else "+00:00"
    return ResolvedScheduleTime(
        scheduled_local=local_naive.isoformat(timespec="seconds"),
        timezone=timezone_name,
        scheduled_at_utc=utc_value.isoformat(timespec="seconds").replace("+00:00", "Z"),
        utc_offset=offset,
    )
