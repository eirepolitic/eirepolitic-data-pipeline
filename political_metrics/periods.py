from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from calendar import monthrange
from zoneinfo import ZoneInfo

DUBLIN = ZoneInfo("Europe/Dublin")


@dataclass(frozen=True)
class MetricPeriod:
    start: date
    end: date
    label: str
    kind: str

    def contains(self, value: date) -> bool:
        return self.start <= value <= self.end


def _as_date(value: date | datetime | str) -> date:
    if isinstance(value, datetime):
        return value.astimezone(DUBLIN).date() if value.tzinfo else value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)


def _month_period(year: int, month: int, label: str | None = None) -> MetricPeriod:
    end_day = monthrange(year, month)[1]
    return MetricPeriod(date(year, month, 1), date(year, month, end_day), label or f"{year:04d}-{month:02d}", "month")


def resolve_period(spec: str | tuple[date | str, date | str], *, today: date | datetime | str | None = None) -> MetricPeriod:
    """Resolve common public metric period specifications to inclusive Dublin dates.

    Supported forms:
    - YYYY-MM
    - YYYY
    - YYYY-Q1 .. YYYY-Q4
    - last_completed_month
    - rolling_7d / rolling_30d / rolling_90d
    - (start, end) tuple using ISO dates or date objects
    """
    if isinstance(spec, tuple):
        start, end = map(_as_date, spec)
        if end < start:
            raise ValueError("period end cannot be before period start")
        return MetricPeriod(start, end, f"{start.isoformat()}_{end.isoformat()}", "date_range")

    current = _as_date(today or datetime.now(DUBLIN))

    if spec == "last_completed_month":
        first_this_month = current.replace(day=1)
        previous_day = first_this_month - timedelta(days=1)
        return _month_period(previous_day.year, previous_day.month, "last_completed_month")

    if spec.startswith("rolling_") and spec.endswith("d"):
        try:
            days = int(spec[len("rolling_") : -1])
        except ValueError as exc:
            raise ValueError(f"unsupported period: {spec}") from exc
        if days not in {7, 30, 90}:
            raise ValueError(f"unsupported rolling period: {spec}")
        return MetricPeriod(current - timedelta(days=days - 1), current, spec, "rolling")

    if len(spec) == 7 and spec[4] == "-":
        year, month = map(int, spec.split("-"))
        return _month_period(year, month)

    if len(spec) == 7 and spec[4:6] == "-Q":
        year = int(spec[:4])
        quarter = int(spec[-1])
        if quarter not in {1, 2, 3, 4}:
            raise ValueError(f"unsupported quarter: {spec}")
        start_month = 1 + ((quarter - 1) * 3)
        start = date(year, start_month, 1)
        end_month = start_month + 2
        end = date(year, end_month, monthrange(year, end_month)[1])
        return MetricPeriod(start, end, spec, "quarter")

    if len(spec) == 4 and spec.isdigit():
        year = int(spec)
        return MetricPeriod(date(year, 1, 1), date(year, 12, 31), spec, "year")

    raise ValueError(f"unsupported period specification: {spec}")
