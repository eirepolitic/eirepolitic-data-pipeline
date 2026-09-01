from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import calendar
import re


_PERIOD_RE = re.compile(r"^(\d{4})-(\d{2})$")


@dataclass(frozen=True)
class MonthlyPeriod:
    key: str
    start: date
    end: date

    @property
    def label(self) -> str:
        return self.start.strftime("%B %Y")


def resolve_monthly_period(value: str, *, today: date | None = None) -> MonthlyPeriod:
    """Resolve YYYY-MM or last_completed_month to an inclusive calendar month."""
    today = today or date.today()
    value = (value or "last_completed_month").strip().lower()

    if value == "last_completed_month":
        first_this_month = today.replace(day=1)
        end = first_this_month - timedelta(days=1)
        start = end.replace(day=1)
        return MonthlyPeriod(key=start.strftime("%Y-%m"), start=start, end=end)

    match = _PERIOD_RE.match(value)
    if not match:
        raise ValueError("period must be YYYY-MM or last_completed_month")

    year, month = (int(part) for part in match.groups())
    if month < 1 or month > 12:
        raise ValueError("period month must be between 01 and 12")

    start = date(year, month, 1)
    end = date(year, month, calendar.monthrange(year, month)[1])
    if end >= today:
        raise ValueError(f"period {value} is not a completed month as of {today.isoformat()}")
    return MonthlyPeriod(key=value, start=start, end=end)
