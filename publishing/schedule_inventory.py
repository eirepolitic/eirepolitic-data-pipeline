from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from publishing.dynamodb_ledger import DynamoDBPublicationLedger
from publishing.scheduler import EventBridgePublicationScheduler, SchedulerTarget


@dataclass(frozen=True)
class ScheduledPublicationInventoryItem:
    publication_id: str
    publication_version: int
    project_id: str
    period: str
    scheduled_local: str
    timezone: str
    scheduled_at_utc: str
    schedule_name: str
    scheduler_state: str
    scheduler_matches_ledger: bool


def list_upcoming_scheduled_publications(
    *,
    table: Any,
    scheduler_client: Any,
    scheduler_target: SchedulerTarget,
    now_utc: str | None = None,
) -> tuple[ScheduledPublicationInventoryItem, ...]:
    ledger = DynamoDBPublicationLedger(table)
    scheduler = EventBridgePublicationScheduler(scheduler_client, scheduler_target)
    now = now_utc or datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

    records = ledger.list_by_state("scheduled", scheduled_from_utc=now)
    items: list[ScheduledPublicationInventoryItem] = []

    for record in records:
        if record.schedule is None:
            continue
        schedule_name = scheduler.schedule_name(record.schedule)
        try:
            live = scheduler_client.get_schedule(Name=schedule_name, GroupName=scheduler_target.group_name)
            scheduler_state = str(live.get("State", "UNKNOWN"))
            scheduler_matches = scheduler.verify(record.schedule)
        except Exception:
            scheduler_state = "MISSING"
            scheduler_matches = False

        items.append(
            ScheduledPublicationInventoryItem(
                publication_id=record.request.publication_id,
                publication_version=record.request.publication_version,
                project_id=record.request.project_id,
                period=record.request.period,
                scheduled_local=record.schedule.scheduled_local,
                timezone=record.schedule.timezone,
                scheduled_at_utc=record.schedule.scheduled_at_utc,
                schedule_name=schedule_name,
                scheduler_state=scheduler_state,
                scheduler_matches_ledger=scheduler_matches,
            )
        )

    return tuple(items)
