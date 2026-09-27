from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from publishing.canary import (
    CANARY_ACCOUNT_REF,
    CANARY_ALT_TEXT,
    CANARY_ASSET_PACKAGE_ID,
    CANARY_CAPTION,
    CANARY_HASHTAGS,
    CANARY_PERIOD,
    CANARY_PROJECT_ID,
)
from publishing.control import PublicationControlService
from publishing.dynamodb_ledger import DynamoDBPublicationLedger
from publishing.dynamodb_runtime import DynamoDBAssetPackageStore
from publishing.models import InstagramOptions, PublicationRequest
from publishing.scheduler import EventBridgePublicationScheduler, SchedulerConflict, SchedulerTarget

GATE5_PUBLICATION_ID = "instagram-scheduled-canary-20260926"
GATE5_VERSION = 1
GATE5_SCHEDULE_ID = "gate5-scheduled-canary-20260926"
GATE5_LOCAL = "2026-09-26T17:45:00"
GATE5_TIMEZONE = "America/Vancouver"


@dataclass(frozen=True)
class Gate5PreparationResult:
    publication_id: str
    version: int
    state: str
    scheduled_at_utc: str
    schedule_name: str
    verified: bool


def prepare_scheduled_canary(
    *,
    dynamodb_resource: Any,
    scheduler_client: Any,
    target: SchedulerTarget,
    table_name: str = "eirepolitic-publications",
) -> Gate5PreparationResult:
    table = dynamodb_resource.Table(table_name)
    ledger = DynamoDBPublicationLedger(table)
    assets = DynamoDBAssetPackageStore(table)
    control = PublicationControlService(ledger)
    package = assets.get(CANARY_ASSET_PACKAGE_ID)

    request = PublicationRequest(
        publication_id=GATE5_PUBLICATION_ID,
        publication_version=GATE5_VERSION,
        platform="instagram",
        account_ref=CANARY_ACCOUNT_REF,
        project_id=CANARY_PROJECT_ID,
        period=CANARY_PERIOD,
        asset_package_id=CANARY_ASSET_PACKAGE_ID,
        caption=CANARY_CAPTION,
        hashtags=CANARY_HASHTAGS,
        caption_mentions=(),
        instagram=InstagramOptions(
            post_type="image",
            media_tags=(),
            collaborators=(),
            location_id=None,
            first_comment=None,
        ),
    )

    try:
        record = control.create_draft(request, package)
    except Exception:
        record = control.get(GATE5_PUBLICATION_ID)
        if record.request != request:
            raise

    if record.state == "draft":
        record = control.approve(
            GATE5_PUBLICATION_ID,
            package,
            approval_id="gate5-scheduled-canary-approval-20260926",
            approved_by="user-explicit-gate5-chat",
            approved_at_utc="2026-09-27T00:07:00Z",
        )
    if record.state == "approved":
        record = control.schedule(
            GATE5_PUBLICATION_ID,
            schedule_id=GATE5_SCHEDULE_ID,
            scheduled_local=GATE5_LOCAL,
            timezone_name=GATE5_TIMEZONE,
        )
    if record.state != "scheduled" or record.schedule is None:
        raise RuntimeError(f"Gate 5 canary is not scheduled: {record.state}")

    scheduler = EventBridgePublicationScheduler(scheduler_client, target)
    try:
        schedule_name = scheduler.create(record.schedule)
    except SchedulerConflict:
        schedule_name = scheduler.schedule_name(record.schedule)
    verified = scheduler.verify(record.schedule)
    if not verified:
        raise RuntimeError("EventBridge Scheduler verification failed for Gate 5 canary")

    return Gate5PreparationResult(
        publication_id=record.request.publication_id,
        version=record.request.publication_version,
        state=record.state,
        scheduled_at_utc=record.schedule.scheduled_at_utc,
        schedule_name=schedule_name,
        verified=verified,
    )
