from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from publishing.control import PublicationControlService
from publishing.dynamodb_ledger import DynamoDBPublicationLedger
from publishing.dynamodb_runtime import DynamoDBAssetPackageStore
from publishing.models import InstagramOptions, InstagramUserTag, PublicationRequest
from publishing.s3_assets import S3ApprovedAssetStore, SourceAsset
from publishing.scheduler import EventBridgePublicationScheduler, SchedulerTarget

_HASHTAG = re.compile(r"(?<!\w)#[A-Za-z0-9_]+")
_MENTION = re.compile(r"(?<!\w)@[A-Za-z0-9._]+")


@dataclass(frozen=True)
class StandardPublicationResult:
    publication_id: str
    publication_version: int
    state: str
    asset_package_id: str
    mode: str
    scheduled_at_utc: str | None = None
    schedule_name: str | None = None


def _slug(value: str) -> str:
    clean = "".join(ch.lower() if ch.isalnum() else "-" for ch in value)
    return "-".join(part for part in clean.split("-") if part)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _load_options(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("options_json must be a JSON object")
    return parsed


def prepare_standard_publication(
    *,
    artifact_root: str | Path,
    factory_run_id: str,
    approved_by: str,
    mode: Literal["immediate", "scheduled"],
    bucket: str,
    dynamodb_resource: Any,
    s3_client: Any,
    scheduler_client: Any | None = None,
    scheduler_target: SchedulerTarget | None = None,
    scheduled_local: str | None = None,
    timezone_name: str = "America/Vancouver",
    caption_override: str | None = None,
    options_json: str | None = None,
    table_name: str = "eirepolitic-publications",
) -> StandardPublicationResult:
    root = Path(artifact_root)
    handoff_path = root / "publication_handoff.json"
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    if handoff.get("schema_version") != 1:
        raise ValueError("unsupported publication handoff schema")
    if not handoff.get("qa_all_passed"):
        raise ValueError("factory handoff QA did not pass")
    if handoff.get("factory_publication_enabled") is not False:
        raise ValueError("factory publication guard must remain false")

    project_id = str(handoff["project_id"])
    period = str(handoff["period_key"])
    publication_id = f"ig-{_slug(project_id)}-{_slug(period)}-{factory_run_id}"
    asset_package_id = f"assets-{publication_id}"

    options = _load_options(options_json)
    alt_texts = options.get("alt_text") or []
    if alt_texts and len(alt_texts) != len(handoff["slides"]):
        raise ValueError("alt_text must contain exactly one value per slide")

    sources: list[SourceAsset] = []
    for index, slide in enumerate(handoff["slides"]):
        path = (root / slide["path"]).resolve()
        if root.resolve() not in path.parents:
            raise ValueError(f"slide path escapes artifact root: {slide['path']}")
        sources.append(SourceAsset(path, alt_text=(alt_texts[index] if alt_texts else "")))

    caption_path = handoff.get("caption_path")
    if caption_override is not None and caption_override != "":
        caption = caption_override
    elif caption_path:
        caption = (root / caption_path).read_text(encoding="utf-8").strip()
    else:
        raise ValueError("factory run has no caption; supply caption_override")

    table = dynamodb_resource.Table(table_name)
    package_store = DynamoDBAssetPackageStore(table)
    ledger = DynamoDBPublicationLedger(table)
    control = PublicationControlService(ledger)

    package = S3ApprovedAssetStore(s3_client, bucket).finalize_package(
        project_id=project_id,
        period=period,
        asset_package_id=asset_package_id,
        sources=sources,
    )
    package_store.put(package)

    media_tags = tuple(
        InstagramUserTag(
            username=str(tag["username"]),
            media_ordinal=int(tag["media_ordinal"]),
            x=(float(tag["x"]) if tag.get("x") is not None else None),
            y=(float(tag["y"]) if tag.get("y") is not None else None),
        )
        for tag in options.get("media_tags", [])
    )
    request = PublicationRequest(
        publication_id=publication_id,
        publication_version=1,
        platform="instagram",
        account_ref="eirepolitic-instagram",
        project_id=project_id,
        period=period,
        asset_package_id=asset_package_id,
        caption=caption,
        hashtags=tuple(dict.fromkeys(_HASHTAG.findall(caption))),
        caption_mentions=tuple(dict.fromkeys(_MENTION.findall(caption))),
        instagram=InstagramOptions(
            post_type="image" if len(package.media) == 1 else "carousel",
            media_tags=media_tags,
            collaborators=tuple(options.get("collaborators", [])),
            location_id=options.get("location_id"),
            first_comment=options.get("first_comment"),
        ),
    )

    try:
        record = control.create_draft(request, package)
    except Exception:
        record = control.get(publication_id)
        if record.request != request:
            raise

    if record.state == "draft":
        record = control.approve(
            publication_id,
            package,
            approval_id=f"approval-{publication_id}",
            approved_by=approved_by,
            approved_at_utc=_utc_now(),
        )

    if mode == "immediate":
        if record.state not in {"approved", "published"}:
            raise ValueError(f"immediate publication has unexpected state: {record.state}")
        return StandardPublicationResult(
            publication_id=publication_id,
            publication_version=record.request.publication_version,
            state=record.state,
            asset_package_id=asset_package_id,
            mode=mode,
        )

    if not scheduled_local:
        raise ValueError("scheduled_local is required for scheduled mode")
    if record.state == "approved":
        record = control.schedule(
            publication_id,
            schedule_id=f"schedule-{publication_id}",
            scheduled_local=scheduled_local,
            timezone_name=timezone_name,
        )
    if record.state != "scheduled" or record.schedule is None:
        raise ValueError(f"scheduled publication has unexpected state: {record.state}")
    if scheduler_client is None or scheduler_target is None:
        raise ValueError("scheduler client/target are required for scheduled mode")
    scheduler = EventBridgePublicationScheduler(scheduler_client, scheduler_target)
    try:
        schedule_name = scheduler.create(record.schedule)
    except Exception:
        schedule_name = scheduler.schedule_name(record.schedule)
        if not scheduler.verify(record.schedule):
            raise
    if not scheduler.verify(record.schedule):
        raise RuntimeError("EventBridge schedule verification failed")
    return StandardPublicationResult(
        publication_id=publication_id,
        publication_version=record.request.publication_version,
        state=record.state,
        asset_package_id=asset_package_id,
        mode=mode,
        scheduled_at_utc=record.schedule.scheduled_at_utc,
        schedule_name=schedule_name,
    )


def result_json(result: StandardPublicationResult) -> str:
    return json.dumps(asdict(result), indent=2, sort_keys=True)
