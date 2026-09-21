from __future__ import annotations

import pytest

from publishing.control import PublicationControlError, PublicationControlService
from publishing.ledger import InMemoryPublicationLedger, LedgerConflict
from publishing.models import AssetPackage, InstagramOptions, MediaAsset, PublicationRequest


def _package() -> AssetPackage:
    return AssetPackage(
        asset_package_id="pkg-1",
        project_id="demo",
        period="2026-09",
        media=(
            MediaAsset(
                asset_id="slide-1",
                ordinal=1,
                bucket="eirepolitic-data",
                key="instagram/approved/demo/2026-09/pkg-1/media/01.jpg",
                sha256="abc123",
                mime_type="image/jpeg",
                width=1080,
                height=1350,
                size_bytes=12345,
                alt_text="Demo slide",
            ),
        ),
        publication_ready=True,
        review_status="approved",
    )


def _request() -> PublicationRequest:
    return PublicationRequest(
        publication_id="pub-1",
        publication_version=1,
        platform="instagram",
        account_ref="eirepolitic_instagram",
        project_id="demo",
        period="2026-09",
        asset_package_id="pkg-1",
        caption="Demo post #EirePolitic",
        hashtags=("#EirePolitic",),
        instagram=InstagramOptions(post_type="image"),
    )


def _service() -> PublicationControlService:
    return PublicationControlService(InMemoryPublicationLedger())


def _approved(service: PublicationControlService) -> None:
    service.create_draft(_request(), _package())
    service.approve(
        "pub-1",
        _package(),
        approval_id="approval-1",
        approved_by="human-1",
        approved_at_utc="2026-09-05T20:00:00Z",
    )


def test_create_approve_schedule_and_query() -> None:
    service = _service()
    service.create_draft(_request(), _package())
    assert service.get("pub-1").state == "draft"

    approved = service.approve(
        "pub-1",
        _package(),
        approval_id="approval-1",
        approved_by="human-1",
        approved_at_utc="2026-09-05T20:00:00Z",
    )
    assert approved.state == "approved"
    assert approved.approval is not None

    scheduled = service.schedule(
        "pub-1",
        schedule_id="schedule-1",
        scheduled_local="2026-09-08T19:30:00",
    )
    assert scheduled.state == "scheduled"
    assert scheduled.schedule is not None
    assert scheduled.schedule.scheduled_at_utc == "2026-09-08T18:30:00Z"
    assert [record.request.publication_id for record in service.scheduled()] == ["pub-1"]


def test_unapproved_publication_cannot_be_scheduled() -> None:
    service = _service()
    service.create_draft(_request(), _package())
    with pytest.raises(PublicationControlError, match="approved"):
        service.schedule("pub-1", schedule_id="schedule-1", scheduled_local="2026-09-08T19:30:00")


def test_material_edit_invalidates_existing_approval_and_schedule() -> None:
    service = _service()
    _approved(service)
    service.schedule("pub-1", schedule_id="schedule-1", scheduled_local="2026-09-08T19:30:00")

    edited = service.edit_draft("pub-1", _package(), caption="Changed #EirePolitic")
    assert edited.request.publication_version == 2
    assert edited.state == "draft"
    assert edited.approval is None
    assert edited.schedule is None


def test_reschedule_preserves_content_approval() -> None:
    service = _service()
    _approved(service)
    before = service.schedule("pub-1", schedule_id="schedule-1", scheduled_local="2026-09-08T19:30:00")
    approval_id = before.approval.approval_id

    after = service.reschedule("pub-1", scheduled_local="2026-09-08T20:00:00")
    assert after.state == "scheduled"
    assert after.approval.approval_id == approval_id
    assert after.schedule.scheduled_at_utc == "2026-09-08T19:00:00Z"


def test_cancel_marks_schedule_cancelled() -> None:
    service = _service()
    _approved(service)
    service.schedule("pub-1", schedule_id="schedule-1", scheduled_local="2026-09-08T19:30:00")

    cancelled = service.cancel("pub-1")
    assert cancelled.state == "cancelled"
    assert cancelled.schedule.status == "cancelled"
    assert service.scheduled() == []


def test_duplicate_publication_id_is_rejected() -> None:
    service = _service()
    service.create_draft(_request(), _package())
    with pytest.raises(LedgerConflict, match="already exists"):
        service.create_draft(_request(), _package())


def test_skipped_version_is_rejected() -> None:
    service = _service()
    service.create_draft(_request(), _package())
    current = service.get("pub-1").request
    from dataclasses import replace

    with pytest.raises(LedgerConflict, match="exactly one"):
        service.ledger.put_version(replace(current, publication_version=3))
