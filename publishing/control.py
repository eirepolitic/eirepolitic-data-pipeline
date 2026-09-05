from __future__ import annotations

from dataclasses import replace

from publishing.fingerprints import publication_request_fingerprint
from publishing.ledger import PublicationLedger, PublicationRecord
from publishing.models import AssetPackage, PublicationApproval, PublicationRequest, PublicationSchedule
from publishing.timezone import resolve_local_time
from publishing.validation import next_publication_version, validate_approval, validate_request_against_package


class PublicationControlError(ValueError):
    pass


class PublicationControlService:
    """Deterministic mutation boundary intended to sit behind High Director."""

    def __init__(self, ledger: PublicationLedger) -> None:
        self.ledger = ledger

    @staticmethod
    def fingerprint(request: PublicationRequest, package: AssetPackage) -> str:
        validate_request_against_package(request, package)
        hashes = [media.sha256 for media in package.media]
        return publication_request_fingerprint(request, hashes)

    def create_draft(self, request: PublicationRequest, package: AssetPackage) -> PublicationRecord:
        validate_request_against_package(request, package)
        return self.ledger.create_publication(request)

    def edit_draft(self, publication_id: str, package: AssetPackage, **changes: object) -> PublicationRecord:
        current = self.ledger.get_publication(publication_id)
        if current.state not in {"draft", "approved", "scheduled"}:
            raise PublicationControlError(f"publication cannot be edited from state {current.state}")
        request = next_publication_version(current.request, **changes)
        validate_request_against_package(request, package)
        return self.ledger.put_version(request)

    def approve(
        self,
        publication_id: str,
        package: AssetPackage,
        *,
        approval_id: str,
        approved_by: str,
        approved_at_utc: str,
    ) -> PublicationRecord:
        current = self.ledger.get_publication(publication_id)
        if current.state != "draft":
            raise PublicationControlError("only a draft publication can be approved")
        fingerprint = self.fingerprint(current.request, package)
        approval = PublicationApproval(
            approval_id=approval_id,
            publication_id=current.request.publication_id,
            publication_version=current.request.publication_version,
            request_fingerprint=fingerprint,
            approved_by=approved_by,
            approved_at_utc=approved_at_utc,
        )
        validate_approval(approval, current.request, fingerprint)
        return self.ledger.record_approval(approval)

    def schedule(
        self,
        publication_id: str,
        *,
        schedule_id: str,
        scheduled_local: str,
        timezone_name: str = "Europe/Dublin",
        fold: int | None = None,
    ) -> PublicationRecord:
        current = self.ledger.get_publication(publication_id)
        if current.state != "approved":
            raise PublicationControlError("publication must be approved before scheduling")
        resolved = resolve_local_time(scheduled_local, timezone_name, fold=fold)
        schedule = PublicationSchedule(
            schedule_id=schedule_id,
            publication_id=current.request.publication_id,
            publication_version=current.request.publication_version,
            scheduled_local=resolved.scheduled_local,
            timezone=resolved.timezone,
            scheduled_at_utc=resolved.scheduled_at_utc,
            status="scheduled",
        )
        return self.ledger.put_schedule(schedule)

    def reschedule(
        self,
        publication_id: str,
        *,
        scheduled_local: str,
        timezone_name: str = "Europe/Dublin",
        fold: int | None = None,
    ) -> PublicationRecord:
        current = self.ledger.get_publication(publication_id)
        if current.state != "scheduled" or current.schedule is None:
            raise PublicationControlError("only a scheduled publication can be rescheduled")
        resolved = resolve_local_time(scheduled_local, timezone_name, fold=fold)
        schedule = replace(
            current.schedule,
            scheduled_local=resolved.scheduled_local,
            timezone=resolved.timezone,
            scheduled_at_utc=resolved.scheduled_at_utc,
            status="scheduled",
        )
        return self.ledger.put_schedule(schedule)

    def cancel(self, publication_id: str) -> PublicationRecord:
        return self.ledger.cancel(publication_id)

    def get(self, publication_id: str) -> PublicationRecord:
        return self.ledger.get_publication(publication_id)

    def scheduled(self) -> list[PublicationRecord]:
        return self.ledger.list_by_state("scheduled")
