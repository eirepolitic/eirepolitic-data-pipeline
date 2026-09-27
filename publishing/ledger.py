from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Protocol

from publishing.models import PublicationApproval, PublicationRequest, PublicationSchedule


class LedgerError(RuntimeError):
    pass


class LedgerConflict(LedgerError):
    pass


class LedgerNotFound(LedgerError):
    pass


@dataclass(frozen=True)
class PublicationRecord:
    request: PublicationRequest
    approval: PublicationApproval | None = None
    schedule: PublicationSchedule | None = None
    state: str = "draft"


class PublicationLedger(Protocol):
    def create_publication(self, request: PublicationRequest) -> PublicationRecord: ...
    def get_publication(self, publication_id: str) -> PublicationRecord: ...
    def put_version(self, request: PublicationRequest) -> PublicationRecord: ...
    def record_approval(self, approval: PublicationApproval) -> PublicationRecord: ...
    def put_schedule(self, schedule: PublicationSchedule) -> PublicationRecord: ...
    def cancel(self, publication_id: str) -> PublicationRecord: ...
    def list_by_state(self, state: str) -> list[PublicationRecord]: ...


class InMemoryPublicationLedger:
    """Thread-safe reference implementation used for control-plane tests.

    DynamoDB will implement the same behavioral contract later. This class is not
    intended as production persistence.
    """

    def __init__(self) -> None:
        self._records: dict[str, PublicationRecord] = {}
        self._lock = RLock()

    def create_publication(self, request: PublicationRequest) -> PublicationRecord:
        with self._lock:
            if request.publication_id in self._records:
                raise LedgerConflict(f"publication already exists: {request.publication_id}")
            record = PublicationRecord(request=request)
            self._records[request.publication_id] = record
            return record

    def get_publication(self, publication_id: str) -> PublicationRecord:
        with self._lock:
            try:
                return self._records[publication_id]
            except KeyError as exc:
                raise LedgerNotFound(f"publication not found: {publication_id}") from exc

    def put_version(self, request: PublicationRequest) -> PublicationRecord:
        with self._lock:
            current = self.get_publication(request.publication_id)
            if request.publication_version != current.request.publication_version + 1:
                raise LedgerConflict("new publication version must increment current version by exactly one")
            record = PublicationRecord(request=request, state="draft")
            self._records[request.publication_id] = record
            return record

    def record_approval(self, approval: PublicationApproval) -> PublicationRecord:
        with self._lock:
            current = self.get_publication(approval.publication_id)
            if current.request.publication_version != approval.publication_version:
                raise LedgerConflict("approval version does not match current publication version")
            if current.state != "draft":
                raise LedgerConflict(f"cannot approve publication in state {current.state}")
            record = PublicationRecord(
                request=current.request,
                approval=approval,
                schedule=current.schedule,
                state="approved",
            )
            self._records[approval.publication_id] = record
            return record

    def put_schedule(self, schedule: PublicationSchedule) -> PublicationRecord:
        with self._lock:
            current = self.get_publication(schedule.publication_id)
            if current.request.publication_version != schedule.publication_version:
                raise LedgerConflict("schedule version does not match current publication version")
            if current.approval is None:
                raise LedgerConflict("publication must be approved before scheduling")
            if current.state not in {"approved", "scheduled"}:
                raise LedgerConflict(f"cannot schedule publication in state {current.state}")
            record = PublicationRecord(
                request=current.request,
                approval=current.approval,
                schedule=schedule,
                state="scheduled" if schedule.status == "scheduled" else current.state,
            )
            self._records[schedule.publication_id] = record
            return record

    def cancel(self, publication_id: str) -> PublicationRecord:
        with self._lock:
            current = self.get_publication(publication_id)
            if current.state not in {"approved", "scheduled"}:
                raise LedgerConflict(f"cannot cancel publication in state {current.state}")
            schedule = current.schedule
            if schedule is not None:
                schedule = PublicationSchedule(
                    schedule_id=schedule.schedule_id,
                    publication_id=schedule.publication_id,
                    publication_version=schedule.publication_version,
                    scheduled_local=schedule.scheduled_local,
                    timezone=schedule.timezone,
                    scheduled_at_utc=schedule.scheduled_at_utc,
                    status="cancelled",
                )
            record = PublicationRecord(
                request=current.request,
                approval=current.approval,
                schedule=schedule,
                state="cancelled",
            )
            self._records[publication_id] = record
            return record

    def list_by_state(self, state: str) -> list[PublicationRecord]:
        with self._lock:
            return [record for record in self._records.values() if record.state == state]
