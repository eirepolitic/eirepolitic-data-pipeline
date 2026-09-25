from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Literal


ExecutionState = Literal[
    "pending",
    "publishing",
    "published",
    "outcome_uncertain",
    "needs_attention",
    "failed",
]

OperationState = Literal["pending", "started", "succeeded", "uncertain", "failed"]


class ExecutionConflict(RuntimeError):
    pass


@dataclass(frozen=True)
class OperationRecord:
    operation_key: str
    state: OperationState = "pending"
    provider_id: str | None = None
    error_code: str | None = None


@dataclass(frozen=True)
class ExecutionAttempt:
    publication_id: str
    publication_version: int
    attempt_id: str
    state: ExecutionState = "pending"
    lease_owner: str | None = None
    lease_expires_at_utc: str | None = None
    published_media_id: str | None = None
    operations: tuple[OperationRecord, ...] = ()

    @property
    def idempotency_key(self) -> str:
        return f"{self.publication_id}:v{self.publication_version}"


def _utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def acquire_execution_lease(
    attempt: ExecutionAttempt,
    *,
    owner: str,
    now_utc: str,
    lease_seconds: int = 300,
) -> ExecutionAttempt:
    if attempt.state == "published" or attempt.published_media_id:
        raise ExecutionConflict("publication already has a permanent published result")
    now = _utc(now_utc)
    if attempt.lease_owner and attempt.lease_expires_at_utc and _utc(attempt.lease_expires_at_utc) > now:
        if attempt.lease_owner != owner:
            raise ExecutionConflict("another worker holds the execution lease")
    expiry = now + timedelta(seconds=lease_seconds)
    return replace(
        attempt,
        state="publishing",
        lease_owner=owner,
        lease_expires_at_utc=expiry.isoformat(timespec="seconds").replace("+00:00", "Z"),
    )


def begin_operation(attempt: ExecutionAttempt, operation_key: str) -> ExecutionAttempt:
    existing = next((op for op in attempt.operations if op.operation_key == operation_key), None)
    if existing and existing.state == "succeeded":
        return attempt
    replacement = OperationRecord(operation_key=operation_key, state="started", provider_id=existing.provider_id if existing else None)
    return _replace_operation(attempt, replacement)


def record_operation_success(attempt: ExecutionAttempt, operation_key: str, provider_id: str | None = None) -> ExecutionAttempt:
    return _replace_operation(attempt, OperationRecord(operation_key=operation_key, state="succeeded", provider_id=provider_id))


def record_operation_uncertain(attempt: ExecutionAttempt, operation_key: str, provider_id: str | None = None) -> ExecutionAttempt:
    return replace(
        _replace_operation(attempt, OperationRecord(operation_key=operation_key, state="uncertain", provider_id=provider_id)),
        state="outcome_uncertain",
    )


def record_published(attempt: ExecutionAttempt, media_id: str) -> ExecutionAttempt:
    if not media_id:
        raise ExecutionConflict("published media ID is required")
    return replace(
        attempt,
        state="published",
        published_media_id=media_id,
        lease_owner=None,
        lease_expires_at_utc=None,
    )


def should_execute_operation(attempt: ExecutionAttempt, operation_key: str) -> bool:
    if attempt.state == "published":
        return False
    existing = next((op for op in attempt.operations if op.operation_key == operation_key), None)
    return existing is None or existing.state not in {"succeeded", "uncertain"}


def _replace_operation(attempt: ExecutionAttempt, operation: OperationRecord) -> ExecutionAttempt:
    operations = [op for op in attempt.operations if op.operation_key != operation.operation_key]
    operations.append(operation)
    operations.sort(key=lambda item: item.operation_key)
    return replace(attempt, operations=tuple(operations))
