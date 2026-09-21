from __future__ import annotations

import json

import pytest
from botocore.exceptions import ClientError

from publishing.execution import (
    ExecutionAttempt,
    ExecutionConflict,
    acquire_execution_lease,
    begin_operation,
    record_operation_success,
    record_operation_uncertain,
    record_published,
    should_execute_operation,
)
from publishing.models import PublicationSchedule
from publishing.scheduler import EventBridgePublicationScheduler, SchedulerTarget


class FakeSchedulerClient:
    def __init__(self) -> None:
        self.schedules = {}

    def create_schedule(self, **kwargs):
        key = (kwargs["GroupName"], kwargs["Name"])
        if key in self.schedules:
            raise ClientError({"Error": {"Code": "ConflictException", "Message": "exists"}}, "CreateSchedule")
        self.schedules[key] = kwargs
        return {"ScheduleArn": "arn:aws:scheduler:region:acct:schedule/test"}

    def get_schedule(self, **kwargs):
        key = (kwargs["GroupName"], kwargs["Name"])
        if key not in self.schedules:
            raise ClientError({"Error": {"Code": "ResourceNotFoundException", "Message": "missing"}}, "GetSchedule")
        return self.schedules[key]

    def delete_schedule(self, **kwargs):
        key = (kwargs["GroupName"], kwargs["Name"])
        if key not in self.schedules:
            raise ClientError({"Error": {"Code": "ResourceNotFoundException", "Message": "missing"}}, "DeleteSchedule")
        del self.schedules[key]
        return {}


def _schedule() -> PublicationSchedule:
    return PublicationSchedule(
        schedule_id="schedule-1",
        publication_id="pub/one",
        publication_version=3,
        scheduled_local="2026-09-08T19:30:00",
        timezone="Europe/Dublin",
        scheduled_at_utc="2026-09-08T18:30:00Z",
        status="scheduled",
    )


def _scheduler(client=None):
    return EventBridgePublicationScheduler(
        client or FakeSchedulerClient(),
        SchedulerTarget(
            lambda_arn="arn:aws:lambda:eu-west-1:123:function:publisher",
            role_arn="arn:aws:iam::123:role/scheduler-invoke-publisher",
            dlq_arn="arn:aws:sqs:eu-west-1:123:publishing-dlq",
        ),
    )


def test_scheduler_payload_contains_identity_only() -> None:
    payload = json.loads(_scheduler().payload(_schedule()))
    assert payload == {"publication_id": "pub/one", "expected_version": 3}
    assert "caption" not in payload
    assert "token" not in payload


def test_scheduler_creates_exact_utc_job_and_verifies_it() -> None:
    client = FakeSchedulerClient()
    scheduler = _scheduler(client)
    name = scheduler.create(_schedule())
    assert name == "instagram-pub-one-v3"
    assert scheduler.verify(_schedule()) is True
    stored = client.schedules[("eirepolitic-instagram", name)]
    assert stored["ScheduleExpression"] == "at(2026-09-08T18:30:00)"
    assert stored["FlexibleTimeWindow"] == {"Mode": "OFF"}
    assert stored["ActionAfterCompletion"] == "DELETE"
    assert stored["Target"]["RetryPolicy"]["MaximumRetryAttempts"] == 3
    assert stored["Target"]["DeadLetterConfig"]["Arn"].endswith("publishing-dlq")


def test_cancel_is_idempotent_when_schedule_is_missing() -> None:
    scheduler = _scheduler()
    scheduler.cancel(_schedule())
    scheduler.create(_schedule())
    scheduler.cancel(_schedule())
    assert scheduler.verify(_schedule()) is False


def test_execution_lease_blocks_concurrent_worker_but_allows_expired_takeover() -> None:
    attempt = ExecutionAttempt("pub-1", 1, "attempt-1")
    leased = acquire_execution_lease(attempt, owner="worker-a", now_utc="2026-09-05T20:00:00Z", lease_seconds=60)
    with pytest.raises(ExecutionConflict, match="another worker"):
        acquire_execution_lease(leased, owner="worker-b", now_utc="2026-09-05T20:00:30Z", lease_seconds=60)
    takeover = acquire_execution_lease(leased, owner="worker-b", now_utc="2026-09-05T20:01:01Z", lease_seconds=60)
    assert takeover.lease_owner == "worker-b"


def test_succeeded_operation_is_not_reexecuted_on_replay() -> None:
    attempt = ExecutionAttempt("pub-1", 1, "attempt-1")
    attempt = begin_operation(attempt, "create_child:slide-1")
    attempt = record_operation_success(attempt, "create_child:slide-1", provider_id="container-123")
    assert should_execute_operation(attempt, "create_child:slide-1") is False
    replay = begin_operation(attempt, "create_child:slide-1")
    assert replay == attempt


def test_uncertain_publish_requires_reconciliation_not_duplicate_publish() -> None:
    attempt = ExecutionAttempt("pub-1", 1, "attempt-1")
    attempt = record_operation_uncertain(attempt, "publish_parent", provider_id="container-parent")
    assert attempt.state == "outcome_uncertain"
    assert should_execute_operation(attempt, "publish_parent") is False


def test_published_result_is_permanent_do_not_republish_guard() -> None:
    attempt = ExecutionAttempt("pub-1", 1, "attempt-1")
    published = record_published(attempt, "ig-media-123")
    assert published.state == "published"
    assert published.published_media_id == "ig-media-123"
    assert should_execute_operation(published, "publish_parent") is False
    with pytest.raises(ExecutionConflict, match="already"):
        acquire_execution_lease(published, owner="worker-b", now_utc="2026-09-05T21:00:00Z")
