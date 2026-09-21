from __future__ import annotations

from copy import deepcopy

import pytest
from botocore.exceptions import ClientError

from publishing.dynamodb_ledger import DynamoDBPublicationLedger
from publishing.ledger import LedgerConflict, LedgerNotFound
from publishing.models import InstagramOptions, PublicationApproval, PublicationRequest, PublicationSchedule


class FakeDynamoTable:
    """Small fake for adapter contract tests; it intentionally does not emulate AWS broadly."""

    def __init__(self) -> None:
        self.item = None
        self.last_call = None
        self.fail_condition = False

    def put_item(self, **kwargs):
        self.last_call = ("put_item", kwargs)
        if self.fail_condition:
            raise _conditional_failure()
        self.item = deepcopy(kwargs["Item"])
        return {}

    def get_item(self, **kwargs):
        self.last_call = ("get_item", kwargs)
        return {"Item": deepcopy(self.item)} if self.item else {}

    def update_item(self, **kwargs):
        self.last_call = ("update_item", kwargs)
        if self.fail_condition:
            raise _conditional_failure()
        if self.item is None:
            raise _conditional_failure()

        values = kwargs["ExpressionAttributeValues"]
        expression = kwargs["UpdateExpression"]
        if ":new_version" in values:
            self.item["publication_version"] = values[":new_version"]
            self.item["state"] = "draft"
            self.item["request"] = deepcopy(values[":request"])
            self.item.pop("approval", None)
            self.item.pop("schedule", None)
        elif ":approval" in values:
            self.item["approval"] = deepcopy(values[":approval"])
            self.item["state"] = "approved"
        elif ":target_state" in values:
            self.item["schedule"] = deepcopy(values[":schedule"])
            self.item["state"] = values[":target_state"]
            self.item["scheduled_at_utc"] = values[":schedule"]["scheduled_at_utc"]
        elif ":cancelled" in values:
            self.item["state"] = "cancelled"
            if ":schedule" in values:
                self.item["schedule"] = deepcopy(values[":schedule"])
        return {"Attributes": deepcopy(self.item)}

    def query(self, **kwargs):
        self.last_call = ("query", kwargs)
        if self.item and self.item.get("state") == kwargs["ExpressionAttributeValues"][":state"]:
            return {"Items": [deepcopy(self.item)]}
        return {"Items": []}


def _conditional_failure() -> ClientError:
    return ClientError(
        {"Error": {"Code": "ConditionalCheckFailedException", "Message": "condition failed"}},
        "UpdateItem",
    )


def _request(version: int = 1) -> PublicationRequest:
    return PublicationRequest(
        publication_id="pub-1",
        publication_version=version,
        platform="instagram",
        account_ref="eirepolitic_instagram",
        project_id="demo",
        period="2026-09",
        asset_package_id="pkg-1",
        caption="Demo #EirePolitic",
        hashtags=("#EirePolitic",),
        instagram=InstagramOptions(post_type="image"),
    )


def _approval(version: int = 1) -> PublicationApproval:
    return PublicationApproval(
        approval_id="approval-1",
        publication_id="pub-1",
        publication_version=version,
        request_fingerprint="sha256:abc",
        approved_by="human-1",
        approved_at_utc="2026-09-05T20:00:00Z",
    )


def _schedule(version: int = 1) -> PublicationSchedule:
    return PublicationSchedule(
        schedule_id="schedule-1",
        publication_id="pub-1",
        publication_version=version,
        scheduled_local="2026-09-08T19:30:00",
        timezone="Europe/Dublin",
        scheduled_at_utc="2026-09-08T18:30:00Z",
        status="scheduled",
    )


def test_create_and_read_roundtrip() -> None:
    table = FakeDynamoTable()
    ledger = DynamoDBPublicationLedger(table)
    created = ledger.create_publication(_request())
    loaded = ledger.get_publication("pub-1")
    assert created == loaded
    assert table.last_call[1]["ConsistentRead"] is True


def test_create_uses_no_overwrite_condition() -> None:
    table = FakeDynamoTable()
    ledger = DynamoDBPublicationLedger(table)
    ledger.create_publication(_request())
    assert table.last_call[1]["ConditionExpression"] == "attribute_not_exists(pk)"

    table.fail_condition = True
    with pytest.raises(LedgerConflict, match="already exists"):
        ledger.create_publication(_request())


def test_missing_publication_raises_not_found() -> None:
    with pytest.raises(LedgerNotFound):
        DynamoDBPublicationLedger(FakeDynamoTable()).get_publication("missing")


def test_new_version_conditionally_replaces_content_and_clears_approval() -> None:
    table = FakeDynamoTable()
    ledger = DynamoDBPublicationLedger(table)
    ledger.create_publication(_request())
    ledger.record_approval(_approval())
    updated = ledger.put_version(_request(version=2))
    assert updated.request.publication_version == 2
    assert updated.state == "draft"
    assert updated.approval is None
    assert ":previous_version" in table.last_call[1]["ExpressionAttributeValues"]


def test_approval_and_schedule_are_conditional() -> None:
    table = FakeDynamoTable()
    ledger = DynamoDBPublicationLedger(table)
    ledger.create_publication(_request())
    approved = ledger.record_approval(_approval())
    assert approved.state == "approved"
    assert "#state = :draft" in table.last_call[1]["ConditionExpression"]

    scheduled = ledger.put_schedule(_schedule())
    assert scheduled.state == "scheduled"
    assert scheduled.schedule.scheduled_at_utc == "2026-09-08T18:30:00Z"
    assert "attribute_exists(approval)" in table.last_call[1]["ConditionExpression"]


def test_conditional_failure_becomes_ledger_conflict() -> None:
    table = FakeDynamoTable()
    ledger = DynamoDBPublicationLedger(table)
    ledger.create_publication(_request())
    table.fail_condition = True
    with pytest.raises(LedgerConflict):
        ledger.record_approval(_approval())


def test_cancel_uses_current_version_and_state_guard() -> None:
    table = FakeDynamoTable()
    ledger = DynamoDBPublicationLedger(table)
    ledger.create_publication(_request())
    ledger.record_approval(_approval())
    ledger.put_schedule(_schedule())
    cancelled = ledger.cancel("pub-1")
    assert cancelled.state == "cancelled"
    assert cancelled.schedule.status == "cancelled"
    assert "publication_version = :version" in table.last_call[1]["ConditionExpression"]


def test_state_query_uses_gsi() -> None:
    table = FakeDynamoTable()
    ledger = DynamoDBPublicationLedger(table)
    ledger.create_publication(_request())
    ledger.record_approval(_approval())
    ledger.put_schedule(_schedule())
    rows = ledger.list_by_state("scheduled")
    assert len(rows) == 1
    assert table.last_call[1]["IndexName"] == "state-scheduled_at-index"
