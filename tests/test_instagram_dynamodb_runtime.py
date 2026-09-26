from __future__ import annotations

import pytest
from botocore.exceptions import ClientError

from publishing.dynamodb_runtime import DynamoDBAssetPackageStore, DynamoDBExecutionStore
from publishing.execution import ExecutionConflict, begin_operation, record_operation_success
from publishing.models import AssetPackage, MediaAsset


class FakeTable:
    def __init__(self) -> None:
        self.items: dict[tuple[str, str], dict] = {}

    @staticmethod
    def _identity(key: dict[str, str]) -> tuple[str, str]:
        return key["pk"], key["sk"]

    def put_item(self, *, Item, ConditionExpression=None):
        identity = self._identity(Item)
        if ConditionExpression == "attribute_not_exists(pk)" and identity in self.items:
            raise ClientError({"Error": {"Code": "ConditionalCheckFailedException"}}, "PutItem")
        self.items[identity] = Item.copy()
        return {}

    def get_item(self, *, Key, ConsistentRead=False):
        item = self.items.get(self._identity(Key))
        return {"Item": item.copy()} if item else {}

    def update_item(
        self,
        *,
        Key,
        UpdateExpression,
        ConditionExpression,
        ExpressionAttributeValues,
    ):
        identity = self._identity(Key)
        item = self.items[identity]
        if ConditionExpression == "revision = :expected_revision":
            if item["revision"] != ExpressionAttributeValues[":expected_revision"]:
                raise ClientError({"Error": {"Code": "ConditionalCheckFailedException"}}, "UpdateItem")
        item = item.copy()
        item["attempt"] = ExpressionAttributeValues[":attempt"]
        item["revision"] = ExpressionAttributeValues[":next_revision"]
        self.items[identity] = item
        return {}


def _package() -> AssetPackage:
    return AssetPackage(
        asset_package_id="pkg-1",
        project_id="demo",
        period="2026-09",
        media=(
            MediaAsset(
                asset_id="slide_01",
                ordinal=1,
                bucket="approved-bucket",
                key="instagram/approved/demo/2026-09/pkg-1/media/01-deadbeef.jpg",
                sha256="deadbeef",
                mime_type="image/jpeg",
                width=1080,
                height=1350,
                size_bytes=1234,
                alt_text="Demo",
            ),
        ),
        publication_ready=True,
        review_status="approved",
    )


def test_asset_package_is_immutable_and_round_trips() -> None:
    table = FakeTable()
    store = DynamoDBAssetPackageStore(table)
    package = _package()

    assert store.put(package) == package
    assert store.get("pkg-1") == package
    assert store.put(package) == package

    changed = AssetPackage(**{**package.__dict__, "review_status": "changed"})
    with pytest.raises(Exception, match="different content"):
        store.put(changed)


def test_execution_session_recovers_provider_ids_across_replay() -> None:
    table = FakeTable()
    store = DynamoDBExecutionStore(table)
    session = store.open(publication_id="pub-1", publication_version=3, attempt_id="attempt-a")

    attempt = begin_operation(session.attempt, "create_child:slide_01")
    session.persist(attempt)
    attempt = record_operation_success(attempt, "create_child:slide_01", provider_id="container-123")
    session.persist(attempt)

    replay = store.open(publication_id="pub-1", publication_version=3, attempt_id="attempt-b")
    assert replay.attempt.attempt_id == "attempt-a"
    assert replay.revision == 2
    assert replay.attempt.operations[0].provider_id == "container-123"


def test_execution_revision_blocks_stale_worker_write() -> None:
    table = FakeTable()
    store = DynamoDBExecutionStore(table)
    first = store.open(publication_id="pub-1", publication_version=1, attempt_id="attempt-a")
    second = store.open(publication_id="pub-1", publication_version=1, attempt_id="attempt-b")

    first.persist(begin_operation(first.attempt, "publish_parent"))
    with pytest.raises(ExecutionConflict, match="concurrently"):
        second.persist(begin_operation(second.attempt, "create_parent"))
