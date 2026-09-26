from __future__ import annotations

from dataclasses import asdict
from typing import Any

from botocore.exceptions import ClientError

from publishing.execution import ExecutionAttempt, ExecutionConflict, OperationRecord
from publishing.ledger import LedgerConflict, LedgerNotFound
from publishing.models import AssetPackage, MediaAsset


class DynamoDBAssetPackageStore:
    def __init__(self, table: Any) -> None:
        self.table = table

    @staticmethod
    def _key(asset_package_id: str) -> dict[str, str]:
        return {"pk": f"ASSET#{asset_package_id}", "sk": "PACKAGE"}

    def put(self, package: AssetPackage) -> AssetPackage:
        item = {
            **self._key(package.asset_package_id),
            "entity_type": "asset_package",
            "asset_package_id": package.asset_package_id,
            "project_id": package.project_id,
            "period": package.period,
            "package": asdict(package),
        }
        try:
            self.table.put_item(Item=item, ConditionExpression="attribute_not_exists(pk)")
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") == "ConditionalCheckFailedException":
                existing = self.get(package.asset_package_id)
                if existing == package:
                    return existing
                raise LedgerConflict(f"asset package already exists with different content: {package.asset_package_id}") from exc
            raise
        return package

    def get(self, asset_package_id: str) -> AssetPackage:
        response = self.table.get_item(Key=self._key(asset_package_id), ConsistentRead=True)
        item = response.get("Item")
        if not item:
            raise LedgerNotFound(f"asset package not found: {asset_package_id}")
        data = item["package"]
        media = tuple(MediaAsset(**asset) for asset in data.get("media", []))
        return AssetPackage(
            asset_package_id=data["asset_package_id"],
            project_id=data["project_id"],
            period=data["period"],
            media=media,
            publication_ready=bool(data["publication_ready"]),
            review_status=data["review_status"],
            safety_notes=tuple(data.get("safety_notes", [])),
        )


class DynamoDBExecutionSession:
    def __init__(self, table: Any, attempt: ExecutionAttempt, revision: int) -> None:
        self.table = table
        self.attempt = attempt
        self.revision = revision

    @staticmethod
    def _key(publication_id: str, publication_version: int) -> dict[str, str]:
        return {"pk": f"PUB#{publication_id}", "sk": f"EXEC#v{publication_version:08d}"}

    @staticmethod
    def _serialize(attempt: ExecutionAttempt) -> dict[str, Any]:
        return asdict(attempt)

    @staticmethod
    def _deserialize(data: dict[str, Any]) -> ExecutionAttempt:
        return ExecutionAttempt(
            publication_id=data["publication_id"],
            publication_version=int(data["publication_version"]),
            attempt_id=data["attempt_id"],
            state=data.get("state", "pending"),
            lease_owner=data.get("lease_owner"),
            lease_expires_at_utc=data.get("lease_expires_at_utc"),
            published_media_id=data.get("published_media_id"),
            operations=tuple(OperationRecord(**operation) for operation in data.get("operations", [])),
        )

    @classmethod
    def open(
        cls,
        table: Any,
        *,
        publication_id: str,
        publication_version: int,
        attempt_id: str,
    ) -> "DynamoDBExecutionSession":
        key = cls._key(publication_id, publication_version)
        attempt = ExecutionAttempt(publication_id, publication_version, attempt_id)
        item = {
            **key,
            "entity_type": "execution_attempt",
            "publication_id": publication_id,
            "publication_version": publication_version,
            "revision": 0,
            "attempt": cls._serialize(attempt),
        }
        try:
            table.put_item(Item=item, ConditionExpression="attribute_not_exists(pk)")
            return cls(table, attempt, 0)
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") != "ConditionalCheckFailedException":
                raise

        response = table.get_item(Key=key, ConsistentRead=True)
        existing = response.get("Item")
        if not existing:
            raise ExecutionConflict("execution attempt disappeared during concurrent creation")
        return cls(table, cls._deserialize(existing["attempt"]), int(existing.get("revision", 0)))

    def persist(self, attempt: ExecutionAttempt) -> None:
        if (
            attempt.publication_id != self.attempt.publication_id
            or attempt.publication_version != self.attempt.publication_version
        ):
            raise ExecutionConflict("cannot persist execution state for a different publication identity/version")

        expected = self.revision
        next_revision = expected + 1
        try:
            self.table.update_item(
                Key=self._key(attempt.publication_id, attempt.publication_version),
                UpdateExpression="SET attempt = :attempt, revision = :next_revision",
                ConditionExpression="revision = :expected_revision",
                ExpressionAttributeValues={
                    ":attempt": self._serialize(attempt),
                    ":next_revision": next_revision,
                    ":expected_revision": expected,
                },
            )
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") == "ConditionalCheckFailedException":
                raise ExecutionConflict("execution state changed concurrently; reload before retrying") from exc
            raise
        self.attempt = attempt
        self.revision = next_revision


class DynamoDBExecutionStore:
    def __init__(self, table: Any) -> None:
        self.table = table

    def open(
        self,
        *,
        publication_id: str,
        publication_version: int,
        attempt_id: str,
    ) -> DynamoDBExecutionSession:
        return DynamoDBExecutionSession.open(
            self.table,
            publication_id=publication_id,
            publication_version=publication_version,
            attempt_id=attempt_id,
        )
