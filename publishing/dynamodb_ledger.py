from __future__ import annotations

from dataclasses import asdict
from typing import Any

from botocore.exceptions import ClientError

from publishing.ledger import LedgerConflict, LedgerNotFound, PublicationRecord
from publishing.models import (
    InstagramOptions,
    InstagramUserTag,
    PublicationApproval,
    PublicationRequest,
    PublicationSchedule,
)


class DynamoDBPublicationLedger:
    """DynamoDB implementation of the publication-ledger behavioral contract.

    One item represents the current control-plane snapshot for a publication.
    Historical versions/execution attempts will use separate item types when the
    runtime layer is added. All mutations use conditional writes so stale callers
    cannot silently overwrite a newer publication state.
    """

    def __init__(self, table: Any) -> None:
        self.table = table

    @staticmethod
    def _key(publication_id: str) -> dict[str, str]:
        return {"pk": f"PUB#{publication_id}", "sk": "CONTROL"}

    @staticmethod
    def _serialize_request(request: PublicationRequest) -> dict[str, Any]:
        return asdict(request)

    @staticmethod
    def _deserialize_request(data: dict[str, Any]) -> PublicationRequest:
        instagram = data.get("instagram") or {}
        tags = tuple(InstagramUserTag(**tag) for tag in instagram.get("media_tags", []))
        options = InstagramOptions(
            post_type=instagram["post_type"],
            media_tags=tags,
            collaborators=tuple(instagram.get("collaborators", [])),
            location_id=instagram.get("location_id"),
            first_comment=instagram.get("first_comment"),
        )
        return PublicationRequest(
            publication_id=data["publication_id"],
            publication_version=int(data["publication_version"]),
            platform=data["platform"],
            account_ref=data["account_ref"],
            project_id=data["project_id"],
            period=data["period"],
            asset_package_id=data["asset_package_id"],
            caption=data["caption"],
            hashtags=tuple(data.get("hashtags", [])),
            caption_mentions=tuple(data.get("caption_mentions", [])),
            instagram=options,
        )

    @staticmethod
    def _deserialize_approval(data: dict[str, Any] | None) -> PublicationApproval | None:
        return PublicationApproval(**data) if data else None

    @staticmethod
    def _deserialize_schedule(data: dict[str, Any] | None) -> PublicationSchedule | None:
        return PublicationSchedule(**data) if data else None

    @classmethod
    def _to_record(cls, item: dict[str, Any]) -> PublicationRecord:
        return PublicationRecord(
            request=cls._deserialize_request(item["request"]),
            approval=cls._deserialize_approval(item.get("approval")),
            schedule=cls._deserialize_schedule(item.get("schedule")),
            state=item["state"],
        )

    @staticmethod
    def _is_conditional_failure(exc: ClientError) -> bool:
        return exc.response.get("Error", {}).get("Code") == "ConditionalCheckFailedException"

    def create_publication(self, request: PublicationRequest) -> PublicationRecord:
        item = {
            **self._key(request.publication_id),
            "entity_type": "publication_control",
            "publication_id": request.publication_id,
            "publication_version": request.publication_version,
            "account_ref": request.account_ref,
            "state": "draft",
            "request": self._serialize_request(request),
        }
        try:
            self.table.put_item(Item=item, ConditionExpression="attribute_not_exists(pk)")
        except ClientError as exc:
            if self._is_conditional_failure(exc):
                raise LedgerConflict(f"publication already exists: {request.publication_id}") from exc
            raise
        return self._to_record(item)

    def get_publication(self, publication_id: str) -> PublicationRecord:
        response = self.table.get_item(Key=self._key(publication_id), ConsistentRead=True)
        item = response.get("Item")
        if not item:
            raise LedgerNotFound(f"publication not found: {publication_id}")
        return self._to_record(item)

    def put_version(self, request: PublicationRequest) -> PublicationRecord:
        previous_version = request.publication_version - 1
        try:
            response = self.table.update_item(
                Key=self._key(request.publication_id),
                UpdateExpression=(
                    "SET publication_version = :new_version, #state = :draft, request = :request "
                    "REMOVE approval, schedule"
                ),
                ConditionExpression="publication_version = :previous_version",
                ExpressionAttributeNames={"#state": "state"},
                ExpressionAttributeValues={
                    ":new_version": request.publication_version,
                    ":previous_version": previous_version,
                    ":draft": "draft",
                    ":request": self._serialize_request(request),
                },
                ReturnValues="ALL_NEW",
            )
        except ClientError as exc:
            if self._is_conditional_failure(exc):
                raise LedgerConflict("publication version changed or publication does not exist") from exc
            raise
        return self._to_record(response["Attributes"])

    def record_approval(self, approval: PublicationApproval) -> PublicationRecord:
        try:
            response = self.table.update_item(
                Key=self._key(approval.publication_id),
                UpdateExpression="SET approval = :approval, #state = :approved",
                ConditionExpression="publication_version = :version AND #state = :draft",
                ExpressionAttributeNames={"#state": "state"},
                ExpressionAttributeValues={
                    ":approval": asdict(approval),
                    ":approved": "approved",
                    ":draft": "draft",
                    ":version": approval.publication_version,
                },
                ReturnValues="ALL_NEW",
            )
        except ClientError as exc:
            if self._is_conditional_failure(exc):
                raise LedgerConflict("publication is not an approvable current draft") from exc
            raise
        return self._to_record(response["Attributes"])

    def put_schedule(self, schedule: PublicationSchedule) -> PublicationRecord:
        target_state = "scheduled" if schedule.status == "scheduled" else "approved"
        try:
            response = self.table.update_item(
                Key=self._key(schedule.publication_id),
                UpdateExpression="SET schedule = :schedule, #state = :target_state",
                ConditionExpression=(
                    "publication_version = :version AND attribute_exists(approval) "
                    "AND (#state = :approved OR #state = :scheduled)"
                ),
                ExpressionAttributeNames={"#state": "state"},
                ExpressionAttributeValues={
                    ":schedule": asdict(schedule),
                    ":target_state": target_state,
                    ":version": schedule.publication_version,
                    ":approved": "approved",
                    ":scheduled": "scheduled",
                },
                ReturnValues="ALL_NEW",
            )
        except ClientError as exc:
            if self._is_conditional_failure(exc):
                raise LedgerConflict("publication is not an approved schedulable version") from exc
            raise
        return self._to_record(response["Attributes"])

    def cancel(self, publication_id: str) -> PublicationRecord:
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

        expression = "SET #state = :cancelled"
        values: dict[str, Any] = {
            ":cancelled": "cancelled",
            ":expected_state": current.state,
            ":version": current.request.publication_version,
        }
        if schedule is not None:
            expression += ", schedule = :schedule"
            values[":schedule"] = asdict(schedule)

        try:
            response = self.table.update_item(
                Key=self._key(publication_id),
                UpdateExpression=expression,
                ConditionExpression="publication_version = :version AND #state = :expected_state",
                ExpressionAttributeNames={"#state": "state"},
                ExpressionAttributeValues=values,
                ReturnValues="ALL_NEW",
            )
        except ClientError as exc:
            if self._is_conditional_failure(exc):
                raise LedgerConflict("publication changed before cancellation completed") from exc
            raise
        return self._to_record(response["Attributes"])

    def list_by_state(self, state: str) -> list[PublicationRecord]:
        response = self.table.query(
            IndexName="state-scheduled_at-index",
            KeyConditionExpression="#state = :state",
            ExpressionAttributeNames={"#state": "state"},
            ExpressionAttributeValues={":state": state},
        )
        return [self._to_record(item) for item in response.get("Items", [])]
