from __future__ import annotations

from dataclasses import replace
from typing import Callable

from publishing.execution import (
    ExecutionAttempt,
    begin_operation,
    record_operation_success,
    record_operation_uncertain,
    record_published,
    should_execute_operation,
)
from publishing.meta_provider import ApprovedAssetUrlProvider, InstagramMetaProvider
from publishing.models import AssetPackage, MediaAsset, PublicationRequest
from publishing.validation import validate_request_against_package


class PublishingError(RuntimeError):
    pass


class PublishingNeedsAttention(PublishingError):
    pass


class PublishingOutcomeUncertain(PublishingError):
    pass


PersistAttempt = Callable[[ExecutionAttempt], None]


class InstagramPublisher:
    """Deterministic orchestration over an approved request and Meta provider.

    Persistence is injected so every provider ID can be durably recorded before
    the next external side effect. The worker contains no LLM/editorial logic.
    """

    def __init__(
        self,
        provider: InstagramMetaProvider,
        asset_urls: ApprovedAssetUrlProvider,
        persist_attempt: PersistAttempt,
    ) -> None:
        self.provider = provider
        self.asset_urls = asset_urls
        self.persist_attempt = persist_attempt

    def publish(
        self,
        request: PublicationRequest,
        package: AssetPackage,
        attempt: ExecutionAttempt,
    ) -> ExecutionAttempt:
        validate_request_against_package(request, package)
        if attempt.publication_id != request.publication_id or attempt.publication_version != request.publication_version:
            raise PublishingError("execution attempt does not match publication identity/version")
        if attempt.state == "published":
            return attempt

        child_ids: list[str] = []
        for asset in package.media:
            operation = f"create_child:{asset.asset_id}"
            existing = self._provider_id(attempt, operation)
            if existing:
                child_ids.append(existing)
                continue
            if not should_execute_operation(attempt, operation):
                raise PublishingNeedsAttention(f"operation {operation} requires reconciliation")

            attempt = begin_operation(attempt, operation)
            self.persist_attempt(attempt)
            container_id = self.provider.create_image_container(
                image_url=self.asset_urls.url_for(bucket=asset.bucket, key=asset.key),
                caption=request.caption if len(package.media) == 1 else None,
                alt_text=asset.alt_text or None,
                user_tags=self._tags_for_asset(request, asset),
                collaborators=request.instagram.collaborators if len(package.media) == 1 else (),
                location_id=request.instagram.location_id if len(package.media) == 1 else None,
                is_carousel_item=len(package.media) > 1,
            )
            attempt = record_operation_success(attempt, operation, provider_id=container_id)
            self.persist_attempt(attempt)
            child_ids.append(container_id)

        if len(child_ids) == 1:
            parent_id = child_ids[0]
        else:
            parent_id, attempt = self._ensure_carousel_parent(request, tuple(child_ids), attempt)

        status = self.provider.get_container_status(parent_id)
        if status.status_code == "PUBLISHED":
            media_id = status.media_id
            if not media_id:
                attempt = record_operation_uncertain(attempt, "publish_parent", provider_id=parent_id)
                self.persist_attempt(attempt)
                raise PublishingOutcomeUncertain("container is PUBLISHED but media ID requires reconciliation")
            attempt = record_published(attempt, media_id)
            self.persist_attempt(attempt)
            return self._first_comment(request, attempt)
        if status.status_code != "FINISHED":
            raise PublishingNeedsAttention(f"container {parent_id} is not publishable: {status.status_code}")

        if should_execute_operation(attempt, "publish_parent"):
            attempt = begin_operation(attempt, "publish_parent")
            self.persist_attempt(attempt)
            try:
                media_id = self.provider.publish_container(parent_id)
            except Exception:
                attempt = record_operation_uncertain(attempt, "publish_parent", provider_id=parent_id)
                self.persist_attempt(attempt)
                raise PublishingOutcomeUncertain("publish outcome is unknown; reconcile the same container before retrying")
            if not media_id:
                attempt = record_operation_uncertain(attempt, "publish_parent", provider_id=parent_id)
                self.persist_attempt(attempt)
                raise PublishingOutcomeUncertain("Meta did not return a media ID; reconcile before retrying")
            attempt = record_operation_success(attempt, "publish_parent", provider_id=parent_id)
            attempt = record_published(attempt, media_id)
            self.persist_attempt(attempt)
        elif attempt.state != "published":
            raise PublishingNeedsAttention("publish operation requires reconciliation")

        return self._first_comment(request, attempt)

    def _ensure_carousel_parent(
        self,
        request: PublicationRequest,
        child_ids: tuple[str, ...],
        attempt: ExecutionAttempt,
    ) -> tuple[str, ExecutionAttempt]:
        existing = self._provider_id(attempt, "create_parent")
        if existing:
            return existing, attempt
        if not should_execute_operation(attempt, "create_parent"):
            raise PublishingNeedsAttention("carousel parent requires reconciliation")
        attempt = begin_operation(attempt, "create_parent")
        self.persist_attempt(attempt)
        parent_id = self.provider.create_carousel_container(
            child_container_ids=child_ids,
            caption=request.caption,
            collaborators=request.instagram.collaborators,
            location_id=request.instagram.location_id,
        )
        attempt = record_operation_success(attempt, "create_parent", provider_id=parent_id)
        self.persist_attempt(attempt)
        return parent_id, attempt

    def _first_comment(self, request: PublicationRequest, attempt: ExecutionAttempt) -> ExecutionAttempt:
        message = request.instagram.first_comment
        if not message or not attempt.published_media_id:
            return attempt
        if not should_execute_operation(attempt, "first_comment"):
            return attempt
        attempt = begin_operation(attempt, "first_comment")
        self.persist_attempt(attempt)
        try:
            comment_id = self.provider.create_comment(attempt.published_media_id, message)
        except Exception:
            attempt = record_operation_uncertain(attempt, "first_comment")
            # Media remains permanently published; secondary-action uncertainty must never republish it.
            attempt = replace(attempt, state="published")
            self.persist_attempt(attempt)
            return attempt
        attempt = record_operation_success(attempt, "first_comment", provider_id=comment_id)
        self.persist_attempt(attempt)
        return attempt

    @staticmethod
    def _provider_id(attempt: ExecutionAttempt, operation_key: str) -> str | None:
        operation = next((op for op in attempt.operations if op.operation_key == operation_key), None)
        return operation.provider_id if operation and operation.state == "succeeded" else None

    @staticmethod
    def _tags_for_asset(request: PublicationRequest, asset: MediaAsset) -> tuple[dict[str, object], ...]:
        tags: list[dict[str, object]] = []
        for tag in request.instagram.media_tags:
            if tag.media_ordinal != asset.ordinal:
                continue
            value: dict[str, object] = {"username": tag.username}
            if tag.x is not None and tag.y is not None:
                value.update({"x": tag.x, "y": tag.y})
            tags.append(value)
        return tuple(tags)
