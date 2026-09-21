from __future__ import annotations

import pytest

from publishing.execution import ExecutionAttempt, OperationRecord
from publishing.meta_provider import MetaContainerStatus
from publishing.models import AssetPackage, InstagramOptions, MediaAsset, PublicationRequest
from publishing.publisher import InstagramPublisher, PublishingNeedsAttention, PublishingOutcomeUncertain


class FakeUrls:
    def url_for(self, *, bucket: str, key: str) -> str:
        return f"https://assets.example.invalid/{bucket}/{key}"


class FakeMeta:
    def __init__(self) -> None:
        self.image_calls = []
        self.carousel_calls = []
        self.publish_calls = []
        self.comment_calls = []
        self.status = "FINISHED"
        self.status_media_id = None
        self.fail_publish = False
        self.fail_comment = False

    def create_image_container(self, **kwargs):
        self.image_calls.append(kwargs)
        return f"child-{len(self.image_calls)}"

    def create_carousel_container(self, **kwargs):
        self.carousel_calls.append(kwargs)
        return "parent-1"

    def get_container_status(self, container_id):
        return MetaContainerStatus(container_id, self.status, self.status_media_id)

    def publish_container(self, container_id):
        self.publish_calls.append(container_id)
        if self.fail_publish:
            raise TimeoutError("response lost")
        return "ig-media-1"

    def create_comment(self, media_id, message):
        self.comment_calls.append((media_id, message))
        if self.fail_comment:
            raise TimeoutError("response lost")
        return "comment-1"


def _asset(n):
    return MediaAsset(f"slide-{n}", n, "bucket", f"key-{n}.jpg", f"hash-{n}", "image/jpeg", 1080, 1350, 1000, f"Alt {n}")


def _package(count=1):
    return AssetPackage("pkg", "demo", "2026-09", tuple(_asset(n) for n in range(1, count + 1)), True, "approved")


def _request(count=1, first_comment=None):
    return PublicationRequest(
        "pub", 1, "instagram", "account", "demo", "2026-09", "pkg",
        "Exact caption #EirePolitic", ("#EirePolitic",), (),
        InstagramOptions("image" if count == 1 else "carousel", first_comment=first_comment),
    )


def _publisher(meta, persisted):
    return InstagramPublisher(meta, FakeUrls(), persisted.append)


def test_single_image_publishes_exact_caption_and_records_result():
    meta = FakeMeta()
    persisted = []
    result = _publisher(meta, persisted).publish(_request(), _package(), ExecutionAttempt("pub", 1, "attempt"))
    assert result.state == "published"
    assert result.published_media_id == "ig-media-1"
    assert meta.image_calls[0]["caption"] == "Exact caption #EirePolitic"
    assert meta.publish_calls == ["child-1"]
    assert persisted[-1].published_media_id == "ig-media-1"


def test_carousel_creates_children_then_parent_and_caption_only_on_parent():
    meta = FakeMeta()
    result = _publisher(meta, []).publish(_request(2), _package(2), ExecutionAttempt("pub", 1, "attempt"))
    assert result.state == "published"
    assert len(meta.image_calls) == 2
    assert all(call["caption"] is None for call in meta.image_calls)
    assert meta.carousel_calls[0]["child_container_ids"] == ("child-1", "child-2")
    assert meta.carousel_calls[0]["caption"] == "Exact caption #EirePolitic"
    assert meta.publish_calls == ["parent-1"]


def test_replay_reuses_persisted_child_container_instead_of_creating_duplicate():
    meta = FakeMeta()
    attempt = ExecutionAttempt(
        "pub", 1, "attempt",
        operations=(OperationRecord("create_child:slide-1", "succeeded", "child-existing"),),
    )
    result = _publisher(meta, []).publish(_request(), _package(), attempt)
    assert result.state == "published"
    assert meta.image_calls == []
    assert meta.publish_calls == ["child-existing"]


def test_publish_timeout_becomes_uncertain_and_is_not_retried_blindly():
    meta = FakeMeta()
    meta.fail_publish = True
    persisted = []
    with pytest.raises(PublishingOutcomeUncertain):
        _publisher(meta, persisted).publish(_request(), _package(), ExecutionAttempt("pub", 1, "attempt"))
    uncertain = persisted[-1]
    assert uncertain.state == "outcome_uncertain"
    assert len(meta.publish_calls) == 1

    meta.fail_publish = False
    with pytest.raises(PublishingNeedsAttention, match="reconciliation"):
        _publisher(meta, []).publish(_request(), _package(), uncertain)
    assert len(meta.publish_calls) == 1


def test_provider_reports_already_published_is_reconciled_without_publish_call():
    meta = FakeMeta()
    meta.status = "PUBLISHED"
    meta.status_media_id = "ig-existing"
    result = _publisher(meta, []).publish(_request(), _package(), ExecutionAttempt("pub", 1, "attempt"))
    assert result.state == "published"
    assert result.published_media_id == "ig-existing"
    assert meta.publish_calls == []


def test_published_without_media_id_is_uncertain_not_republished():
    meta = FakeMeta()
    meta.status = "PUBLISHED"
    persisted = []
    with pytest.raises(PublishingOutcomeUncertain):
        _publisher(meta, persisted).publish(_request(), _package(), ExecutionAttempt("pub", 1, "attempt"))
    assert persisted[-1].state == "outcome_uncertain"
    assert meta.publish_calls == []


def test_first_comment_failure_never_republishes_media():
    meta = FakeMeta()
    meta.fail_comment = True
    persisted = []
    result = _publisher(meta, persisted).publish(_request(first_comment="First comment"), _package(), ExecutionAttempt("pub", 1, "attempt"))
    assert result.state == "published"
    assert result.published_media_id == "ig-media-1"
    assert len(meta.publish_calls) == 1
    assert len(meta.comment_calls) == 1

    replay = _publisher(meta, []).publish(_request(first_comment="First comment"), _package(), result)
    assert replay == result
    assert len(meta.publish_calls) == 1
