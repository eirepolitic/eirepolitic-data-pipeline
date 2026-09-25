from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class MediaAsset:
    asset_id: str
    ordinal: int
    bucket: str
    key: str
    sha256: str
    mime_type: str
    width: int
    height: int
    size_bytes: int
    alt_text: str = ""


@dataclass(frozen=True)
class AssetPackage:
    asset_package_id: str
    project_id: str
    period: str
    media: tuple[MediaAsset, ...]
    publication_ready: bool
    review_status: str
    safety_notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class InstagramUserTag:
    username: str
    media_ordinal: int
    x: float | None = None
    y: float | None = None


@dataclass(frozen=True)
class InstagramOptions:
    post_type: Literal["image", "carousel"]
    media_tags: tuple[InstagramUserTag, ...] = ()
    collaborators: tuple[str, ...] = ()
    location_id: str | None = None
    first_comment: str | None = None


@dataclass(frozen=True)
class PublicationRequest:
    publication_id: str
    publication_version: int
    platform: Literal["instagram"]
    account_ref: str
    project_id: str
    period: str
    asset_package_id: str
    caption: str
    hashtags: tuple[str, ...] = ()
    caption_mentions: tuple[str, ...] = ()
    instagram: InstagramOptions = field(default_factory=lambda: InstagramOptions(post_type="image"))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PublicationApproval:
    approval_id: str
    publication_id: str
    publication_version: int
    request_fingerprint: str
    approved_by: str
    approved_at_utc: str


@dataclass(frozen=True)
class PublicationSchedule:
    schedule_id: str
    publication_id: str
    publication_version: int
    scheduled_local: str
    timezone: str
    scheduled_at_utc: str
    status: Literal["draft", "scheduled", "cancelled"] = "draft"
