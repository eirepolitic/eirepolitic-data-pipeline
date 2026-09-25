from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class MetaContainerStatus:
    container_id: str
    status_code: str
    media_id: str | None = None
    error_message: str | None = None


class InstagramMetaProvider(Protocol):
    """Provider boundary for the Meta Instagram Content Publishing API."""

    def create_image_container(
        self,
        *,
        image_url: str,
        caption: str | None,
        alt_text: str | None,
        user_tags: tuple[dict[str, object], ...] = (),
        collaborators: tuple[str, ...] = (),
        location_id: str | None = None,
        is_carousel_item: bool = False,
    ) -> str: ...

    def create_carousel_container(
        self,
        *,
        child_container_ids: tuple[str, ...],
        caption: str,
        collaborators: tuple[str, ...] = (),
        location_id: str | None = None,
    ) -> str: ...

    def get_container_status(self, container_id: str) -> MetaContainerStatus: ...

    def publish_container(self, container_id: str) -> str | None: ...

    def create_comment(self, media_id: str, message: str) -> str: ...


class ApprovedAssetUrlProvider(Protocol):
    """Returns a short-lived HTTPS retrieval URL for one immutable approved asset."""

    def url_for(self, *, bucket: str, key: str) -> str: ...
