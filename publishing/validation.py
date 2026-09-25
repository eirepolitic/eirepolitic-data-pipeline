from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from publishing.models import AssetPackage, PublicationApproval, PublicationRequest


class PublicationValidationError(ValueError):
    pass


def validate_asset_package(package: AssetPackage) -> None:
    if not package.publication_ready:
        raise PublicationValidationError("asset package is not publication-ready")
    if package.review_status.lower() != "approved":
        raise PublicationValidationError("asset package review status is not approved")
    if package.safety_notes:
        raise PublicationValidationError("asset package contains unresolved safety notes")
    if not package.media:
        raise PublicationValidationError("asset package contains no media")

    ordinals = [item.ordinal for item in package.media]
    expected = list(range(1, len(package.media) + 1))
    if ordinals != expected:
        raise PublicationValidationError("media ordinals must be contiguous and ordered from 1")

    for item in package.media:
        if not item.sha256:
            raise PublicationValidationError(f"media {item.asset_id} is missing SHA-256")
        if item.width <= 0 or item.height <= 0 or item.size_bytes <= 0:
            raise PublicationValidationError(f"media {item.asset_id} has invalid dimensions or file size")
        if item.mime_type != "image/jpeg":
            raise PublicationValidationError(f"media {item.asset_id} is not Instagram-ready JPEG")


def validate_request_against_package(request: PublicationRequest, package: AssetPackage) -> None:
    validate_asset_package(package)
    if request.project_id != package.project_id:
        raise PublicationValidationError("publication project does not match asset package")
    if request.period != package.period:
        raise PublicationValidationError("publication period does not match asset package")
    if request.asset_package_id != package.asset_package_id:
        raise PublicationValidationError("publication references the wrong asset package")
    if not request.caption.strip():
        raise PublicationValidationError("caption must not be empty")

    expected_type = "image" if len(package.media) == 1 else "carousel"
    if request.instagram.post_type != expected_type:
        raise PublicationValidationError(
            f"post type {request.instagram.post_type!r} does not match {len(package.media)} media assets"
        )

    max_ordinal = len(package.media)
    for tag in request.instagram.media_tags:
        if tag.media_ordinal < 1 or tag.media_ordinal > max_ordinal:
            raise PublicationValidationError(f"media tag @{tag.username} references invalid media ordinal")
        if (tag.x is None) ^ (tag.y is None):
            raise PublicationValidationError("media tag coordinates must provide both x and y or neither")
        if tag.x is not None and not (0.0 <= tag.x <= 1.0 and 0.0 <= tag.y <= 1.0):
            raise PublicationValidationError("media tag coordinates must be normalized between 0 and 1")

    for hashtag in request.hashtags:
        if hashtag and not hashtag.startswith("#"):
            raise PublicationValidationError(f"invalid hashtag {hashtag!r}")
        if hashtag and hashtag not in request.caption:
            raise PublicationValidationError(f"structured hashtag {hashtag!r} is missing from exact caption")

    for mention in request.caption_mentions:
        normalized = mention if mention.startswith("@") else f"@{mention}"
        if normalized not in request.caption:
            raise PublicationValidationError(f"structured mention {normalized!r} is missing from exact caption")


def validate_approval(
    approval: PublicationApproval,
    request: PublicationRequest,
    expected_fingerprint: str,
) -> None:
    if approval.publication_id != request.publication_id:
        raise PublicationValidationError("approval publication ID does not match request")
    if approval.publication_version != request.publication_version:
        raise PublicationValidationError("approval publication version does not match request")
    if approval.request_fingerprint != expected_fingerprint:
        raise PublicationValidationError("approval fingerprint does not match exact publication request")


def next_publication_version(request: PublicationRequest, **changes: object) -> PublicationRequest:
    """Return an immutable next version after a material content/account change."""
    return replace(request, publication_version=request.publication_version + 1, **changes)
