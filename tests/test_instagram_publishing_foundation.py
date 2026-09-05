from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from publishing.assets import finalize_image_to_jpeg, sha256_file
from publishing.caption_templates import CaptionTemplateError, load_caption_template, validate_final_caption
from publishing.fingerprints import publication_request_fingerprint
from publishing.models import (
    AssetPackage,
    InstagramOptions,
    InstagramUserTag,
    MediaAsset,
    PublicationApproval,
    PublicationRequest,
)
from publishing.timezone import TimeResolutionError, resolve_local_time
from publishing.validation import (
    PublicationValidationError,
    next_publication_version,
    validate_approval,
    validate_request_against_package,
)


def _asset(ordinal: int = 1) -> MediaAsset:
    return MediaAsset(
        asset_id=f"slide_{ordinal:02d}",
        ordinal=ordinal,
        bucket="eirepolitic-data",
        key=f"instagram/approved/demo/2026-08/pkg/media/{ordinal:02d}.jpg",
        sha256=f"hash-{ordinal}",
        mime_type="image/jpeg",
        width=1080,
        height=1350,
        size_bytes=1000 + ordinal,
        alt_text=f"Slide {ordinal}",
    )


def _package(count: int = 1) -> AssetPackage:
    return AssetPackage(
        asset_package_id="pkg-1",
        project_id="demo",
        period="2026-08",
        media=tuple(_asset(i) for i in range(1, count + 1)),
        publication_ready=True,
        review_status="approved",
    )


def _request(count: int = 1) -> PublicationRequest:
    return PublicationRequest(
        publication_id="pub-1",
        publication_version=1,
        platform="instagram",
        account_ref="eirepolitic_instagram",
        project_id="demo",
        period="2026-08",
        asset_package_id="pkg-1",
        caption="Demo caption @example #EirePolitic",
        hashtags=("#EirePolitic",),
        caption_mentions=("@example",),
        instagram=InstagramOptions(
            post_type="image" if count == 1 else "carousel",
            media_tags=(InstagramUserTag(username="example", media_ordinal=1, x=0.5, y=0.5),),
        ),
    )


def test_fingerprint_changes_when_caption_or_asset_order_changes() -> None:
    request = _request(count=2)
    hashes = [asset.sha256 for asset in _package(count=2).media]
    original = publication_request_fingerprint(request, hashes)

    changed_caption = publication_request_fingerprint(replace(request, caption="Changed #EirePolitic @example"), hashes)
    changed_order = publication_request_fingerprint(request, list(reversed(hashes)))

    assert original.startswith("sha256:")
    assert original != changed_caption
    assert original != changed_order
    assert original == publication_request_fingerprint(request, hashes)


def test_request_validation_enforces_review_and_exact_caption_metadata() -> None:
    package = _package()
    request = _request()
    validate_request_against_package(request, package)

    with pytest.raises(PublicationValidationError, match="hashtag"):
        validate_request_against_package(replace(request, caption="Demo caption @example"), package)

    blocked_package = replace(package, safety_notes=("review required",))
    with pytest.raises(PublicationValidationError, match="safety notes"):
        validate_request_against_package(request, blocked_package)


def test_material_edit_creates_new_version_and_old_approval_no_longer_matches() -> None:
    request = _request()
    fingerprint = publication_request_fingerprint(request, [_asset().sha256])
    approval = PublicationApproval(
        approval_id="approval-1",
        publication_id="pub-1",
        publication_version=1,
        request_fingerprint=fingerprint,
        approved_by="human-1",
        approved_at_utc="2026-09-05T12:00:00Z",
    )
    validate_approval(approval, request, fingerprint)

    changed = next_publication_version(request, caption="Updated @example #EirePolitic")
    changed_fingerprint = publication_request_fingerprint(changed, [_asset().sha256])
    assert changed.publication_version == 2
    with pytest.raises(PublicationValidationError, match="version"):
        validate_approval(approval, changed, changed_fingerprint)


def test_dublin_summer_time_resolves_to_utc() -> None:
    resolved = resolve_local_time("2026-09-08T19:30:00")
    assert resolved.timezone == "Europe/Dublin"
    assert resolved.scheduled_at_utc == "2026-09-08T18:30:00Z"
    assert resolved.utc_offset == "+01:00"


def test_dublin_nonexistent_and_ambiguous_times_are_not_guessed() -> None:
    # In 2026 Europe/Dublin advances clocks on 29 March and repeats 01:30 on 25 October.
    with pytest.raises(TimeResolutionError, match="does not exist"):
        resolve_local_time("2026-03-29T01:30:00")

    with pytest.raises(TimeResolutionError, match="occurs twice"):
        resolve_local_time("2026-10-25T01:30:00")

    first = resolve_local_time("2026-10-25T01:30:00", fold=0)
    second = resolve_local_time("2026-10-25T01:30:00", fold=1)
    assert first.scheduled_at_utc != second.scheduled_at_utc


def test_png_finalizer_creates_repeatable_rgb_jpeg(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    first_output = tmp_path / "first.jpg"
    second_output = tmp_path / "second.jpg"

    image = Image.new("RGBA", (32, 24), (20, 40, 60, 128))
    image.save(source, format="PNG")

    first = finalize_image_to_jpeg(source, first_output)
    second = finalize_image_to_jpeg(source, second_output)

    assert first.mime_type == "image/jpeg"
    assert first.width == 32
    assert first.height == 24
    assert first.sha256 == second.sha256
    assert first.sha256 == sha256_file(first_output)

    with Image.open(first_output) as rendered:
        assert rendered.format == "JPEG"
        assert rendered.mode == "RGB"


def test_member_profile_template_loads_current_default_hashtags() -> None:
    template = load_caption_template("instagram/caption_templates/member_profile.yml")
    assert template.template_id == "member_profile"
    assert template.version == 1
    assert "#EirePolitic" in template.default_hashtags
    validate_final_caption("A complete final caption.", template)


def test_required_template_component_is_enforced(tmp_path: Path) -> None:
    template_path = tmp_path / "required.yml"
    template_path.write_text(
        """
template_id: required_example
version: 1
components:
  attribution:
    required: true
hashtags:
  default: []
attribution: "Source: EirePolitic."
disclaimer: null
""".strip(),
        encoding="utf-8",
    )
    template = load_caption_template(template_path)
    with pytest.raises(CaptionTemplateError, match="attribution"):
        validate_final_caption("Caption without source.", template)
    validate_final_caption("Caption.\n\nSource: EirePolitic.", template)
