from __future__ import annotations

from pathlib import Path

from botocore.exceptions import ClientError
from PIL import Image

from publishing.s3_assets import S3ApprovedAssetStore, SourceAsset, approved_asset_key


class FakeS3:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], dict[str, object]] = {}
        self.put_calls = 0

    def put_object(self, **kwargs):
        self.put_calls += 1
        identity = (kwargs["Bucket"], kwargs["Key"])
        self.objects[identity] = kwargs
        return {"ETag": "fake"}

    def head_object(self, *, Bucket: str, Key: str):
        identity = (Bucket, Key)
        if identity not in self.objects:
            raise ClientError(
                {
                    "Error": {"Code": "404", "Message": "not found"},
                    "ResponseMetadata": {"HTTPStatusCode": 404},
                },
                "HeadObject",
            )
        value = self.objects[identity]
        return {"Metadata": value["Metadata"]}


def test_approved_asset_key_is_content_addressed() -> None:
    key = approved_asset_key(
        project_id="demo",
        period="2026-09",
        asset_package_id="pkg-1",
        ordinal=2,
        sha256="0123456789abcdef" * 4,
    )
    assert key == "instagram/approved/demo/2026-09/pkg-1/media/02-0123456789abcdef.jpg"


def test_finalize_package_uploads_private_delivery_jpeg_once(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    Image.new("RGBA", (40, 50), (10, 20, 30, 128)).save(source)

    s3 = FakeS3()
    store = S3ApprovedAssetStore(s3, "approved-bucket")

    first = store.finalize_package(
        project_id="demo",
        period="2026-09",
        asset_package_id="pkg-1",
        sources=[SourceAsset(source, alt_text="Demo alt text")],
    )
    second = store.finalize_package(
        project_id="demo",
        period="2026-09",
        asset_package_id="pkg-1",
        sources=[SourceAsset(source, alt_text="Demo alt text")],
    )

    assert first == second
    assert first.publication_ready is True
    asset = first.media[0]
    assert asset.bucket == "approved-bucket"
    assert asset.mime_type == "image/jpeg"
    assert asset.alt_text == "Demo alt text"
    assert asset.key.startswith("instagram/approved/demo/2026-09/pkg-1/media/01-")
    uploaded = s3.objects[("approved-bucket", asset.key)]
    assert uploaded["ContentType"] == "image/jpeg"
    assert uploaded["Metadata"]["sha256"] == asset.sha256
    assert s3.put_calls == 1


def test_existing_mismatched_object_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    Image.new("RGB", (40, 50), (10, 20, 30)).save(source)

    s3 = FakeS3()
    store = S3ApprovedAssetStore(s3, "approved-bucket")
    package = store.finalize_package(
        project_id="demo",
        period="2026-09",
        asset_package_id="pkg-1",
        sources=[SourceAsset(source)],
    )
    key = package.media[0].key
    s3.objects[("approved-bucket", key)]["Metadata"]["sha256"] = "different"

    try:
        store.finalize_package(
            project_id="demo",
            period="2026-09",
            asset_package_id="pkg-1",
            sources=[SourceAsset(source)],
        )
    except RuntimeError as exc:
        assert "different content" in str(exc)
    else:
        raise AssertionError("expected immutable object mismatch to fail")
