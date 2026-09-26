from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Sequence

from botocore.exceptions import ClientError

from publishing.assets import finalize_image_to_jpeg
from publishing.models import AssetPackage, MediaAsset


@dataclass(frozen=True)
class SourceAsset:
    path: Path
    alt_text: str = ""


def approved_asset_key(
    *,
    project_id: str,
    period: str,
    asset_package_id: str,
    ordinal: int,
    sha256: str,
) -> str:
    digest = sha256.removeprefix("sha256:")
    return (
        f"instagram/approved/{project_id}/{period}/{asset_package_id}/media/"
        f"{ordinal:02d}-{digest[:16]}.jpg"
    )


class S3ApprovedAssetStore:
    """Finalize reviewed media and upload it to immutable, content-addressed S3 keys."""

    def __init__(self, s3_client, bucket: str) -> None:
        self.s3 = s3_client
        self.bucket = bucket

    def finalize_package(
        self,
        *,
        project_id: str,
        period: str,
        asset_package_id: str,
        sources: Sequence[SourceAsset],
        review_status: str = "approved",
        safety_notes: Sequence[str] = (),
    ) -> AssetPackage:
        if not sources:
            raise ValueError("at least one source asset is required")

        media: list[MediaAsset] = []
        with TemporaryDirectory(prefix="instagram-approved-") as temp_dir:
            temp = Path(temp_dir)
            for ordinal, source in enumerate(sources, start=1):
                output = temp / f"{ordinal:02d}.jpg"
                finalized = finalize_image_to_jpeg(source.path, output)
                key = approved_asset_key(
                    project_id=project_id,
                    period=period,
                    asset_package_id=asset_package_id,
                    ordinal=ordinal,
                    sha256=finalized.sha256,
                )
                self._put_once(
                    key=key,
                    body=output.read_bytes(),
                    sha256=finalized.sha256,
                    asset_package_id=asset_package_id,
                )
                media.append(
                    MediaAsset(
                        asset_id=f"slide_{ordinal:02d}",
                        ordinal=ordinal,
                        bucket=self.bucket,
                        key=key,
                        sha256=finalized.sha256,
                        mime_type=finalized.mime_type,
                        width=finalized.width,
                        height=finalized.height,
                        size_bytes=finalized.size_bytes,
                        alt_text=source.alt_text,
                    )
                )

        return AssetPackage(
            asset_package_id=asset_package_id,
            project_id=project_id,
            period=period,
            media=tuple(media),
            publication_ready=review_status == "approved" and not safety_notes,
            review_status=review_status,
            safety_notes=tuple(safety_notes),
        )

    def _put_once(self, *, key: str, body: bytes, sha256: str, asset_package_id: str) -> None:
        expected_hash = sha256.removeprefix("sha256:")
        existing = self._head_if_exists(key)
        if existing is not None:
            self._verify_existing(key, existing, expected_hash, asset_package_id)
            return

        self.s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=body,
            ContentType="image/jpeg",
            Metadata={
                "sha256": expected_hash,
                "asset-package-id": asset_package_id,
            },
        )
        uploaded = self.s3.head_object(Bucket=self.bucket, Key=key)
        self._verify_existing(key, uploaded, expected_hash, asset_package_id)

    def _head_if_exists(self, key: str):
        try:
            return self.s3.head_object(Bucket=self.bucket, Key=key)
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if code in {"404", "NoSuchKey", "NotFound"} or status == 404:
                return None
            raise

    @staticmethod
    def _verify_existing(key: str, head: dict, expected_hash: str, asset_package_id: str) -> None:
        metadata = head.get("Metadata", {})
        if metadata.get("sha256") != expected_hash:
            raise RuntimeError(f"immutable S3 key already exists with different content: {key}")
        if metadata.get("asset-package-id") != asset_package_id:
            raise RuntimeError(f"immutable S3 key already exists for a different asset package: {key}")
