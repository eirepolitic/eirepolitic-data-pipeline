import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from botocore.exceptions import ClientError

from process.party_assets_upload import collect_uploads, upload_build


class FakeS3:
    def __init__(self, existing=None):
        self.existing = set(existing or [])
        self.uploaded = []

    def head_object(self, Bucket, Key):
        if Key in self.existing:
            return {"ContentLength": 1}
        raise ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject")

    def upload_file(self, filename, bucket, key, ExtraArgs=None):
        self.uploaded.append((filename, bucket, key, ExtraArgs or {}))


class PartyAssetUploadTests(unittest.TestCase):
    def _build_root(self, root: Path, success: bool = True) -> Path:
        build = root / "build"
        (build / "assets/example-party").mkdir(parents=True)
        (build / "assets/example-party/logo.png").write_bytes(b"png")
        (build / "assets/example-party/source.png").write_bytes(b"source")
        (build / "contact_sheet.png").write_bytes(b"sheet")
        (build / "manifest.json").write_text(json.dumps({"success": success}), encoding="utf-8")
        return build

    def test_collect_uploads_uses_versioned_relative_paths(self):
        with TemporaryDirectory() as temp_dir:
            build = self._build_root(Path(temp_dir))
            uploads = collect_uploads(build, "processed/reference/party_assets/v1")
            keys = {key for _, key in uploads}
            self.assertIn("processed/reference/party_assets/v1/assets/example-party/logo.png", keys)
            self.assertIn("processed/reference/party_assets/v1/manifest.json", keys)
            self.assertIn("processed/reference/party_assets/v1/contact_sheet.png", keys)

    def test_dry_run_does_not_upload(self):
        with TemporaryDirectory() as temp_dir:
            build = self._build_root(Path(temp_dir))
            client = FakeS3()
            report = upload_build(build, "bucket", "processed/reference/party_assets/v1", False, client=client)
            self.assertFalse(report["apply"])
            self.assertEqual(client.uploaded, [])

    def test_existing_object_blocks_upload(self):
        with TemporaryDirectory() as temp_dir:
            build = self._build_root(Path(temp_dir))
            existing = {"processed/reference/party_assets/v1/assets/example-party/logo.png"}
            client = FakeS3(existing=existing)
            with self.assertRaisesRegex(ValueError, "already exist"):
                upload_build(build, "bucket", "processed/reference/party_assets/v1", True, client=client)
            self.assertEqual(client.uploaded, [])

    def test_unsuccessful_manifest_blocks_upload(self):
        with TemporaryDirectory() as temp_dir:
            build = self._build_root(Path(temp_dir), success=False)
            with self.assertRaisesRegex(ValueError, "not successful"):
                upload_build(build, "bucket", "processed/reference/party_assets/v1", False, client=FakeS3())


if __name__ == "__main__":
    unittest.main()
