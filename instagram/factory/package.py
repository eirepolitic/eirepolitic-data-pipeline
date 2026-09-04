from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_zip(source_root: str | Path, output_zip: str | Path) -> dict:
    source_root = Path(source_root)
    output_zip = Path(output_zip)
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(path for path in source_root.rglob("*") if path.is_file() and path != output_zip)
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            relative = path.relative_to(source_root).as_posix()
            info = zipfile.ZipInfo(relative, date_time=ZIP_EPOCH)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())
    manifest = {
        "zip": str(output_zip),
        "file_count": len(files),
        "sha256": sha256_file(output_zip),
        "files": [path.relative_to(source_root).as_posix() for path in files],
    }
    output_zip.with_suffix(output_zip.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
