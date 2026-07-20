from __future__ import annotations

from pathlib import Path
from typing import Any


def upload_directory_to_s3(output_root: str | Path, bucket: str, prefix: str) -> dict[str, Any]:
    import boto3

    root = Path(output_root)
    if not root.is_dir():
        raise ValueError(f"Output root does not exist: {root}")
    client = boto3.client("s3")
    uploaded: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        key = f"{prefix.rstrip('/')}/{path.relative_to(root).as_posix()}"
        client.upload_file(str(path), bucket, key)
        uploaded.append(key)
    return {"success": True, "bucket": bucket, "prefix": prefix, "uploaded_file_count": len(uploaded)}
