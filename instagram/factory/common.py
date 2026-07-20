from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def slugify(value: str) -> str:
    from .constituency_pilot import normalize_text
    return normalize_text(value).replace(" ", "-")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_run_id(project_id: str, project_version: int, batch_id: str, git_sha: str) -> str:
    identity = f"{project_id}|{project_version}|{batch_id}|{git_sha}"
    suffix = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    return f"{project_id}-v{project_version}-{suffix}"


def source_batch_id(source_manifest: dict[str, Any]) -> str:
    for source in source_manifest.values():
        if not isinstance(source, dict):
            continue
        resolution = source.get("resolution", {})
        if isinstance(resolution, dict) and resolution.get("batch_id"):
            return str(resolution["batch_id"])
    return "local-fixture"


def replace_tokens(value: Any, context: dict[str, Any]) -> Any:
    if not isinstance(value, str):
        return value
    result = value
    for key, replacement in context.items():
        if isinstance(replacement, (str, int, float)):
            result = result.replace("{{ " + str(key) + " }}", str(replacement))
    return result
