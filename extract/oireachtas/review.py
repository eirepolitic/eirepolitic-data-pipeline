"""Review-output helpers for unified Oireachtas builds.

These helpers create the small generated files that a GitHub Actions workflow
can publish to the `oireachtas-review-output` branch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from .normalize import stable_json_dumps


REVIEW_ROOT = Path("oireachtas_review_output")


def table_review_dir(table: str, *, root: Path = REVIEW_ROOT) -> Path:
    """Return local review directory for a table."""
    return root / "review" / table / "latest"


def write_review_bundle(
    *,
    table: str,
    manifest: Mapping[str, Any],
    schema: Mapping[str, Any],
    dq: Mapping[str, Any],
    sample_rows: Sequence[Mapping[str, Any]] | None = None,
    root: Path = REVIEW_ROOT,
) -> Path:
    """Write local review files and return the table directory."""
    out_dir = table_review_dir(table, root=root)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = list(sample_rows or [])
    pd.DataFrame(rows).to_csv(out_dir / "sample.csv", index=False)
    (out_dir / "sample.md").write_text(_sample_markdown(table, rows), encoding="utf-8")
    (out_dir / "manifest.json").write_text(stable_json_dumps(manifest) + "\n", encoding="utf-8")
    (out_dir / "schema.json").write_text(stable_json_dumps(schema) + "\n", encoding="utf-8")
    (out_dir / "dq.json").write_text(stable_json_dumps(dq) + "\n", encoding="utf-8")

    write_index(root=root)
    return out_dir


def write_index(*, root: Path = REVIEW_ROOT) -> None:
    """Write a simple generated review index."""
    review_root = root / "review"
    review_root.mkdir(parents=True, exist_ok=True)
    table_dirs = sorted(path for path in review_root.iterdir() if path.is_dir())

    lines = [
        "# Oireachtas review output",
        "",
        "This branch/folder is machine-updated by the Oireachtas table-test workflow.",
        "Only small review samples should be published here.",
        "",
        "## Latest outputs",
        "",
    ]

    if not table_dirs:
        lines.append("No table review outputs found.")
    for table_dir in table_dirs:
        table = table_dir.name
        lines.extend(
            [
                f"### `{table}`",
                f"- `review/{table}/latest/sample.csv`",
                f"- `review/{table}/latest/sample.md`",
                f"- `review/{table}/latest/manifest.json`",
                f"- `review/{table}/latest/schema.json`",
                f"- `review/{table}/latest/dq.json`",
                "",
            ]
        )

    (review_root / "index.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    (root / "README.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def raw_review_url(*, repo: str, branch: str, table: str, filename: str = "manifest.json") -> str:
    """Return a raw.githubusercontent.com URL for a review file."""
    return f"https://raw.githubusercontent.com/{repo}/{branch}/review/{table}/latest/{filename}"


def _sample_markdown(table: str, rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return f"# `{table}` sample\n\nNo sample rows.\n"

    df = pd.DataFrame(rows)
    display = df.copy()
    for column in display.columns:
        display[column] = display[column].astype(str).str.slice(0, 250)
    return f"# `{table}` sample\n\n" + display.to_markdown(index=False) + "\n"
