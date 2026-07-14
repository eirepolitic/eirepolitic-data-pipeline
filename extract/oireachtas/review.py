"""Review artifact helpers for the unified Oireachtas pipeline."""

from __future__ import annotations

import csv
import json
import re
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping

from .normalize import stable_json_dumps


REVIEW_ROOT = Path("review")
DEFAULT_SAMPLE_ROWS = 10
MAX_SAMPLE_ROWS = 100
MAX_LIST_ITEMS = 100
MAX_TEXT_LENGTH = 20_000


def write_review_bundle(
    *,
    table: str,
    manifest: Mapping[str, Any],
    schema: Mapping[str, Any],
    dq: Mapping[str, Any],
    sample_rows: Iterable[Mapping[str, Any]],
    root: Path = REVIEW_ROOT,
    sample_limit: int = DEFAULT_SAMPLE_ROWS,
) -> Path:
    """Write immutable run-scoped review files and a compact latest copy."""
    run_id = _safe_component(
        str(manifest.get("run_id") or manifest.get("workflow_run_id") or manifest.get("created_at_utc") or "unknown-run")
    )
    table_root = root / table
    run_dir = table_root / "runs" / run_id
    latest_dir = table_root / "latest"
    rows = [dict(row) for row in sample_rows][: max(1, min(int(sample_limit), MAX_SAMPLE_ROWS))]

    _write_bundle(run_dir, manifest=manifest, schema=schema, dq=dq, rows=rows)
    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    shutil.copytree(run_dir, latest_dir)
    (table_root / "latest_pointer.json").write_text(
        stable_json_dumps({"table": table, "run_id": run_id, "run_path": f"runs/{run_id}"}) + "\n",
        encoding="utf-8",
    )
    return run_dir


def _write_bundle(
    output_dir: Path,
    *,
    manifest: Mapping[str, Any],
    schema: Mapping[str, Any],
    dq: Mapping[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    compact_manifest = _compact_payload(dict(manifest))
    compact_schema = _compact_payload(dict(schema))
    compact_dq = _compact_payload(dict(dq))
    (output_dir / "manifest.json").write_text(stable_json_dumps(compact_manifest) + "\n", encoding="utf-8")
    (output_dir / "schema.json").write_text(stable_json_dumps(compact_schema) + "\n", encoding="utf-8")
    (output_dir / "dq.json").write_text(stable_json_dumps(compact_dq) + "\n", encoding="utf-8")
    _write_csv(output_dir / "sample.csv", rows)
    (output_dir / "sample.md").write_text(_markdown_table(rows), encoding="utf-8")
    (output_dir / "report.md").write_text(
        _report_markdown(manifest=compact_manifest, schema=compact_schema, dq=compact_dq, sample_rows=rows),
        encoding="utf-8",
    )


def raw_review_url(*, repo: str, branch: str, table: str, filename: str = "report.md") -> str:
    """Return a raw GitHub URL for a latest review file."""
    return f"https://raw.githubusercontent.com/{repo}/{branch}/review/{table}/latest/{filename}"


def _compact_payload(value: Any, *, depth: int = 0) -> Any:
    if depth >= 8:
        return "<truncated-depth>"
    if isinstance(value, Mapping):
        return {str(key): _compact_payload(child, depth=depth + 1) for key, child in value.items()}
    if isinstance(value, list):
        compact = [_compact_payload(child, depth=depth + 1) for child in value[:MAX_LIST_ITEMS]]
        if len(value) > MAX_LIST_ITEMS:
            compact.append({"_truncated_items": len(value) - MAX_LIST_ITEMS})
        return compact
    if isinstance(value, tuple):
        return _compact_payload(list(value), depth=depth)
    if isinstance(value, str) and len(value) > MAX_TEXT_LENGTH:
        return value[:MAX_TEXT_LENGTH] + f"...<truncated {len(value) - MAX_TEXT_LENGTH} chars>"
    return value


def _safe_component(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-.")
    return text[:128] or "unknown-run"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _cell(row.get(column)) for column in columns})


def _cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_compact_payload(value), ensure_ascii=False, sort_keys=True, default=str)
    return value


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No sample rows._\n"
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, divider]
    for row in rows:
        values = [str(_cell(row.get(column, ""))).replace("|", "\\|").replace("\n", " ") for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def _report_markdown(
    *,
    manifest: Mapping[str, Any],
    schema: Mapping[str, Any],
    dq: Mapping[str, Any],
    sample_rows: list[dict[str, Any]],
) -> str:
    checks = dq.get("checks", []) if isinstance(dq.get("checks"), list) else []
    lines = [
        f"# Review: `{manifest.get('table', schema.get('table', 'unknown'))}`",
        "",
        f"- Status: `{manifest.get('status', 'unknown')}`",
        f"- Mode: `{manifest.get('mode', 'unknown')}`",
        f"- DQ status: `{dq.get('dq_status', 'unknown')}`",
        f"- Output rows: `{manifest.get('output_rows', manifest.get('row_count', 'unknown'))}`",
        f"- Primary key: `{', '.join(schema.get('primary_key', []))}`",
        "",
        "## DQ checks",
        "",
    ]
    if checks:
        lines.extend(["| Check | Status | Metric | Threshold |", "| --- | --- | ---: | ---: |"]) 
        for check in checks:
            lines.append(f"| {check.get('check_name', '')} | {check.get('status', '')} | {check.get('metric_value', '')} | {check.get('threshold', '')} |")
    else:
        lines.append("_No DQ checks recorded._")
    lines.extend(["", "## Sample", "", _markdown_table(sample_rows)])
    return "\n".join(lines)
