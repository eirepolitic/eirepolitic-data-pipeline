from __future__ import annotations

import argparse
import io
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from instagram.renderer.template_renderer import render_template_file

DEFAULT_REGION = "ca-central-1"


def slugify(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "item"


def read_csv(path: str) -> pd.DataFrame:
    if path.startswith("s3://"):
        _, _, rest = path.partition("s3://")
        bucket, _, key = rest.partition("/")
        s3 = boto3.client("s3", region_name=DEFAULT_REGION)
        obj = s3.get_object(Bucket=bucket, Key=key)
        return pd.read_csv(io.StringIO(obj["Body"].read().decode("utf-8-sig", errors="replace")))
    return pd.read_csv(path)


def select_rows(df: pd.DataFrame, selector: dict[str, Any], limit_override: int | None = None) -> pd.DataFrame:
    mode = selector.get("mode", "top_speech_count")
    limit = int(limit_override or selector.get("limit", 10))
    if mode == "top_speech_count":
        sort_col = "speech_count_2025" if "speech_count_2025" in df.columns else "speech_count"
        return df.sort_values(by=[sort_col, "full_name"], ascending=[False, True]).head(limit).copy()
    if mode == "all_with_photos":
        photo_col = df["photo_url"] if "photo_url" in df.columns else pd.Series([""] * len(df))
        return df[photo_col.fillna("").astype(str).str.len() > 0].head(limit).copy()
    if mode == "explicit_members":
        names = {str(x).strip().lower() for x in selector.get("names", [])}
        return df[df["full_name"].fillna("").str.lower().isin(names)].head(limit).copy()
    raise RuntimeError(f"Unsupported selector mode: {mode}")


def value(row: pd.Series, col: str, default: str = "N/A") -> str:
    raw = row.get(col, default)
    if pd.isna(raw) or str(raw).strip() == "":
        return default
    return str(raw).strip()


def row_bindings(row: pd.Series, footer_text: str) -> dict[str, str]:
    rank = value(row, "speech_rank_2025")
    vote = value(row, "vote_participation_pct_2025")
    if vote not in {"N/A", "0"} and not vote.endswith("%"):
        vote = f"{vote}%"
    speech_count = value(row, "speech_count_2025", "0")
    return {
        "member_name": value(row, "full_name"),
        "party": value(row, "party"),
        "constituency": value(row, "constituency"),
        "member_photo": value(row, "photo_url", ""),
        "top_issue": value(row, "top_issue_2025", "No classified issue yet"),
        "vote_participation": vote,
        "speech_rank": rank,
        "speech_count_text": f"{speech_count} issue-labelled speeches in 2025",
        "footer_text": footer_text,
    }


def relpath(path: str | Path, root: Path) -> str:
    try:
        return Path(path).relative_to(root).as_posix()
    except ValueError:
        return Path(path).as_posix()


def html_escape(value: Any) -> str:
    text = str(value or "")
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def write_review(output_root: Path, rows: list[dict[str, Any]], spec: dict[str, Any], spec_path: str | Path) -> None:
    review_dir = output_root / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    enriched_rows: list[dict[str, Any]] = []
    for row in rows:
        warnings = str(row.get("warnings", "")).strip()
        has_photo = bool(str(row.get("photo_url", "")).strip())
        enriched = {
            "review_status": "needs_review",
            "review_notes": "",
            "publish_ready": "no",
            "needs_photo_check": "yes" if not has_photo else "no",
            "has_render_warnings": "yes" if warnings else "no",
            **row,
            "output_file_rel": relpath(row["output_file"], output_root),
            "bindings_file_rel": relpath(row.get("bindings_file", ""), output_root),
            "render_manifest_file_rel": relpath(row.get("render_manifest_file", ""), output_root),
        }
        enriched_rows.append(enriched)

    df = pd.DataFrame(enriched_rows)
    df.to_csv(review_dir / "review_table.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "success": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "campaign": spec.get("campaign"),
        "post_type": spec.get("post_type"),
        "spec_path": str(spec_path),
        "output_root": str(output_root),
        "rendered": len(enriched_rows),
        "review_table": str(review_dir / "review_table.csv"),
        "review_index": str(review_dir / "review_index.html"),
        "items": enriched_rows,
        "review_checklist": [
            "Confirm member name, party, and constituency are correct.",
            "Confirm image belongs to the correct member, or replace missing/incorrect images before publishing.",
            "Check every metric against the source table.",
            "Check long text for clipping or misleading truncation.",
            "Check generated warnings before setting publish_ready=yes.",
        ],
    }
    (review_dir / "review_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    cards = []
    for row in enriched_rows:
        rel = html_escape(row["output_file_rel"])
        warning_text = html_escape(row.get("warnings", "") or "None")
        cards.append(
            "<article>"
            f"<h2>{html_escape(row['full_name'])}</h2>"
            f"<p><strong>{html_escape(row['party'])}</strong> · {html_escape(row['constituency'])}</p>"
            f"<img src='../{rel}' alt='{html_escape(row['full_name'])}' />"
            "<dl>"
            f"<dt>Top issue</dt><dd>{html_escape(row['top_issue_2025'])}</dd>"
            f"<dt>Vote participation</dt><dd>{html_escape(row['vote_participation_pct_2025'])}</dd>"
            f"<dt>Speech rank</dt><dd>{html_escape(row['speech_rank_2025'])}</dd>"
            f"<dt>Speech count</dt><dd>{html_escape(row['speech_count_2025'])}</dd>"
            f"<dt>Warnings</dt><dd>{warning_text}</dd>"
            f"<dt>Bindings</dt><dd><code>{html_escape(row['bindings_file_rel'])}</code></dd>"
            f"<dt>Render manifest</dt><dd><code>{html_escape(row['render_manifest_file_rel'])}</code></dd>"
            "</dl>"
            "<p class='status'>Review status: needs_review · Publish ready: no</p>"
            "</article>"
        )

    html = """<!doctype html>
<html>
<head>
  <meta charset='utf-8'>
  <title>Member Profile Batch v1 Review</title>
  <style>
    body{font-family:Arial,sans-serif;background:#0f2f24;color:#f4ead7;margin:0;padding:32px}
    h1{margin-top:0} .summary{color:#cbbf9f;margin-bottom:24px}
    article{margin:24px 0;padding:18px;border:1px solid #cbbf9f;border-radius:14px;background:#173d30;max-width:980px}
    img{max-width:360px;width:100%;display:block;border-radius:10px;border:1px solid #cbbf9f;margin:14px 0}
    dl{display:grid;grid-template-columns:180px 1fr;gap:8px 16px} dt{color:#d8b45f;font-weight:bold} dd{margin:0}
    code{color:#f4ead7}.status{padding:10px 12px;background:#214a3b;border-radius:10px;display:inline-block}
  </style>
</head>
<body>
  <h1>Member Profile Batch v1 Review</h1>
  <p class='summary'>Use this page for human review only. Set publish status in <code>review_table.csv</code> after checking every item.</p>
""" + "\n".join(cards) + "\n</body></html>"
    (review_dir / "review_index.html").write_text(html, encoding="utf-8")


def render_campaign(spec_path: str | Path, limit_override: int | None = None) -> dict[str, Any]:
    spec = yaml.safe_load(Path(spec_path).read_text(encoding="utf-8"))
    if spec.get("campaign") != "member_profile_batch_v1":
        raise RuntimeError("Only member_profile_batch_v1 is supported by this first campaign renderer.")
    df = read_csv(spec["data"]["source_table"])
    selected = select_rows(df, spec["variation"]["selector"], limit_override)
    output_root = Path(spec["render"].get("output_root", "generated_posts/member_profile_batch_v1"))
    png_dir = output_root / "png"
    png_dir.mkdir(parents=True, exist_ok=True)
    bindings_dir = output_root / "metadata" / "bindings"
    bindings_dir.mkdir(parents=True, exist_ok=True)

    review_rows: list[dict[str, Any]] = []
    for _, row in selected.iterrows():
        slug = slugify(row.get("full_name"))
        bindings = row_bindings(row, spec.get("review", {}).get("footer_text", "Source: Oireachtas data pipeline"))
        bindings_path = bindings_dir / f"{slug}.yml"
        bindings_path.write_text(yaml.safe_dump({"bindings": bindings}, sort_keys=False, allow_unicode=True), encoding="utf-8")
        out_path = png_dir / f"{slug}.png"
        manifest = render_template_file(spec["render"]["template"], bindings_path, out_path, spec["render"].get("palette"))
        review_rows.append({
            "output_file": str(out_path),
            "bindings_file": str(bindings_path),
            "render_manifest_file": str(output_root / "metadata" / "manifests" / f"{slug}.render_manifest.json"),
            "full_name": bindings["member_name"],
            "party": bindings["party"],
            "constituency": bindings["constituency"],
            "top_issue_2025": bindings["top_issue"],
            "vote_participation_pct_2025": bindings["vote_participation"],
            "speech_count_2025": value(row, "speech_count_2025", "0"),
            "speech_rank_2025": bindings["speech_rank"],
            "photo_url": bindings["member_photo"],
            "warnings": ";".join(manifest.get("warnings", [])),
        })
    write_review(output_root, review_rows, spec, spec_path)
    return {"success": True, "campaign": spec["campaign"], "rendered": len(review_rows), "output_root": str(output_root)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render an Instagram campaign from render_spec.yml.")
    parser.add_argument("--campaign", required=True, help="Path to campaign render_spec.yml")
    parser.add_argument("--limit", type=int, help="Optional limit override")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(render_campaign(args.campaign, args.limit), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
