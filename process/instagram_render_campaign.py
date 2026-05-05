from __future__ import annotations

import argparse
import io
import json
import re
import sys
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


def write_review(output_root: Path, rows: list[dict[str, Any]]) -> None:
    review_dir = output_root / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(review_dir / "review_table.csv", index=False, encoding="utf-8-sig")
    cards = []
    for row in rows:
        rel = Path(row["output_file"]).relative_to(output_root)
        cards.append(
            f"<article><h2>{row['full_name']}</h2><p>{row['party']} · {row['constituency']}</p>"
            f"<img src='../{rel.as_posix()}' /></article>"
        )
    html = """<!doctype html><html><head><meta charset='utf-8'><title>Review Index</title><style>body{font-family:sans-serif;background:#0f2f24;color:#f4ead7}article{margin:24px;padding:16px;border:1px solid #cbbf9f;border-radius:12px}img{max-width:360px;width:100%;display:block}</style></head><body><h1>Member Profile Batch v1 Review</h1>""" + "\n".join(cards) + "</body></html>"
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
    write_review(output_root, review_rows)
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
