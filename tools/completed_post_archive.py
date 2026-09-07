from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

REQUIRED_SUMMARY_KEYS = {
    "schema_version", "post_id", "title", "status", "repository", "created_by",
    "created_at", "platform", "assets", "tools_used", "process", "qa", "decisions",
}


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate_summary(summary: dict, repo: str, post_id: str) -> None:
    missing = sorted(REQUIRED_SUMMARY_KEYS - set(summary))
    if missing:
        raise ValueError(f"summary missing required keys: {missing}")
    if summary["schema_version"] != 1:
        raise ValueError("schema_version must be 1")
    if summary["post_id"] != post_id:
        raise ValueError("summary post_id does not match workflow post_id")
    if summary["repository"] != repo:
        raise ValueError(f"summary repository must be {repo}")
    if summary["status"] != "completed":
        raise ValueError("only completed posts may be archived")
    for key in ("assets", "tools_used", "process", "qa", "decisions"):
        if not isinstance(summary[key], list) or not summary[key]:
            raise ValueError(f"{key} must be a non-empty list")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare(source_dir: Path, summary_path: Path, out_dir: Path, repo: str, post_id: str) -> None:
    summary = _load(summary_path)
    validate_summary(summary, repo, post_id)
    if not source_dir.is_dir():
        raise ValueError(f"source directory does not exist: {source_dir}")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    assets_dir = out_dir / "assets"
    assets_dir.mkdir(parents=True)
    copied = []
    for rel in summary["assets"]:
        src = source_dir / rel
        if not src.is_file():
            raise ValueError(f"declared asset missing: {rel}")
        dst = assets_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append({"path": f"assets/{rel}", "bytes": dst.stat().st_size, "sha256": sha256(dst)})
    (out_dir / "agent-summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md = [f"# {summary['title']}", "", f"Post ID: `{post_id}`", f"Repository: `{repo}`", f"Created by: {summary['created_by']}", f"Created at: {summary['created_at']}", "", "## Tools used"]
    md += [f"- {item}" for item in summary["tools_used"]]
    md += ["", "## Process"] + [f"{i+1}. {item}" for i, item in enumerate(summary["process"])]
    md += ["", "## Decisions"] + [f"- {item}" for item in summary["decisions"]]
    md += ["", "## QA"] + [f"- {item}" for item in summary["qa"]]
    for section in ("sources", "limitations", "related_workflows", "related_pull_requests"):
        values = summary.get(section) or []
        if values:
            md += ["", f"## {section.replace('_', ' ').title()}"] + [f"- {item}" for item in values]
    (out_dir / "agent-summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    provenance = {
        "schema_version": 1,
        "post_id": post_id,
        "repository": repo,
        "archived_at": datetime.now(timezone.utc).isoformat(),
        "files": copied,
    }
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")


def update_index(index_path: Path, summary_path: Path, s3_uri: str) -> None:
    summary = _load(summary_path)
    index = _load(index_path) if index_path.exists() else {"schema_version": 1, "posts": []}
    entry = {
        "post_id": summary["post_id"],
        "title": summary["title"],
        "created_at": summary["created_at"],
        "platform": summary["platform"],
        "repository": summary["repository"],
        "s3_uri": s3_uri,
    }
    existing = next((x for x in index["posts"] if x["post_id"] == summary["post_id"]), None)
    if existing is not None:
        if existing != entry:
            raise ValueError(f"post_id already exists with different metadata: {summary['post_id']}")
        return
    index["posts"].append(entry)
    index["posts"] = sorted(index["posts"], key=lambda x: (x["created_at"], x["post_id"]), reverse=True)
    index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--source-dir", type=Path, required=True)
    prep.add_argument("--summary", type=Path, required=True)
    prep.add_argument("--out-dir", type=Path, required=True)
    prep.add_argument("--repo", required=True)
    prep.add_argument("--post-id", required=True)
    idx = sub.add_parser("update-index")
    idx.add_argument("--index", type=Path, required=True)
    idx.add_argument("--summary", type=Path, required=True)
    idx.add_argument("--s3-uri", required=True)
    args = p.parse_args()
    if args.command == "prepare":
        prepare(args.source_dir, args.summary, args.out_dir, args.repo, args.post_id)
    else:
        update_index(args.index, args.summary, args.s3_uri)


if __name__ == "__main__":
    main()
