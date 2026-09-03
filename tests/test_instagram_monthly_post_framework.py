from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import yaml

from instagram.factory.package import deterministic_zip
from instagram.visuals.renderers import horizontal_bar

ROOT = Path(__file__).resolve().parents[1]
PROJECT_PATH = ROOT / "instagram/projects/party_issue_monthly_profile_v2/project.yml"
FIXTURE_PATH = ROOT / "instagram/reference/regression/party_issue_monthly_profile_v2_july_2026.json"
REGISTRY_PATH = ROOT / "configs/reference/party_assets_v1.csv"


def _project() -> dict:
    return yaml.safe_load(PROJECT_PATH.read_text(encoding="utf-8"))


def _fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _git_blob_sha(path: Path) -> str:
    payload = path.read_bytes()
    return hashlib.sha1(f"blob {len(payload)}\0".encode("ascii") + payload).hexdigest()


def test_project_definition_matches_approved_july_contract() -> None:
    project = _project()
    fixture = _fixture()
    slides = {row["id"]: row for row in project["slides"]["definitions"]}

    assert project["project_id"] == "party_issue_monthly_profile_v2"
    assert project["period"]["cadence"] == "monthly"
    assert project["period"]["default"] == "last_completed_month"
    assert project["source"]["require_immutable_batch"] is True
    assert project["source"]["require_batch_status"] == "validated"
    assert project["metrics"]["average_party_baseline"] == "unweighted_all_displayed_groups_including_zero"
    assert project["metrics"]["per_td_denominator"] == "period_end_active_dail_tds"

    assert project["render"]["width"] == fixture["batch"]["dimensions"][0]
    assert project["render"]["height"] == fixture["batch"]["dimensions"][1]
    assert project["render"]["min_visual_rows"] == fixture["chart"]["min_visual_rows"] == 4
    assert project["qa"]["expected_party_count"] == fixture["batch"]["party_count"] == 11
    assert project["qa"]["expected_slide_count"] == fixture["batch"]["slide_count"] == 55

    assert slides["cover"]["title"] == fixture["titles"]["cover"]
    assert slides["most_discussed_issues"]["title"] == fixture["titles"]["most_discussed_issues"]
    assert slides["more_than_average"]["title"] == fixture["titles"]["more_than_average"]
    assert slides["more_per_td"]["title"] == fixture["titles"]["more_per_td"]
    assert slides["glossary"]["title"] == fixture["titles"]["glossary"]
    assert slides["most_discussed_issues"]["value_format"] == fixture["value_formats"]["most_discussed_issues"]
    assert slides["more_than_average"]["value_format"] == fixture["value_formats"]["more_than_average"]
    assert slides["more_per_td"]["value_format"] == fixture["value_formats"]["more_per_td"]

    assert project["publication"]["enabled"] is False
    assert project["review"]["state"] == "pending_human_review"


def test_cover_geometry_and_palette_are_locked() -> None:
    project = _project()
    fixture = _fixture()
    cover = project["render"]["cover"]
    palette = project["render"]["palette"]

    assert cover["logo_size"] == 500
    assert cover["logo_top"] == 300
    assert cover["logo_border_width"] == 6
    assert cover["logo_border_color"] == fixture["cover"]["border_color"] == "#d8b45f"
    assert cover["scale_overrides"] == fixture["cover"]["scale_overrides"]
    assert "social-democrats" in cover["neutral_cleanup_party_keys"]

    assert palette == {
        "background": fixture["chart"]["background"],
        "text": fixture["chart"]["text"],
        "muted": fixture["chart"]["muted"],
        "accent": fixture["chart"]["accent"],
        "grid": fixture["chart"]["grid"],
    }


def test_corner_assets_are_exact_approved_blobs() -> None:
    fixture = _fixture()
    asset_root = ROOT / "instagram/templates/assets"
    for name, expected in fixture["assets"]["corner_git_blobs"].items():
        assert _git_blob_sha(asset_root / name) == expected


def test_party_registry_is_complete_approved_and_canonical() -> None:
    fixture = _fixture()
    with REGISTRY_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 11
    assert all(row["asset_status"] == "approved" for row in rows)
    assert len({row["party_key"] for row in rows}) == 11
    for row in rows:
        expected = fixture["assets"]["canonical_uri_template"].format(party_key=row["party_key"])
        assert row["logo_s3_uri"] == expected

    social = next(row for row in rows if row["party_key"] == "social-democrats")
    assert social["logo_s3_uri"] == fixture["assets"]["social_democrats_uri"]
    independent = next(row for row in rows if row["party_key"] == "independent")
    assert "Independents" in independent["party_aliases"].split(";")


def _render_bar(tmp_path: Path, count: int) -> dict:
    rows = [{"label": f"Issue {idx + 1}", "value": float(count - idx)} for idx in range(count)]
    template = {
        "template_id": "regression",
        "params": {
            "width": 1032,
            "height": 1210,
            "max_items": 7,
            "sort": "descending",
            "value_format": "integer",
            "min_visual_rows": 4,
        },
        "palette": {
            "background": "#0f2f24",
            "text": "#f4ead7",
            "muted": "#c8bda8",
            "accent": "#d8b45f",
            "grid": "#f4ead7",
        },
    }
    sample = {"visual_id": f"rows-{count}", "bindings": {"label": "label", "value": "value"}}
    return horizontal_bar.render(
        template,
        sample,
        rows,
        tmp_path / f"rows-{count}.png",
        tmp_path / f"rows-{count}.metadata.json",
        tmp_path / f"rows-{count}.manifest.json",
        {"test": True},
    )


def test_short_chart_bar_thickness_never_exceeds_four_row_layout(tmp_path: Path) -> None:
    fixture = _fixture()
    manifests = {count: _render_bar(tmp_path, count) for count in (1, 2, 3, 4)}
    four_row_thickness = manifests[4]["readability"]["bar_thickness_px"]
    assert four_row_thickness <= fixture["chart"]["max_short_chart_bar_thickness_px"] + 0.01

    for count in (1, 2, 3):
        readability = manifests[count]["readability"]
        assert readability["effective_visual_row_count"] == 4
        assert readability["min_visual_rows"] == 4
        assert readability["bar_thickness_px"] <= four_row_thickness + 0.01
        assert readability["category_text_clipped_count"] == 0
        assert readability["value_text_clipped_count"] == 0
        assert readability["truncated_label_count"] == 0


def test_empty_analytical_state_is_truthful_and_not_a_fake_bar(tmp_path: Path) -> None:
    template = {
        "template_id": "empty-regression",
        "params": {"width": 1032, "height": 1210, "min_visual_rows": 4, "value_format": "plus_pp_1"},
        "palette": {
            "background": "#0f2f24",
            "text": "#f4ead7",
            "muted": "#c8bda8",
            "accent": "#d8b45f",
            "grid": "#f4ead7",
        },
    }
    sample = {
        "visual_id": "empty",
        "bindings": {"label": "label", "value": "value"},
        "empty_message": "No issues above average",
    }
    manifest = horizontal_bar.render(
        template,
        sample,
        [],
        tmp_path / "empty.png",
        tmp_path / "empty.metadata.json",
        tmp_path / "empty.manifest.json",
        {"test": True},
    )
    readability = manifest["readability"]
    assert manifest["warnings"] == []
    assert readability["empty_state"] is True
    assert readability["empty_message"] == "No issues above average"
    assert readability["displayed_item_count"] == 0
    assert readability["bar_thickness_px"] == 0.0


def test_deterministic_zip_is_reproducible(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.txt").write_text("alpha\n", encoding="utf-8")
    (source / "b.txt").write_text("beta\n", encoding="utf-8")
    first = deterministic_zip(source, tmp_path / "first.zip")
    second = deterministic_zip(source, tmp_path / "second.zip")
    assert first["sha256"] == second["sha256"]
    assert first["files"] == second["files"] == ["a.txt", "b.txt"]
