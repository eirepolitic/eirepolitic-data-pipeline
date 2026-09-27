from __future__ import annotations

from pathlib import Path

from instagram.factory.contract import QASummary, RenderResult, SlideRef
from instagram.factory.publication_handoff import build_publication_handoff, write_publication_handoff


def test_publication_handoff_uses_portable_relative_paths(tmp_path: Path) -> None:
    slide = tmp_path / "slides" / "slide-01.png"
    caption = tmp_path / "caption.txt"
    manifest = tmp_path / "metadata" / "manifest.json"
    slide.parent.mkdir(parents=True)
    manifest.parent.mkdir(parents=True)
    slide.write_bytes(b"png")
    caption.write_text("Test caption #Test", encoding="utf-8")
    manifest.write_text("{}", encoding="utf-8")

    result = RenderResult(
        project_id="demo_project",
        period_key="2026-09",
        output_root=str(tmp_path),
        slides=(SlideRef(id="slide-01", path=str(slide), width=1080, height=1350),),
        contact_sheets=(),
        caption_path=str(caption),
        manifest_path=str(manifest),
        package=None,
        qa=QASummary(expected_slide_count=1, actual_slide_count=1),
        review_state="pending_human_review",
        publication_enabled=False,
    )

    value = build_publication_handoff(result, artifact_root=tmp_path)
    assert value["schema_version"] == 1
    assert value["qa_all_passed"] is True
    assert value["factory_publication_enabled"] is False
    assert value["slides"][0]["path"] == "slides/slide-01.png"
    assert value["caption_path"] == "caption.txt"
    assert value["source_manifest_path"] == "metadata/manifest.json"
    assert value["operator_contract"]["requires_explicit_publish_approval"] is True

    path = write_publication_handoff(result, artifact_root=tmp_path)
    assert path.name == "publication_handoff.json"
    assert path.exists()
