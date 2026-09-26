from __future__ import annotations

from pathlib import Path

from publishing.assets import finalize_image_to_jpeg
from publishing.canary import (
    CANARY_ALT_TEXT,
    CANARY_CAPTION,
    CANARY_HASHTAGS,
    render_canary_source,
)


def test_canary_content_is_neutral_and_has_no_real_tags() -> None:
    assert CANARY_CAPTION == (
        "Eirepolitic Instagram publishing pipeline test. Temporary verification post; "
        "this will be deleted after the test. #EirepoliticTest #PipelineTest"
    )
    assert CANARY_HASHTAGS == ("#EirepoliticTest", "#PipelineTest")
    assert "@" not in CANARY_CAPTION
    assert CANARY_ALT_TEXT.startswith("Abstract geometric test card")


def test_canary_image_is_exact_and_reproducible(tmp_path: Path) -> None:
    source = render_canary_source(tmp_path / "canary.png")
    finalized = finalize_image_to_jpeg(source, tmp_path / "canary.jpg")

    assert finalized.width == 1080
    assert finalized.height == 1350
    assert finalized.sha256 == "077729ceabc1371ac2e4d9f46baa990141d79a247e6f13c0f06563879886357d"
