"""Declarative QA runner (EirePolitic Director, Phase 3, §3.3).

Reads the `qa:` block each project already declares in its own `project.yml`
and evaluates it against a normalized `contract.RenderResult`, producing a
list of `contract.QACheck`. This replaces the inline Python `assert` blocks
that today live hand-written inside each bespoke render workflow (plan
§1.3/§3.3) — the checks themselves are not weakened, they move somewhere a
new project inherits them from just by declaring the same `qa:` keys.

Both projects' current `qa:` vocabularies (as of Phase 3) are supported:

  party_issue_monthly_profile_v2:
    expected_party_count, expected_slide_count, require_no_text_clipping,
    require_no_text_truncation, require_glossary, require_asset_registry,
    require_publication_disabled, require_review_state

  ipi_polling_factory_v1:
    expected_slide_count, expected_dimensions, require_source_footer

An unrecognized `qa:` key is not silently ignored and not treated as a hard
failure either — it is recorded as a QACheck whose detail says no checker
exists for it, so a new project's declared-but-unenforced checks stay
visible instead of disappearing quietly. See `_UNVERIFIED_DETAIL`.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Callable

from instagram.factory.contract import QACheck, RenderResult

_UNVERIFIED_DETAIL = "no checker registered for this qa: key; declared but not independently verified"


def _read_qa_summary_csv(result: RenderResult) -> list[dict[str, str]] | None:
    """party_issue_monthly_profile_v2-specific: the adapter writes a
    per-slide QA row CSV (qa_summary.csv) to its period root. Not all
    projects produce this file — callers must handle None."""
    candidate = Path(result.output_root) / "qa_summary.csv"
    if not candidate.is_file():
        return None
    with candidate.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _check_expected_slide_count(value: Any, *, project: dict, result: RenderResult) -> QACheck:
    return QACheck(
        name="expected_slide_count",
        passed=result.qa.slide_count_ok,
        detail=f"declared {value}, actual {result.qa.actual_slide_count}",
    )


def _check_expected_party_count(value: Any, *, project: dict, result: RenderResult) -> QACheck:
    actual = result.raw.get("party_count")
    return QACheck(
        name="expected_party_count",
        passed=actual == int(value),
        detail=f"declared {value}, actual {actual!r}",
    )


def _check_expected_dimensions(value: Any, *, project: dict, result: RenderResult) -> QACheck:
    declared = tuple(int(v) for v in value)
    passed = result.qa.dimensions == declared and all(
        (slide.width, slide.height) == declared for slide in result.slides
    )
    return QACheck(
        name="expected_dimensions",
        passed=passed,
        detail=f"declared {declared}, contract dimensions {result.qa.dimensions}",
    )


def _check_require_no_text_clipping(value: Any, *, project: dict, result: RenderResult) -> QACheck:
    if not value:
        return QACheck(name="require_no_text_clipping", passed=True, detail="not required")
    rows = _read_qa_summary_csv(result)
    if rows is None:
        return QACheck(
            name="require_no_text_clipping",
            passed=False,
            detail="required but no qa_summary.csv found at output_root to verify against",
        )
    clipped = [
        row
        for row in rows
        if row.get("no_category_clipping") == "False" or row.get("no_value_clipping") == "False"
    ]
    return QACheck(
        name="require_no_text_clipping",
        passed=not clipped,
        detail="all rows clean" if not clipped else f"{len(clipped)} row(s) flagged clipping",
    )


def _check_require_no_text_truncation(value: Any, *, project: dict, result: RenderResult) -> QACheck:
    if not value:
        return QACheck(name="require_no_text_truncation", passed=True, detail="not required")
    rows = _read_qa_summary_csv(result)
    if rows is None:
        return QACheck(
            name="require_no_text_truncation",
            passed=False,
            detail="required but no qa_summary.csv found at output_root to verify against",
        )
    truncated = [row for row in rows if row.get("no_label_truncation") == "False"]
    return QACheck(
        name="require_no_text_truncation",
        passed=not truncated,
        detail="all rows clean" if not truncated else f"{len(truncated)} row(s) flagged truncation",
    )


def _check_require_glossary(value: Any, *, project: dict, result: RenderResult) -> QACheck:
    if not value:
        return QACheck(name="require_glossary", passed=True, detail="not required")
    has_glossary = any(Path(slide.path).stem.endswith("glossary") for slide in result.slides)
    return QACheck(name="require_glossary", passed=has_glossary, detail="glossary slide present" if has_glossary else "no slide id ends in 'glossary'")


def _check_require_asset_registry(value: Any, *, project: dict, result: RenderResult) -> QACheck:
    actual = result.raw.get("run_manifest", {}).get("party_asset_registry") or result.raw.get("party_asset_registry")
    return QACheck(
        name="require_asset_registry",
        passed=actual == value,
        detail=f"declared {value!r}, actual {actual!r}",
    )


def _check_require_publication_disabled(value: Any, *, project: dict, result: RenderResult) -> QACheck:
    if not value:
        return QACheck(name="require_publication_disabled", passed=True, detail="not required")
    # Already a hard invariant of RenderResult.__post_init__ (contract.py) —
    # re-checked here explicitly so it shows up in the declarative QA report
    # a new project inherits, not just as an opaque constructor guarantee.
    return QACheck(
        name="require_publication_disabled",
        passed=result.publication_enabled is False,
        detail=f"publication_enabled={result.publication_enabled!r}",
    )


def _check_require_review_state(value: Any, *, project: dict, result: RenderResult) -> QACheck:
    return QACheck(
        name="require_review_state",
        passed=result.review_state == value,
        detail=f"declared {value!r}, actual {result.review_state!r}",
    )


def _check_require_source_footer(value: Any, *, project: dict, result: RenderResult) -> QACheck:
    if not value:
        return QACheck(name="require_source_footer", passed=True, detail="not required")
    # Honest limitation: neither this checker nor the bespoke workflow it
    # replaces (instagram_polling_factory_render.yml) verifies the footer's
    # pixels are actually present — that would need OCR, out of scope here.
    # This only checks the adapter's own declaration is internally
    # consistent, which is the same level of rigor the current system has.
    declared_in_raw = (result.raw.get("qa") or {}).get("source_footer_required")
    return QACheck(
        name="require_source_footer",
        passed=declared_in_raw is True,
        detail=(
            "adapter declares source_footer_required=True (pixel presence not independently "
            "verified by this checker, matching the current bespoke workflow's coverage)"
            if declared_in_raw is True
            else f"adapter's own qa.source_footer_required={declared_in_raw!r}, expected True"
        ),
    )


_CHECKERS: dict[str, Callable[..., QACheck]] = {
    "expected_slide_count": _check_expected_slide_count,
    "expected_party_count": _check_expected_party_count,
    "expected_dimensions": _check_expected_dimensions,
    "require_no_text_clipping": _check_require_no_text_clipping,
    "require_no_text_truncation": _check_require_no_text_truncation,
    "require_glossary": _check_require_glossary,
    "require_asset_registry": _check_require_asset_registry,
    "require_publication_disabled": _check_require_publication_disabled,
    "require_review_state": _check_require_review_state,
    "require_source_footer": _check_require_source_footer,
}


def run_declarative_checks(project: dict[str, Any], result: RenderResult) -> list[QACheck]:
    """Evaluate every `qa:` key declared in `project.yml` against `result`.

    Returns one QACheck per declared key, in declaration order. Does not
    raise — callers decide what to do with a failing check (the generic
    workflow, per plan §3.4, should treat any failure as a hard stop before
    publishing a preview, exactly as the existing bespoke workflows already
    hard-fail on a failed assert).
    """
    declared = project.get("qa") or {}
    checks: list[QACheck] = []
    for key, value in declared.items():
        checker = _CHECKERS.get(str(key))
        if checker is None:
            checks.append(QACheck(name=str(key), passed=True, detail=f"{_UNVERIFIED_DETAIL} (declared value: {value!r})"))
            continue
        checks.append(checker(value, project=project, result=result))
    return checks


def qa_with_declarative_checks(project: dict[str, Any], result: RenderResult) -> RenderResult:
    """Convenience wrapper: run the declarative checks and return a new
    RenderResult with result.qa.checks populated. RenderResult is frozen, so
    this rebuilds it rather than mutating in place."""
    from dataclasses import replace

    checks = tuple(run_declarative_checks(project, result))
    return replace(result, qa=replace(result.qa, checks=checks))
