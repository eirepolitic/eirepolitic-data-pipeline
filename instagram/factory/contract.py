"""Normalized render-result contract (EirePolitic Director, Phase 3, §3.2).

Every project adapter's `generate()` returns its own ad-hoc dict shape today
(see the implementation plan §1.4 for the documented divergence between
`party_issue_monthly_profile_v2` and `ipi_polling_factory_v1`). Neither
adapter is rewritten and neither changes what it renders — this module only
defines the common shape a *shim* (see `instagram/factory/normalize.py`)
translates each adapter's raw return value into, so that generic downstream
code (QA, the review page, the generic workflow) can be written once against
one contract instead of once per project.

`party_issue_monthly_profile_v2/adapter.py`, `project.yml`, and
`instagram/factory/recurring.py` are all frozen and byte-identity-checked
against commit 386b933 by `.github/workflows/director_factory_v1_identity_ci.yml`
(see plan §1.8/§1.10). Nothing in this module or its siblings edits those
files; normalization happens entirely on the *output side*, after
`recurring.run_project()` has already run and already enforced its own
`publication_enabled is False` / `review_state == "pending_human_review"`
guardrails.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SlideRef:
    """One rendered slide image."""

    id: str
    path: str
    width: int | None = None
    height: int | None = None


@dataclass(frozen=True)
class ContactSheetRef:
    """One contact-sheet / overview image."""

    id: str
    path: str


@dataclass(frozen=True)
class QACheck:
    """One declarative QA check result (see `instagram/factory/qa.py`)."""

    name: str
    passed: bool
    detail: str = ""


@dataclass(frozen=True)
class QASummary:
    expected_slide_count: int
    actual_slide_count: int
    dimensions: tuple[int, int] | None = None
    checks: tuple[QACheck, ...] = field(default_factory=tuple)

    @property
    def slide_count_ok(self) -> bool:
        return self.actual_slide_count == self.expected_slide_count

    @property
    def all_passed(self) -> bool:
        return self.slide_count_ok and all(check.passed for check in self.checks)

    def failed_checks(self) -> list[QACheck]:
        failures = [check for check in self.checks if not check.passed]
        if not self.slide_count_ok:
            failures.append(
                QACheck(
                    name="slide_count",
                    passed=False,
                    detail=f"expected {self.expected_slide_count}, got {self.actual_slide_count}",
                )
            )
        return failures


@dataclass(frozen=True)
class RenderResult:
    """The normalized output of one project render, regardless of which
    adapter produced it. Built by a `normalize.py` shim, never constructed
    directly by an adapter."""

    project_id: str
    period_key: str
    output_root: str
    slides: tuple[SlideRef, ...]
    contact_sheets: tuple[ContactSheetRef, ...]
    caption_path: str | None
    manifest_path: str | None
    package: dict[str, Any] | None
    qa: QASummary
    review_state: str
    publication_enabled: bool
    source_batch_id: str | None = None
    # The adapter's original, untouched return value. Kept so that
    # project-specific declarative QA checks (instagram/factory/qa.py) and
    # the review page can pull fields the normalized shape doesn't carry
    # (e.g. party's per-slide clipping/truncation flags, polling's trend
    # waves) without a second call into the adapter.
    raw: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # These two invariants are the same ones instagram/factory/recurring.py
        # already hard-fails on for the raw adapter result (see plan §1.4) —
        # re-asserted here so nothing can silently normalize a bad result into
        # a contract that claims to be safe.
        if self.review_state != "pending_human_review":
            raise ValueError(
                f"RenderResult.review_state must be 'pending_human_review' for "
                f"{self.project_id!r}, got {self.review_state!r}"
            )
        if self.publication_enabled is not False:
            raise ValueError(
                f"RenderResult.publication_enabled must be False for {self.project_id!r}, "
                f"got {self.publication_enabled!r}"
            )
        if not self.slides:
            raise ValueError(f"RenderResult for {self.project_id!r} has no slides")
