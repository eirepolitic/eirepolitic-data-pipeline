# Decisions

## Pilot grain

Use `constituency` as the first Phase 2 grain.

## Slide sequence

Use two slides:

1. generated constituency cover card;
2. classified issue ranking chart.

Both slides use the existing `title_text_media_v1` layout. The issue chart reuses `horizontal_bar_draft_v1`.

## Scenario policy

- Minimum and maximum are synthetic layout and regression tests and cannot be published.
- Synthetic scenarios only need to render correctly without clipping, overflow, missing bindings, or invalid provenance; they are not visual-design approval candidates.
- The real example is internally consistent actual data selected by median complexity.
- Human visual and factual review applies to the real example.
- All scenarios remain `no_publication` until explicit approval.

## Review decision — 2026-07-19

The minimum and maximum outputs are accepted as automated QA fixtures. No further aesthetic polishing is required unless they expose a rendering defect. Visual refinement and approval will focus on the real-data example.

The reusable constituency design is approved as the Phase 3 batch baseline. Approval does not approve individual generated posts for publication; every generated item remains `unreviewed`, `approved: false`, and `publishing_allowed: false`.

## Batch policy

- Generate one deterministic two-slide post set for every constituency returned by the active production dataset.
- Preserve a stable run ID derived from project version, source batch ID, and Git commit.
- Write run, item, slide, visual, and review-state manifests.
- Store each run under the existing project S3 prefix.
- Isolate item failures and retain partial results.
- Do not publish, schedule, or mark generated posts approved automatically.

## Recurring-generation cadence — 2026-07-20

Recurring readiness checks remain manual. The workflow may be started explicitly when a new draft check is wanted, but no cron or automatic schedule will be configured. Duplicate-batch prevention and all review-only safeguards remain active.

## Data policy

Resolve unified compatibility keys through the production pointer, retain legacy fallback keys, and record join coverage in the project manifest.

## Scope boundary

Phases 1–5 are implemented for this constituency project. Generic multi-project orchestration remains future work. Automatic publishing, scheduling, approval, and recurring cadence remain disabled.
