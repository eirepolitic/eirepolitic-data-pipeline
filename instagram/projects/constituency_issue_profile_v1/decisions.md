# Constituency issue profile decisions

## Grain

Use `constituency` as the granularity. Each current constituency receives one two-slide draft post set.

## Slide sequence

1. Cover slide titled with the constituency name and a tall metric card showing current TDs, classified speeches, and displayed issue categories.
2. Horizontal bar chart titled `Top issues: <constituency>` showing the top seven classified issues.

Both slides use `title_text_media_v1`. The issue chart reuses `horizontal_bar_draft_v1`.

The cover media does not repeat the constituency name and contains no small footer/status copy.

## Validation policy

- Validation is real-data-first.
- Every required visual scenario first searches the current production batch.
- A missing current scenario searches up to 12 recent historical unified-model batches.
- Synthetic contract-edge data is not generated for convenience.
- A scenario is waived only after current and searched historical production data contain no qualifying real case.
- Every scenario records the full decision path: current real, historical real, synthetic contract edge, and waiver.
- Legacy `minimum` and `maximum` remain compatibility aliases only; the review matrix uses visual-specific scenario names.
- All validation outputs remain `no_publication` until explicit approval.

## Review outputs

- `validation_contact_sheet.png` is the deduplicated approval summary.
- The representative cover appears once.
- Identical chart renders are grouped by hash and labelled with all scenarios covered.
- Waivers appear in one compact block.
- `validation_audit_contact_sheet.png` preserves all scenario evidence.
- Full source, batch, selection, and quality details remain in JSON manifests.

## Automated quality gates

Validation fails before human review when declared thresholds are breached for:

- slide whitespace and occupied height
- media-slot vertical and area fill
- chart plot utilization
- category, value, and axis font sizes
- bar thickness
- label wrapping
- value-label right-edge headroom

A dedicated constituency regression test verifies the same quality contract used by the party project.

## Batch policy

- Generate one deterministic two-slide post set for every constituency returned by the active production dataset.
- Preserve a stable run ID derived from project version, source batch ID, and Git commit.
- Write run, item, slide, visual, and review-state manifests.
- Store each run under the existing project S3 prefix.
- Isolate item failures and retain partial results.
- Do not publish, schedule, or mark generated posts approved automatically.

## Recurring-generation cadence

Recurring readiness checks remain manual. The workflow may be started explicitly when a new draft check is needed, but no cron or automatic schedule is configured. Duplicate-batch prevention and review safeguards remain active.

## Data policy

Resolve unified compatibility keys through the production pointer, retain legacy fallback keys for current-source resolution, and record join coverage. Historical validation uses physical compatibility objects from prior unified-model batches.

## Scope boundary

The generic factory supports party and constituency production projects through adapter contracts. Automatic publishing, scheduling, approval, and recurring cadence remain disabled.
