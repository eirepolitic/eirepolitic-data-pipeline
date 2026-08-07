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
- A non-qualifying current record is treated as no match and cannot block historical fallback.
- Every scenario records the full decision path: current real, historical real, synthetic contract edge, and waiver.
- Legacy `minimum` and `maximum` remain compatibility aliases only; the review matrix uses visual-specific scenario names.
- All validation outputs remain `no_publication` until explicit approval.

Horizontal-bar scenarios use the same hard semantic contract as the party project:

- `item_count_min`: at most 3 displayed categories
- `item_count_max`: at least 6 displayed categories
- `labels_short`: longest displayed category label at most 20 characters
- `labels_long`: longest displayed category label at least 35 characters
- `values_small`: displayed maximum value at most 20
- `values_large`: displayed maximum value at least 1,000
- `values_tight`: relative spread at most 20%, including 0% spread
- `values_wide`: positive max/min ratio at least 5x
- `single_outlier`: top/second ratio at least 3x
- `ties`: exact tied displayed values
- `all_equal`: exact all-equal displayed values
- `zeros`: exact displayed zero value
- `real_example`: at least 5 displayed categories

## Review outputs

- `validation_contact_sheet.png` is the primary two-column review sheet and shows every defined non-legacy scenario exactly once as rendered or waived.
- The representative cover appears once.
- Rendered scenario cards retain large 760x950 previews and metric-first metadata.
- Waived scenarios use separate compact 360px cards; two waivers share a row and an odd final waiver spans both columns.
- `validation_summary_contact_sheet.png` is the separate deduplicated render summary.
- `validation_audit_contact_sheet.png` preserves complete scenario-by-scenario evidence.
- `validation_contact_sheet_manifest.json`, live validation JSON, and scenario manifests retain source, batch, search-stage, selection, and quality details.

## Automated quality gates

Validation fails before human review when declared thresholds are breached for:

- semantic scenario qualification
- slide whitespace and occupied height
- media-slot vertical and area fill
- chart plot utilization
- title final font size, line count, truncation, or clipping
- dynamically selected category and value font sizes
- direct category-label text bounds
- direct value-label text bounds
- unsupported category-label truncation
- bar thickness
- label wrapping
- value-label right-edge headroom
- primary-sheet scenario completeness
- compact waiver-row isolation and final odd-waiver spanning
- required review evidence export

The horizontal-bar renderer measures text in pixels, selects a suitable label font and plot margin, and expands the value axis until value labels fit. All final text bounds and sizing decisions are retained in validation manifests.

A dedicated constituency regression test verifies the same quality and semantic contract used by the party project.

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
