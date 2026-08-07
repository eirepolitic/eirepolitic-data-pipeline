# Validation review improvement plan

## Goal

Make project validation faster to review, less repetitive, and more reliable while preserving a complete audit trail.

## Principles

- Use current real production data first.
- Search historical production data before synthetic fallback or waiver.
- Treat a current record that fails a hard scenario threshold as no match so historical fallback can run.
- Do not generate synthetic contract-edge data merely to fill the party/constituency validation sheet.
- Show every defined non-legacy scenario exactly once in the primary review as rendered or waived.
- Keep a separate deduplicated render summary for rapid visual comparison.
- Preserve every scenario, search stage, selection decision, and waiver reason in audit output and manifests.
- Fail automatically on measurable semantic/layout/rendering defects; reserve human review for design and factual judgement.

## Phase 1 — primary, summary, and audit review outputs

**Status: completed**

Implemented:

- primary two-column `validation_contact_sheet.png` showing every defined non-legacy scenario exactly once as rendered or waived
- separate deduplicated `validation_summary_contact_sheet.png`
- complete `validation_audit_contact_sheet.png`
- representative cover shown once
- legacy minimum/maximum aliases omitted from the primary scenario matrix
- identical visuals grouped by SHA-256 in the summary
- metric-first metadata on primary cards
- large 760x950 slide previews retained
- compact 360px waiver cards on separate compact rows
- two waiver cards per row, with an odd final waiver spanning both columns
- full technical detail retained in JSON/manifests

## Phase 2 — cover and title simplification

**Status: completed for party and constituency production projects**

Implemented:

- duplicate names removed from cover media
- small footer/status copy removed
- large useful metrics added
- titles shortened to `Top issues: <item>`
- party and constituency cover assets standardized to the tall media-slot aspect ratio

## Phase 3 — threshold-based real-data scenario selection and historical fallback

**Status: completed**

Implemented selection order:

1. current production real data
2. historical production real data
3. synthetic contract-edge only when explicitly allowed by policy
4. waiver

For the active party/constituency horizontal-bar validation matrix, synthetic data is not generated for convenience. A scenario is waived after current and searched historical production data contain no qualifying real case.

Hard semantic qualification:

- `item_count_min`: at most 3 displayed categories
- `item_count_max`: at least 6 displayed categories
- `labels_short`: longest displayed label at most 20 characters
- `labels_long`: longest displayed label at least 35 characters
- `values_small`: displayed maximum at most 20
- `values_large`: displayed maximum at least 1,000
- `values_tight`: relative spread at most 20%, including 0%
- `values_wide`: positive max/min ratio at least 5x
- `single_outlier`: top/second ratio at least 3x
- `ties`, `all_equal`, and `zeros`: exact semantic conditions
- `real_example`: at least 5 displayed categories

Historical fallback behavior:

- current production data searched first
- up to 12 recent historical production batches searched second
- a non-qualifying current candidate does not block historical fallback
- batch, item, source-key, search-stage, and replacement provenance recorded
- waiver only after current and historical searches fail

## Phase 4 — automated whitespace and content-utilization checks

**Status: completed**

Implemented:

- occupied-content bounding-box measurement
- media-slot vertical and area fill checks
- plot-area and plot-height checks
- excessive whitespace failure
- aspect-ratio letterboxing failure
- machine-readable layout and visual thresholds

## Phase 5 — renderer readability checks

**Status: completed for `title_text_media_v1` and `horizontal_bar_draft_v1`**

Implemented:

- adaptive category-label font sizing based on measured label width
- pixel-measured two-line label wrapping
- dynamic plot-left allocation
- adaptive value-label font sizing
- automatic value-axis headroom expansion
- direct category-label bounding boxes
- direct value-label bounding boxes
- zero-clipping enforcement for category and value text
- zero-unapproved-truncation enforcement
- title requested/actual/minimum font telemetry
- title shrink, wrap, truncation, and rendered-bounds telemetry
- title clipping and minimum-font enforcement
- minimum axis font size and bar thickness
- duplicate-render detection

Acceptance met:

- titles, category labels, and value labels fail validation when clipped
- text cannot shrink below declared minimums
- chart labels adapt to actual content complexity
- production and fixture renders record measured text bounds in manifests

Scope note:

Equivalent measured-text logic must be defined separately for future visual families. Completion here applies to the active title layout and horizontal-bar renderer.

## Phase 6 — regression, workflow gates, and evidence packaging

**Status: completed**

Completed:

- generic-core regression coverage
- party production-project regression coverage
- constituency production-project regression coverage
- historical fallback precedence and waiver coverage
- shared party/constituency semantic-contract coverage
- explicit semantic threshold tests for item count, labels, small/large/tight/wide values, outlier, ties/equality/zero, and representative density
- image-dimension, media-fill, scenario-coverage, contact-sheet, and text-bound checks
- compact waiver-height, row-isolation, pairing, and final-span regression coverage
- workflow gates for semantic thresholds and exact semantic conditions
- workflow gates for primary-sheet scenario completeness and waiver-row layout
- verified evidence export containing primary, summary, audit, contact-sheet manifest, live validation JSON, and verification JSON
- party and constituency decision records updated
- successful clean live S3 validation and artifact upload

## Latest verified validation

- workflow run: `31200674855`
- validated SHA: `cf6ff63ea4662418266dd71b873442251ea11c58`
- artifact: `verified-dense-two-column-contact-sheet-31200674855`
- artifact ID: `9002709884`

The artifact exists and was uploaded by that workflow run. All required generic, historical, semantic, constituency, party, live S3, contact-sheet verification, evidence-export, and artifact-upload steps passed.

## Delivery status

1. Primary/summary/audit redesign — completed
2. Party and constituency cover/title cleanup — completed
3. Threshold-based semantic selection and historical fallback — completed
4. Utilization checks — completed
5. Renderer readability and measured text bounds — completed for active layout/visual
6. Regression, workflow gates, evidence packaging, and documentation — completed
