# Validation review improvement plan

## Goal

Make project validation faster to review, less repetitive, and more reliable while preserving a complete audit trail.

## Principles

- Use current real production data first.
- Search historical production data before synthetic fallback or waiver.
- Keep synthetic contract-edge tests separate and non-publishable.
- Show each unique rendered result once in the summary review.
- Preserve every scenario and decision in the audit output and manifests.
- Fail automatically on measurable layout/rendering defects; reserve human review for design and factual judgement.

## Phase 1 — summary and audit review outputs

**Status: completed**

Implemented:

- concise deduplicated `validation_contact_sheet.png`
- complete `validation_audit_contact_sheet.png`
- representative cover shown once
- legacy minimum/maximum aliases omitted from the summary
- identical visuals grouped by SHA-256
- compact waiver section
- full technical detail retained in JSON

## Phase 2 — cover and title simplification

**Status: completed for party and constituency production projects**

Implemented:

- duplicate names removed from cover media
- small footer/status copy removed
- large useful metrics added
- titles shortened to `Top issues: <item>`
- party and constituency cover assets standardized to the tall media-slot aspect ratio

## Phase 3 — historical real-data fallback

**Status: completed**

Implemented:

- current production data searched first
- up to 12 recent historical production batches searched second
- batch, item, source-key, and search-stage provenance recorded
- synthetic contract-edge data not generated for convenience
- waiver only after current and historical searches fail

Live evidence:

- eight historical batches loaded
- zero current waivers replaced
- three scenarios remained waived

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

## Phase 6 — regression tests and documentation

**Status: completed**

Completed:

- generic-core regression coverage
- party production-project regression coverage
- constituency production-project regression coverage
- historical fallback precedence and waiver coverage
- image-dimension, media-fill, scenario-coverage, contact-sheet, and text-bound checks
- party and constituency decision records updated
- canonical factory plan status updated
- successful clean local and live S3 validation
- validation evidence artifact uploaded

## Delivery status

1. Summary/audit redesign — completed
2. Party and constituency cover/title cleanup — completed
3. Historical fallback — completed
4. Utilization checks — completed
5. Renderer readability and measured text bounds — completed for active layout/visual
6. Regression and documentation — completed
