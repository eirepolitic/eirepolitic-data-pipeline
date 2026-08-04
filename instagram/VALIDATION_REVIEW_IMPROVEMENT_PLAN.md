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

**Status: in progress**

Implement:

- make `validation_contact_sheet.png` a concise summary sheet
- show the cover once, preferably from `real_example`
- show only the visual slide for visual-shape scenarios
- remove legacy `minimum`/`maximum` rows when `item_count_min`/`item_count_max` exist
- group scenarios that produce the same visual hash
- display scenario coverage as badges on one unique render
- move waived scenarios into one compact summary section
- reduce visible metadata to source, status, tested property, and key metrics
- retain full technical details in JSON
- create a separate `validation_audit_contact_sheet.png` containing all scenario outputs

Acceptance:

- the summary contains no duplicate rendered chart
- the cover appears once
- every required scenario is represented by a render group or waiver
- summary and audit manifests record scenario-to-render grouping
- generated artifacts remain non-publishable

## Phase 2 — cover and title simplification

Implement:

- remove duplicated party/constituency names from cover media when already present as the slide title
- remove small draft/status/footer-like text from cover media
- use the cover media area for large useful metrics
- shorten long visual titles where needed
- keep detailed meaning in alt text and manifests

Acceptance:

- no duplicated headline text within one slide
- no unreadably small cover copy
- title remains within its configured line and font-size limits

## Phase 3 — historical real-data fallback

Implement:

- define historical source discovery for each production adapter
- search current production data first
- search historical batches second
- record selected batch, period, item key, and source path
- permit synthetic contract-edge data only for recurring/future conditions allowed by the metric contract
- waive only after current and historical searches fail

Acceptance:

- every scenario manifest records the search stages attempted
- a waiver records why current and historical data were insufficient
- no synthetic scenario is created for convenience

## Phase 4 — automated whitespace and content-utilization checks

Implement:

- calculate the occupied-content bounding box for final slides
- calculate media-slot utilization
- calculate chart plot-area utilization
- warn or fail on excessive top/bottom whitespace
- detect letterboxing caused by incompatible source and slot aspect ratios
- record thresholds in the relevant catalogue profile

Acceptance:

- the previously observed vertical letterboxing would fail automatically
- thresholds are visual/layout specific and machine-readable

## Phase 5 — renderer readability checks

Implement:

- minimum final font-size validation
- title wrap and shrink reporting
- category-label clipping checks
- value-label clipping checks
- minimum bar thickness
- dynamic label wrapping and sizing based on actual labels
- duplicate-render detection as a required validation check

Acceptance:

- clipping or undersized text fails validation before human review
- all renderer warnings appear in the summary manifest

## Phase 6 — regression tests and documentation

Implement:

- unit tests for deduplication, aliases, waivers, cover selection, and pagination
- image-dimension and scenario-coverage tests
- fixture cases for identical hashes and historical fallback
- update the canonical factory plan and project decisions
- generate a fresh live S3 validation pack

Acceptance:

- generic, party, and constituency regression suites pass
- live S3 validation succeeds
- the review artifact contains both summary and audit outputs

## Delivery order

1. Phase 1 summary/audit redesign
2. Phase 2 active party cover cleanup
3. Phase 4 basic utilization checks
4. Phase 5 renderer checks
5. Phase 3 historical source fallback
6. Phase 2 constituency/general cover cleanup
7. Phase 6 final regression and documentation

Historical fallback is scheduled after the immediate review-usability and automated-layout work because it requires adapter-specific source-contract design.
