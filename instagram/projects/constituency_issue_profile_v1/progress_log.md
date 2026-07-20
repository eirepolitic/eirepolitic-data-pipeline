# Progress log

## Phase 2

The constituency pilot established deterministic minimum, maximum, and real-example complete-slide validation using the active Oireachtas production data.

Validation runs:

- `29703058481` — local complete-slide validation
- `29703087934` — initial live S3 validation
- `29703335986` — final live S3 validation
- `29704116144` — corrected scenario-contract validation

## Phase 3

Implemented deterministic constituency batch generation with complete slides, hashes, provenance, item/run manifests, review-state initialization, isolated failure handling, immutable S3 storage, and artifact upload.

Validation evidence:

- live batch workflow: `29711056766`
- artifact: `constituency-factory-batch-constituency_issue_profile_v1-v1-1cd892b3cf6f`
- S3 root: `s3://eirepolitic-data/processed/instagram_factory/projects/constituency_issue_profile_v1/`

## Phase 4

Implemented item/slide review states, append-only review history, immutable derived runs, and selective regeneration that preserves unaffected files byte-for-byte.

Validation workflow: `29711566533`.

## Phase 5

Implemented production readiness checks, duplicate-batch prevention, and manual recurring draft generation.

Validation workflow: `29711883200`.

The live source batch `current-government-backfill-20260716-1` was correctly blocked as already generated. Recurring execution remains manual.

## Phase 6 — review index and ready gate

Implemented:

- static HTML review index for every generated item and slide
- review index manifest and status counts
- `build-review-index` CLI command
- strict `mark-ready` CLI command
- exact blocking reasons for every unapproved item or slide
- audited reviewer, timestamp, and note when a run is marked ready
- `ready_for_posting` state in run and review manifests
- publishing remains `false` even after the ready gate passes

Validation evidence:

- Phase 1–6 workflow: `29755339072`
- catalogue validation: passed
- project validation: passed
- all Phase 1–6 tests: passed
- unreviewed runs are blocked from ready status
- fully approved runs can be marked ready
- ready status does not enable publishing

## Current status

The constituency Content Factory v1 workflow is implemented from project definition through review, selective regeneration, recurring readiness, review indexing, and an auditable ready-for-posting checkpoint.

Automatic Instagram publishing, social scheduling, automatic approval, and automatic recurring cadence remain disabled and outside this completed v1 workflow.
