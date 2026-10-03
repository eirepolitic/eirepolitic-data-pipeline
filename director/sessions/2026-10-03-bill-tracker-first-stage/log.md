# First Stage Bill Tracker development session — 2026-10-03

The task is to develop the First Stage variant of the merged `bill_tracker_factory_v1` Instagram series, beginning with a live inventory refresh and a single representative Bill-slide prototype. No publishing or scheduling is authorised.

## Repository review

Read the required Director operating documentation and inspected the merged Bill Tracker project, its current content, adapter, renderer, generic factory workflow, and the two completed Enacted-post summaries. Current repository truth confirms `bill_tracker_series` is merged and the existing factory must be extended rather than replaced.

The canonical Enacted family establishes the 1080×1350 geometry, dark green/cream/gold palette, corner ornaments, bounded title fitting, source/footer treatment, shared-fit comparable typography, and left-to-right legislative-process timeline. The current adapter is Enacted-specific: it only accepts `post1`/`post2`, requires exactly three Bills, emits two slides per Bill, and hard-codes nine slides. Those assumptions will need to become period-specific for First Stage.

## Data-source status

The public Irish Politics Data Model page currently documents the older September production snapshot, so it cannot by itself establish the live 3 October inventory. The merged factory resolves the validated production batch from S3 via `instagram/factory/oireachtas_source.py`, using `processed/oireachtas_unified/pointers/production.json`, and validates the referenced immutable batch manifest before loading tables.

Current official Oireachtas Bill pages will be used to cross-check the selected Bills before any First Stage label is treated as editorially current.

## Current status

A session branch has been created at `sly/session/2026-10-03-bill-tracker-first-stage`. The next steps are to resolve the current production batch and stage inventory, reconcile it against live Oireachtas pages, choose a prototype Bill based only on data/layout complexity, then add one review-only First Stage prototype period to the existing factory and render it through the generic hosted-review workflow. Publication controls remain disabled.
