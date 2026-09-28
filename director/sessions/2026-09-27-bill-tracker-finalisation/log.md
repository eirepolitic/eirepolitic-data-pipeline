# Bill Tracker finalisation session

Started from current `main` after reading the Director guide and required canonical references. `director/refs.yml` still classifies `bill_tracker_series` as draft, while the standard Instagram publisher is a separate production path. The factory therefore remains review-only and must not be modified to publish directly.

The human has already approved the two nine-slide review carousels hosted on `previews/bill-tracker-full-posts-review` with the statement: “These look good and are ready to post.” Those assets are the visual/editorial migration target. They are not themselves the production artifact and will not be published directly.

PR #130 (`feature/bill-content-snapshot-v1`) is still open and draft. It contains the reusable Bill Tracker project and older factory work, but its own description records an earlier eight-slide validation run. It must be inspected and reconciled with the later approved review rather than merged unchanged.

A Director session branch was created at `sly/session/2026-09-27-bill-tracker-finalisation`. Next steps are to inspect the existing Bill Tracker project and temporary review implementation, migrate the approved design/content into the generic factory with publication flags disabled, revalidate vote semantics/data against production and official records, render and QA both carousels through `instagram_factory_render.yml`, host the exact production review, and obtain explicit approval of that exact artifact before caption finalisation and scheduling.
