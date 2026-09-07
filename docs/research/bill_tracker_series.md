# Bill tracker Instagram series

Status: **prototype validated against production; editorial test draft in progress**  
Date: **6 September 2026**

## Objective

Create a repeatable EirePolitic content product that periodically explains Bills which are newly introduced, have moved stage, or have become enacted. The same deterministic dataset should be reusable for Instagram, Appsmith, Power BI or future editorial surfaces.

The social format is six Bills per carousel. Each Bill gets one overview card. Detailed single-Bill posts can follow where audience interest warrants them.

## Important correction from live production

The earlier exploratory 45-Bill view was not the full production universe. Resolving the current production pointer and rebuilding from the active immutable batch returned **406 Bills**:

- Current: **259**
- Enacted: **56**
- Lapsed: **68**
- Defeated: **21**
- Withdrawn: **2**

An exhaustive current-Bill catalogue would therefore require roughly 43 six-Bill carousels and is not suitable as a recurring Instagram edition.

## Recommended recurring model

Use two layers:

1. **Full deterministic Bill snapshot** — one row per Bill, preserving its current source state and certified context.
2. **Editorial change digest** — on later runs, compare the new snapshot with the previous snapshot and select only new Bills or Bills whose deterministic state changed.

The deterministic state key is:

`status | latest stage | latest stage House | latest stage date`

The first run has no previous snapshot, so it can use a recent-activity lookback to seed the first edition. Later runs should use snapshot deltas rather than repeating every Bill still sitting at the same stage.

## Cadence tests

Two read-only production tests were run:

### 180-day baseline

- selected Current/Enacted Bills: **96**
- six-Bill carousel batches: **18**
- Second Stage: 45
- Enacted: 34
- Committee Stage: 10
- First Stage: 4
- Fifth Stage: 3

### 90-day baseline

- selected Current/Enacted Bills: **59**
- six-Bill carousel batches: **13**
- Enacted: 25
- Second Stage: 22
- Committee Stage: 7
- First Stage: 3
- Fifth Stage: 2

Both baselines remain too large to publish as a complete edition. This supports **quarterly snapshotting with delta-only editorial output** as the best starting design. A six-month cadence remains possible if the observed delta after several runs is small.

No automatic schedule is enabled yet. The workflow is manual until the first two snapshots establish real change volume.

## Stage grouping

Stage is the correct structural backbone, but the precise source House remains visible on every card because the same stage names can occur in both Houses.

Public editorial buckets:

- Enacted
- First Stage
- Second Stage
- Committee Stage
- Report Stage
- Fifth Stage
- Returned amendments

The source stage `Cream List` is presented publicly as **Returned amendments**. It describes amendments made by the second House being returned to the originating House for consideration. The original source stage remains stored in the dataset.

Terminal statuses such as Lapsed and Withdrawn are retained in the full snapshot but excluded from the core Current/Enacted tracker. They can support a separate occasional "Bills that stopped" explainer later.

## Reusable deterministic dataset

Prototype module: `political_metrics/bill_content_snapshot.py`

One row per Bill includes:

- Bill identifier, number, year and titles;
- source status;
- latest stage, date and House;
- originating House;
- source sponsor name/role/URI where available;
- sponsor attribution status;
- certified Bill-linked debate-section count;
- certified Bill-linked transcript-intervention count;
- certified Bill-linked division count;
- latest linked division metadata and recorded Tá/Níl/abstain counts;
- stable current-state key;
- explicit safety/status fields for editorial use.

The builder resolves all logical `latest` keys through the active production pointer so one run cannot mix datasets from different production batches.

## Editorial change layer

Prototype module: `political_metrics/bill_editorial_series.py`

Modes:

- `baseline_recent`: first edition; Current/Enacted Bills whose last event falls inside the requested lookback.
- `snapshot_delta`: subsequent editions; only new Bills or Bills whose deterministic state key differs from the previous snapshot.

The output is deterministically batched at six Bills per carousel within editorial stage/status buckets.

## Persistence

Runner: `process/build_bill_content_snapshot.py`

By default the runner is read-only and writes local artifacts only.

An optional `--state-prefix` supports durable editorial state in a separate S3 namespace. If enabled, the runner:

1. reads the prior `latest/bill_content_snapshot.csv` if it exists;
2. generates delta candidates;
3. validates the new snapshot and editorial series;
4. writes a dated snapshot plus `latest` snapshot under the supplied editorial prefix.

This does **not** alter the Oireachtas production pointer or political metric datasets. State persistence has been implemented but has not been enabled during this prototype investigation.

Recommended eventual prefix:

`processed/editorial/bill_tracker`

## Card content contract

The base Bill card should contain:

1. **Bill name**
2. **Status / stage + House**
3. **What it does** — one short sourced plain-English summary
4. **Introduced by** — exact source sponsor person or ministerial office; do not invent a person from an office label
5. **Main debate** — one short case-for point and one short concern/criticism where the certified debate record supports both
6. **Recorded vote** — only if the proposition being voted on can be identified and described accurately
7. **Source/date footer**

## Critical support/opposition rule

A speaker appearing in a Bill debate is **not** evidence that the speaker supports or opposes the Bill.

A Bill-linked division is also not automatically the final vote on the Bill. It may concern an amendment, stage motion or another proposition. For example, the current Israeli-settlements Bill sample has a linked 67–79 division whose proposition is an amendment; those numbers must not be presented as overall support/opposition to the Bill.

Therefore:

- never infer supporters/detractors from speech participation;
- never label a vote as Bill support/opposition until its proposition/stage is certified;
- where proposition certification is unavailable, show only a neutral debate summary or the number of recorded linked divisions;
- final-passage-without-division should be stated as such only from an explicit source.

A future enhancement should materialize **all Bill-linked divisions with proposition/stage labels**, rather than relying on the latest linked division alone.

## Validation completed

Focused unit tests cover:

- latest-stage selection;
- six-Bill batching;
- House preservation;
- `Cream List` public relabel;
- terminal status bucketing;
- no support/opposition inference without certified vote evidence;
- baseline recent filtering;
- snapshot-delta selection.

Read-only GitHub validation runs included:

- `34052215919` — full live snapshot after production-pointer resolver fix;
- `34052387775` — 180-day scoped tracker;
- `34052495878` — 90-day scoped tracker.

No production data changed and no classifier calls were made.

## First test carousel

For the design/content prototype, use the first six most recently enacted Bills returned by the 90-day run:

1. Development (Strategic Gas Reserve) Bill 2026
2. Israeli Settlements in the Occupied Palestinian Territory (Prohibition of Importation of Goods) Bill 2026
3. Criminal Law, Civil Law and Defence (Miscellaneous Provisions) Bill 2026
4. Housing and Residential Tenancies (Miscellaneous Provisions) Bill 2026
5. Health (Provision of Contraception Prescribing Service in Retail Pharmacy Businesses) Bill 2026
6. Regulation of Artificial Intelligence Bill 2026

These provide a useful test because they span energy, foreign affairs/trade, justice/defence, housing, health and AI regulation, while all sharing the same clear terminal status.

## Living next-step plan

1. Finish and editorially review the six-Bill enacted test carousel.
2. Add proposition/stage classification for Bill-linked divisions before any recurring "supporters vs detractors" treatment.
3. Restore the temporary validation workflow changes and remove diagnostic files before opening the feature PR.
4. Keep the permanent `bill_content_snapshot.yml` workflow manual initially.
5. After the first content review, decide whether to enable the separate S3 editorial state prefix.
6. Capture a second snapshot before choosing 3-month versus 6-month automation; use the observed delta count, not an assumed cadence.
7. If approved, schedule the workflow and persist each audited snapshot so future editions automatically contain only changed Bills.
