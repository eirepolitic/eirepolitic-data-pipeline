# Bill tracker Instagram series

Status: **prototype validated against production; two-post enacted prototype in visual review**  
Date: **26 September 2026**

## Objective

Create a repeatable EirePolitic content product that periodically explains Bills which are newly introduced, have moved stage, or have become enacted. The same deterministic dataset should be reusable for Instagram, Appsmith, Power BI or future editorial surfaces.

The approved social format for the current prototype is **two posts with three Bills each**. Each Bill receives **two consecutive slides**: a plain-English explainer followed immediately by the exact linked vote breakdown. Each post therefore contains eight slides: cover, three explainer/vote pairs, and methodology/sources.

## Important correction from live production

The earlier exploratory 45-Bill view was not the full production universe. Resolving the current production pointer and rebuilding from the active immutable batch returned **406 Bills**:

- Current: **259**
- Enacted: **56**
- Lapsed: **68**
- Defeated: **21**
- Withdrawn: **2**

An exhaustive current-Bill catalogue is not suitable as a recurring Instagram edition.

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
- six-Bill data batches: **18**
- Second Stage: 45
- Enacted: 34
- Committee Stage: 10
- First Stage: 4
- Fifth Stage: 3

### 90-day baseline

- selected Current/Enacted Bills: **59**
- six-Bill data batches: **13**
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

The deterministic layer may batch six Bills internally, but the approved Instagram production format now splits each six-Bill editorial set into **two three-Bill posts**.

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

## Instagram post contract

Each three-Bill post uses this fixed order:

1. Cover — `Bills of the Current Session`, category/part identifier, short explanation, and exact three-Bill list.
2. Bill 1 explainer.
3. Bill 1 vote breakdown.
4. Bill 2 explainer.
5. Bill 2 vote breakdown.
6. Bill 3 explainer.
7. Bill 3 vote breakdown.
8. Methodology / sources.

### Explainer slide contract

The explainer must work for a reader with no assumed knowledge of Irish politics or parliamentary procedure. It should explain:

- what the Bill does in plain English;
- practical effects;
- who introduced it where useful;
- why supporters argued for it;
- concerns or arguments raised against it;
- what the specific next-slide vote was deciding;
- what a Tá meant;
- what a Níl meant;
- what carrying or defeating that proposition did to the Bill.

### Vote slide contract

The vote slide must show the exact proposition/stage/date, overall 100% stacked result, Tá, Níl, abstain where present, **no recorded vote** as a separate category, date-correct party rows, aligned numerical columns, and smaller groups below.

`No recorded vote` must not be relabelled as `absent` without separate evidence.

## Critical support/opposition rule

A speaker appearing in a Bill debate is **not** evidence that the speaker supports or opposes the Bill.

A Bill-linked division is also not automatically the final vote on the Bill. It may concern an amendment, stage motion or another proposition. For example, the current Israeli-settlements Bill sample has a linked 67–79 division whose proposition is amendment No. 16; those numbers must not be presented as overall support/opposition to the Bill.

Therefore:

- never infer supporters/detractors from speech participation;
- never label a vote as Bill support/opposition until its proposition/stage is certified;
- where proposition certification is unavailable, show only a neutral debate summary or the number of recorded linked divisions;
- final-passage-without-division should be stated as such only from an explicit source.

A future enhancement should materialize **all Bill-linked divisions with proposition/stage labels**, rather than relying on the latest linked division alone.

## Strategic Gas Reserve verified vote context

For the **Development (Strategic Gas Reserve) Bill 2026**, the 30 June 2026 Dáil division is suitable for a passage framing because the Chair put a combined question covering the remaining sections, Title, Fourth Stage and passage of the Bill.

Editorial interpretation for this specific division:

- **Tá** = pass the Bill through the Dáil in the form then before the House and send it onward;
- **Níl** = reject that passage motion;
- result = **90 Tá, 57 Níl, 27 no recorded vote, 174 eligible TDs**;
- effect = the motion carried and the Bill moved to the Seanad;
- the Bill was subsequently enacted on **23 July 2026**.

The explainer copy should remain simple. The current visual-review wording explains that TDs are members of the Dáil and that the Seanad is Ireland's second parliamentary chamber rather than assuming those terms are already understood.

## Validation completed

Focused unit tests cover:

- latest-stage selection;
- six-Bill deterministic batching;
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

## Current enacted prototype split

### Post 1 — Enacted · Part 1

1. Development (Strategic Gas Reserve) Bill 2026
2. Israeli Settlements in the Occupied Palestinian Territory (Prohibition of Importation of Goods) Bill 2026
3. Criminal Law, Civil Law and Defence (Miscellaneous Provisions) Bill 2026

### Post 2 — Enacted · Part 2

1. Housing and Residential Tenancies (Miscellaneous Provisions) Bill 2026
2. Health (Provision of Contraception Prescribing Service in Retail Pharmacy Businesses) Bill 2026
3. Regulation of Artificial Intelligence Bill 2026

## Living next-step plan

1. Finish visual review of the Strategic Gas Reserve explainer while preserving the already-approved B3-derived vote layout.
2. Build the Israeli Settlements pair next; label the 7 July 2026, 67–79 division explicitly as **amendment No. 16**, not a final Bill passage vote.
3. Build the Criminal Law, Civil Law and Defence pair.
4. Complete Post 1 cover and methodology slide using the approved recurring factory visual treatment.
5. After Post 1 is stable, repeat the same two-slide-per-Bill pattern for Bills 4–6 in Post 2.
6. Migrate the reviewed slide-specific work into `instagram/projects/bill_tracker_factory_v1/` and keep publication disabled with `pending_human_review` until explicit approval.
7. Keep the permanent `bill_content_snapshot.yml` workflow manual initially; capture a second snapshot before choosing an automated cadence.
8. If recurring output is approved, persist audited snapshots under the separate editorial state prefix so later editions can focus on changed Bills only.
