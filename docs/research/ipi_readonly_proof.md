# Irish Polling Indicator read-only ingestion proof

## Purpose

Run a bounded, read-only feasibility proof for the Irish Polling Indicator (IPI), the highest-ranked polling source from the Irish political data-source investigation.

This is not a production implementation. No production architecture, schemas, pipelines, infrastructure, secrets, jobs, or production data were changed.

## Final conclusion

**Status: Ready for a later bounded non-production ingestion proof.**

IPI provides two directly machine-readable development datasets with no authentication and no scraping:

- `data_polls.csv` — raw published polls;
- `data_pollingindicator.csv` — daily modeled estimates and 95% credible intervals.

The current development release inspected was **Development Version: 2026-08-02**, Git commit `09f77f03654bf66551a56fb02a11a242ae9bcf51`.

The source is operationally straightforward, but a future loader must preserve three distinctions:

1. raw polls versus modeled IPI estimates;
2. mutable development data versus stable DOI releases;
3. raw polling percentages (`0–100`) versus modeled proportions (`0–1`).

## Current source structure

Repository: https://github.com/Irish-Polling-Indicator/ipi-data

Current repository files include:

- `README.md`;
- `codebook_ipi-data.pdf`;
- `data_polls.csv`;
- `data_polls.xlsx`;
- `data_polls.dta`;
- `data_polls.rds`;
- `data_pollingindicator.csv`;
- `data_pollingindicator.dta`;
- `data_pollingindicator.rds`.

For a later Python proof, CSV is the simplest source format. XLSX/Stata/R copies should not be independently ingested unless there is a specific validation need.

## 1. Raw polls — `data_polls.csv`

### Current size and coverage

- 773 CSV lines: one header plus **772 poll rows**.
- First poll publication date: **1982-10-30**.
- Latest poll publication date: **2026-08-02**.
- Latest row is an Ireland Thinks poll with fieldwork on **2026-07-31**, sample size **1,384**.
- Current file size shown by GitHub: approximately 80.5 KB.

### Exact current columns

The file currently has **23 columns**:

```text
date
date_start
date_end
date_middle
pollster
sample_size
FF
FG
SF
LAB
GP
PD
WP
DL
SPBP
RENUA
SD
AU
II
IND_OTH_IT
PREV_INDOTH_II
PREV_II
OTH_IND
```

### Field interpretation relevant to ingestion

- `date` is the poll publication/release date.
- `date_start` / `date_end` identify fieldwork period.
- `date_middle` is the midpoint used for plotting/analysis by IPI.
- `pollster` retains source pollster identity.
- `sample_size` is supplied for the poll.
- Party/result columns are raw poll percentages, generally expressed on a **0–100 percentage-point scale**.

The website explicitly plots individual polls at the middle of the fieldwork period rather than publication date. A future EirePolitic poll timeline should retain both dates rather than choosing one and discarding the other.

### Party-column history is structural, not ordinary missingness

The raw schema contains both current party columns and historical/transitional fields. `NA` is therefore expected when a party did not exist, was not separately reported, or was grouped differently in a historical period.

The current schema includes historical grouping fields such as:

- `IND_OTH_IT`;
- `PREV_INDOTH_II`;
- `PREV_II`;
- `OTH_IND`.

These should not be collapsed or filled automatically. Their exact codebook meaning should be preserved during any future normalization.

### Numeric edge cases visible in bounded inspection

- Poll percentages are not guaranteed to be integers; recent rows contain decimal values.
- One historical 1983 row exposes `OTH_IND=-1`. This proof does **not** label that value erroneous because it may be a documented historical/grouping adjustment. It should instead trigger a codebook-aware validation review.
- `NA` appears extensively in party columns and is often semantically valid.

A future parser must therefore support numeric floats and nullable party values. It must not enforce a naive rule that every party value is between 0 and 100 until the historical grouping/codebook semantics have been incorporated.

### Candidate source identity

Publication date alone is not unique: multiple pollsters can publish on the same date. A later proof should test uniqueness of a composite such as:

```text
pollster + date + date_start + date_end + sample_size
```

This is only a candidate source key. It must be verified against the full file rather than assumed.

## 2. Modeled estimates — `data_pollingindicator.csv`

### Current size and coverage

- 14,407 CSV lines: one header plus **14,406 daily estimate rows**.
- First modeled date: **1987-02-17**.
- Latest modeled date in the inspected development file: **2026-07-31**.
- Current GitHub file size: approximately 3.06 MB.

The latest raw poll is published on 2026-08-02 but has fieldwork midpoint/end on 2026-07-31, consistent with the modeled file ending on 2026-07-31 while the public website reports a 2026-08-02 update.

### Exact current columns

The modeled file currently has **41 columns**:

```text
date
cycle
FF, FF_lo, FF_hi
FG, FG_lo, FG_hi
LAB, LAB_lo, LAB_hi
PD, PD_lo, PD_hi
WP, WP_lo, WP_hi
OTH, OTH_lo, OTH_hi
GP, GP_lo, GP_hi
DL, DL_lo, DL_hi
SF, SF_lo, SF_hi
SD, SD_lo, SD_hi
SPBP, SPBP_lo, SPBP_hi
AU, AU_lo, AU_hi
II, II_lo, II_hi
```

For each tracked party/grouping:

- base field = modeled support estimate;
- `_lo` = lower 95% credible bound;
- `_hi` = upper 95% credible bound.

`cycle` identifies the relevant electoral cycle, for example current rows use `2024-`.

### Critical unit difference

Modeled values are proportions on a **0–1 scale**. Example: `0.2136` represents approximately 21.36%.

Raw `data_polls.csv` results use percentage points on a **0–100 scale**.

This is the most important ingestion trap found in the proof. A future normalized model must make the unit explicit or convert both sources consistently. It must never mix `21` and `0.21` in one metric field without normalization.

### Structural `NA` values

Parties not modeled in a historical cycle have `NA` for estimate/lower/upper fields. This is expected. Null validation must therefore be cycle-aware.

### Schema evolution risk

The current modeled schema includes Independent Ireland (`II`). IPI documentation around the post-2024 cycle shows that party classifications can change between electoral cycles. A future fetch must fingerprint the header and alert on added/removed party columns rather than assuming a permanent fixed party list.

## 3. Development versus stable versions

### Development version

The GitHub repository is explicitly the **development version**. IPI states that it is updated after new polls and can change over time. Historical modeled values within the current electoral term can be revised when new polling information is incorporated.

Current inspected development version:

- label: `Development Version: 2026-08-02`;
- commit date: 2026-08-03;
- full commit SHA: `09f77f03654bf66551a56fb02a11a242ae9bcf51`.

The commit added the 2026-08-02 raw Ireland Thinks poll and recalculated the modeled estimate file.

### Stable version

IPI also publishes a stable dataset in Harvard Dataverse:

- DOI: `10.7910/DVN/8YVVYX`;
- current documented stable citation: Louwerse and Müller (2025), Stable Version, V1.

New stable releases are published after an election cycle. IPI states that daily estimates in a stable release will not change because all polls in that electoral cycle are already included.

### Recommended future source-version fields

A later proof should capture at minimum:

```text
source_name
source_dataset
source_repository
source_version_type        # development | stable
source_version_label
source_commit_sha          # development
source_doi                 # stable
retrieved_at_utc
source_file
source_file_header_hash    # or equivalent schema fingerprint
```

For reproducible public historical graphics, prefer a stable DOI release when it covers the required period. For current recurring graphics, use the latest development snapshot but store its exact commit SHA and retrieval timestamp.

## 4. Bounded data-quality observations

This browser-based proof verified structure, row counts, endpoints, current version and representative historical/current values. It did **not** load the entire CSV into a local dataframe, so exact full-file duplicate/null counts were not computed.

Confirmed observations:

- raw file has 772 poll rows;
- modeled file has 14,406 daily rows;
- key source dates use ISO-style `YYYY-MM-DD` in inspected rows;
- same publication date can legitimately occur for different pollsters;
- party `NA` values are frequently legitimate historical/cycle structure;
- raw poll results can be decimal values;
- modeled estimates are decimals/proportions with lower/upper intervals;
- historical party/grouping treatment changes over time;
- development estimates are intentionally mutable;
- source schema can evolve when party classification changes.

Exact duplicate tests, exact null counts, date-order continuity and sum/range diagnostics should be done only in the later bounded non-production fetch.

## 5. Minimum validation rules for a later proof

A future non-production proof should fail or warn on these conditions before any normalized output is trusted.

### File/source checks

1. Record exact Git commit SHA before downloading development files.
2. Record retrieval timestamp and canonical URL.
3. Fingerprint/compare the CSV header against the previously observed schema.
4. Warn on new, removed or renamed columns rather than silently ignoring them.
5. Keep raw source snapshots immutable inside the proof environment.

### Raw poll checks

1. Require non-null `date`, `date_start`, `date_end`, `pollster` and `sample_size` unless the source codebook explicitly allows otherwise.
2. Parse dates strictly and require `date_start <= date_end <= date` only where historically valid; investigate rather than auto-correct exceptions.
3. Confirm `date_middle` lies inside the fieldwork interval.
4. Treat party/result columns as nullable floats.
5. Do not replace party `NA` with zero.
6. Test candidate composite poll identity for duplicates.
7. Flag exact duplicate rows separately from same-date/different-pollster rows.
8. Flag negative or >100 raw result values for codebook/manual review rather than automatic deletion.
9. Preserve pollster spelling as source data and normalize pollster identity only in a separate derived field if later needed.

### Modeled-estimate checks

1. Require unique `date` within the intended modeled series/grain.
2. Parse base/`_lo`/`_hi` fields as nullable floats.
3. For non-null estimates, validate `0 <= lo <= estimate <= hi <= 1`.
4. Do not fill historically unavailable parties with zero.
5. Verify that current-cycle parties correspond to the source header rather than a hard-coded list.
6. Track `cycle` explicitly.
7. Expect past current-term estimates to change between development commits.

### Cross-file checks

1. Never join raw poll percentages directly to modeled proportions without unit normalization.
2. Retain publication date and fieldwork midpoint separately.
3. Do not treat a modeled daily row as an observed poll.
4. Compare latest raw fieldwork midpoint to modeled coverage as a sanity check.
5. Keep citations/source-version metadata available to every downstream graphic.

## 6. Recommended later bounded proof shape

If implementation is authorized separately, the first proof should remain outside production and do only this:

1. Resolve latest IPI development commit SHA.
2. Fetch `data_polls.csv` and `data_pollingindicator.csv` from that exact commit rather than a moving `main` URL.
3. Save source metadata and schema fingerprints.
4. Load both CSVs into memory/local temporary storage only.
5. Produce row counts, date coverage, exact null/duplicate diagnostics and party-column inventories.
6. Normalize units only in a temporary derived dataframe.
7. Produce one sample polling chart/table for validation, clearly separating raw polls and the modeled indicator.
8. Delete/ignore temporary outputs after the proof; do not deploy or alter production schemas.

## Attribution and interpretation

IPI states that users are welcome to use the estimates and polling results as long as the corresponding dataset is cited.

Recommended source citation for the current development dataset should follow the repository's requested citation for Louwerse and Müller, and any EirePolitic graphic should also:

- state that polling reflects opinion during fieldwork and is not an election prediction;
- label the modeled line as the Irish Polling Indicator rather than presenting it as a raw poll;
- preserve pollster attribution for individual poll points.

## Research log

### 2026-09-03 — plan

- Created the read-only IPI proof plan.
- Production changes: none.

### 2026-09-03 — raw polling file

- Confirmed exact raw filename, 23-column schema, 772 poll rows and coverage from 1982-10-30 through 2026-08-02.
- Confirmed fieldwork dates, midpoint, pollster and sample size are present.
- Identified expected historical `NA` values, decimal poll values and historical grouping fields.
- Production changes: none.

### 2026-09-03 — modeled estimate file

- Confirmed exact modeled filename, 41-column schema, 14,406 daily rows and coverage from 1987-02-17 through 2026-07-31.
- Confirmed per-party estimate/lower/upper fields and electoral-cycle field.
- Identified the critical unit difference: raw polls use percentage points while modeled values use proportions.
- Production changes: none.

### 2026-09-03 — versioning and readiness

- Confirmed current development version and exact commit SHA.
- Confirmed development data are mutable and stable DOI releases are the reproducible historical anchor.
- Defined minimum validation and source-version rules for a future non-production proof.
- Conclusion: **Ready for bounded proof.**
- Production changes: none.

## Evidence references

- IPI repository: https://github.com/Irish-Polling-Indicator/ipi-data
- Raw polls CSV: https://github.com/Irish-Polling-Indicator/ipi-data/blob/main/data_polls.csv
- Modeled estimates CSV: https://github.com/Irish-Polling-Indicator/ipi-data/blob/main/data_pollingindicator.csv
- Current inspected commit: https://github.com/Irish-Polling-Indicator/ipi-data/commit/09f77f03654bf66551a56fb02a11a242ae9bcf51
- IPI site/data page: https://pollingindicator.com/
- IPI method: https://pollingindicator.com/method
- Stable DOI: https://doi.org/10.7910/DVN/8YVVYX

## Living next step

The source-level read-only proof is complete. The next step, only if separately continued, is a **bounded non-production fetch-and-validate proof pinned to an exact IPI commit**. That proof should calculate exact duplicate/null diagnostics and produce one temporary normalized sample without changing any production schema or pipeline.
