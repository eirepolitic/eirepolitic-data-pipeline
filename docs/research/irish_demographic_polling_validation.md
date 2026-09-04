# Irish demographic polling dataset validation

## Purpose

Run a bounded, read-only validation of the Irish Demographic Polling Datasets, the second-ranked free polling source from the Irish political data-source investigation.

This remained research-only. No production architecture, schemas, pipelines, secrets, infrastructure, jobs, or production data were changed.

## Final conclusion

**Status: Technically ready for a bounded prototype, but production/republication permission should be clarified first.**

The dataset is useful and unusually rich for demographic polling graphics. Seven public CSVs were validated at upstream commit:

`c15fd6acfe0a161bccb0bfa3f3fbcdcf19d997ca`

The files have complete core poll-date/sample metadata, no exact duplicate rows, no impossible fieldwork dates, and almost all proportion cells fall in the expected `0–1` range. The main caveats are:

1. **The file families have different end dates.** B&A government satisfaction, leader satisfaction and vote intention currently end in October 2023, while the Red C vote-intention file reaches February 2026.
2. **One Red C proportion is clearly invalid:** `15.0` rather than a value between `0` and `1`.
3. **B&A “counts” are weighted table counts, not literal respondent headcounts.** They reproduce the published proportions, but their total denominator can be roughly two to four times the nominal survey sample size.
4. **Counts/proportions schemas are not perfectly symmetric.** The B&A vote-proportion file contains `const_seats_3` and `const_seats_4`, which are absent from the counts file.
5. **Some proportion cells are missing even where counts are present.** A future loader must preserve source missingness rather than derive or backfill automatically.
6. **The repository explicitly invites use in news reports with citation, but no standard open-data licence or explicit blanket commercial-redistribution licence was found.** EirePolitic should obtain clarification before treating the source as a production raw-data feed or redistributing it wholesale.

## Source and maintenance behavior

Repository:

https://github.com/Irish-Dem-Polling/datasets

The repository is actively maintained. At the time of validation it had more than 500 commits and repeated 2026 `Update data` / `Update report` activity. However, prose metadata is stale/inconsistent across project pages: the repository README still says 2011–2023, related project pages describe later coverage, and the file-level dates differ by sub-dataset.

**Operational rule:** derive coverage from the actual pinned CSV files, not from README prose or project-page summaries.

A later prototype should pin an exact upstream Git commit and store both the commit SHA and file hash.

## Published citation/reuse wording

The repository README states that users of the datasets in **news reports or academic research** should cite:

Stefan Müller, Thomas Pluck, and Paula Montano (2024), *Irish Demographic Polling Datasets*.

It also asks users of individual surveys to cite/reference the original pollsters' reports rather than only the aggregate dataset. The README warns explicitly that subgroup sample sizes can be very small and must be treated cautiously.

No standard repository `LICENSE` file was visible, and the package `DESCRIPTION` file contains no `License:` field. Therefore:

- editorial/news-report use with citation is clearly contemplated;
- a blanket open-data licence was **not** verified;
- commercial bulk redistribution, mirroring, or exposing the raw dataset through a public product is **not clearly authorised by an explicit licence**.

**Recommendation:** before production ingestion, request written clarification covering EirePolitic's intended recurring editorial use, internal storage, automated retrieval, derived graphics, and whether any raw/subset data may be exposed publicly.

## Exact files validated

### Behaviour & Attitudes — government satisfaction

- `government-satisfaction/data_banda_govsat_counts.csv`
- `government-satisfaction/data_banda_govsat_prop.csv`

Both files:

- rows: **255**
- columns: **42**
- distinct poll dates: **85**
- coverage: **2011-08-24 to 2023-10-15**
- response categories: `Satisfied`, `Dissatisfied`, `No opinion`

Core metadata fields:

- `date`
- `date_start`
- `date_end`
- `date_middle`
- `sample_size`

Subgroups include gender, age, social class, region, urban/rural, constituency-seat magnitude, likelihood to vote, future vote and past vote.

### Behaviour & Attitudes — party-leader satisfaction

- `party-leaders/data_banda_leaders_counts.csv`
- `party-leaders/data_banda_leaders_prop.csv`

Both files:

- rows: **1,170**
- columns: **44**
- distinct poll dates: **85**
- coverage: **2011-08-24 to 2023-10-15**
- leader parties represented: **5**
- leader names represented: **10**
- response categories: `Satisfied`, `Dissatisfied`, `No opinion`

The same broad demographic/geographic subgroup structure is present as for government satisfaction.

### Behaviour & Attitudes — first-preference vote intention

- `vote-intention/data_banda_firstpref_counts.csv`
- `vote-intention/data_banda_firstpref_prop.csv`

Both files:

- rows: **965**
- distinct poll dates: **78**
- coverage: **2015-08-15 to 2023-10-15**
- parties/labels represented: **15**

Schema difference:

- counts file: **22 columns**
- proportion file: **24 columns**
- proportion-only fields: `const_seats_3`, `const_seats_4`

The counts and proportions otherwise align at row-key level.

### Red C — first-preference vote intention

- `vote-intention/data_redc_firstpref_prop.csv`

Current file:

- rows: **574**
- columns: **18**
- distinct poll dates: **61**
- coverage: **2017-11-26 to 2026-02-22**
- parties/labels represented: **12**

The Red C schema is narrower than B&A. It has no paired counts CSV and uses `region_leinster_rest` rather than B&A's `region_leinster`. It does not expose the same constituency-seat or urban/rural dimensions.

**Future rule:** use pollster-specific source adapters and a controlled normalized dimension mapping; do not assume the B&A and Red C CSVs share one schema.

## Core metadata quality

Across all seven files, exact validation found:

- missing `date`: **0**
- missing `date_start`: **0**
- missing `date_end`: **0**
- missing `date_middle`: **0**
- missing `sample_size`: **0**
- exact duplicate extra rows: **0**
- `date_start > date_end`: **0**
- `date_middle` outside fieldwork interval: **0**
- fieldwork end after publication date: **0**
- invalid numeric measure cells: **0**
- negative measure cells: **0**

This is substantially cleaner source metadata than the historical IPI raw-poll file.

## Proportion validation

All B&A proportion files passed the `0–1` range check.

The Red C vote-intention file contains **one proportion cell over 1**:

- poll date: **2025-05-25**
- party: **Fianna Fáil**
- field: `region_connacht_ulster`
- source value: **15.0**

This is almost certainly a source transcription/scale error, but this investigation does **not** invent the corrected value. A future loader should retain the raw value, flag it as invalid, and exclude it from public graphics until verified against the original Red C report.

## B&A counts-versus-proportions test

The three available B&A counts/proportion pairs were compared at row and subgroup-cell level.

### Government satisfaction

- headers equal: **Yes**
- row counts equal: **Yes**
- row-key sets equal: **Yes**
- comparable subgroup cells tested: **7,854**
- cells differing by more than `0.015`: **0**
- maximum absolute ratio/proportion difference: **0.005**

### Leader satisfaction

- headers equal: **Yes**
- row counts equal: **Yes**
- row-key sets equal: **Yes**
- comparable subgroup cells tested: **36,711**
- cells differing by more than `0.015`: **0**
- maximum absolute ratio/proportion difference: **0.005**

### B&A vote intention

- headers equal: **No**, because `const_seats_3` and `const_seats_4` are proportion-only
- row counts equal: **Yes**
- row-key sets equal: **Yes**
- comparable subgroup cells tested: **14,911**
- cells differing by more than `0.015`: **0**
- maximum absolute ratio/proportion difference: **0.005**

This strongly confirms that the published B&A proportions are consistent with the paired weighted counts within ordinary rounding tolerance.

## Why the counts are not literal sample sizes

For each B&A poll/question group, the validator summed the `total` counts across response categories/parties and compared that denominator with the nominal `sample_size`.

Observed denominator-to-sample ratios:

| Dataset | Minimum | Median | Maximum |
| --- | ---: | ---: | ---: |
| Government satisfaction | 3.142 | 3.613 | 3.904 |
| Leader satisfaction | 1.024 | 3.618 | 3.931 |
| Vote intention | 1.923 | 2.313 | 2.891 |

Therefore the `*_counts.csv` values must **not** be interpreted as raw respondent headcounts or used directly as subgroup `n` without understanding the source weighting/table construction.

The repository itself describes the data as weighted proportions and counts. The exact ratio test confirms that these are useful for reproducing proportions, but not interchangeable with the survey's nominal sample size.

**Publication safeguard:** do not display a subgroup “n=” derived directly from these count fields unless the maintainers confirm the intended denominator semantics.

## Missingness asymmetry

The paired files contain cases where a count is present but the corresponding published proportion is missing:

- government satisfaction: **100 cells**
- leader satisfaction: **161 cells**
- B&A vote intention: **415 cells**

The reverse case—proportion present while the paired count is missing—was **0** in shared fields.

This means the counts can contain more low-level information than the proportion files. A future loader should **not automatically calculate missing proportions** unless maintainers confirm that such derivation is intended and statistically appropriate.

## Missing subgroup data

Missing subgroup values are common and expected because not every poll/report exposes every crosstab consistently.

Examples from this snapshot:

- government counts: 1,226 missing measure cells of 9,180
- government proportions: 1,326 / 9,180
- leader counts: 5,248 / 42,120
- leader proportions: 5,409 / 42,120
- B&A vote counts: 114 / 15,440
- B&A vote proportions: 596 / 17,370
- Red C vote proportions: 0 / 6,888

Missingness must be preserved as `NA`, never treated as zero support/satisfaction.

## Recommended future source model

If permission is clarified and a later non-production prototype is authorised, use separate source adapters:

1. B&A government satisfaction.
2. B&A leader satisfaction.
3. B&A vote intention.
4. Red C vote intention.

Each source record should retain at least:

- upstream repository;
- upstream commit SHA;
- source file path;
- file SHA-256;
- retrieval timestamp;
- poll date and fieldwork dates;
- nominal sample size;
- pollster;
- question family;
- party/leader/response label;
- subgroup dimension and subgroup value;
- published proportion;
- weighted count if available;
- validation flags;
- citation/source-report reference where available.

Do not force every pollster/question family into an identical raw schema before preserving the original source fields.

## Minimum validation rules for a later prototype

1. Pin an exact upstream commit before fetching files.
2. Record a SHA-256 and header fingerprint for each CSV.
3. Fail/warn on added, removed or renamed files/columns.
4. Require the five core date/sample fields.
5. Validate `date_start <= date_middle <= date_end <= date`.
6. Treat proportions as nullable numeric values expected in `0–1`; flag rather than silently fix invalid values.
7. Preserve `NA`; never replace missing subgroup values with zero.
8. Keep B&A and Red C schemas separate at ingestion.
9. Never interpret the B&A weighted count as a literal respondent count without explicit source confirmation.
10. For B&A pairs, periodically verify that count-derived ratios match published proportions within rounding tolerance.
11. Apply a publication eligibility rule for small subgroups only after the denominator/base semantics are confirmed.
12. Preserve original pollster attribution and cite the individual survey/report when publishing a specific poll result.

## Licensing/readiness decision

### Technical readiness

**High.** Direct public CSV files, no authentication, clean core metadata, clear source versioning through Git, and internally consistent B&A counts/proportions.

### Data-quality readiness

**Medium–High.** One obvious Red C outlier and some structural missingness/schema asymmetry require validation flags, but these are manageable.

### Licensing confidence

**Medium.** News-report use with citation is explicitly invited, but no formal open-data licence or blanket commercial redistribution terms were found.

### Overall decision

**Technically ready, but permission clarification recommended before production use.**

For EirePolitic's immediate polling roadmap, the Irish Polling Indicator remains the safer first production candidate because its source/version model is simpler. The demographic dataset remains the strongest second candidate for distinctive crosstab, government-satisfaction and leader-satisfaction graphics once reuse terms are clarified.

## Research log

### 2026-09-03 — plan

- Created the bounded validation plan.
- Production changes: none.

### 2026-09-03 — source structure and rights

- Confirmed seven current CSV files spanning B&A vote intention, B&A government satisfaction, B&A leader satisfaction and Red C vote intention.
- Confirmed citation wording explicitly contemplates news-report use.
- Confirmed no standard open-data licence or `License:` field was identified.
- Production changes: none.

### 2026-09-03 — exact diagnostics

- Ran a pinned, isolated validation against upstream commit `c15fd6acfe0a161bccb0bfa3f3fbcdcf19d997ca`.
- Verified row/column counts, date coverage, hashes, duplicates, date consistency, numeric ranges and paired B&A counts/proportions.
- Production changes: none.

### 2026-09-03 — anomaly review

- Identified one invalid Red C proportion (`15.0`).
- Confirmed B&A counts/proportions match within rounding tolerance.
- Confirmed the B&A counts are not literal respondent headcounts.
- Confirmed counts/proportion schema and missingness asymmetries.
- Production changes: none.

### 2026-09-03 — cleanup

- Temporary validation workflow and temporary reports are removed before merge.
- Durable documentation only is retained.
- Production changes: none.

## Evidence references

- Repository: https://github.com/Irish-Dem-Polling/datasets
- Repository README/citation guidance: https://github.com/Irish-Dem-Polling/datasets#readme
- Related-project description: https://pollingindicator.com/
- B&A government satisfaction CSVs: `government-satisfaction/`
- B&A leader satisfaction CSVs: `party-leaders/`
- B&A / Red C vote-intention CSVs: `vote-intention/`
- Tested upstream commit: `c15fd6acfe0a161bccb0bfa3f3fbcdcf19d997ca`

## Living next step

The demographic polling validation is complete. Before any production recommendation, obtain clarification on recurring commercial/editorial use and raw-data storage/republication. The next purely technical research target is the official Irish election count/transfer data, where the main question is normalization across the 2016/2020 CSV-style releases and the 2024 XLSX workbook.