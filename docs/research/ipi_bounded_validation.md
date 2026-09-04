# Irish Polling Indicator bounded validation

## Purpose

Follow the source-level read-only proof with one isolated, non-production validation run against an exact Irish Polling Indicator (IPI) development commit.

The validation fetched the two CSV files from a pinned upstream Git commit, calculated exact structural/data-quality diagnostics, and wrote only temporary results on a research branch. No production architecture, schema, pipeline, data store, secret, scheduled job or deployed workflow was changed.

## Source version tested

- Upstream repository: https://github.com/Irish-Polling-Indicator/ipi-data
- Development release: **2026-08-02**
- Upstream commit: `09f77f03654bf66551a56fb02a11a242ae9bcf51`
- Validation run 1: GitHub Actions run `33826163523`
- Validation run 2 (anomaly detail): GitHub Actions run `33826214129`

Pinned source files:

- `data_polls.csv`
- `data_pollingindicator.csv`

The temporary validation workflow and temporary output files are deliberately removed from the research branch before this documentation is merged.

# 1. Raw polls exact diagnostics

## File identity

`data_polls.csv`

- rows: **772**
- columns: **23**
- date coverage: **1982-10-30 to 2026-08-02**
- SHA-256 of tested file: `f19ae4a67a101f9c61d034f71d9c56076bd0b4f880438676d55ffde9e000b875`

Current header:

```text
date,date_start,date_end,date_middle,pollster,sample_size,FF,FG,SF,LAB,GP,PD,WP,DL,SPBP,RENUA,SD,AU,II,IND_OTH_IT,PREV_INDOTH_II,PREV_II,OTH_IND
```

## Required metadata completeness

Exact missing counts in the six core fields were all zero:

| Field | Missing |
| --- | ---: |
| `date` | 0 |
| `date_start` | 0 |
| `date_end` | 0 |
| `date_middle` | 0 |
| `pollster` | 0 |
| `sample_size` | 0 |

This is a strong operational result: every current row has the basic metadata needed for source attribution, fieldwork timing and sample-size display.

## Exact duplicate rows

There are **2 duplicate extra rows**, forming two exact duplicate pairs:

1. TNS-MRBI, published **2003-05-16**, fieldwork 2003-05-12 to 2003-05-13, sample 1,000 — source file lines 205/206 in the validation parser's one-based CSV record output.
2. Red C, published **2010-06-27**, fieldwork 2010-06-21 to 2010-06-23, sample 1,003 — lines 304/305.

The proposed composite source key (`pollster + publication date + fieldwork start/end + sample size`) identifies exactly these same two duplicate groups.

### Future handling

A future proof may de-duplicate **exact duplicate rows** for analytical output, but should retain a source-row count/audit flag so the upstream duplication is visible. Do not silently treat same-date polls from different pollsters as duplicates.

## Date consistency anomalies

### Internally impossible fieldwork midpoint records

Two rows fail the rule `date_start <= date_middle <= date_end`:

- poll published 2005-06-11, TNS-MRBI: recorded start `2006-06-07`, end `2005-06-08`, middle `2005-12-07`;
- poll published 2011-09-17, Millward Brown: recorded start `2011-08-30`, end `2011-07-14`, middle `2011-08-06`.

These are clearly internally inconsistent source dates. A future loader should **flag/quarantine them for review and preserve the raw source values**, not invent corrected dates.

### Fieldwork end after publication date

Eight rows have `date_end > date`:

- 1982-11-23, TNS-MRBI;
- 1984-01-01, IMS;
- 1984-04-12, IMS;
- 2003-10-15, Red C;
- 2005-03-15, Red C;
- 2005-09-15, Red C;
- 2009-03-01, Red C;
- 2011-06-22, Millward Brown.

Some early records appear to use approximate/month-level date conventions while others may be transcription errors. The validator should flag these records, but should not automatically rewrite them without authoritative source evidence.

## Party-value diagnostics

- values over 100: **0**
- negative values: **1**
- non-integer numeric values: **66**

The one negative value is:

- 1983-05-22, IMS, `OTH_IND = -1`.

This should remain a source-level review flag. It must not be automatically replaced because historical grouping/rounding semantics may explain it.

Decimal values are legitimate in the source, so party columns must be parsed as nullable floating-point/decimal values rather than integers.

## Structural null counts by party/grouping column

`NA` values are expected when parties did not exist or were not separately modeled/reported. Exact counts in this snapshot:

| Field | NA rows |
| --- | ---: |
| FF | 0 |
| FG | 0 |
| SF | 119 |
| LAB | 0 |
| GP | 59 |
| PD | 519 |
| WP | 638 |
| DL | 699 |
| SPBP | 451 |
| RENUA | 647 |
| SD | 451 |
| AU | 585 |
| II | 734 |
| IND_OTH_IT | 770 |
| PREV_INDOTH_II | 770 |
| PREV_II | 770 |
| OTH_IND | 0 |

These counts confirm that generic `fillna(0)` logic would be unsafe.

# 2. Modeled indicator exact diagnostics

## File identity

`data_pollingindicator.csv`

- rows: **14,406**
- columns: **41**
- date coverage: **1987-02-17 to 2026-07-31**
- SHA-256 of tested file: `412a26fc585147b919f28a598e22a4833fcea511241d5df0278dd776c23200aa`

Electoral cycles present:

```text
1987-1989
1989-1992
1992-1997
1997-2002
2002-2007
2007-2011
2011-2016
2016-2020
2020-2024
2024-
```

## Interval/range checks

Across every non-null party estimate triplet:

- invalid or partial estimate/lower/upper triplets: **0**
- values outside `0–1`: **0**

Thus all tested modeled values satisfy the expected bound/order rule:

```text
0 <= lower <= estimate <= upper <= 1
```

This is a strong data-quality result.

## Duplicate dates are cycle-boundary records

There are **7 duplicate calendar dates** in the modeled file, but they are not accidental exact duplicates. Each date appears once at the end of one electoral cycle and once at the start of the next:

- 1989-06-15: `1987-1989` and `1989-1992`
- 1992-11-25: `1989-1992` and `1992-1997`
- 1997-06-06: `1992-1997` and `1997-2002`
- 2002-05-17: `1997-2002` and `2002-2007`
- 2007-05-24: `2002-2007` and `2007-2011`
- 2011-02-25: `2007-2011` and `2011-2016`
- 2016-02-26: `2011-2016` and `2016-2020`

### Future handling

`date` alone is **not a valid unique key** for the modeled dataset. Use at least:

```text
(date, cycle)
```

This is the clearest schema rule discovered in the bounded validation.

## Calendar gaps

After deduplicating calendar dates for the continuity check, only two multi-day gaps were found:

- 2020-01-31 → 2020-02-08 (8-day difference)
- 2024-11-23 → 2024-11-28 (5-day difference)

Both occur around general-election cycle transitions. A future validator should therefore not require one globally continuous daily series across election boundaries. Continuity should be tested **within an electoral cycle**.

## Structural null counts

Exact `NA` counts for party estimate columns:

| Party/group | NA rows |
| --- | ---: |
| FF | 0 |
| FG | 0 |
| LAB | 0 |
| PD | 5,626 |
| WP | 10,641 |
| OTH | 0 |
| GP | 850 |
| DL | 11,491 |
| SF | 3,765 |
| SD | 10,608 |
| SPBP | 10,608 |
| AU | 12,044 |
| II | 13,795 |

Again, these are primarily historical/cycle structure rather than generic missing-data failures.

# 3. Cross-file check

Latest raw poll fieldwork midpoint:

- **2026-07-31**

Latest modeled-indicator date:

- **2026-07-31**

Result: **match**.

This is a useful future sanity check when a new poll is ingested: after a release, the modeled development file should normally cover the newest fieldwork midpoint/current modeling endpoint expected by IPI.

# 4. Revised validation rules

The exact run changes several rules from the earlier source-level proof.

## Raw poll identity

- Do not require every source row to be unique; the upstream file currently contains exact duplicates.
- Preserve raw rows/source hash, then create a clearly audited deduplicated analytical view if needed.
- Candidate composite key duplicates should trigger comparison of all fields before deduplication.

## Date rules

- Treat impossible fieldwork ordering as a **source anomaly**, not an ingestion failure that should be auto-corrected.
- Preserve raw dates and attach a validation flag.
- For public polling graphics, exclude/quarantine rows with impossible fieldwork dates until an authoritative correction is found if the fieldwork date is required for positioning.

## Modeled series identity

- Unique key should be `(date, cycle)`, not `date`.
- Check daily continuity within cycle, not across election transitions.
- Keep election-cycle transitions explicit in historical graphics.

## Units

- raw polls: percentage-point scale (`0–100`);
- modeled indicator: proportion scale (`0–1`).

Normalize deliberately in a derived layer and keep the original unit metadata.

## Source pinning

Every future run should retain:

- upstream commit SHA;
- retrieval timestamp;
- file SHA-256;
- header/schema fingerprint;
- row count/date coverage;
- validation result counts.

These controls are especially important because IPI's development estimates are expected to change after new polls.

# 5. Readiness decision

**Decision: Ready for a later non-production ingestion prototype, with explicit source-quality flags.**

The source passed the most important operational checks:

- direct public machine-readable access;
- no authentication;
- complete core raw-poll metadata;
- valid modeled uncertainty intervals/ranges;
- clear source versioning through Git;
- stable historical DOI alternative;
- manageable and now well-understood anomalies.

The discovered duplicate/date anomalies do not make the source unsuitable. They do mean that a production-quality design must preserve raw provenance and expose validation status instead of silently “cleaning” historical records.

# Research log

### 2026-09-03 — exact diagnostics

- Ran isolated validation against pinned upstream commit `09f77f03654bf66551a56fb02a11a242ae9bcf51`.
- Confirmed exact row/column counts, hashes, key-field completeness, duplicate counts, party null counts, date anomalies and modeled interval validity.
- Production changes: none.

### 2026-09-03 — anomaly classification

- Identified the two exact duplicate raw poll pairs.
- Identified two internally impossible fieldwork-date records and eight field-end-after-publication records.
- Confirmed all seven duplicate modeled dates are electoral-cycle boundary rows.
- Confirmed only two calendar gaps, both around election-cycle transitions.
- Production changes: none.

### 2026-09-03 — cleanup

- Temporary validation workflow and temporary reports are removed before merge.
- Durable documentation only is retained.
- Production changes: none.

# Living next step

The IPI bounded validation is complete. If research continues, the next highest-value polling step is to perform the same exact, pinned-file diagnostic pass on the **Irish Demographic Polling Datasets**, while also resolving its commercial/republication licensing confidence before any production recommendation.
