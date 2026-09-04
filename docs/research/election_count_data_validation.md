# Irish general-election count and transfer data validation

## Purpose

Run a bounded, read-only validation of the official Department of Housing, Local Government and Heritage general-election count/transfer datasets for 2016, 2020 and 2024.

This is research-only. No production architecture, schemas, pipelines, jobs, secrets, infrastructure or production data were changed.

## Final conclusion

**Status: Ready for a later bounded prototype with year-specific source adapters.**

The official data are strong enough for recurring election graphics and historical analysis, but the delivery format changed materially in 2024:

- **2016** — clean candidate/count CSV, 5,685 rows;
- **2020** — clean candidate/count CSV, 5,015 rows;
- **2024** — formula-heavy XLSX workbook with one sheet per constituency plus general/statistical sheets.

The 2016 and 2020 raw structures are nearly identical and differ mainly in the spelling of the non-transferable column (`Non_Transferable` vs `Non-Transferable`). The 2024 workbook should **not** be forced into the same raw schema before preserving its source structure.

The Department's CKAN datastore exposes the complete 2016/2020 count tables, but for the 2024 workbook it exposes only the **General Statistics** sheet. Full 2024 candidate transfers/counts therefore require parsing the XLSX workbook itself.

All checked resources are published under **CC BY-SA 4.0**.

## Official resources validated

### 2016 count details

Official resource ID:

`6bc45750-652d-4e82-b422-0bdcfc2a4c53`

Direct download:

https://opendata.housing.gov.ie/dataset/71dad45b-ce3c-4779-9eb2-775de1b290f6/resource/6bc45750-652d-4e82-b422-0bdcfc2a4c53/download/generalelection2016countdetails.csv

Validation findings:

- format: CSV;
- encoding: CP1252;
- rows: **5,685**;
- columns: **14**;
- constituencies: **40**;
- candidate records: **551**;
- count range: **1–16**;
- result values: blank / `Elected` / `Excluded`;
- exact duplicate rows: **0**;
- duplicate candidate/count keys: **0**;
- constituencies with count-number gaps: **0**;
- SHA-256: `37f7f19615b2d553f93137ddd35e477bd4d4c5dbcf929cdd675801c12e2775df`.

Exact columns:

```text
Constituency Name
Candidate surname
Candidate First Name
Result
Count Number
Non_Transferable
Occurred On Count
Required To Reach Quota
Required To Save Deposit
Transfers
Votes
Total Votes
Constituency Number
Candidate Id
```

The CKAN datastore exposes the same logical fields and all 5,685 records.

### 2020 count details

Official resource ID:

`6feac8a2-85a6-46ae-ad22-0c77f8065e23`

Direct download:

https://opendata.housing.gov.ie/dataset/a9d1a550-fbdb-46ce-84ee-24ae9199c406/resource/6feac8a2-85a6-46ae-ad22-0c77f8065e23/download/general_election_2020_count_details.csv

Validation findings:

- format: CSV;
- encoding: UTF-8 with BOM;
- rows: **5,015**;
- columns: **14**;
- constituencies: **39**;
- candidate records: **534**;
- count range: **1–15**;
- result values: blank / `Elected` / `Excluded`;
- exact duplicate rows: **0**;
- duplicate candidate/count keys: **0**;
- constituencies with count-number gaps: **0**;
- SHA-256: `3801fb3807a33099673a09a353124781a371eed4d96cf109e5fb0dba7cdd27a3`.

Exact columns:

```text
Constituency Name
Candidate surname
Candidate First Name
Result
Count Number
Non-Transferable
Occurred On Count
Required To Reach Quota
Required To Save Deposit
Transfers
Votes
Total Votes
Constituency Number
Candidate Id
```

The 2016 and 2020 headers are not byte-for-byte identical, but after simple punctuation normalization they match exactly.

## Candidate IDs are not reliable cross-election identities

The exact checks found candidate IDs mapping to multiple name/constituency combinations within each file when treated naively:

- 2016: **62** candidate IDs map to multiple name/constituency combinations;
- 2020: **48** candidate IDs map to multiple name/constituency combinations.

This means `Candidate Id` should **not** be assumed to be a globally unique political-person identifier or reused across election years without further source documentation.

A later ingestion proof should treat candidate identity at minimum as election-scoped, and should preserve the source ID without interpreting it as a stable cross-election person key.

## Missing values in 2016/2020

Blank `Result` values are normal for rows before a candidate is elected/excluded:

- 2016 blank result rows: **3,103**;
- 2020 blank result rows: **2,782**.

Core constituency/candidate/count fields are otherwise complete except for **3 missing candidate first names in 2020**.

These records should be preserved as source data rather than dropped.

## 2024 workbook

Official resource ID:

`fb8a1f97-2683-4515-bac4-e2617222b20d`

Direct workbook:

https://opendata.housing.gov.ie/dataset/7ca52e7b-dd4d-4edd-b159-6ae111bbe538/resource/fb8a1f97-2683-4515-bac4-e2617222b20d/download/2024-dail-general-election-results.xlsx

Validation findings:

- format: XLSX;
- file size: approximately **936 KB**;
- SHA-256: `53e83a24caba263a0808a83badf84413fb77f381411050e92ca5181674f52467`;
- sheets: **49**;
- formula cells across workbook: **33,335**.

### Workbook sheet families

The workbook contains:

- `General Statistics 2024`;
- `Postal and Special Ballots`;
- `Invalid Ballot Papers`;
- `Party Table 2025`;
- `Breakdown by Gender `;
- one worksheet for each constituency;
- `Party_Codes`.

The workbook therefore contains substantially more reusable election information than the older count-detail CSVs, including turnout, invalid ballots, gender breakdowns, party summaries and party-code mappings.

### CKAN datastore limitation for 2024

The datastore reports only **46 rows** for the 2024 resource and exposes the `General Statistics 2024` table. It does **not** expose the full constituency transfer/count matrices.

Therefore:

- datastore/API is suitable for general constituency statistics;
- workbook parsing is required for detailed transfers and count-by-count candidate totals.

Do not assume datastore ingestion alone reproduces the 2024 workbook.

## 2024 General Statistics structure

The general-statistics table exposes, by constituency:

- seats;
- electorate;
- total votes cast;
- turnout percentage;
- invalid votes;
- invalid-vote percentage;
- total valid poll;
- quota;
- number of candidates;
- number of women candidates;
- number of men candidates;
- number of counts;
- number of candidates who lost election expenses.

The workbook total row reports:

- seats: **174**;
- electorate: **3,715,285**;
- votes cast: **2,218,302**;
- turnout: approximately **59.7%**;
- invalid votes: **15,849**;
- valid poll: **2,202,453**;
- candidates: **686**;
- women candidates: **246**;
- men candidates: **440**;
- counts: **507**.

These are workbook-provided values/formulas and should remain attributed to the official dataset.

## 2024 constituency sheet structure

Each constituency worksheet is roughly **92 rows** tall and around **40 columns** wide, with count columns extending horizontally.

A representative sheet (`Dublin West`) confirms the pattern:

1. quota and expense-saving threshold near the top;
2. header row naming `First Count`, `Second Count`, etc.;
3. a descriptive transfer row beneath the count header, e.g. transfer of a candidate's surplus/votes;
4. for each candidate, **two physical rows**:
   - candidate row containing first-preference vote and per-count transfer increments/decrements;
   - running-total row containing the candidate's cumulative total after each count;
5. source party code in a separate column;
6. final `Saved` / `Lost` expense indicator;
7. election-status/elected-candidate text in trailing columns;
8. bottom-of-sheet totals and gender/elected summaries.

Example pattern from Dublin West:

- candidate row: `COPPINGER, RUTH (PBP)` with first count 3,552 and subsequent transfer increments;
- following row: cumulative totals 3,552 → 3,639 → 3,666 → ... → 7,165.

This confirms that a future 2024 adapter can extract both:

- **transfer delta by candidate/count**;
- **running total by candidate/count**.

The negative values on candidate rows are meaningful election mechanics: they represent votes removed/distributed when a candidate is elected/excluded. They must not be treated as invalid numeric data.

## 2024 Party_Codes sheet

The workbook includes a direct code-to-party mapping, for example:

- `FF` — Fianna Fáil;
- `FG` — Fine Gael;
- `SF` — Sinn Féin;
- `ANT` — Aontú;
- `GRN` — Green Party/An Comhaontas Glas;
- `PBP` — People Before Profit–Solidarity;
- `LAB` — The Labour Party;
- `INDI` — Independent Ireland;
- `SD` — Social Democrats;
- `NON-P` — Non-Party.

A later adapter should ingest the source party code mapping from this sheet rather than hard-code a party list.

## Safe historical normalization strategy

### 2016/2020 adapter

These two elections can share one parser with a small column-alias layer:

```text
Non_Transferable -> non_transferable
Non-Transferable -> non_transferable
```

Recommended source grain:

```text
election_year + constituency + candidate_source_id + count_number
```

But candidate source ID remains election-scoped and should not be treated as a stable cross-election political-person ID.

Useful source fields to preserve:

- constituency name and number;
- candidate source ID/name;
- count number;
- result/status;
- transfer amount;
- non-transferable votes;
- occurred-on-count indicator;
- votes;
- total votes;
- required to reach quota;
- required to save deposit.

### 2024 adapter

Use a distinct workbook parser:

1. ingest/capture `General Statistics 2024` separately;
2. ingest `Party_Codes` separately;
3. detect constituency sheets dynamically from workbook names, excluding known summary sheets;
4. determine count columns from the `First Count`, `Second Count`, ... header row;
5. parse candidate rows and their immediately following running-total rows as a paired structure;
6. preserve transfer-description text for every count where available;
7. retain gender and source party code;
8. keep quota and count metadata from the constituency sheet;
9. validate extracted constituency totals against `General Statistics 2024`.

Do not flatten workbook formulas without retaining the original workbook hash/version.

## Recommended common analytical model

A later prototype could normalize all three elections into a **derived** analytical grain such as:

```text
election_year
constituency
candidate_source_key
candidate_name
party_source_code
count_number
transfer_delta
running_total
result_status
quota
```

This should remain derived from preserved source-specific raw structures. Do not replace the original source semantics with this common model at ingestion time.

## Minimum validation rules for a later prototype

1. Pin the exact resource URL and record file SHA-256.
2. Store licence/attribution metadata with the source snapshot.
3. For 2016/2020, fingerprint CSV headers and accept only explicit known aliases.
4. Require unique `(constituency, candidate source ID, count number)` within a single election source.
5. Validate contiguous count numbers within each constituency for 2016/2020.
6. Do not use `Candidate Id` as a cross-election identity.
7. Preserve blank `Result` values as normal intermediate-count state.
8. For 2024, validate all expected constituency sheets against the general-statistics constituency list.
9. Parse 2024 candidate-row/running-total-row pairs explicitly.
10. Allow negative transfer deltas in the 2024 workbook.
11. Validate 2024 quota, valid poll, number of counts and constituency totals against the General Statistics sheet.
12. Read party mappings from `Party_Codes` rather than a hard-coded mapping.
13. Preserve workbook/file hash and retrieval timestamp because the 2024 workbook is formula-heavy.
14. Treat the official CKAN datastore as a useful access layer, but do not assume its 2024 representation contains the full workbook.

## Licensing

The checked election resources are published under **CC BY-SA 4.0**.

A later implementation should preserve:

- source agency attribution;
- resource URL/ID;
- licence name/link;
- source file hash;
- retrieval date.

Share-alike implications for any redistributed derived dataset should be reviewed before exposing normalized election data as a standalone downloadable dataset. Editorial charts/analysis should include clear source attribution.

## Readiness decision

### 2016/2020

**Ready for bounded prototype — Easy.**

They are clean, machine-readable, consistent and directly queryable through the portal datastore.

### 2024

**Ready for bounded prototype — Moderate.**

The workbook is more complex but highly structured and richer than prior releases. The separate adapter is justified by the additional information and clear candidate/count layout.

### Overall

**High-value, high-confidence official source.** The main engineering requirement is versioned, election-specific adapters rather than pretending all years share one raw schema.

## Research log

### 2026-09-03 — plan

- Created the election count/transfer validation plan.
- Production changes: none.

### 2026-09-03 — 2016/2020 exact validation

- Confirmed direct CSV and CKAN datastore access.
- Confirmed 5,685 rows for 2016 and 5,015 rows for 2020.
- Confirmed no exact duplicates, no duplicate candidate/count keys and no constituency count-number gaps.
- Confirmed headers match after the non-transferable punctuation alias is normalized.
- Production changes: none.

### 2026-09-03 — 2024 workbook validation

- Confirmed 49-sheet XLSX workbook with 33,335 formula cells.
- Confirmed CKAN datastore exposes General Statistics only, not full count/transfer sheets.
- Confirmed constituency-sheet candidate/transfer/running-total structure and party-code sheet.
- Confirmed a dedicated workbook adapter is feasible and preferable.
- Production changes: none.

### 2026-09-03 — cleanup

- Temporary workflows and reports are removed before merge.
- Durable documentation only is retained.
- Production changes: none.

## Evidence references

- 2016 dataset: https://data.gov.ie/dataset/general-election-2016-count-details
- 2020 dataset: https://data.gov.ie/dataset/general-election-2020-count-details
- 2024 dataset: https://data.gov.ie/dataset/34th-dail-general-election-29-november-2024-election-results
- Department open-data portal: https://opendata.housing.gov.ie/

## Living next step

The election count/transfer validation is complete. The next top-five source to validate is **CSO PxStat**, using one bounded API table to confirm JSON-stat structure, table/version metadata and geography-vintage safeguards before any production design is considered.