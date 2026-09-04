# Irish referendum result data validation

## Purpose

Run a bounded, read-only validation of official Irish referendum result data across multiple years to determine whether it can support recurring EirePolitic historical graphics and dashboards.

This remained research-only. No production architecture, schemas, pipelines, jobs, secrets, infrastructure or production data were changed.

## Final conclusion

**Status: Ready for a later bounded prototype, with one documented 2015 source anomaly.**

The official Department of Housing, Local Government and Heritage referendum data is one of the easiest historical political datasets to normalize safely:

- all four tested resources are free and published under **CC BY-SA 4.0**;
- all expose a clean CKAN datastore API without authentication;
- the underlying logical schema is the same across 1986, 1992, 2015 and 2018 after trivial header normalization;
- there are no duplicate constituency rows, missing fields or negative values in the datastore results;
- published turnout percentages are internally consistent with `Total Poll / Electorate` within rounding tolerance;
- 1986, 1992 and 2018 pass the vote arithmetic check exactly;
- the 2015 dataset contains one source-level 900-vote inconsistency in Dublin Central, which propagates to the published national total.

The safest future ingestion path is the **CKAN datastore API**, not the raw CSV file, because several older CSVs include a human-readable title row before the actual header.

## Official resources validated

### 1986 — Tenth Amendment

- resource ID: `9680852f-a565-4e7c-a3d2-bcda06b3aeb4`
- direct CSV: https://opendata.housing.gov.ie/dataset/8355d3a5-a3c7-41b5-94ad-2037bfa90f67/resource/9680852f-a565-4e7c-a3d2-bcda06b3aeb4/download/1986tenthamendment.csv
- licence: CC BY-SA 4.0
- datastore rows: **42** including the aggregate total row
- logical constituency/result rows: **41 constituencies + total**

Datastore fields:

```text
Constituency
Total  Electorate
Total Poll
Percentage Poll
Votes in favour of proposal
Votes against proposal
Spoilt votes
```

Exact validation:

- duplicate constituency rows: 0;
- missing fields: 0;
- negative rows: 0;
- vote-total mismatches: 0;
- turnout mismatches over 0.11 percentage points: 0.

## 1992 — Thirteenth Amendment

- resource ID: `393fd6af-58fc-4cfb-9a18-e86b4c2c2129`
- direct CSV: https://opendata.housing.gov.ie/dataset/cbaf3fc4-59a9-4111-871b-e8763692a736/resource/393fd6af-58fc-4cfb-9a18-e86b4c2c2129/download/1992thirteenthamendment.csv
- licence: CC BY-SA 4.0
- datastore rows: **42** including the aggregate total row

Datastore fields are exactly the same as 1986.

Exact validation:

- duplicate constituency rows: 0;
- missing fields: 0;
- negative rows: 0;
- vote-total mismatches: 0;
- turnout mismatches over 0.11 percentage points: 0.

## 2015 — Thirty-fifth Amendment

- resource ID: `b09cbfc7-944c-468d-a742-573d2337c9fb`
- direct CSV: https://opendata.housing.gov.ie/dataset/51ce9e79-22dc-4e23-a866-c3c4b2f6ea3e/resource/b09cbfc7-944c-468d-a742-573d2337c9fb/download/2015thirty-fifthamendment.csv
- licence: CC BY-SA 4.0
- datastore rows: **44** including the aggregate total row

Datastore fields:

```text
Constituency
Electorate
Total Poll
Percentage Poll
Votes in favour of proposal
Votes against proposal
Spoilt votes
```

The only schema change from 1986/1992 is that `Total  Electorate` becomes `Electorate`.

Exact validation:

- duplicate constituency rows: 0;
- missing fields: 0;
- negative rows: 0;
- turnout mismatches over 0.11 percentage points: 0;
- vote-total mismatches: **2 rows** — Dublin Central and the aggregate total.

### 2015 Dublin Central arithmetic anomaly

Published values:

- electorate: 57,193;
- total poll: 33,142;
- votes in favour: 12,912;
- votes against: 20,796;
- spoilt: 334.

The component votes sum to **34,042**, which is **900 more** than the published total poll of 33,142.

The aggregate total row has the same 900-vote difference:

- total poll: 1,949,438;
- favour: 521,798;
- against: 1,412,602;
- spoilt: 15,938;
- components sum: 1,950,338.

This shows the national mismatch is fully explained by the Dublin Central row.

**Do not auto-correct this value.** A future loader should preserve the official source values, attach a validation flag, and exclude or annotate the affected arithmetic-derived metric until an authoritative correction is found.

## 2018 — Thirty-sixth Amendment

- resource ID: `fc638461-6f69-463d-b330-ed05fd1de549`
- direct CSV: https://opendata.housing.gov.ie/dataset/0a5ca75d-56c7-4cae-bdbd-730b0a5a371d/resource/fc638461-6f69-463d-b330-ed05fd1de549/download/referendum_results_on_the_thirty-sixth_amendment_of_the_constitution_bill_2018.csv
- licence: CC BY-SA 4.0
- datastore rows: **41** including the aggregate total row

Datastore fields:

```text
Constituency
Electorate
Total Poll
Percentage Poll
Votes in Favour of proposal
Votes against proposal
Spoilt Votes
```

Only capitalization differs materially from 2015.

Exact validation:

- duplicate constituency rows: 0;
- missing fields: 0;
- negative rows: 0;
- vote-total mismatches: 0;
- turnout mismatches over 0.11 percentage points: 0.

## Raw CSV versus datastore API

### Important raw-file quirk

The 1986, 1992 and 2015 direct CSV files begin with a human-readable referendum title row before the actual tabular header.

A naive `csv.DictReader` therefore interprets the title as the header and produces malformed field names.

The 2018 CSV does not have this problem and begins directly with the tabular header.

### Recommended ingestion path

Prefer:

```text
https://opendata.housing.gov.ie/api/3/action/datastore_search?resource_id=<RESOURCE_ID>
```

for these historical resources.

Benefits:

- clean logical field names;
- no title-row special case;
- consistent JSON records;
- no authentication;
- row-count metadata;
- easier schema fingerprinting.

Direct CSV should remain a source/archive fallback and can be hashed for provenance.

## Safe common schema

After trivial alias normalization, all four resources fit this logical model:

```text
referendum_id
referendum_year
referendum_title
constituency_source_label
electorate
total_poll
turnout_pct
votes_for
votes_against
spoilt_votes
is_total_row
source_resource_id
source_url
source_licence
validation_flags
```

Header aliases required:

```text
Total  Electorate -> electorate
Electorate        -> electorate
Votes in favour of proposal -> votes_for
Votes in Favour of proposal -> votes_for
Spoilt votes -> spoilt_votes
Spoilt Votes -> spoilt_votes
```

Do not change constituency labels at raw-ingestion time.

## Constituency geography changes across years

The tested files clearly demonstrate that constituency labels and boundaries change across referendums.

### 1986 versus 1992

Different labels include:

- 1986: `Longford-Westmeath`, `Roscommon`;
- 1992: `Longford-Roscommon`, `Westmeath`.

### 1992 versus 2015

Older labels such as:

- `Kildare`;
- `Mayo East` / `Mayo West`;
- `Meath`;
- `Limerick East` / `Limerick West`;
- `Sligo-Leitrim`;

are replaced by later constituencies such as:

- `Kildare North` / `Kildare South`;
- `Mayo`;
- `Meath East` / `Meath West`;
- `Limerick` / `Limerick City`;
- `Sligo-North Leitrim`.

### 2015 versus 2018

Further changes include:

- Donegal North-East / Donegal South-West -> Donegal;
- Dublin North and other older Dublin labels -> Dublin Bay North, Dublin Bay South, Dublin Fingal, Dublin Rathdown, etc.;
- Kerry North - West Limerick / Kerry South -> Kerry;
- Laois-Offaly -> Laois and Offaly;
- Tipperary North / South -> Tipperary.

There is also a text-encoding/name-normalization issue visible in the 2018 source for Dún Laoghaire (`Dœn Laoghaire` in the datastore response observed during validation).

**Rule:** historical referendum constituency results must be treated as belonging to the boundary/label system in force for that referendum. Never create a fake same-boundary time series by matching constituency names alone.

## Validation rules for a later prototype

1. Use the official resource ID as the primary source locator.
2. Prefer the CKAN datastore API for logical records.
3. Record the direct CSV URL and file hash as provenance where practical.
4. Normalize only known header aliases; preserve original field names in source metadata.
5. Require one row per source constituency plus one aggregate total row where supplied.
6. Validate `Total Poll = votes_for + votes_against + spoilt_votes`.
7. Validate turnout against `100 * Total Poll / Electorate` within an explicit rounding tolerance.
8. Preserve source anomalies as flags rather than correcting them silently.
9. Separate aggregate `Total`/`TOTAL` rows from constituency rows.
10. Preserve referendum year/title and source constituency label.
11. Do not infer stable constituency identity across referendums from labels alone.
12. Join historical maps only to the matching boundary vintage.
13. Normalize obvious text encoding only in a derived display field; preserve the source text.
14. Store CC BY-SA 4.0 attribution/licence metadata with each resource.

## Suitability for EirePolitic

### Value

**High.** The data can support referendum maps, turnout comparisons, strongest/weakest constituencies, historical amendment comparisons, and poll-versus-result retrospectives.

### Access difficulty

**Easy.** Public CKAN datastore and CSV access; no authentication.

### Automation potential

**High.** Each referendum is a small static table with a consistent logical schema.

### Maintenance burden

**Low.** Historical resources are static. New referendums can be onboarded as new resource IDs with the same validation rules.

### Licensing confidence

**High.** All tested resources are explicitly CC BY-SA 4.0.

## Readiness decision

**Ready for bounded prototype — Easy.**

Among the official non-polling sources tested, referendum data is simpler to normalize than general-election count transfers because there is one compact constituency row per referendum rather than count-by-count candidate state.

The 2015 anomaly is manageable if validation flags and raw provenance are preserved.

## Research log

### 2026-09-03 — plan

- Created the referendum validation plan.
- Confirmed the selected resources are official CSVs under CC BY-SA 4.0.
- Production changes: none.

### 2026-09-03 — raw-file validation

- Confirmed older CSV title-row behavior for 1986, 1992 and 2015.
- Confirmed 2018 begins with a clean header.
- Confirmed all four resources have active CKAN datastore representations.
- Production changes: none.

### 2026-09-03 — datastore validation

- Validated all four complete resource tables through the CKAN API.
- Confirmed no duplicate constituency rows, missing fields or negative values.
- Confirmed turnout arithmetic in every row within rounding tolerance.
- Confirmed vote-component arithmetic for 1986, 1992 and 2018.
- Identified the 2015 Dublin Central 900-vote inconsistency and its matching national-total effect.
- Production changes: none.

### 2026-09-03 — geography review

- Confirmed material constituency-label/boundary changes between the tested referendum years.
- Defined boundary-vintage and source-label preservation rules.
- Production changes: none.

### 2026-09-03 — cleanup

- Temporary workflows and reports are removed before merge.
- Durable documentation only is retained.
- Production changes: none.

## Evidence references

- 1986 dataset: https://data.gov.ie/dataset/referendum-on-the-tenth-amendment-of-the-constitution-bill-1986
- 1992 dataset: https://data.gov.ie/dataset/referendum-on-the-thirteenth-amendment-of-the-constitution-bill-1992
- 2015 dataset: https://data.gov.ie/dataset/referendum-on-the-thirty-fifth-amendment-of-the-constitution-bill-2015
- 2018 dataset: https://data.gov.ie/dataset/referendum-results-on-the-thirty-sixth-amendment-of-the-constitution-bill-2018
- Department open-data portal: https://opendata.housing.gov.ie/

## Living next step

The current top-five exact validation sequence is complete. The next step is to update the overall ingestion-feasibility record with the validated readiness order and identify which sources are ready for future non-production prototypes versus which still require rights clarification.