# CSO PxStat validation

## Purpose

Run a bounded, read-only validation of the Central Statistics Office PxStat API as a future EirePolitic data source.

This remained research-only. No production architecture, schemas, pipelines, jobs, secrets, infrastructure or production data were changed.

## Final conclusion

**Status: Ready for a later bounded prototype.**

The CSO PxStat source is technically strong:

- public REST access without authentication;
- JSON-stat 1.0 and 2.0 plus CSV/PX/XLSX outputs;
- stable table identifiers;
- machine-readable dimension/category codes and labels;
- no duplicates or nulls in the tested table;
- exact value-order agreement between JSON-stat 2.0 and CSV;
- CC BY 4.0 licence.

The main risk is **semantic/geographic correctness**, not connectivity. A table ID must be selected because its dimensions and geography are appropriate for the intended political question; titles alone are not sufficient.

The validation target `F4061` is a good technical API test, but it is **not a Dáil-constituency population table**. It is a percentage table broken down by Census year, general health, sex, county/city and age group.

## Source validated

### Table

- ID: `F4061`
- title: `Population`
- publisher: Central Statistics Office
- licence: **CC BY 4.0**
- catalogue record: https://data.gov.ie/dataset/f4061-population

The data.gov.ie resource metadata recorded the resource as updated on **17 November 2025** during this investigation.

### Endpoints

JSON-stat 2.0:

https://ws.cso.ie/public/api.restful/PxStat.Data.Cube_API.ReadDataset/F4061/JSON-stat/2.0/en

JSON-stat 1.0:

https://ws.cso.ie/public/api.restful/PxStat.Data.Cube_API.ReadDataset/F4061/JSON-stat/1.0/en

CSV:

https://ws.cso.ie/public/api.restful/PxStat.Data.Cube_API.ReadDataset/F4061/CSV/1.0/en

All three endpoints were reachable without authentication.

## Exact response diagnostics

### JSON-stat 2.0

- content type: `application/json`
- response size: **177,950 bytes**
- SHA-256: `4172575c62579cd677b0ddbf49c8ff71b55e6a036c5ef2693edb52c6ea35b940`
- class: `dataset`
- label: `Population`
- dimensions: **6**
- observations: **37,107**
- null observations: **0**
- status field: not present
- numeric range: **0–100**

Exact dimension order:

```text
STATISTIC
TLIST(A1)
C02832V03406
C02199V02655
C04104V04868
C02076V03371
```

Exact dimension sizes:

```text
1 x 3 x 7 x 3 x 31 x 19 = 37,107
```

### CSV

- response size: **6,051,936 bytes**
- SHA-256: `217fd122846a21cc73eae02d224c4568ce22d39b395a0cbb0d87b5ebfea5521e`
- rows: **37,107**
- columns: **14**
- exact duplicate rows: **0**
- missing cells in all tested columns: **0**

Header:

```text
STATISTIC
Statistic Label
TLIST(A1)
Census Year
C02832V03406
General Health
C02199V02655
Sex
C04104V04868
County and City
C02076V03371
Age Group
UNIT
VALUE
```

The CSV `VALUE` sequence exactly matches the JSON-stat 2.0 value array.

## Dimension semantics

### Statistic

- code: `F4061C01`
- label: `Population`

### Census Year

3 categories:

- 2011
- 2016
- 2022

### General Health

7 categories:

- all;
- very good;
- good;
- fair;
- bad;
- very bad;
- not stated.

### Sex

3 categories:

- both sexes;
- male;
- female.

### County and City

31 categories including:

- State (`IE0`);
- Cork City and Cork County;
- Clare;
- Cavan;
- Carlow;
- Dublin City;
- Donegal;
- Dún Laoghaire-Rathdown;
- Fingal;
- Galway City;
- Galway County;
- and the remaining county/city authorities.

Most local geography category codes are opaque UUID-style identifiers rather than human-readable county abbreviations.

### Age Group

19 categories:

- all ages;
- 0–4;
- 5–9;
- ...;
- 80–84;
- 85+.

## Critical semantic finding: this table is percentage-based

The CSV `UNIT` is `%`, and values are on a 0–100 scale.

For example, the combination:

- Population;
- Census 2011;
- General health — All;
- Both sexes;
- State;
- All ages;

has value `100`.

Therefore `F4061` must **not** be interpreted as a table of population headcounts merely because the table/statistic label is `Population`.

This is a strong general lesson for future CSO ingestion: inspect `UNIT`, dimensions and category semantics before deciding what a table measures.

## Geography/version findings

The tested geography dimension is **County and City**, not Dáil constituency.

That means this table cannot be joined directly to election-constituency results without a separate, justified geographic transformation. EirePolitic should not infer a constituency mapping from county labels.

A future politically useful CSO source must record at least:

```text
source_table_id
source_statistic_code
source_statistic_label
reference_period_or_census_year
geography_dimension_code
geography_dimension_label
geography_category_code
geography_category_label
unit
retrieved_at
source_response_hash
```

If a chosen table uses Dáil constituencies, local electoral areas, electoral divisions or small areas, that exact geography/vintage must remain explicit.

## Why category codes matter

PxStat exposes both codes and labels. Future ingestion should preserve both.

Examples:

- Census year code/label: `2022` / `2022`;
- State geography: `IE0` / `State`;
- age code: `205` / `0 - 4 years`.

Opaque county/city codes demonstrate why labels alone are insufficient for durable joins. Conversely, codes should not be assumed to have the same meaning across unrelated geography dimensions/tables unless CSO metadata confirms it.

## JSON-stat handling rules

A future Python loader must respect the JSON-stat dimension order and category index order.

For this table:

```text
id   = [STATISTIC, TLIST(A1), C02832V03406, C02199V02655, C04104V04868, C02076V03371]
size = [1, 3, 7, 3, 31, 19]
```

The flattened `value` array is ordered according to those dimensions. It should be expanded using JSON-stat semantics or a proven library rather than ad-hoc nested-loop assumptions.

The CSV endpoint is easier to inspect/debug and is a useful cross-check, but JSON-stat is the better machine-readable source when dimensions and codes need to be preserved exactly.

## Version/update strategy

PxStat table IDs are strong retrieval identifiers, but a future ingestion process should not assume responses are immutable.

For every fetch, record:

- table ID;
- endpoint/version format;
- retrieval timestamp;
- response SHA-256;
- dimension IDs and sizes;
- category-code fingerprint;
- row/value count;
- catalogue `data last updated` value where available.

If a table is revised in place, the response hash and possibly dimension/category fingerprints will detect the change.

## Minimum validation rules for a later prototype

1. Use an explicit table ID; never select a table by title alone.
2. Record and verify `UNIT` before interpreting values.
3. Record JSON-stat dimension `id` and `size` arrays.
4. Preserve category codes and category labels.
5. Fail/warn if dimension IDs, dimension sizes or unit change unexpectedly.
6. Verify product of dimension sizes equals expected observation slots.
7. Record null/status handling rather than replacing suppressed/missing values with zero.
8. Cross-check JSON-stat and CSV values during initial onboarding of a table.
9. Store reference year and geography dimension explicitly.
10. Do not join a CSO geography to an election geography unless the boundary/vintage relationship has been explicitly validated.
11. Record response hash and catalogue update date for reproducibility.
12. Prefer one deliberately selected table per editorial use case rather than bulk-ingesting PxStat indiscriminately.

## Suitability for EirePolitic

### Technical value

**High.** The API is clean, free, open and automation-friendly.

### Editorial value

**High when tables are selected deliberately.** CSO can support recurring graphics on population, housing, migration, labour, health, education and constituency/area demographics.

### Ingestion difficulty

**Moderate.** Connectivity is easy; dimensional metadata and geography semantics require care.

### Maintenance burden

**Low–Medium.** Stable table IDs and structured APIs reduce operational work, but table revisions/classification changes need monitoring.

### Licensing confidence

**High.** The tested resource is explicitly CC BY 4.0.

## Readiness decision

**Ready for a bounded non-production prototype.**

The first real editorial prototype should **not** use `F4061` automatically. It should select a table whose geography and unit match a concrete EirePolitic feature—for example a constituency or small-area demographic profile—and repeat the same metadata/structure validation for that exact table.

## Research log

### 2026-09-03 — plan

- Created the CSO PxStat validation plan.
- Confirmed F4061 is published by the CSO under CC BY 4.0 and exposes JSON-stat, CSV, PX and XLSX resources.
- Production changes: none.

### 2026-09-03 — exact API validation

- Fetched JSON-stat 1.0, JSON-stat 2.0 and CSV without authentication.
- Confirmed 37,107 observations and exact CSV/JSON-stat value-sequence agreement.
- Confirmed no duplicate CSV rows and no null observations in the tested table.
- Confirmed the six-dimensional structure and exact category counts.
- Production changes: none.

### 2026-09-03 — semantic/geography review

- Confirmed the table unit is percentage, not headcount.
- Confirmed geography is County and City, not Dáil constituency.
- Defined explicit table/unit/geography/version safeguards for future ingestion.
- Production changes: none.

### 2026-09-03 — cleanup

- Temporary validation workflow and temporary report are removed before merge.
- Durable documentation only is retained.
- Production changes: none.

## Evidence references

- CSO PxStat/data portal: https://data.cso.ie/
- F4061 catalogue: https://data.gov.ie/dataset/f4061-population
- JSON-stat 2.0 endpoint: https://ws.cso.ie/public/api.restful/PxStat.Data.Cube_API.ReadDataset/F4061/JSON-stat/2.0/en
- CSV endpoint: https://ws.cso.ie/public/api.restful/PxStat.Data.Cube_API.ReadDataset/F4061/CSV/1.0/en
- CSO PxStat user guidance: https://www.cso.ie/en/databases/userguides/pxstatuserguide/

## Living next step

The CSO API validation is complete. The final source in the current top-five validation sequence is **official referendum results**. Validate several referendum CSVs across years, confirm schema drift and geography caveats, then update the overall top-five readiness order.