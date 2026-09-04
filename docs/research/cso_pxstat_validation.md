# CSO PxStat validation

## Purpose

Run a bounded, read-only validation of the Central Statistics Office PxStat API as a future EirePolitic data source.

This remains research-only. No production architecture, schemas, pipelines, jobs, secrets, infrastructure or production data may be changed.

## Scope

Use one small, politically useful Census 2022 table as the validation target:

- table ID: `F4061` — Population;
- publisher: Central Statistics Office;
- licence: CC BY 4.0;
- formats: JSON-stat, CSV, PX and XLSX.

The purpose is not to choose a permanent CSO topic yet. It is to prove the API shape, identifiers, metadata, geography/version behaviour and minimum safeguards needed for later use.

## Questions to answer

1. Does the JSON-stat endpoint work without authentication?
2. What exact JSON-stat dimensions, category codes and value ordering are returned?
3. Is the table ID a stable enough source key for retrieval?
4. What geography dimension/vintage is encoded in this table?
5. How do CSV and JSON-stat outputs compare structurally?
6. Are there null/status values or other data-quality conditions a future loader must preserve?
7. What update/version metadata can be recorded?
8. What minimum validation rules are required to avoid joining the wrong geography vintage to election data?

## Evidence to collect

- canonical table ID and API URLs;
- exact JSON-stat `id`, `size`, dimensions and category counts;
- exact CSV header/row count;
- date/reference-period fields;
- geography labels/codes;
- missing/null/status counts;
- file/response hashes where practical;
- catalogue update date and licence;
- comparison of CSV and JSON-stat values for a bounded sample;
- recommended source metadata fields;
- future failure/monitoring rules;
- readiness rating.

## Method

1. Verify the official data.gov.ie catalogue record and licence.
2. Fetch JSON-stat and CSV from the official PxStat endpoint without credentials.
3. Use an isolated temporary research workflow only if whole-file diagnostics are needed.
4. Do not build a production connector or schema.
5. Remove temporary workflow/output files before documentation merge.

## Research log

### 2026-09-03 — plan

- Created the CSO PxStat validation plan.
- Confirmed F4061 is published by the CSO under CC BY 4.0 and exposes JSON-stat, CSV, PX and XLSX resources.
- Production changes: none.
- Next: inspect the exact JSON-stat/CSV structures and geography metadata.

## Living next step

Pin the current F4061 responses, validate their exact structure and document the geography/version safeguards needed before any later ingestion proof.