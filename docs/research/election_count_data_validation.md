# Irish general-election count and transfer data validation

## Purpose

Run a bounded, read-only validation of the official Department of Housing, Local Government and Heritage general-election count/transfer datasets for 2016, 2020 and 2024.

This is research-only. No production architecture, schemas, pipelines, jobs, secrets, infrastructure or production data may be changed.

## Questions to answer

1. What exact official files/resources exist for 2016, 2020 and 2024?
2. Are 2016 and 2020 structurally compatible in practice, not just in their catalogue descriptions?
3. What sheets, columns and grains exist in the 2024 XLSX workbook?
4. Can count-by-count candidate vote totals and transfers be reconstructed consistently across years?
5. Are candidate, constituency, quota, status and transfer semantics sufficiently explicit for recurring graphics?
6. What duplicate, null, numeric-range or count-sequence anomalies are present?
7. Are candidate/constituency identifiers stable within an election, and should they ever be reused across elections?
8. What minimum per-election adapter and validation rules would a later prototype require?
9. How should CC BY-SA 4.0 attribution/share-alike obligations be recorded in any later derived-data design?

## Evidence to collect

For each election/resource:

- official dataset/resource URL and resource ID;
- licence;
- file type, size and hash where practical;
- row/sheet counts;
- exact columns;
- constituency and candidate counts;
- count-number range;
- result/status values;
- candidate/constituency identifiers;
- transfer and non-transferable fields;
- duplicate/null diagnostics;
- count-sequence continuity;
- schema differences by election;
- transformation required for a common analytical model;
- risks around boundary/name/party changes;
- future ingestion mode and readiness rating.

## Method

1. Verify official catalogue metadata and download URLs.
2. Inspect 2016 and 2020 CSV resources directly.
3. Inspect the 2024 XLSX workbook in an isolated temporary research workflow if needed.
4. Run exact bounded diagnostics on all three files.
5. Document source-specific adapters rather than forcing one raw schema.
6. Remove every temporary workflow/output before documentation merge.

## Known source evidence before exact validation

- 2016 official count details: CSV, CC BY-SA 4.0.
- 2020 official count details: CSV/JSON/XML, CC BY-SA 4.0.
- 2024 official results/transfers: XLSX, CC BY-SA 4.0.
- 2016 and 2020 catalogue dictionaries are nearly identical, with at least one naming difference (`Non_Transferable` vs `Non-Transferable`).
- The 2024 release is a different workbook-style publication and must be treated as a separate source adapter until inspected.

## Research log

### 2026-09-03 — plan

- Created the election count/transfer validation plan.
- Production changes: none.
- Next: inspect the exact 2016/2020 CSVs and 2024 workbook structure.

## Living next step

Pin the three official resources, run exact schema/data-quality diagnostics, then decide the safest historical normalization strategy without implementing it.