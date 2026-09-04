# Irish demographic polling dataset validation

## Purpose

Run a bounded, read-only validation of the Irish Demographic Polling Datasets, the second-ranked free polling source from the Irish political data-source investigation.

This remains research-only. No production architecture, schemas, pipelines, secrets, infrastructure, jobs, or production data may be changed.

## Questions to answer

1. What exact files and sub-datasets are currently published?
2. What dates, pollsters, measures and demographic breakdowns are present?
3. Are paired counts/proportions files structurally consistent?
4. Are subgroup base sizes available and usable for publication-quality safeguards?
5. Are there exact duplicate rows, impossible dates, invalid proportions/counts or asymmetric schemas?
6. How frequently is the repository updated, and how should an exact source version be pinned later?
7. What attribution and reuse language is actually published?
8. Is commercial/public EirePolitic use sufficiently clear, or should permission be sought before production use?

## Evidence to collect

- exact repository/file paths;
- latest upstream commit/version used for checks;
- current row/column counts and date coverage;
- field/column inventories by file family;
- pollster/source identifiers;
- demographic variables and subgroup counts;
- duplicate/null/range/date diagnostics;
- counts-versus-proportions consistency where directly comparable;
- licensing/citation/republication language;
- source-update behavior and schema-evolution risk;
- recommended future ingestion mode and validation rules;
- final readiness classification: Ready / Technically ready but permission clarification needed / Defer.

## Method

1. Inspect repository structure and published documentation first.
2. Verify exact current file paths and latest source commit.
3. Use bounded read-only file inspection.
4. If exact whole-file diagnostics are needed, use an isolated temporary research workflow pinned to the upstream commit.
5. Remove all temporary workflow/output files before merging documentation.
6. Do not build or deploy a production connector.

## Research log

### 2026-09-03 — plan

- Created the bounded validation plan.
- Production changes: none.
- Next: inspect the current repository structure, attribution language and exact CSV families.

## Living next step

Inspect the current upstream repository and published terms/citation guidance, then pin an exact commit for bounded diagnostics.
