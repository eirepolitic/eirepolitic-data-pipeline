# Irish referendum result data validation

## Purpose

Run a bounded, read-only validation of official Irish referendum result CSVs across multiple years to determine whether they can support recurring EirePolitic historical graphics and dashboards.

This remains research-only. No production architecture, schemas, pipelines, jobs, secrets, infrastructure or production data may be changed.

## Scope

Validate four official Department of Housing, Local Government and Heritage datasets:

1. Tenth Amendment referendum — 1986.
2. Thirteenth Amendment referendum — 1992.
3. Thirty-fifth Amendment referendum — 2015.
4. Thirty-sixth Amendment referendum — 2018.

All four are published as CSV and are listed under CC BY-SA 4.0.

## Questions to answer

1. Are the four CSV schemas structurally compatible?
2. What header/name drift exists across decades?
3. Are electorate, poll, turnout, yes/no and spoilt-vote fields complete and internally consistent?
4. Are there duplicate constituency rows or invalid numeric values?
5. Does `Total Poll = votes for + votes against + spoilt` hold?
6. Does published turnout percentage match `Total Poll / Electorate` within normal rounding tolerance?
7. How many constituencies exist in each referendum, and how much geography changes between years?
8. Can the CKAN datastore be used directly as an API alternative to CSV download?
9. What source metadata/licensing fields must a later loader preserve?
10. What normalization rules are safe without erasing historical constituency boundaries?

## Evidence to collect

For each file:

- official dataset/resource URL and resource ID;
- licence;
- file hash, row count and exact columns;
- constituency count and labels;
- duplicate/null counts;
- numeric ranges;
- turnout consistency;
- vote-total consistency;
- datastore availability;
- schema differences;
- boundary/geography caveats;
- future ingestion mode and readiness rating.

## Method

1. Verify official catalogue metadata and direct download URLs.
2. Fetch all four CSVs read-only.
3. Query the CKAN datastore for each resource.
4. Run exact bounded cross-year diagnostics in an isolated temporary research workflow.
5. Document only the durable findings.
6. Remove temporary workflow/output files before documentation merge.

## Research log

### 2026-09-03 — plan

- Created the referendum validation plan.
- Confirmed the selected 1986, 1992, 2015 and 2018 datasets are official CSV resources under CC BY-SA 4.0.
- Production changes: none.
- Next: run exact cross-year schema and arithmetic validation.

## Living next step

Validate the four pinned referendum resources, classify schema drift and geography differences, then update the overall top-five ingestion readiness order.