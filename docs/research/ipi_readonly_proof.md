# Irish Polling Indicator read-only ingestion proof

## Purpose

Run a bounded, read-only feasibility proof for the Irish Polling Indicator (IPI), the highest-ranked polling source from the Irish political data-source investigation.

This is not a production implementation. It must not change production architecture, schemas, pipelines, infrastructure, secrets, jobs, or production data.

## Questions to answer

1. What are the exact current machine-readable files?
2. What columns/fields do the raw polls and modeled estimates expose?
3. What date coverage is present now?
4. What party columns are present now?
5. Are there obvious duplicate, null, or type-consistency issues visible from bounded read-only inspection?
6. How should source version/retrieval metadata be recorded later?
7. Which files should be treated as mutable development data versus stable historical snapshots?
8. What minimum checks should a future proof-of-concept enforce before any production design is considered?

## Evidence to record

For each inspected file/resource:

- canonical source URL/path;
- file format;
- current source version/date where exposed;
- column names;
- first/last date or documented coverage;
- pollster/sample/fieldwork metadata availability;
- party/result fields;
- modeled uncertainty fields where applicable;
- duplicate/null observations visible from bounded inspection;
- update/version behaviour;
- attribution/citation requirement;
- recommended future source key/version fields;
- failure/monitoring risks.

## Method

- Use only public read-only web/GitHub access.
- Do not create an account or token.
- Do not write any source data into production systems.
- Do not add connector code or workflows.
- Do not treat modeled estimates as raw polls.
- Do not infer missing methodology fields.
- Record limitations when file-level inspection is not possible through available read-only tooling.

## Phase order

1. Identify exact current IPI development files and repository structure.
2. Inspect raw polling file structure and current coverage.
3. Inspect modeled-estimate file structure and current coverage.
4. Check versioning/update behaviour and stable DOI release relationship.
5. Record bounded data-quality observations and recommended future validation rules.
6. Conclude whether IPI is ready for a later non-production ingestion proof.

## Research log

### 2026-09-03 — plan

- Created the read-only IPI proof plan.
- Production changes: none.
- Next: inspect the exact current development files and schemas.

## Living next step

Inspect the exact IPI repository files and document current raw-poll and estimate schemas before any implementation work is considered.
