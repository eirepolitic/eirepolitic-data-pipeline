# Oireachtas Unified Build Packet Status

**Branch:** `main`  
**Last updated:** 2026-06-12  
**Current packet:** W02 — monthly refresh workflow

This is the compact operational handoff for `docs/oireachtas_unified_data_model_plan.md`. Continue from `main`. Existing legacy pipelines remain untouched while unified replacements are built and validated table-by-table.

## Shared implementation state

- CLI: `python -m extract.oireachtas.build_table`
- Registry: `configs/oireachtas/tables.yml`
- Manual test workflow: `.github/workflows/oireachtas_table_test.yml`
- Weekly refresh workflow: `.github/workflows/oireachtas_weekly_refresh.yml`
- AWS region: `ca-central-1`
- S3 bucket: `eirepolitic-data`
- Review branch: `oireachtas-review-output`
- Review publishing preserves existing table folders and runs after table/DQ failure when local review output exists.
- Standard confirmed outputs: raw API/source files, partitioned CSV, partitioned Parquet, latest CSV/Parquet pointers, run manifest, review sample/schema/manifest/DQ.

## Completed foundation packets

- **F01** package/registry skeleton.
- **F02** S3 and review-branch smoke test, run `26832499568`, success.
- **F03** API discovery, run `26832847170`, success. Confirmed `/houses`, `/members`, `/debates`, `/divisions`, `/votes`, `/questions`, `/legislation`, `/parties`, `/constituencies`. `/divisions` is canonical; `/votes` is fallback.

## Confirmed silver packets

- **T01 — `silver_houses`**: run `26847237939`; 25 rows; PK `house_uri`; DQ pass.
- **T02 — `silver_constituencies`**: run `27069529002`; 43 rows; PK `constituency_uri`; DQ pass.
- **T03 — `silver_parties`**: run `27069711527`; 11 rows; PK `party_uri`; DQ pass.
- **T04 — `silver_members`**: run `27070132888`; 25 rows; PK `member_code`; DQ pass.
- **T05 — `silver_member_memberships`**: run `27070298915`; 25 rows; PK `membership_id`; DQ pass.
- **T06 — `silver_member_parties`**: run `27097902733`; 25 rows; PK `member_party_id`; DQ pass.
- **T07 — `silver_member_constituencies`**: run `27098119595`; 25 rows; PK `member_constituency_id`; DQ pass.
- **T08 — `silver_member_offices`**: run `27098313330`; 77 rows; PK `member_office_id`; DQ pass.
- **T09 — `silver_source_files`**: run `27098621113`; 25 rows; PK `source_file_id`; DQ pass.
- **T10 — `silver_debate_records`**: run `27098769263`; 2 rows; PK `debate_id`; DQ pass.
- **T11 — `silver_debate_sections`**: run `27099679458`; 8 rows; PK `debate_section_id`; DQ pass.
- **T12 — `silver_speeches`**: run `27222202849`; 357 rows; PK `speech_id`; DQ pass.
- **T13 — `silver_divisions`**: run `27222935479`; 3 rows; PK `division_id`; DQ pass.
- **T14 — `silver_division_tallies`**: run `27236879805`; 9 rows; PK `division_tally_id`; DQ pass.
- **T15 — `silver_member_votes`**: run `27291681684`; 512 rows; PK `member_vote_id`; DQ pass.
- **T16 — `silver_questions`**: run `27292008182`; 10 rows; PK `question_id`; DQ pass.
- **T17 — `silver_bills`**: run `27325455277`; 10 rows; PK `bill_id`; DQ pass.
- **T18 — `silver_bill_versions`**: run `27326814396`; 10 rows; PK `bill_version_id`; DQ pass.
- **T19 — `silver_bill_stages`**: run `27327648268`; 16 rows; PK `bill_stage_id`; DQ pass.
- **T20 — `silver_bill_related_docs`**: run `27328140775`; 1 row; PK `related_doc_id`; DQ pass.
- **T21 — `silver_bill_sponsors`**: run `27328994935`; 36 rows; PK `bill_sponsor_id`; DQ pass.
- **T22 — `silver_bill_debates`**: run `27356675022`; 12 rows; PK `bill_debate_id`; DQ pass.
- **T23 — `silver_bill_events`**: run `27359680226`; 19 rows; PK `bill_event_id`; DQ pass.

## Confirmed gold packets

- **G01 — `gold_current_members`**: run `27362834154`; 10 rows; PK `member_code`; DQ pass.
- **G02 — `gold_member_activity_yearly`**: run `27363058073`; 10 rows; 173 member-year rows before limit; PK `member_code`, `year`; DQ pass.
- **G03 — `gold_member_activity_monthly`**: run `27364362461`; 10 rows; 173 member-month rows before limit; PK `member_code`, `year_month`; DQ pass.
- **G04 — `gold_constituency_activity_yearly`**: run `27364432622`; 10 rows; PK `constituency_name`, `year`; DQ pass.
- **G05 — `gold_content_fact_pool`**: run `27364496528`; 10 rows; 60 candidate facts before limit; PK `fact_id`; DQ pass.

## Confirmed control packets

### C01 — `control_pipeline_runs`

- Builder: `extract/oireachtas/table_control_pipeline_runs.py`
- Run: `27365125297`; run number 46; success.
- Manifest objects found/read: 40/40; read errors 0.
- Output rows: 10; PK `run_id`; DQ pass.
- Review: `review/control_pipeline_runs/latest/{manifest.json,sample.csv,dq.json}`.

### C02 — `control_table_manifests`

- Builder: `extract/oireachtas/table_control_table_manifests.py`
- CLI/workflow updates:
  - `extract/oireachtas/build_table.py`
  - `.github/workflows/oireachtas_table_test.yml`
- First validation run: `27396546367`; run number 47; success.
- Latest W01 refresh validation also passed with run ID `control_table_manifests_20260612T053120Z`.
- Manifest objects found/read in latest review: 60/60; read errors 0.
- Latest table rows before limit: 31
- Output rows: 10, limited by workflow test default `limit=10`.
- PK: `table_name`, unique
- DQ: pass
- Review: `review/control_table_manifests/latest/{manifest.json,sample.csv,dq.json}`.

### C03 — `control_data_quality_results`

- Builder: `extract/oireachtas/table_control_data_quality_results.py`
- CLI/workflow updates:
  - `extract/oireachtas/build_table.py`
  - `.github/workflows/oireachtas_table_test.yml`
- First validation run: `27396584533`; run number 48; success.
- Latest W01 refresh validation also passed with run ID `control_data_quality_results_20260612T053125Z`.
- Manifest objects found in latest review: 61
- Candidate DQ rows before limit: 244
- Output rows: 10, limited by workflow test default `limit=10`.
- PK: `dq_result_id`, unique
- DQ: pass
- Review: `review/control_data_quality_results/latest/{manifest.json,sample.csv,dq.json}`.

## Confirmed workflow packets

### W01 — weekly refresh workflow

- Workflow file: `.github/workflows/oireachtas_weekly_refresh.yml`
- Workflow ID: `294426406`
- Manual validation run: `27396638715`
- Run number: 1
- Result: success
- Event: `workflow_dispatch`
- Schedule: `20 3 * * 0`
- Default manual mode: `test`
- Scheduled mode: `incremental`
- Manual run validated the weekly table set step and review publication.
- Weekly table set includes current member/member-bridge tables, recent debate/speech/vote/question tables, current gold activity/fact tables, and control tables.

## Next packet batch

### W02 — monthly refresh workflow

Goal:

- add `.github/workflows/oireachtas_monthly_refresh.yml`;
- include monthly dimensions and legislation tables: constituencies, parties, bills, bill versions, bill stages, related docs, sponsors, bill debates, bill events, gold constituency yearly, content fact pool, and control tables;
- add manual dispatch and monthly cron;
- validate with a manual test run before trusting schedule.

### W03 — yearly refresh workflow

Goal:

- add `.github/workflows/oireachtas_yearly_refresh.yml`;
- include yearly/full-refresh dimensions and annual gold/control outputs;
- add manual dispatch and yearly cron;
- validate with a limited manual test run.

### X01 — cutover comparison report

Goal:

- add deterministic comparison report between old outputs and unified outputs without changing downstream consumers;
- compare old member, speech, vote, and profile-metric outputs against unified silver/gold outputs;
- publish small comparison report to `oireachtas-review-output`;
- no cutover or destructive changes.

Handoff instruction:

```text
Continue from main.
Process packets three at a time.
Start W02 monthly refresh workflow, then W03 yearly refresh workflow, then X01 cutover comparison report.
Workflow default currently points to control_data_quality_results.
Weekly workflow exists and manual run 27396638715 succeeded.
```
