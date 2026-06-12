# Oireachtas Unified Build Packet Status

**Branch:** `main`  
**Last updated:** 2026-06-12  
**Current packet:** P01 — latest publishing control

This is the compact operational handoff for `docs/oireachtas_unified_data_model_plan.md`. Continue from `main`. Existing legacy pipelines remain untouched while unified replacements are built and validated table-by-table.

## Shared implementation state

- CLI: `python -m extract.oireachtas.build_table`
- Registry: `configs/oireachtas/tables.yml`
- Manual test workflow: `.github/workflows/oireachtas_table_test.yml`
- Weekly refresh workflow: `.github/workflows/oireachtas_weekly_refresh.yml`
- Monthly refresh workflow: `.github/workflows/oireachtas_monthly_refresh.yml`
- Yearly refresh workflow: `.github/workflows/oireachtas_yearly_refresh.yml`
- Cutover comparison workflow: `.github/workflows/oireachtas_cutover_comparison.yml`
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

- **C01 — `control_pipeline_runs`**: run `27365125297`; 10 rows; PK `run_id`; DQ pass.
- **C02 — `control_table_manifests`**: run `27396546367`; 10 rows; PK `table_name`; DQ pass.
- **C03 — `control_data_quality_results`**: run `27396584533`; 10 rows; PK `dq_result_id`; DQ pass.

## Confirmed workflow packets

- **W01 — weekly refresh workflow**: workflow ID `294426406`; run `27396638715`; success; schedule `20 3 * * 0`.
- **W02 — monthly refresh workflow**: workflow ID `294432002`; run `27397121321`; success; schedule `35 4 1 * *`.
- **W03 — yearly refresh workflow**: workflow ID `294432103`; run `27397123885`; success; schedule `15 5 2 1 *`.

## Confirmed comparison/planning packets

### X01 — cutover comparison report

- Builder: `extract/oireachtas/cutover_comparison.py`
- Workflow: `.github/workflows/oireachtas_cutover_comparison.yml`
- Workflow ID: `294432488`
- Final validation run: `27397256307`; run number 2; success.
- Output rows: 5; PK `comparison_name`; DQ pass.
- Review: `review/cutover_comparison_report/latest/{manifest.json,sample.csv,dq.json,report.md}`.
- Note: all compared unified latest outputs existed, but many unified outputs are still limited by test/default limits, so row-count differences are expected until full refresh modes are run with higher limits.

### X02 — downstream cutover planning

- Document: `docs/oireachtas_downstream_cutover_plan.md`
- Result: complete.
- Identified downstream consumers and old-to-new S3 key candidates.
- Recommendation: do not cut over yet; build compatibility adapters and run side-by-side profile metric trial first.
- Legacy workflows and Instagram consumers were not changed.

### X03 — production run configuration review

- Document: `docs/oireachtas_production_run_config_review.md`
- Result: complete.
- Key finding: scheduled workflows are validated but still use fixed historic date windows; manual test runs can overwrite latest pointers with limited outputs.
- Recommendation: add latest-publishing controls and dynamic date windows before production cutover.

### X04 — registry/status cleanup

- File: `configs/oireachtas/tables.yml`
- Result: complete.
- Updated all validated silver, gold, and control tables from `planned` to `confirmed`.
- Runtime table schemas and builders were not changed.

## Next packet batch

### P01 — latest publishing control

Goal:

- add a CLI/workflow control so `mode=test` writes partitioned/review outputs but does not overwrite `processed/oireachtas_unified/latest/*`;
- preserve current latest behavior for non-test/manual production runs after explicit choice;
- validate with a safe smoke test.

### P02 — dynamic date windows

Goal:

- add dynamic scheduled date windows for weekly/monthly/yearly workflows;
- remove fixed January 2025 scheduled windows where appropriate;
- keep manual override inputs available.

### P03 — downstream compatibility adapters

Goal:

- build compatibility CSVs under `processed/oireachtas_unified/compat/...` for member roster and member votes;
- do not overwrite old keys;
- enable side-by-side member-profile metric trial.

Handoff instruction:

```text
Continue from main.
Process packets three at a time.
Start P01 latest publishing control, then P02 dynamic date windows, then P03 downstream compatibility adapters.
Do not repoint downstream consumers or disable old workflows without explicit user approval.
Latest validated workflows: W02 run 27397121321, W03 run 27397123885, X01 run 27397256307.
Documentation added: docs/oireachtas_downstream_cutover_plan.md and docs/oireachtas_production_run_config_review.md.
Registry statuses are confirmed for all validated tables.
```
