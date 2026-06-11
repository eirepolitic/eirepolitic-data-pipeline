# Oireachtas Unified Build Packet Status

**Branch:** `main`  
**Last updated:** 2026-06-11  
**Current packet:** G01 — `gold_current_members`

This is the compact operational handoff for `docs/oireachtas_unified_data_model_plan.md`. Continue from `main`. Existing legacy pipelines remain untouched while unified replacements are built and validated table-by-table.

## Shared implementation state

- CLI: `python -m extract.oireachtas.build_table`
- Registry: `configs/oireachtas/tables.yml`
- Manual test workflow: `.github/workflows/oireachtas_table_test.yml`
- AWS region: `ca-central-1`
- S3 bucket: `eirepolitic-data`
- Review branch: `oireachtas-review-output`
- Review publishing preserves existing table folders and runs after table/DQ failure when local review output exists.
- Standard confirmed outputs: raw API/source files, partitioned CSV, partitioned Parquet, latest CSV/Parquet pointers, run manifest, review sample/schema/manifest/DQ.

## Completed foundation packets

- **F01** package/registry skeleton.
- **F02** S3 and review-branch smoke test, run `26832499568`, success.
- **F03** API discovery, run `26832847170`, success. Confirmed `/houses`, `/members`, `/debates`, `/divisions`, `/votes`, `/questions`, `/legislation`, `/parties`, `/constituencies`. `/divisions` is canonical; `/votes` is fallback.

## Confirmed table packets

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
- **T12 — `silver_speeches`**: run `27222202849`; 357 rows; PK `speech_id`; DQ pass; speaker member-code enrichment 344/357 rows, 96.36%.
- **T13 — `silver_divisions`**: run `27222935479`; 3 rows; PK `division_id`; DQ pass.
- **T14 — `silver_division_tallies`**: run `27236879805`; 9 rows; PK `division_tally_id`; DQ pass.
- **T15 — `silver_member_votes`**: run `27291681684`; 512 rows; PK `member_vote_id`; DQ pass.
- **T16 — `silver_questions`**: run `27292008182`; 10 rows; PK `question_id`; DQ pass.

### T17 — `silver_bills`

- Builder: `extract/oireachtas/table_bills.py`
- Final run: `27325455277`; run number 33; success.
- Raw legislation rows: 10; output rows: 10; PK `bill_id`, unique; DQ pass.
- Final run ID: `silver_bills_20260611T051524Z`.
- Confirmed nested shapes:
  - versions: `bill.versions[].version.{uri,showAs,date,docType,lang,formats}`;
  - stages: `bill.stages[].event.{uri,showAs,stageURI,stageOutcome,stageCompleted,progressStage,dates,house,chamber}`;
  - related docs: `bill.relatedDocs[].relatedDoc.{uri,showAs,date,docType,lang,formats}`;
  - sponsors: `bill.sponsors[].sponsor.{by,as,isPrimary}`;
  - debates: `bill.debates[].{uri,showAs,date,debateSectionId,chamber}`;
  - events: `bill.events[].event.{uri,eventURI,showAs,dates,chamber}`.

### T18 — `silver_bill_versions`

- Builder: `extract/oireachtas/table_bill_versions.py`
- Final run: `27326814396`; run number 35; success.
- Raw legislation rows: 10; raw version rows: 10; output rows: 10; bills with versions: 10.
- PK `bill_version_id`, unique; DQ pass.
- Final run ID: `silver_bill_versions_20260611T055216Z`.
- PDF source rows: 10; XML source rows: 0 because `version.formats.xml` was null in the sample payload.
- Source-file ID pattern: `source_file:{stable_hash(['legislation', bill_id, format_type, format_uri, format_url], length=24)}`.
- Review: `review/silver_bill_versions/latest/{manifest.json,sample.csv,dq.json}`.

### T19 — `silver_bill_stages`

- Builder: `extract/oireachtas/table_bill_stages.py`
- Final run: `27327648268`; run number 36; success.
- Raw legislation rows: 10; raw stage rows: 16; output rows: 16; bills with stages: 10.
- PK `bill_stage_id`, unique; DQ pass.
- Final run ID: `silver_bill_stages_20260611T061231Z`.
- Stage names observed: `Committee Stage`, `First Stage`, `Report Stage`, `Second Stage`.
- Stage outcome values observed: `Current`; blank outcomes allowed where API returns null.
- House names observed: `26th Seanad`, `27th Seanad`.
- Review: `review/silver_bill_stages/latest/{manifest.json,sample.csv,dq.json}`.

### T20 — `silver_bill_related_docs`

- Builder: `extract/oireachtas/table_bill_related_docs.py`
- Final run: `27328140775`; run number 37; success.
- Raw legislation rows: 10; raw related-doc rows: 1; output rows: 1; bills with related docs: 1.
- PK `related_doc_id`, unique; DQ pass.
- Final run ID: `silver_bill_related_docs_20260611T062419Z`.
- PDF source rows: 1; XML source rows: 0 because `relatedDoc.formats.xml` was null.
- T09-compatible source-file IDs: pass.
- Observed doc type/language: `memo`, `eng`.
- Review: `review/silver_bill_related_docs/latest/{manifest.json,sample.csv,dq.json}`.

### T21 — `silver_bill_sponsors`

- Builder: `extract/oireachtas/table_bill_sponsors.py`
- Final run: `27328994935`; run number 38; success.
- Raw legislation rows: 10; raw sponsor rows: 36; output rows: 36; bills with sponsors: 10; primary sponsor rows: 10.
- PK `bill_sponsor_id`, unique; DQ pass.
- Final run ID: `silver_bill_sponsors_20260611T064401Z`.
- `is_primary` normalized to `true`/`false`.
- `sponsor.as` was null in the sample payload, so role fields are blank and allowed.
- Review: `review/silver_bill_sponsors/latest/{manifest.json,sample.csv,dq.json}`.

### T22 — `silver_bill_debates`

- Builder: `extract/oireachtas/table_bill_debates.py`
- Final run: `27356675022`; run number 39; success.
- Raw legislation rows: 10; raw debate rows: 12; output rows: 12; bills with debates: 8; debate-section rows: 12.
- PK `bill_debate_id`, unique; DQ pass.
- Final run ID: `silver_bill_debates_20260611T150745Z`.
- Chamber values observed: `Seanad Éireann`.
- Review: `review/silver_bill_debates/latest/{manifest.json,sample.csv,dq.json}`.

### T23 — `silver_bill_events`

- Registry added: `configs/oireachtas/tables.yml`
- Builder: `extract/oireachtas/table_bill_events.py`
- CLI/workflow updates:
  - `extract/oireachtas/build_table.py`
  - `.github/workflows/oireachtas_table_test.yml`
- Final run: `27359680226`
- Run number: 40
- Result: success
- Raw legislation rows: 10
- Raw event rows: 19
- Output rows: 19
- Bills with events: 10
- PK: `bill_event_id`, unique
- DQ: pass
- Endpoint: `/legislation?chamber=dail&house_no=34&date_start=2025-01-01&date_end=2025-01-31&limit=10`.
- Final run ID: `silver_bill_events_20260611T155424Z`.
- Normalized columns:
  - `bill_event_id`
  - `bill_id`
  - `event_uri`
  - `event_type_uri`
  - `event_name`
  - `event_date`
  - `chamber_uri`
  - `chamber_name`
  - `event_order`
  - `snapshot_date`
- DQ checks passed:
  - row count > 0;
  - required columns present;
  - primary key non-null and unique;
  - `bill_id` populated, preserving join to `silver_bills.bill_id`;
  - event URI/type URI/name/date populated;
  - chamber URI/name populated;
  - event order populated.
- Event names observed: `Admissibility for Introduction`, `Approved for Initiation`, `Bill Lapsed`, `Bill Restored`.
- Chamber values observed: `Dáil Éireann`, `Seanad Éireann`.
- Review:
  - `review/silver_bill_events/latest/manifest.json`
  - `review/silver_bill_events/latest/sample.csv`
  - `review/silver_bill_events/latest/dq.json`

## Next packet

### G01 — `gold_current_members`

Goal:

- add the current-member roster mart using confirmed silver member tables;
- build one row per current member from `silver_members`, `silver_member_memberships`, `silver_member_parties`, `silver_member_constituencies`, and `silver_member_offices`;
- preserve `member_code` as the primary key;
- output `member_code`, `full_name`, `party_name`, `constituency_name`, `house_no`, `office_name`, and `snapshot_date`;
- use latest/current bridge rows where available;
- publish CSV, Parquet, latest pointers, manifest, schema, DQ, and review sample;
- validate row count > 0, `member_code` unique, member names populated, and current party/constituency fields populated where silver exposes them.

Expected files:

- update `configs/oireachtas/tables.yml` if `gold_current_members` needs a schema adjustment
- `extract/oireachtas/table_gold_current_members.py`
- update `extract/oireachtas/build_table.py`
- update `.github/workflows/oireachtas_table_test.yml` default to `gold_current_members`
- update this status file after validation

Suggested test command:

```bash
python -m extract.oireachtas.build_table \
  --table gold_current_members \
  --mode test \
  --chamber dail \
  --house-no 34 \
  --limit 25 \
  --write-review-sample
```

Handoff instruction:

```text
Continue from main.
Start G01 — gold_current_members.
Workflow default currently points to silver_bill_events.
Use confirmed latest silver member tables as inputs or rebuild/fetch them consistently from the API if no local latest-table reader exists yet.
```
