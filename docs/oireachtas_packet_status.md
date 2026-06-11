# Oireachtas Unified Build Packet Status

**Branch:** `main`  
**Last updated:** 2026-06-11  
**Current packet:** T23 — `silver_bill_events`

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

- Registry added: `configs/oireachtas/tables.yml`
- Builder: `extract/oireachtas/table_bill_debates.py`
- CLI/workflow updates:
  - `extract/oireachtas/build_table.py`
  - `.github/workflows/oireachtas_table_test.yml`
- Final run: `27356675022`
- Run number: 39
- Result: success
- Raw legislation rows: 10
- Raw debate rows: 12
- Output rows: 12
- Bills with debates: 8
- Debate-section rows: 12
- PK: `bill_debate_id`, unique
- DQ: pass
- Endpoint: `/legislation?chamber=dail&house_no=34&date_start=2025-01-01&date_end=2025-01-31&limit=10`.
- Final run ID: `silver_bill_debates_20260611T150745Z`.
- Normalized columns:
  - `bill_debate_id`
  - `bill_id`
  - `debate_id`
  - `debate_uri`
  - `debate_date`
  - `debate_show_as`
  - `debate_section_id`
  - `chamber_uri`
  - `chamber_name`
  - `debate_order`
  - `snapshot_date`
- DQ checks passed:
  - row count > 0;
  - required columns present;
  - primary key non-null and unique;
  - `bill_id` populated, preserving join to `silver_bills.bill_id`;
  - debate ID/URI/date/show-as populated;
  - debate section ID populated;
  - chamber URI/name populated;
  - debate order populated.
- Chamber values observed: `Seanad Éireann`.
- Review:
  - `review/silver_bill_debates/latest/manifest.json`
  - `review/silver_bill_debates/latest/sample.csv`
  - `review/silver_bill_debates/latest/dq.json`

## Next packet

### T23 — `silver_bill_events`

Goal:

- add bill event bridge from `bill.events[].event`;
- add registry entry if absent;
- build one row per bill event, preserving join to `silver_bills.bill_id`;
- normalize `bill_event_id`, `bill_id`, `event_uri`, `event_type_uri`, `event_name`, `event_date`, `chamber_uri`, `chamber_name`, `event_order`, and `snapshot_date`;
- derive `event_date` from `event.dates[].date`, using earliest parseable date unless multiple event-date rows are required later;
- use `event.uri` as stable identity where available, with deterministic fallback hash;
- publish raw JSON, CSV, Parquet, latest pointers, manifest, schema, DQ, and review sample;
- validate row count > 0, primary key unique, `bill_id` populated, event URI/name/date populated where API exposes them, and event order populated.

Expected files:

- update `configs/oireachtas/tables.yml` if `silver_bill_events` is absent
- `extract/oireachtas/table_bill_events.py`
- update `extract/oireachtas/build_table.py`
- update `.github/workflows/oireachtas_table_test.yml` default to `silver_bill_events`
- update this status file after validation

Suggested test command:

```bash
python -m extract.oireachtas.build_table \
  --table silver_bill_events \
  --mode test \
  --chamber dail \
  --house-no 34 \
  --date-start 2025-01-01 \
  --date-end 2025-01-31 \
  --limit 10 \
  --write-review-sample
```

Handoff instruction:

```text
Continue from main.
Start T23 — silver_bill_events.
Workflow default currently points to silver_bill_debates.
Use bill.events[].event from the confirmed T17 payload.
```
