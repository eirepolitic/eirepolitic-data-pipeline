# Oireachtas Unified Data Model + Pipeline Plan

**Repository:** `eirepolitic-data-pipeline`  
**Document role:** Source of truth for the Oireachtas database layer behind the Instagram content creation platform.  
**Created:** 2026-06-02  
**Last updated:** 2026-06-02  
**Status:** Plan created. Unified pipelines not yet built. Existing pipelines remain active.  
**Primary S3 bucket:** `s3://eirepolitic-data`  
**AWS region observed:** `ca-central-1`  
**Scope:** deterministic Oireachtas API extraction, normalization, CSV/Parquet writing, S3 layout, refresh schedules, GitHub Actions runners, and table validation.  
**Out of scope:** LLM issue classification, LLM summarisation, Instagram publishing, scheduling posts, automatic approval.

---

## 1. Mandatory operating rules for future agents

1. Read this file before changing any Oireachtas database work.
2. Inspect the current repo before editing; other chats may have changed files.
3. Keep old pipelines until the unified replacement is confirmed table-by-table.
4. Build new pipelines in parallel under new paths/prefixes.
5. Use existing repo scripts to learn working API parameters and parsing patterns.
6. Use Oireachtas documentation plus live test pulls because the API is messy and docs are imperfect.
7. Work table-by-table: design, build test, run, inspect small sample, patch, retry, confirm, document, move on.
8. Ask the user only when a functional, cost, legal, or architecture decision cannot safely be made from context.
9. Keep chat replies short.
10. Never say a GitHub/AWS tool cannot be used without first checking available tools and prior chat/tool history.
11. With the GitHub action tool, pass only the repository name:

```json
{"repo": "eirepolitic-data-pipeline"}
```

Do **not** pass `owner/repo`.

---

## 2. Update rule for this document

Update this file whenever any of these change:

- table added, removed, renamed, or re-keyed;
- pipeline file added or modified;
- workflow/schedule added or modified;
- S3 prefix/path convention changed;
- parser behaviour changed;
- validation rule changed;
- table status changes from planned/tested/confirmed;
- old pipeline is deprecated or replaced;
- API behaviour is discovered that affects extraction;
- user makes a decision that affects design.

Every update must add a dated entry in **Section 15 — Change log**.

---

## 3. Goal

Create a unified, stable, joinable Oireachtas data model for Instagram content and future analytics.

The system must:

1. pull Oireachtas API metadata;
2. download linked XML/PDF/source files where useful;
3. preserve raw API/file data;
4. normalize messy nested payloads into clean tables;
5. write CSV and Parquet versions to S3;
6. expose small review samples that the assistant can inspect;
7. refresh tables weekly, monthly, or yearly based on volatility;
8. support joins across members, memberships, parties, constituencies, debates, speeches, divisions/votes, questions, and legislation;
9. provide deterministic base tables for later LLM enrichment.

---

## 4. Current repo inventory observed on `main`

### 4.1 Existing Oireachtas-related scripts

| File | Current purpose | Reuse notes |
|---|---|---|
| `extract/monthly_extract.py` | Fetches all Dáil 34 debates from `/debates`, downloads XML files, writes to `raw/debates/xml/`. | Reuse pagination, `formats.xml.uri`, `data.oireachtas.ie` file download, retry pattern. |
| `extract/debates_xml_to_csv_s3.py` | Parses raw debate XML from S3 into a denormalized speeches CSV. | Reuse Akoma Ntoso XML namespace approach; improve IDs, nested text handling, speaker matching. |
| `extract/monthly_members_extract.py` | Fetches Dáil 34 members and writes one flattened CSV. | Reuse member/membership/party/constituency traversal. |
| `process/build_dail_votes_member_records.py` | Pulls votes/divisions and writes division/member-vote CSV + Parquet. | Reuse tally flattening; verify endpoint because docs list `/divisions` while script uses `/votes`. |
| `process/debate_speeches_csv_to_parquet.py` | Converts CSV to Parquet with cleaned columns. | Reuse Parquet and column-cleaning conventions. |
| `process/build_member_profile_metrics_2025.py` | Builds downstream Instagram metrics from members, classified speeches, votes, photos. | Later replace inputs with unified gold/silver tables. |

### 4.2 Existing workflows

| File | Current purpose |
|---|---|
| `.github/workflows/monthly_extract.yml` | Monthly debates XML extract, XML-to-CSV, and members extract. |
| `.github/workflows/build_member_profile_metrics_2025.yml` | Manual vote pull and member profile metrics build. |
| `.github/workflows/speech_issue_classifier.yml` | Manual LLM issue classifier and Parquet conversion; out of scope for this deterministic layer. |

### 4.3 Existing S3 paths to avoid overwriting

| Path | Current use |
|---|---|
| `raw/debates/xml/` | Existing raw debate XML. |
| `raw/debates/debate_speeches_extracted.csv` | Existing parsed speech CSV. |
| `raw/members/oireachtas_members_34th_dail.csv` | Existing member CSV. |
| `processed/votes/dail_vote_divisions.csv` | Existing vote division CSV. |
| `processed/votes/parquets/dail_vote_divisions.parquet` | Existing vote division Parquet. |
| `processed/votes/dail_vote_member_records.csv` | Existing member-vote CSV. |
| `processed/votes/parquets/dail_vote_member_records.parquet` | Existing member-vote Parquet. |
| `processed/members/member_profile_metrics_2025.csv` | Existing Instagram metrics CSV. |

---

## 5. External API facts to design around

Observed from Oireachtas open-data/API docs and existing repo code:

1. API base: `https://api.oireachtas.ie/v1`.
2. Linked raw files are usually under `https://data.oireachtas.ie`.
3. API metadata identifies source files via nested `formats` fields.
4. Available documented datasets include debates, divisions/votes, parliamentary questions, legislation, members, houses, parties, and constituencies.
5. Oireachtas documentation says API datasets are intended to be used with `data.oireachtas.ie` raw datasets.
6. Data can appear as soon as published; late edits/revisions must be expected.
7. Current repo uses `/votes`; docs/data.gov list votes as `/divisions`. The unified client must test both and standardize the working documented endpoint with fallback.

---

## 6. Target architecture

```text
Oireachtas API + data.oireachtas.ie
  -> bronze raw JSON/XML/PDF + manifests
  -> silver normalized joinable CSV/Parquet tables
  -> gold deterministic Instagram-ready marts
  -> later LLM enrichment layer
```

### 6.1 New repo structure

Build new code in a separate package so old scripts remain untouched:

```text
extract/oireachtas/
  __init__.py
  client.py              # API client, pagination, retry, endpoint alias handling
  io_s3.py               # S3 CSV/Parquet/JSON/XML helpers
  normalize.py           # text/date/name/URI/hash helpers
  schemas.py             # table schema and validation helpers
  xml_debates.py         # debate XML parser
  xml_questions.py       # question XML parser if needed
  build_table.py         # CLI: build one table
  build_all.py           # cadence orchestrator

configs/oireachtas/
  tables.yml             # table registry
  api_params.yml         # default chamber/house/date params
  schema_overrides.yml   # API quirks and field aliases

tests/oireachtas/
  test_normalize.py
  test_xml_debates.py
  test_table_contracts.py
  fixtures/

.github/workflows/
  oireachtas_table_test.yml
  oireachtas_weekly_refresh.yml
  oireachtas_monthly_refresh.yml
  oireachtas_yearly_refresh.yml
```

### 6.2 Standard CLI shape

All new table builds should use one entry point:

```bash
python -m extract.oireachtas.build_table \
  --table silver_members \
  --mode test \
  --chamber dail \
  --house-no 34 \
  --date-start 2025-01-01 \
  --date-end 2025-01-31 \
  --limit 25 \
  --write-review-sample true \
  --sample-rows 10
```

Modes:

| Mode | Meaning |
|---|---|
| `test` | small sample, review outputs, no production overwrite |
| `incremental` | recent/current changed data |
| `full` | full refresh for a table/scope |
| `backfill` | historical load by date/house range |

---

## 7. S3 layout for unified work

Use `oireachtas_unified` while developing to avoid collisions.

```text
raw/oireachtas_unified/api/<endpoint>/snapshot_date=<YYYY-MM-DD>/run_id=<run_id>/page-00000.json
raw/oireachtas_unified/files/<source_type>/<id_or_date>/<file>
processed/oireachtas_unified/silver/<table>/snapshot_date=<YYYY-MM-DD>/run_id=<run_id>/part-00000.parquet
processed/oireachtas_unified/silver_csv/<table>/snapshot_date=<YYYY-MM-DD>/run_id=<run_id>/<table>.csv
processed/oireachtas_unified/latest/parquet/<table>.parquet
processed/oireachtas_unified/latest/csv/<table>.csv
processed/oireachtas_unified/manifests/<table>/run_id=<run_id>.json
processed/oireachtas_unified/review/<table>/latest/sample.csv
processed/oireachtas_unified/review/<table>/latest/schema.json
```

Rules:

1. Parquet is the analytical source of truth.
2. CSV is kept for Appsmith/Power BI/manual review.
3. Raw API pages and raw downloaded files are preserved.
4. Every confirmed table gets a manifest.
5. Every test run writes a small review sample.
6. Do not overwrite old non-unified paths until cutover is approved.

---

## 8. Data model key strategy

Use stable source identifiers first, generated hashes second, display-name joins last.

| Entity | Preferred key | Fallback |
|---|---|---|
| Member | `member_code`, `member_uri` | normalized name only for fallback match |
| House | `house_uri` | `chamber + house_no` |
| Constituency | `constituency_uri` | normalized name + house/date range |
| Party | `party_uri` | normalized party name + date range |
| Debate | `debate_id` or `debate_uri` | date + chamber + XML URI hash |
| Debate section | source section URI/eId | debate ID + order + heading hash |
| Speech | generated `speech_id` | debate + section + order + text hash |
| Division/vote | `division_id` or `vote_id + date` | subject + date + chamber hash |
| Question | `question_id` or URI | date + number + member hash |
| Bill | bill URI / number + year | title + introduced date hash |

Generated deterministic IDs:

```python
sha256("|".join(stable_fields).lower().strip().encode()).hexdigest()[:16]
```

---

## 9. Silver tables

### 9.1 Core dimensions

#### `silver_houses`

Grain: one row per house/chamber/house number.

Columns:

- `house_uri`
- `house_no`
- `house_code`
- `chamber`
- `show_as`
- `date_start`
- `date_end`
- `is_current`
- `source_endpoint`
- `snapshot_date`
- `source_hash`

Cadence: yearly full + manual.

#### `silver_constituencies`

Grain: one row per constituency/date-range/house.

Columns:

- `constituency_uri`
- `constituency_code`
- `constituency_name`
- `show_as`
- `house_uri`
- `house_no`
- `chamber`
- `date_start`
- `date_end`
- `is_current`
- `source_endpoint`
- `snapshot_date`
- `source_hash`

Cadence: monthly current + yearly full.

#### `silver_parties`

Grain: one row per party/date range where available.

Columns:

- `party_uri`
- `party_code`
- `party_name`
- `show_as`
- `date_start`
- `date_end`
- `is_current`
- `source_endpoint`
- `snapshot_date`
- `source_hash`

Cadence: monthly current + yearly full.

#### `silver_members`

Grain: one row per member/person.

Columns:

- `member_code`
- `member_uri`
- `full_name`
- `first_name`
- `last_name`
- `display_name`
- `gender`
- `member_key`
- `is_current_member`
- `latest_party_name`
- `latest_constituency_name`
- `latest_house_no`
- `source_endpoint`
- `snapshot_date`
- `source_hash`

Cadence: weekly current + monthly full.

#### `silver_member_memberships`

Grain: one row per member membership object.

Columns:

- `membership_id`
- `member_code`
- `member_uri`
- `house_uri`
- `house_no`
- `house_code`
- `chamber`
- `membership_start`
- `membership_end`
- `is_current`
- `source_hash`
- `snapshot_date`

Cadence: weekly current + monthly full.

#### `silver_member_parties`

Grain: one row per member-party assignment.

Columns:

- `member_party_id`
- `membership_id`
- `member_code`
- `party_uri`
- `party_name`
- `party_start`
- `party_end`
- `is_current`
- `snapshot_date`

Cadence: weekly current + monthly full.

#### `silver_member_constituencies`

Grain: one row per member-constituency assignment.

Columns:

- `member_constituency_id`
- `membership_id`
- `member_code`
- `constituency_uri`
- `constituency_name`
- `represent_start`
- `represent_end`
- `is_current`
- `snapshot_date`

Cadence: weekly current + monthly full.

#### `silver_member_offices`

Grain: one row per office/ministerial role assignment where available.

Columns:

- `member_office_id`
- `membership_id`
- `member_code`
- `office_uri`
- `office_name`
- `office_start`
- `office_end`
- `is_current`
- `snapshot_date`

Cadence: weekly current + monthly full.

### 9.2 Debates and speeches

#### `silver_debate_records`

Grain: one row per debate record returned by `/debates`.

Columns:

- `debate_id`
- `debate_uri`
- `context_date`
- `debate_date`
- `chamber`
- `house_uri`
- `house_no`
- `house_code`
- `show_as`
- `source_xml_uri`
- `source_xml_url`
- `source_pdf_uri`
- `source_pdf_url`
- `api_result_hash`
- `snapshot_date`

Cadence: weekly recent incremental + monthly reconciliation + yearly full/backfill.

#### `silver_debate_sections`

Grain: one row per debate section.

Columns:

- `debate_section_id`
- `debate_id`
- `section_eid`
- `section_uri`
- `section_order`
- `heading`
- `show_as`
- `parent_section_id`
- `snapshot_date`

Cadence: follows debate records.

#### `silver_speeches`

Grain: one row per speech element in debate XML.

Columns:

- `speech_id`
- `debate_id`
- `debate_section_id`
- `debate_date`
- `speech_order`
- `speaker_ref`
- `speaker_name`
- `speaker_member_code`
- `speaker_match_method`
- `speech_text`
- `speech_text_hash`
- `word_count`
- `char_count`
- `language`
- `xml_source_key`
- `snapshot_date`

Parser requirements:

1. Use XML namespaces defensively.
2. Extract paragraph text with `itertext()`, not only direct `.text`.
3. Preserve unmatched speakers instead of dropping rows.
4. Resolve speakers via XML person refs first, member code/URI second, normalized-name fallback last.
5. Never dedupe only on date + speaker + text; include debate/section/order/hash.

Cadence: weekly recent incremental + monthly reconciliation.

Optional later table: `silver_speech_paragraphs` if paragraph-level data is useful.

### 9.3 Divisions/votes

#### `silver_divisions`

Grain: one row per division/vote event.

Columns:

- `division_id`
- `vote_id`
- `division_date`
- `chamber`
- `house_uri`
- `house_no`
- `committee_code`
- `subject`
- `outcome`
- `debate_id`
- `debate_section_id`
- `debate_show_as`
- `api_result_hash`
- `snapshot_date`

Cadence: weekly recent/current-year incremental + monthly reconciliation.

Important: unlike the existing script, the unified base table should keep committee rows and let downstream marts filter them.

#### `silver_division_tallies`

Grain: one row per division and vote/tally type.

Columns:

- `division_tally_id`
- `division_id`
- `vote_code`
- `vote_label`
- `show_as`
- `member_count`
- `snapshot_date`

Cadence: follows divisions.

#### `silver_member_votes`

Grain: one row per member per division.

Columns:

- `member_vote_id`
- `division_id`
- `vote_id`
- `division_date`
- `member_code`
- `member_name`
- `vote_code`
- `vote_label`
- `party_name_at_vote`
- `constituency_name_at_vote`
- `snapshot_date`

Cadence: follows divisions.

### 9.4 Parliamentary questions

#### `silver_questions`

Grain: one row per question.

Columns:

- `question_id`
- `question_uri`
- `question_date`
- `question_no`
- `question_type`
- `question_text`
- `answer_text`
- `asked_by_member_code`
- `asked_by_name`
- `to_minister_or_department`
- `debate_section_id`
- `source_xml_uri`
- `source_xml_url`
- `source_pdf_uri`
- `source_pdf_url`
- `snapshot_date`
- `source_hash`

Cadence: weekly recent incremental + monthly reconciliation.

Optional deterministic table only if API exposes topics: `silver_question_topics`. No LLM topic tagging in this workstream.

### 9.5 Legislation

#### `silver_bills`

Grain: one row per bill/act item.

Columns:

- `bill_id`
- `bill_uri`
- `bill_no`
- `bill_year`
- `title`
- `short_title`
- `origin_house_uri`
- `origin_house_name`
- `bill_type`
- `status`
- `introduced_date`
- `last_event_date`
- `source_endpoint`
- `snapshot_date`
- `source_hash`

Cadence: monthly incremental + yearly full.

#### `silver_bill_versions`

Grain: one row per bill version/document.

Columns:

- `bill_version_id`
- `bill_id`
- `version_label`
- `version_date`
- `format_pdf_uri`
- `format_pdf_url`
- `format_xml_uri`
- `format_xml_url`
- `s3_pdf_key`
- `s3_xml_key`
- `snapshot_date`

Cadence: monthly. Use `skip_existing` for immutable files.

#### `silver_bill_stages`

Grain: one row per bill stage/progress event.

Columns:

- `bill_stage_id`
- `bill_id`
- `stage_name`
- `stage_date`
- `house_uri`
- `house_name`
- `stage_outcome`
- `order_in_bill`
- `snapshot_date`

Cadence: monthly.

Optional if sponsors are exposed reliably: `silver_bill_sponsors`.

### 9.6 Control tables

#### `control_pipeline_runs`

One row per table run.

Columns:

- `run_id`
- `workflow_run_id`
- `table_name`
- `mode`
- `cadence`
- `started_at_utc`
- `finished_at_utc`
- `status`
- `input_params_json`
- `raw_rows`
- `output_rows`
- `error_message`
- `manifest_s3_key`

#### `control_table_manifests`

Current summary per table output.

Columns:

- `table_name`
- `latest_run_id`
- `latest_snapshot_date`
- `latest_parquet_key`
- `latest_csv_key`
- `row_count`
- `column_count`
- `schema_hash`
- `primary_key_unique`
- `dq_status`
- `updated_at_utc`

#### `control_data_quality_results`

One row per validation check.

Columns:

- `dq_result_id`
- `run_id`
- `table_name`
- `check_name`
- `status`
- `metric_value`
- `threshold`
- `message`
- `created_at_utc`

---

## 10. Gold tables for Instagram/data apps

Build only after required silver inputs are confirmed.

### `gold_current_members`

One row per current member, joining member identity, current membership, party, constituency, and office where available.

Cadence: weekly.

### `gold_member_activity_yearly`

One row per member per year.

Core metrics:

- speech count;
- distinct debate days;
- division count available;
- votes cast count;
- vote participation percentage;
- yes/no/abstain counts;
- speech rank;
- vote participation rank.

Cadence: weekly for current year, yearly freeze for previous years.

### `gold_member_activity_monthly`

One row per member per month for trend content.

Cadence: weekly for current year.

### `gold_constituency_activity_yearly`

One row per constituency per year.

Cadence: monthly.

### `gold_content_fact_pool`

Deterministic candidate facts for Instagram cards:

- top speakers;
- top constituencies;
- vote participation rankings;
- bill milestones;
- question volumes;
- debate/activity facts.

Cadence: weekly after dependent tables refresh.

No LLM-generated fields in gold until the enrichment layer is explicitly added.

---

## 11. Refresh cadence

### 11.1 Weekly workflow

File: `.github/workflows/oireachtas_weekly_refresh.yml`  
Proposed cron: `20 3 * * 0`

Run:

- current members and membership bridges;
- recent debate records, XML files, sections, speeches;
- recent/current-year divisions, tallies, member votes;
- recent parliamentary questions;
- current-year gold activity tables;
- control manifests and DQ outputs.

### 11.2 Monthly workflow

File: `.github/workflows/oireachtas_monthly_refresh.yml`  
Proposed cron: `35 4 1 * *`

Run:

- constituencies;
- parties;
- legislation/bills/stages/versions;
- current house/year reconciliation for debates, votes, and questions;
- constituency gold marts.

### 11.3 Yearly workflow

File: `.github/workflows/oireachtas_yearly_refresh.yml`  
Proposed cron: `15 5 2 1 *`

Run:

- full historical/current dimension refresh;
- previous-year fact rebuild/freeze;
- schema drift survey;
- annual DQ summary;
- previous-year gold freeze.

### 11.4 Manual test workflow

File: `.github/workflows/oireachtas_table_test.yml`

Inputs:

- `table`
- `mode`
- `chamber`
- `house_no`
- `date_start`
- `date_end`
- `limit`
- `sample_rows`
- `write_review_sample`
- `publish_review_branch`

---

## 12. Assistant-visible test/review loop

The attached handoff established that the assistant can reliably inspect generated outputs through a dedicated GitHub output branch. Reuse that pattern for data samples.

Create/use branch:

```text
oireachtas-review-output
```

Branch layout:

```text
review/index.md
review/<table>/latest/sample.csv
review/<table>/latest/schema.json
review/<table>/latest/manifest.json
```

Per-table loop:

1. Build parser/table in test mode.
2. Run manual workflow with small limits/date range.
3. Write S3 review sample.
4. Publish sample/schema/manifest to `oireachtas-review-output`.
5. Assistant opens raw GitHub sample output.
6. Assistant checks schema, row count, nulls, IDs, joins, parse quality.
7. Patch parser and rerun until acceptable.
8. Mark table confirmed in this document.
9. Move to next table.

Required log output for each test:

```text
TABLE=<table>
MODE=test
ROWS=<n>
COLUMNS=<n>
PRIMARY_KEY=<key>
PRIMARY_KEY_UNIQUE=<true|false>
CSV_KEY=s3://...
PARQUET_KEY=s3://...
REVIEW_SAMPLE_RAW_URL=https://raw.githubusercontent.com/...
```

Sample policy:

- default 10 rows;
- truncate long text in logs;
- full sample CSV in review branch/S3;
- do not include LLM output fields.

---

## 13. Validation rules

### 13.1 Every table

- required columns exist;
- table is not empty unless explicitly allowed in test mode;
- primary key is non-null;
- primary key is unique;
- CSV and Parquet row counts match;
- schema hash is logged;
- manifest is written;
- `snapshot_date` exists;
- date fields parse above threshold;
- row count and warnings appear in logs.

### 13.2 Join checks

| Check | Rule |
|---|---|
| `silver_member_votes.member_code -> silver_members.member_code` | warn below 98%, fail below 90% |
| `silver_speeches.speaker_member_code -> silver_members.member_code` | fail if non-null FK breaks |
| speech speaker match rate | warn below 70%; unmatched non-members may be valid |
| `silver_member_parties.member_code -> silver_members.member_code` | fail if broken FK > 0 |
| `silver_member_constituencies.member_code -> silver_members.member_code` | fail if broken FK > 0 |
| debate XML download coverage | warn below 98%, fail below 90% in full/current runs |

### 13.3 Schema drift

For every endpoint run:

1. hash top-level raw result keys;
2. hash nested key paths to depth 4;
3. compare with previous successful manifest;
4. warn on new fields;
5. fail only when required fields disappear;
6. store drift details in manifest.

---

## 14. Implementation sequence

### Phase 0 — Plan

Status: **done**

- [x] Inspect handoff file.
- [x] Inspect repo tree.
- [x] Inspect existing Oireachtas-related scripts/workflows.
- [x] Verify public API endpoint list.
- [x] Create this document.

### Phase 1 — Foundation

Status: **not started**

- [ ] Create `extract/oireachtas/` package.
- [ ] Create `configs/oireachtas/` files.
- [ ] Implement client, S3 IO, normalization, schema validation.
- [ ] Create manual table test workflow.
- [ ] Add review output branch publishing.
- [ ] Prove one tiny table end-to-end.

Recommended first proof table: `silver_houses`. Fallback: `silver_members`, because the current repo already has working member extraction.

### Phase 2 — Core dimensions

Status: **not started**

Build and confirm in this order:

1. `silver_houses`
2. `silver_constituencies`
3. `silver_parties`
4. `silver_members`
5. `silver_member_memberships`
6. `silver_member_parties`
7. `silver_member_constituencies`
8. `silver_member_offices`

### Phase 3 — Divisions/votes

Status: **not started**

1. Test `/divisions` and `/votes`.
2. Build `silver_divisions`.
3. Build `silver_division_tallies`.
4. Build `silver_member_votes`.
5. Compare against old `process/build_dail_votes_member_records.py` output.

### Phase 4 — Debates/speeches

Status: **not started**

1. Build `silver_debate_records`.
2. Build raw XML downloader/checker.
3. Build `silver_debate_sections`.
4. Build `silver_speeches`.
5. Compare sample output with old `extract/debates_xml_to_csv_s3.py` output.

### Phase 5 — Questions

Status: **not started**

1. Discover `/questions` payload shape.
2. Build `silver_questions` metadata.
3. Add XML/PDF download if useful.
4. Add XML question parser only after sample review.

### Phase 6 — Legislation

Status: **not started**

1. Discover `/legislation` payload shape.
2. Build `silver_bills`.
3. Build `silver_bill_versions`.
4. Build `silver_bill_stages`.
5. Add sponsors only if reliable.

### Phase 7 — Gold marts

Status: **not started**

1. `gold_current_members`
2. `gold_member_activity_yearly`
3. `gold_member_activity_monthly`
4. `gold_constituency_activity_yearly`
5. `gold_content_fact_pool`

### Phase 8 — Schedules

Status: **not started**

- [ ] Add weekly workflow.
- [ ] Add monthly workflow.
- [ ] Add yearly workflow.
- [ ] Add concurrency groups.
- [ ] Add manual dispatch inputs.
- [ ] Confirm successful runs and update this doc.

### Phase 9 — Cutover

Status: **not started**

Do not begin until unified tables are confirmed.

- [ ] Compare old vs new members.
- [ ] Compare old vs new votes.
- [ ] Compare old vs new speeches.
- [ ] Update Instagram metrics inputs.
- [ ] Keep old scripts through at least one successful scheduled cycle.
- [ ] Mark old scripts deprecated.
- [ ] Disable/remove old schedules only after documented approval/decision.

---

## 15. Change log

| Date | Change | Files affected | Status |
|---|---|---|---|
| 2026-06-02 | Created source-of-truth technical plan for unified Oireachtas data model/pipeline workstream. | `docs/oireachtas_unified_data_model_plan.md` | Done |

---

## 16. Status tracker

| Component/table | Status | Notes |
|---|---|---|
| Plan document | Created | This file |
| Foundation package | Not started | Next task |
| Manual test workflow | Not started | Required before table iteration |
| Review output branch loop | Not started | Reuse Instagram preview-output pattern |
| `silver_houses` | Not started | First recommended test table |
| `silver_constituencies` | Not started | Core dimension |
| `silver_parties` | Not started | Core dimension |
| `silver_members` | Not started | Existing extractor available |
| `silver_member_memberships` | Not started | Required for time-aware joins |
| `silver_member_parties` | Not started | Required for party-at-date joins |
| `silver_member_constituencies` | Not started | Required for constituency-at-date joins |
| `silver_member_offices` | Not started | Useful if API exposes offices reliably |
| `silver_divisions` | Not started | Endpoint needs verification |
| `silver_division_tallies` | Not started | Derived from division payload |
| `silver_member_votes` | Not started | Existing old script available |
| `silver_debate_records` | Not started | Existing old script available |
| `silver_debate_sections` | Not started | Needs improved XML/API parser |
| `silver_speeches` | Not started | Existing old parser available |
| `silver_questions` | Not started | Discovery needed |
| `silver_bills` | Not started | Discovery needed |
| `silver_bill_versions` | Not started | Discovery needed |
| `silver_bill_stages` | Not started | Discovery needed |
| Gold marts | Not started | Build after silver confirmed |
| Weekly workflow | Not started | Later |
| Monthly workflow | Not started | Later |
| Yearly workflow | Not started | Later |

---

## 17. Immediate next action

Create Phase 1 foundation files and the manual table-test workflow, then prove `silver_houses` end-to-end in test mode.

Fallback if `/houses` is awkward: prove `silver_members` first using the existing member extractor as the behaviour reference.
