# Oireachtas Unified Data Model + Pipeline Plan

**Repository:** `eirepolitic-data-pipeline`  
**Document role:** Source of truth for the Oireachtas database layer behind the Instagram content creation platform.  
**Created:** 2026-06-02  
**Last updated:** 2026-06-02  
**Current branch:** `gpt/docs-oireachtas_unified_data_model_plan.md-a4926f5c`  
**Merge status:** Not yet merged to `main`. Continue work from this branch unless a later branch supersedes it.  
**Status:** Plan strengthened after review. Unified pipelines not yet built. Existing pipelines remain active.  
**Primary S3 bucket:** `s3://eirepolitic-data`  
**AWS region observed:** `ca-central-1`  
**Scope:** deterministic Oireachtas API extraction, source-file download, normalization, CSV/Parquet writing, S3 layout, GitHub Actions runners, refresh schedules, validation, review samples, and eventual cutover from existing pipelines.  
**Out of scope:** LLM issue classification, LLM summarisation, Instagram publishing, post scheduling, automatic approval, changing visual templates.

---

## 1. Mandatory operating rules for future agents

1. Read this file before changing any Oireachtas/database work.
2. Inspect the current repo before editing; other chats may have changed files after this document was written.
3. Keep old pipelines until the unified replacement is confirmed table-by-table.
4. Build new pipelines in parallel under new code paths and new S3 prefixes.
5. Use existing repo scripts to learn working API parameters, pagination, parsing, S3 writing, and workflow conventions.
6. Use official Oireachtas API/open-data documentation plus live API test pulls because the API is messy and documentation is imperfect.
7. Work table-by-table: design, build test, run, inspect small sample, patch, rerun, confirm, document, move on.
8. Require user input only when a functional, cost, legal, permission, or architecture decision cannot safely be made from repo/context.
9. Keep chat replies short and practical.
10. Do not say a GitHub/AWS tool cannot be used without first checking available tools, previous tool results, and chat history.
11. With the connected GitHub action tool, pass only the repository name:

```json
{"repo": "eirepolitic-data-pipeline"}
```

Do **not** pass `owner/repo`.

12. If AWS/Lambda tools are exposed in a future chat, use them when useful. If not, use GitHub Actions with existing AWS secrets to perform S3/API tests.
13. Do not touch LLM pipelines unless only documenting that they are out of scope.
14. Do not delete or disable old workflows until explicit cutover criteria are met and documented here.

---

## 2. Source evidence reviewed

### 2.1 Uploaded handoff

File reviewed locally:

```text
/mnt/data/instagram_content_system_handoff(1).md
```

Important requirements extracted:

- The Instagram system is review-only: no publishing, no scheduling, no automatic approval.
- Generated outputs must be reviewed from real generated artifacts, not approximations.
- The reliable assistant-visible review path is a dedicated GitHub output branch.
- Existing proven branch for Instagram previews: `instagram-preview-output`.
- S3 remains useful for storage, but public-style S3 URLs are not reliable for assistant review because uploads use `public-read=false` and public access/browser access can fail.
- Review-output branches may be public if the repo is public; only publish safe review samples there.
- The GitHub tool repo parameter must be repo name only.
- The current S3 bucket is `eirepolitic-data`, region `ca-central-1`.

### 2.2 Current repo files inspected

Current `main` files observed during planning:

| File | Relevant finding |
|---|---|
| `extract/monthly_extract.py` | Existing working `/debates` pagination, XML URI extraction, `data.oireachtas.ie` download, S3 raw XML write. |
| `extract/debates_xml_to_csv_s3.py` | Existing Akoma Ntoso XML speech parser, S3 XML listing, speech CSV write. Needs stronger IDs and `itertext()` parsing. |
| `extract/monthly_members_extract.py` | Existing `/members` flattening for Dáil 34 members/memberships/parties/constituencies. |
| `process/build_dail_votes_member_records.py` | Existing vote/division flattening and Parquet writing. Uses `/v1/votes`, while public docs list `/v1/divisions`; must test both. |
| `process/debate_speeches_csv_to_parquet.py` | Existing column cleaning and Parquet write pattern. |
| `process/build_member_profile_metrics_2025.py` | Current downstream Instagram metric builder that should eventually consume unified gold/silver tables. |
| `.github/workflows/monthly_extract.yml` | Existing monthly scheduled debates/members extraction. |
| `.github/workflows/build_member_profile_metrics_2025.yml` | Existing manual votes + profile metrics workflow. |
| `.github/workflows/speech_issue_classifier.yml` | LLM classification workflow; out of scope. |
| `.github/workflows/instagram_s3_preview_test.yml` | Proven pattern for publishing workflow outputs to a dedicated branch with `contents: write`. Use this pattern for table review samples. |
| `instagram/README.md` | Documents current downstream inputs and confirms Instagram relies on S3-backed member/debate/photo/constituency datasets. |
| `requirements.txt` | Already includes `requests`, `boto3`, `pandas`, `pyarrow`, `pyyaml`, and other dependencies needed for the deterministic database layer. |

### 2.3 Public API documentation facts

Useful URLs:

- `https://api.oireachtas.ie/`
- `https://data.oireachtas.ie/`
- `https://www.oireachtas.ie/en/open-data/`
- `https://data.gov.ie/dataset/houses-of-the-oireachtas-open-data-apis`
- `https://github.com/Irishsmurf/OireachtasAPI`

Important API facts to design around:

1. API metadata is served from `https://api.oireachtas.ie/v1`.
2. Oireachtas states the APIs are intended to be used with `https://data.oireachtas.ie` source files.
3. Debate and parliamentary-question source files are XML and comply with Akoma Ntoso-style structure.
4. Bills, Acts, and other documents may be PDF.
5. Source files are retrieved by appending `formats` URI fragments from API JSON to `https://data.oireachtas.ie`.
6. New data are made available as soon as published, so weekly incremental pulls are needed for current facts.
7. APIs are a work in progress; schema drift and endpoint quirks are expected.
8. Documented JSON resources include:
   - `/v1/debates`
   - `/v1/legislation`
   - `/v1/questions`
   - `/v1/divisions`
   - `/v1/constituencies`
   - `/v1/parties`
   - `/v1/houses`
   - `/v1/members`

Known mismatch:

- Existing repo script uses `/v1/votes`.
- Public docs/data.gov list votes as `/v1/divisions`.
- The unified client must run endpoint discovery and either standardize on `/divisions` or keep a documented `/votes` fallback if live API testing proves it is needed.

---

## 3. Document update rule

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
- user makes a design decision;
- S3/GitHub/AWS permissions are discovered or changed;
- review-output branch structure changes.

Every update must add a dated entry in **Section 17 — Change log**.

---

## 4. Goal

Create a unified, stable, joinable Oireachtas data model for Instagram content and future analytics.

The system must:

1. pull Oireachtas API metadata;
2. download linked XML/PDF/source files where useful;
3. preserve raw API/file data;
4. normalize messy nested payloads into clean tables;
5. write CSV and Parquet versions to S3;
6. expose small review samples that the assistant can inspect from a raw GitHub output branch;
7. refresh tables weekly, monthly, or yearly based on volatility;
8. support joins across members, memberships, parties, constituencies, houses, debates, speeches, divisions/votes, questions, bills, bill stages, and source files;
9. provide deterministic base tables for later LLM enrichment;
10. eventually replace existing old pipelines after confirmed equivalence/cutover.

---

## 5. Current production-ish repo state to preserve

### 5.1 Existing Oireachtas-related scripts

| File | Current purpose | Reuse notes |
|---|---|---|
| `extract/monthly_extract.py` | Fetches all Dáil 34 debates from `/debates`, downloads XML files, writes to `raw/debates/xml/`. | Reuse pagination, retry/backoff, XML URI handling, file naming. |
| `extract/debates_xml_to_csv_s3.py` | Parses raw debate XML from S3 into denormalized speeches. | Reuse namespace strategy, but improve text extraction, IDs, metadata, speaker resolution. |
| `extract/monthly_members_extract.py` | Fetches Dáil 34 members and writes one flattened CSV. | Reuse traversal of `memberships`, `parties`, `represents`. |
| `process/build_dail_votes_member_records.py` | Pulls vote/division data and writes division/member-vote CSV + Parquet. | Reuse tally flattening and Parquet writing; do not keep its committee filtering in base tables. |
| `process/build_member_profile_metrics_2025.py` | Creates current Instagram profile metrics from old member/debate/vote/photo inputs. | Treat as downstream consumer to replace later. |
| `process/debate_speeches_csv_to_parquet.py` | Converts classified debate CSV to Parquet. | Reuse column cleaning and Parquet pattern. |

### 5.2 Existing workflows

| File | Current purpose | Preserve? |
|---|---|---|
| `.github/workflows/monthly_extract.yml` | Monthly old debates XML + parsed speech CSV + members CSV. | Yes until cutover. |
| `.github/workflows/build_member_profile_metrics_2025.yml` | Manual old vote/member-profile metric build. | Yes until gold tables replace it. |
| `.github/workflows/speech_issue_classifier.yml` | Manual LLM issue classification. | Yes, but out of scope. |
| `.github/workflows/instagram_s3_preview_test.yml` | Proven raw GitHub output branch publishing pattern. | Use as model for `oireachtas-review-output`. |

### 5.3 Existing S3 paths not to overwrite

| Path | Current use |
|---|---|
| `raw/debates/xml/` | Old raw debate XML. |
| `raw/debates/debate_speeches_extracted.csv` | Old parsed speeches. |
| `raw/members/oireachtas_members_34th_dail.csv` | Old member CSV. |
| `processed/debates/debate_speeches_classified.csv` | LLM-classified speeches; out of scope. |
| `processed/votes/dail_vote_divisions.csv` | Old vote divisions. |
| `processed/votes/parquets/dail_vote_divisions.parquet` | Old vote divisions Parquet. |
| `processed/votes/dail_vote_member_records.csv` | Old member votes. |
| `processed/votes/parquets/dail_vote_member_records.parquet` | Old member votes Parquet. |
| `processed/members/member_profile_metrics_2025.csv` | Old Instagram member metrics. |
| `processed/members/parquets/member_profile_metrics_2025.parquet` | Old member metrics Parquet. |

### 5.4 Current downstream Instagram inputs to replace later

From `instagram/README.md` and current scripts, current downstream inputs include:

```text
raw/members/oireachtas_members_34th_dail.csv
processed/members/members_summaries.csv
processed/members/member_photos/members_photo_urls.csv
processed/members/members_photo_urls.csv
processed/debates/debate_speeches_classified.csv
processed/constituencies/constituency_images.csv
processed/members/member_profile_metrics_2025.csv
```

The unified data model should eventually replace only the deterministic Oireachtas-derived inputs. Photo URLs, constituency images, and LLM summaries/classifications remain separate layers.

---

## 6. Target architecture

```text
Oireachtas API + data.oireachtas.ie
  -> bronze raw JSON/XML/PDF + run manifests
  -> silver normalized joinable CSV/Parquet tables
  -> gold deterministic Instagram-ready marts
  -> later enrichment layer: LLM classifications/summaries/copy assistance
```

### 6.1 Layer definitions

| Layer | Purpose | Writes |
|---|---|---|
| Bronze | Exact raw API pages and source files for replay/debugging. | JSON, XML, PDF, manifest JSON. |
| Silver | Clean, typed, joinable normalized model. | CSV + Parquet for every confirmed table. |
| Gold | Deterministic Instagram/data-app marts built from silver only. | CSV + Parquet. |
| Enrichment | Later LLM-derived classifications/summaries. | Out of this workstream. |

### 6.2 New repo structure

Build new code in a separate package so old scripts remain untouched:

```text
extract/oireachtas/
  __init__.py
  client.py              # API client, pagination, retry, endpoint alias handling
  io_s3.py               # S3 CSV/Parquet/JSON/XML/PDF helpers
  normalize.py           # text/date/name/URI/hash helpers
  schemas.py             # table schema and validation helpers
  discovery.py           # endpoint smoke tests and payload-shape summaries
  xml_debates.py         # debate XML parser
  xml_questions.py       # question XML parser if needed
  build_table.py         # CLI: build one table
  build_all.py           # cadence orchestrator
  review.py              # sample/schema/manifest publishing helpers

configs/oireachtas/
  tables.yml             # table registry
  api_params.yml         # default chamber/house/date params
  schema_overrides.yml   # API quirks, field aliases, nullable exceptions
  cadences.yml           # weekly/monthly/yearly table groups

tests/oireachtas/
  test_normalize.py
  test_client.py
  test_xml_debates.py
  test_table_contracts.py
  fixtures/

.github/workflows/
  oireachtas_table_test.yml
  oireachtas_weekly_refresh.yml
  oireachtas_monthly_refresh.yml
  oireachtas_yearly_refresh.yml
```

### 6.3 Standard CLI shape

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
| `discover` | endpoint smoke test, no production outputs, writes payload-shape summary. |
| `test` | small sample, review outputs, no production overwrite. |
| `incremental` | recent/current changed data. |
| `full` | full refresh for a table/scope. |
| `backfill` | historical load by date/house range. |

### 6.4 Shared client requirements

`client.py` must support:

1. base URL default `https://api.oireachtas.ie/v1`;
2. file base URL default `https://data.oireachtas.ie`;
3. endpoint registry and aliases;
4. `/divisions` with optional `/votes` fallback;
5. `requests.Session` reuse;
6. retries/backoff for `429`, `5xx`, timeout, connection reset;
7. pagination via `skip`/`limit`;
8. result count logging;
9. raw page preservation before transform;
10. manifest per run;
11. schema drift summary;
12. source-file download from `formats` fields;
13. no API key assumption;
14. deterministic test mode with small limits.

---

## 7. S3 layout for unified work

Use `oireachtas_unified` while developing to avoid collisions.

```text
raw/oireachtas_unified/api/<endpoint>/snapshot_date=<YYYY-MM-DD>/run_id=<run_id>/page-00000.json
raw/oireachtas_unified/files/<source_type>/<id_or_date>/<file>
processed/oireachtas_unified/silver/<table>/snapshot_date=<YYYY-MM-DD>/run_id=<run_id>/part-00000.parquet
processed/oireachtas_unified/silver_csv/<table>/snapshot_date=<YYYY-MM-DD>/run_id=<run_id>/<table>.csv
processed/oireachtas_unified/gold/<table>/snapshot_date=<YYYY-MM-DD>/run_id=<run_id>/part-00000.parquet
processed/oireachtas_unified/gold_csv/<table>/snapshot_date=<YYYY-MM-DD>/run_id=<run_id>/<table>.csv
processed/oireachtas_unified/latest/parquet/<table>.parquet
processed/oireachtas_unified/latest/csv/<table>.csv
processed/oireachtas_unified/manifests/<table>/run_id=<run_id>.json
processed/oireachtas_unified/review/<table>/latest/sample.csv
processed/oireachtas_unified/review/<table>/latest/schema.json
processed/oireachtas_unified/review/<table>/latest/manifest.json
```

Rules:

1. Parquet is the analytical source of truth.
2. CSV is kept for Appsmith, Power BI, review, and simple GitHub output samples.
3. Raw API pages and downloaded files are preserved.
4. Every confirmed table gets a manifest.
5. Every test run writes a small review sample.
6. Do not overwrite old non-unified paths until cutover.
7. Add an S3 smoke-test step before broad writes to verify permissions on new prefixes.
8. Keep review samples small and safe because raw GitHub output branches may be public.

### 7.1 S3 permission caveat

The handoff only proves S3 preview permissions around:

```text
s3://eirepolitic-data/instagram/previews/*
```

It does **not** prove write/read permissions for:

```text
s3://eirepolitic-data/raw/oireachtas_unified/*
s3://eirepolitic-data/processed/oireachtas_unified/*
```

Before building many tables, the first workflow must run a narrow smoke test:

1. `PutObject` one JSON manifest to `processed/oireachtas_unified/review/_smoke/latest/manifest.json`.
2. `GetObject` the same file.
3. Optionally `PutObject` one tiny CSV.
4. Fail fast if permissions are missing.

If S3 permissions fail, document exact error in this file and use available AWS/GitHub tools to fix or request the minimum permission change.

---

## 8. Review-output branch design

The uploaded handoff proves the most reliable assistant-review path is a dedicated GitHub branch, not public S3 URLs or artifact ZIP inspection.

Create/use:

```text
branch: oireachtas-review-output
folder: review/
```

Branch layout:

```text
README.md
review/index.md
review/_smoke/latest/manifest.json
review/<table>/latest/sample.csv
review/<table>/latest/sample.md
review/<table>/latest/schema.json
review/<table>/latest/manifest.json
review/<table>/latest/dq.json
```

Workflow requirements:

1. `permissions: contents: write` is required.
2. Use the same worktree/orphan-branch pattern as `.github/workflows/instagram_s3_preview_test.yml`.
3. Treat the output branch as generated, not source.
4. Overwrite only the `review/` folder on each run.
5. Include raw GitHub URLs in `GITHUB_STEP_SUMMARY`.
6. Do not publish large full datasets to the review branch.
7. Do not rely on GitHub artifacts as the primary assistant review path.
8. Do not claim a sample was inspected unless the assistant opened the raw branch file, workflow logs, or S3 object directly.

Required raw URL pattern:

```text
https://raw.githubusercontent.com/eirepolitic/eirepolitic-data-pipeline/oireachtas-review-output/review/<table>/latest/sample.csv
```

Required log block:

```text
TABLE=<table>
MODE=<discover|test|incremental|full|backfill>
ROWS=<n>
COLUMNS=<n>
PRIMARY_KEY=<key>
PRIMARY_KEY_UNIQUE=<true|false>
CSV_KEY=s3://...
PARQUET_KEY=s3://...
MANIFEST_KEY=s3://...
REVIEW_SAMPLE_RAW_URL=https://raw.githubusercontent.com/...
DQ_STATUS=<pass|warn|fail>
```

Sample policy:

- default 10 rows;
- cap sample CSV at 100 rows in manual tests;
- truncate very long text in `sample.md`, but keep full sampled fields in `sample.csv`;
- include `schema.json` and `manifest.json` for machine inspection;
- never include LLM output fields in this deterministic review layer.

---

## 9. Data model key strategy

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
| Source file | `source_file_id` | file URL/S3 key hash |

Generated deterministic IDs:

```python
sha256("|".join(stable_fields).lower().strip().encode()).hexdigest()[:16]
```

Name-based joining rules:

1. Do not use display names as primary keys.
2. Display-name joins are allowed only as fallback.
3. Fallback joins must record `match_method` and preferably `match_confidence`.
4. Unmatched rows must be retained with null FK and clear match status.

---

## 10. Silver data model

### 10.1 Core dimensions

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

### 10.2 Source file inventory

#### `silver_source_files`

Purpose: normalize every XML/PDF/source file discovered from `formats` fields.

Grain: one row per source file URI.

Columns:

- `source_file_id`
- `source_entity_type`
- `source_entity_id`
- `format_type`
- `format_uri`
- `format_url`
- `s3_key`
- `content_type`
- `download_status`
- `downloaded_at_utc`
- `byte_size`
- `etag_or_hash`
- `snapshot_date`

Cadence: follows parent table. Use skip-existing where files are immutable.

### 10.3 Debates and speeches

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
- `source_file_id_xml`
- `source_file_id_pdf`
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
- `speaker_match_confidence`
- `speech_text`
- `speech_text_hash`
- `word_count`
- `char_count`
- `language`
- `source_file_id`
- `xml_source_key`
- `snapshot_date`

Parser requirements:

1. Use XML namespaces defensively.
2. Extract paragraph text with `itertext()`, not only direct `.text`.
3. Preserve unmatched speakers instead of dropping rows.
4. Resolve speakers via XML person refs first, member code/URI second, normalized-name fallback last.
5. Never dedupe only on date + speaker + text; include debate/section/order/hash.
6. Keep procedural/non-member speakers if the XML includes them.
7. Do not run LLM classification here.

Cadence: weekly recent incremental + monthly reconciliation.

Optional later table: `silver_speech_paragraphs` if paragraph-level data is useful.

### 10.4 Divisions/votes

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

### 10.5 Parliamentary questions

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
- `source_file_id_xml`
- `source_file_id_pdf`
- `snapshot_date`
- `source_hash`

Cadence: weekly recent incremental + monthly reconciliation.

Optional deterministic table only if API exposes topics: `silver_question_topics`. No LLM topic tagging in this workstream.

### 10.6 Legislation

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
- `source_file_id_pdf`
- `source_file_id_xml`
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

### 10.7 Control tables

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

## 11. Gold tables for Instagram/data apps

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
- debate/activity facts;
- current-membership facts.

Cadence: weekly after dependent tables refresh.

No LLM-generated fields in gold until the enrichment layer is explicitly added.

---

## 12. Refresh cadence

### 12.1 Weekly workflow

File: `.github/workflows/oireachtas_weekly_refresh.yml`  
Proposed cron: `20 3 * * 0`

Run:

- current members and membership bridges;
- recent debate records, XML files, sections, speeches;
- recent/current-year divisions, tallies, member votes;
- recent parliamentary questions;
- current-year gold activity tables;
- control manifests and DQ outputs.

Default window: last 45 days for recent event/fact tables, plus current year for yearly gold metrics.

### 12.2 Monthly workflow

File: `.github/workflows/oireachtas_monthly_refresh.yml`  
Proposed cron: `35 4 1 * *`

Run:

- constituencies;
- parties;
- legislation/bills/stages/versions;
- current house/year reconciliation for debates, votes, and questions;
- source-file inventory reconciliation;
- constituency gold marts.

### 12.3 Yearly workflow

File: `.github/workflows/oireachtas_yearly_refresh.yml`  
Proposed cron: `15 5 2 1 *`

Run:

- full historical/current dimension refresh;
- previous-year fact rebuild/freeze;
- schema drift survey;
- annual DQ summary;
- previous-year gold freeze.

### 12.4 Manual test workflow

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

Required workflow permissions:

```yaml
permissions:
  contents: write
```

Use `contents: write` because it must publish to `oireachtas-review-output`.

---

## 13. Table-by-table autonomous test loop

This section is the operating loop the user requested.

For each table:

1. Implement only the minimum parser/build logic for that table and its dependencies.
2. Run `discover` mode if payload shape is unknown.
3. Run `test` mode with tiny limits/date windows.
4. Write CSV, Parquet, manifest, schema, and DQ output to S3 test/review paths.
5. Publish sample/schema/manifest/DQ to `oireachtas-review-output`.
6. Assistant fetches raw GitHub sample output or workflow logs.
7. Assistant checks:
   - column names;
   - row count;
   - required key nulls;
   - primary key uniqueness;
   - obvious parsing failures;
   - join feasibility;
   - nested arrays not wrongly discarded;
   - source-file links/S3 keys where applicable.
8. Patch code and rerun until table is acceptable.
9. Mark table confirmed in this document with actual run evidence.
10. Move to next table.

Do not pause for user input unless:

- S3/GitHub/AWS permissions block progress;
- API behaviour creates a real model-design decision;
- a cost-affecting architecture choice appears;
- a cutover/deprecation decision is needed.

---

## 14. Validation rules

### 14.1 Every table

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

### 14.2 Join checks

| Check | Rule |
|---|---|
| `silver_member_votes.member_code -> silver_members.member_code` | warn below 98%, fail below 90% |
| `silver_speeches.speaker_member_code -> silver_members.member_code` | fail if non-null FK breaks |
| speech speaker match rate | warn below 70%; unmatched non-members may be valid |
| `silver_member_parties.member_code -> silver_members.member_code` | fail if broken FK > 0 |
| `silver_member_constituencies.member_code -> silver_members.member_code` | fail if broken FK > 0 |
| `silver_debate_sections.debate_id -> silver_debate_records.debate_id` | fail if broken FK > 0 |
| `silver_speeches.debate_section_id -> silver_debate_sections.debate_section_id` | warn if broken because some XML may be odd; fail after parser confirmed |
| debate XML download coverage | warn below 98%, fail below 90% in full/current runs |
| source file rows with `download_status=success` have S3 key | fail if missing |

### 14.3 Schema drift

For every endpoint run:

1. hash top-level raw result keys;
2. hash nested key paths to depth 4;
3. compare with previous successful manifest;
4. warn on new fields;
5. fail only when required fields disappear;
6. store drift details in manifest.

### 14.4 Equivalence checks before cutover

Before replacing old outputs:

| Old output | New comparison |
|---|---|
| `raw/members/oireachtas_members_34th_dail.csv` | compare to `silver_members` + current membership joins. |
| `raw/debates/debate_speeches_extracted.csv` | compare row counts and text samples to `silver_speeches`. |
| `processed/votes/dail_vote_divisions.csv` | compare to `silver_divisions`, filtered to old Dáil-only assumptions. |
| `processed/votes/dail_vote_member_records.csv` | compare to `silver_member_votes`, filtered to old Dáil-only assumptions. |
| `processed/members/member_profile_metrics_2025.csv` | compare to `gold_member_activity_yearly` plus member/photo inputs. |

---

## 15. Implementation sequence

### Phase 0 — Plan

Status: **done**

- [x] Inspect handoff file.
- [x] Inspect repo tree.
- [x] Inspect existing Oireachtas-related scripts/workflows.
- [x] Verify public API endpoint list.
- [x] Create plan document.
- [x] Review plan against user request, handoff, repo, and API docs.
- [x] Strengthen plan with review branch mechanics, S3 permission caveats, source-file inventory, and exact acceptance gates.

### Phase 1 — Foundation and review loop

Status: **not started**

Build these first:

- [ ] `extract/oireachtas/__init__.py`
- [ ] `extract/oireachtas/client.py`
- [ ] `extract/oireachtas/io_s3.py`
- [ ] `extract/oireachtas/normalize.py`
- [ ] `extract/oireachtas/schemas.py`
- [ ] `extract/oireachtas/discovery.py`
- [ ] `extract/oireachtas/review.py`
- [ ] `extract/oireachtas/build_table.py`
- [ ] `configs/oireachtas/tables.yml`
- [ ] `configs/oireachtas/api_params.yml`
- [ ] `.github/workflows/oireachtas_table_test.yml`

Foundation acceptance criteria:

1. S3 smoke test succeeds for `processed/oireachtas_unified/review/_smoke/latest/manifest.json`.
2. Review branch `oireachtas-review-output` is created/updated by workflow.
3. Raw GitHub URL to review sample is printed in workflow summary.
4. One endpoint runs in `discover` mode and writes payload-shape summary.
5. One tiny table writes CSV + Parquet + manifest + schema + DQ output.

Recommended first proof table: `silver_houses`. Fallback: `silver_members` because the existing repo already has working member extraction.

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
2. Document which endpoint works and why.
3. Build `silver_divisions`.
4. Build `silver_division_tallies`.
5. Build `silver_member_votes`.
6. Compare against old `process/build_dail_votes_member_records.py` output.

### Phase 4 — Debates/speeches

Status: **not started**

1. Build `silver_debate_records`.
2. Build `silver_source_files` support for debate XML/PDF.
3. Build raw XML downloader/checker.
4. Build `silver_debate_sections`.
5. Build `silver_speeches`.
6. Compare sample output with old `extract/debates_xml_to_csv_s3.py` output.

### Phase 5 — Questions

Status: **not started**

1. Discover `/questions` payload shape.
2. Build `silver_questions` metadata.
3. Add source-file rows/downloads if formats fields are present.
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

## 16. Initial implementation tickets

Use this as the next development checklist.

### Ticket 1 — Foundation package skeleton

Create package files and a minimal table registry. No API calls yet.

Acceptance:

- imports work;
- `python -m extract.oireachtas.build_table --help` works;
- no old pipeline files modified.

### Ticket 2 — API discovery client

Implement endpoint smoke tests.

Acceptance:

- can call `/houses`, `/members`, `/debates`, `/divisions`, `/questions`, `/legislation`, `/parties`, `/constituencies` in `discover` mode with tiny limits;
- writes raw JSON pages to S3 test prefix if S3 smoke test passes;
- writes review payload-shape summary to `oireachtas-review-output`.

### Ticket 3 — S3 + review branch smoke test

Implement the no-table smoke test.

Acceptance:

- workflow creates/updates `oireachtas-review-output`;
- raw GitHub URL opens for `review/_smoke/latest/manifest.json`;
- S3 Put/Get works on unified review prefix;
- document updated with result.

### Ticket 4 — First table: `silver_houses`

Acceptance:

- test run writes CSV + Parquet + manifest + schema;
- review sample available through raw GitHub branch;
- primary key uniqueness validated;
- this document updated with row count, S3 keys, workflow run ID, and status.

### Ticket 5 — Fallback first data-rich table: `silver_members`

Only use if `/houses` is unsuitable as first proof.

Acceptance:

- compare row count with old `raw/members/oireachtas_members_34th_dail.csv` for Dáil 34 scope;
- output member identities and membership bridges;
- document differences.

---

## 17. Change log

| Date | Change | Files affected | Status |
|---|---|---|---|
| 2026-06-02 | Created source-of-truth technical plan for unified Oireachtas data model/pipeline workstream. | `docs/oireachtas_unified_data_model_plan.md` | Done |
| 2026-06-02 | Reviewed plan against user request, uploaded handoff, current repo, and public Oireachtas docs. Strengthened with branch status, proven review-output mechanics, S3 permission caveat, source-file inventory, client/discovery requirements, stricter validation/cutover gates, and initial implementation tickets. | `docs/oireachtas_unified_data_model_plan.md` | Done |

---

## 18. Status tracker

| Component/table | Status | Notes |
|---|---|---|
| Plan document | Strengthened | This file on branch `gpt/docs-oireachtas_unified_data_model_plan.md-a4926f5c`. |
| Foundation package | Not started | Next task. |
| S3 smoke test | Not started | Must verify unified-prefix permissions. |
| Manual test workflow | Not started | Required before table iteration. |
| Review output branch loop | Not started | Must create `oireachtas-review-output`. |
| Endpoint discovery | Not started | Needed before table builds. |
| `silver_houses` | Not started | First recommended proof table. |
| `silver_constituencies` | Not started | Core dimension. |
| `silver_parties` | Not started | Core dimension. |
| `silver_members` | Not started | Existing extractor available. |
| `silver_member_memberships` | Not started | Required for time-aware joins. |
| `silver_member_parties` | Not started | Required for party-at-date joins. |
| `silver_member_constituencies` | Not started | Required for constituency-at-date joins. |
| `silver_member_offices` | Not started | Useful if API exposes offices reliably. |
| `silver_source_files` | Not started | Added during plan review. |
| `silver_divisions` | Not started | Endpoint needs verification: `/divisions` vs `/votes`. |
| `silver_division_tallies` | Not started | Derived from division payload. |
| `silver_member_votes` | Not started | Existing old script available. |
| `silver_debate_records` | Not started | Existing old script available. |
| `silver_debate_sections` | Not started | Needs improved XML/API parser. |
| `silver_speeches` | Not started | Existing old parser available. |
| `silver_questions` | Not started | Discovery needed. |
| `silver_bills` | Not started | Discovery needed. |
| `silver_bill_versions` | Not started | Discovery needed. |
| `silver_bill_stages` | Not started | Discovery needed. |
| Gold marts | Not started | Build after silver confirmed. |
| Weekly workflow | Not started | Later. |
| Monthly workflow | Not started | Later. |
| Yearly workflow | Not started | Later. |

---

## 19. Immediate next action

Create Phase 1 foundation files and the manual table-test workflow, then prove the review loop and S3 unified prefix before building real tables.

Recommended first execution order:

1. Add package skeleton and config skeleton.
2. Add `oireachtas_table_test.yml` with `contents: write`.
3. Implement S3/review branch smoke test.
4. Run workflow manually.
5. Fetch raw GitHub review output.
6. Update this document with actual run evidence.
7. Build `silver_houses` in test mode.

Fallback if `/houses` is awkward: prove `silver_members` first using the existing member extractor as behaviour reference.
