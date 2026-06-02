# Oireachtas Unified Data Model + Pipeline Plan

**Repository:** `eirepolitic-data-pipeline`  
**Document role:** Source of truth and handoff guide for the Oireachtas database layer behind the Instagram content creation platform.  
**Created:** 2026-06-02  
**Last updated:** 2026-06-02  
**Current branch:** `gpt/docs-oireachtas_unified_data_model_plan.md-a4926f5c`  
**Merge status:** Not yet merged to `main`. Continue work from this branch unless a later branch supersedes it.  
**Status:** Plan updated for bounded autonomous implementation. Unified pipelines not yet built. Existing pipelines remain active.  
**Primary S3 bucket:** `s3://eirepolitic-data`  
**AWS region observed:** `ca-central-1`  
**Scope:** deterministic Oireachtas API extraction, source-file download, normalization, CSV/Parquet writing, S3 layout, GitHub Actions runners, refresh schedules, validation, review samples, per-table checkpoints, and eventual cutover from existing pipelines.  
**Out of scope:** LLM issue classification, LLM summarisation, Instagram publishing, post scheduling, automatic approval, changing visual templates.

---

## 1. How to use this document

This document is designed so work can move safely between chats. Future GPT agents should:

1. Read this file first.
2. Inspect the current repo branch and latest `main` before editing.
3. Identify the current active packet in **Section 16 — Bounded implementation packets**.
4. Complete only one packet at a time unless the user explicitly asks to continue.
5. After each packet, update:
   - packet status,
   - evidence links/run IDs,
   - S3 keys,
   - review branch sample URLs,
   - known issues,
   - change log.
6. Check in with the user after each packet with a short status summary.
7. If work needs to migrate to a new chat, the new chat should be able to continue by reading this document and the latest branch state.

This structure is intentional: long GPT sessions are more prone to mistakes. Each packet is small enough to build, test, inspect, document, and stop cleanly.

---

## 2. Mandatory operating rules for future agents

1. Keep old pipelines until the unified replacement is confirmed table-by-table.
2. Build new pipelines in parallel under new code paths and new S3 prefixes.
3. Use existing repo scripts to learn working API parameters, pagination, parsing, S3 writing, and workflow conventions.
4. Use Oireachtas documentation plus live API test pulls because the API is messy and documentation is imperfect.
5. Work in bounded packets. Do not drift into multiple unrelated tables in one pass.
6. Ask the user only when a functional, cost, legal, permission, or architecture decision cannot safely be made from repo/context.
7. Keep chat replies short and practical.
8. Do not say a GitHub/AWS tool cannot be used without first checking available tools, previous tool results, and chat history.
9. With the connected GitHub action tool, pass only the repository name:

```json
{"repo": "eirepolitic-data-pipeline"}
```

Do **not** pass `owner/repo`.

10. If AWS/Lambda tools are exposed in a future chat, use them when useful. If not, use GitHub Actions with existing AWS secrets to perform S3/API tests.
11. Do not touch LLM pipelines unless only documenting that they are out of scope.
12. Do not delete or disable old workflows until explicit cutover criteria are met and documented here.
13. Never mark a table confirmed without an actual test run and inspectable output evidence.

---

## 3. Evidence reviewed

### 3.1 Uploaded handoff

File reviewed locally:

```text
/mnt/data/instagram_content_system_handoff(1).md
```

Important requirements extracted:

- Instagram work is review-only: no publishing, no scheduling, no automatic approval.
- Generated outputs must be reviewed from real generated artifacts.
- The reliable assistant-visible review path is a dedicated GitHub output branch.
- Existing proven branch for Instagram previews: `instagram-preview-output`.
- S3 remains useful for storage, but public-style S3 URLs are not reliable for assistant review because uploads use `public-read=false` and browser/public access can fail.
- Review-output branches may be public if the repo is public; only publish safe review samples there.
- The GitHub tool repo parameter must be repo name only.
- Current bucket: `eirepolitic-data`, region `ca-central-1`.

### 3.2 Current repo files inspected

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
| `requirements.txt` | Already includes `requests`, `boto3`, `pandas`, `pyarrow`, `pyyaml`, and other dependencies needed for this layer. |

### 3.3 Public API facts

Useful URLs:

- `https://api.oireachtas.ie/`
- `https://data.oireachtas.ie/`
- `https://www.oireachtas.ie/en/open-data/`
- `https://data.gov.ie/dataset/houses-of-the-oireachtas-open-data-apis`
- `https://github.com/Irishsmurf/OireachtasAPI`

Important facts:

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

Current downstream inputs include:

```text
raw/members/oireachtas_members_34th_dail.csv
processed/members/members_summaries.csv
processed/members/member_photos/members_photo_urls.csv
processed/members/members_photo_urls.csv
processed/debates/debate_speeches_classified.csv
processed/constituencies/constituency_images.csv
processed/members/member_profile_metrics_2025.csv
```

The unified data model should eventually replace only deterministic Oireachtas-derived inputs. Photo URLs, constituency images, and LLM summaries/classifications remain separate layers.

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
| `discover` | Endpoint smoke test, no production outputs, writes payload-shape summary. |
| `test` | Small sample, review outputs, no production overwrite. |
| `incremental` | Recent/current changed data. |
| `full` | Full refresh for a table/scope. |
| `backfill` | Historical load by date/house range. |

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

Columns: `house_uri`, `house_no`, `house_code`, `chamber`, `show_as`, `date_start`, `date_end`, `is_current`, `source_endpoint`, `snapshot_date`, `source_hash`.

Cadence: yearly full + manual.

#### `silver_constituencies`

Grain: one row per constituency/date-range/house.

Columns: `constituency_uri`, `constituency_code`, `constituency_name`, `show_as`, `house_uri`, `house_no`, `chamber`, `date_start`, `date_end`, `is_current`, `source_endpoint`, `snapshot_date`, `source_hash`.

Cadence: monthly current + yearly full.

#### `silver_parties`

Grain: one row per party/date range where available.

Columns: `party_uri`, `party_code`, `party_name`, `show_as`, `date_start`, `date_end`, `is_current`, `source_endpoint`, `snapshot_date`, `source_hash`.

Cadence: monthly current + yearly full.

#### `silver_members`

Grain: one row per member/person.

Columns: `member_code`, `member_uri`, `full_name`, `first_name`, `last_name`, `display_name`, `gender`, `member_key`, `is_current_member`, `latest_party_name`, `latest_constituency_name`, `latest_house_no`, `source_endpoint`, `snapshot_date`, `source_hash`.

Cadence: weekly current + monthly full.

#### `silver_member_memberships`

Grain: one row per member membership object.

Columns: `membership_id`, `member_code`, `member_uri`, `house_uri`, `house_no`, `house_code`, `chamber`, `membership_start`, `membership_end`, `is_current`, `source_hash`, `snapshot_date`.

Cadence: weekly current + monthly full.

#### `silver_member_parties`

Grain: one row per member-party assignment.

Columns: `member_party_id`, `membership_id`, `member_code`, `party_uri`, `party_name`, `party_start`, `party_end`, `is_current`, `snapshot_date`.

Cadence: weekly current + monthly full.

#### `silver_member_constituencies`

Grain: one row per member-constituency assignment.

Columns: `member_constituency_id`, `membership_id`, `member_code`, `constituency_uri`, `constituency_name`, `represent_start`, `represent_end`, `is_current`, `snapshot_date`.

Cadence: weekly current + monthly full.

#### `silver_member_offices`

Grain: one row per office/ministerial role assignment where available.

Columns: `member_office_id`, `membership_id`, `member_code`, `office_uri`, `office_name`, `office_start`, `office_end`, `is_current`, `snapshot_date`.

Cadence: weekly current + monthly full.

### 10.2 Source file inventory

#### `silver_source_files`

Purpose: normalize every XML/PDF/source file discovered from `formats` fields.

Columns: `source_file_id`, `source_entity_type`, `source_entity_id`, `format_type`, `format_uri`, `format_url`, `s3_key`, `content_type`, `download_status`, `downloaded_at_utc`, `byte_size`, `etag_or_hash`, `snapshot_date`.

Cadence: follows parent table. Use skip-existing where files are immutable.

### 10.3 Debates and speeches

#### `silver_debate_records`

Grain: one row per debate record returned by `/debates`.

Columns: `debate_id`, `debate_uri`, `context_date`, `debate_date`, `chamber`, `house_uri`, `house_no`, `house_code`, `show_as`, `source_xml_uri`, `source_xml_url`, `source_pdf_uri`, `source_pdf_url`, `source_file_id_xml`, `source_file_id_pdf`, `api_result_hash`, `snapshot_date`.

Cadence: weekly recent incremental + monthly reconciliation + yearly full/backfill.

#### `silver_debate_sections`

Grain: one row per debate section.

Columns: `debate_section_id`, `debate_id`, `section_eid`, `section_uri`, `section_order`, `heading`, `show_as`, `parent_section_id`, `snapshot_date`.

Cadence: follows debate records.

#### `silver_speeches`

Grain: one row per speech element in debate XML.

Columns: `speech_id`, `debate_id`, `debate_section_id`, `debate_date`, `speech_order`, `speaker_ref`, `speaker_name`, `speaker_member_code`, `speaker_match_method`, `speaker_match_confidence`, `speech_text`, `speech_text_hash`, `word_count`, `char_count`, `language`, `source_file_id`, `xml_source_key`, `snapshot_date`.

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

Columns: `division_id`, `vote_id`, `division_date`, `chamber`, `house_uri`, `house_no`, `committee_code`, `subject`, `outcome`, `debate_id`, `debate_section_id`, `debate_show_as`, `api_result_hash`, `snapshot_date`.

Cadence: weekly recent/current-year incremental + monthly reconciliation.

Important: unlike the existing script, the unified base table should keep committee rows and let downstream marts filter them.

#### `silver_division_tallies`

Grain: one row per division and vote/tally type.

Columns: `division_tally_id`, `division_id`, `vote_code`, `vote_label`, `show_as`, `member_count`, `snapshot_date`.

Cadence: follows divisions.

#### `silver_member_votes`

Grain: one row per member per division.

Columns: `member_vote_id`, `division_id`, `vote_id`, `division_date`, `member_code`, `member_name`, `vote_code`, `vote_label`, `party_name_at_vote`, `constituency_name_at_vote`, `snapshot_date`.

Cadence: follows divisions.

### 10.5 Parliamentary questions

#### `silver_questions`

Grain: one row per question.

Columns: `question_id`, `question_uri`, `question_date`, `question_no`, `question_type`, `question_text`, `answer_text`, `asked_by_member_code`, `asked_by_name`, `to_minister_or_department`, `debate_section_id`, `source_xml_uri`, `source_xml_url`, `source_pdf_uri`, `source_pdf_url`, `source_file_id_xml`, `source_file_id_pdf`, `snapshot_date`, `source_hash`.

Cadence: weekly recent incremental + monthly reconciliation.

Optional deterministic table only if API exposes topics: `silver_question_topics`. No LLM topic tagging in this workstream.

### 10.6 Legislation

#### `silver_bills`

Grain: one row per bill/act item.

Columns: `bill_id`, `bill_uri`, `bill_no`, `bill_year`, `title`, `short_title`, `origin_house_uri`, `origin_house_name`, `bill_type`, `status`, `introduced_date`, `last_event_date`, `source_endpoint`, `snapshot_date`, `source_hash`.

Cadence: monthly incremental + yearly full.

#### `silver_bill_versions`

Grain: one row per bill version/document.

Columns: `bill_version_id`, `bill_id`, `version_label`, `version_date`, `format_pdf_uri`, `format_pdf_url`, `format_xml_uri`, `format_xml_url`, `source_file_id_pdf`, `source_file_id_xml`, `s3_pdf_key`, `s3_xml_key`, `snapshot_date`.

Cadence: monthly. Use `skip_existing` for immutable files.

#### `silver_bill_stages`

Grain: one row per bill stage/progress event.

Columns: `bill_stage_id`, `bill_id`, `stage_name`, `stage_date`, `house_uri`, `house_name`, `stage_outcome`, `order_in_bill`, `snapshot_date`.

Cadence: monthly.

Optional if sponsors are exposed reliably: `silver_bill_sponsors`.

### 10.7 Control tables

`control_pipeline_runs`, `control_table_manifests`, and `control_data_quality_results` are required audit/control tables. Build these during the foundation stage or as soon as the first real table runs.

---

## 11. Gold tables for Instagram/data apps

Build only after required silver inputs are confirmed.

| Table | Grain | Cadence | Notes |
|---|---|---|---|
| `gold_current_members` | one row per current member | weekly | joins member identity, current membership, party, constituency, office |
| `gold_member_activity_yearly` | one row per member/year | weekly for current year; yearly freeze for prior years | speech count, debate days, vote participation, yes/no/abstain counts, ranks |
| `gold_member_activity_monthly` | one row per member/month | weekly for current year | trend content |
| `gold_constituency_activity_yearly` | one row per constituency/year | monthly | constituency rankings |
| `gold_content_fact_pool` | one row per deterministic fact | weekly | top speakers, constituencies, vote rankings, bill milestones, question volumes |

No LLM-generated fields in gold until the enrichment layer is explicitly added.

---

## 12. Refresh cadence

| Workflow | File | Cron | Scope |
|---|---|---|---|
| Weekly | `.github/workflows/oireachtas_weekly_refresh.yml` | `20 3 * * 0` | current members, recent debates/speeches, recent votes, recent questions, current-year gold |
| Monthly | `.github/workflows/oireachtas_monthly_refresh.yml` | `35 4 1 * *` | constituencies, parties, legislation, reconciliation, source-file inventory |
| Yearly | `.github/workflows/oireachtas_yearly_refresh.yml` | `15 5 2 1 *` | full historical/current dimensions, previous-year fact freeze, schema survey |
| Manual test | `.github/workflows/oireachtas_table_test.yml` | manual | one table or discovery/smoke test |

Manual test workflow inputs:

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

---

## 13. Table-by-table autonomous test loop

For each table:

1. Implement only the minimum parser/build logic for that table and its dependencies.
2. Run `discover` mode if payload shape is unknown.
3. Run `test` mode with tiny limits/date windows.
4. Write CSV, Parquet, manifest, schema, and DQ output to S3 test/review paths.
5. Publish sample/schema/manifest/DQ to `oireachtas-review-output`.
6. Assistant fetches raw GitHub sample output or workflow logs.
7. Assistant checks column names, row count, key nulls, key uniqueness, parsing, joins, nested arrays, source-file links.
8. Patch code and rerun until table is acceptable.
9. Mark table confirmed in this document with actual run evidence.
10. Stop/check in with the user unless the user explicitly says to continue to the next packet.

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
| `silver_speeches.debate_section_id -> silver_debate_sections.debate_section_id` | warn if broken because XML may be odd; fail after parser confirmed |
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

| Old output | New comparison |
|---|---|
| `raw/members/oireachtas_members_34th_dail.csv` | compare to `silver_members` + current membership joins |
| `raw/debates/debate_speeches_extracted.csv` | compare row counts and text samples to `silver_speeches` |
| `processed/votes/dail_vote_divisions.csv` | compare to `silver_divisions`, filtered to old Dáil-only assumptions |
| `processed/votes/dail_vote_member_records.csv` | compare to `silver_member_votes`, filtered to old Dáil-only assumptions |
| `processed/members/member_profile_metrics_2025.csv` | compare to `gold_member_activity_yearly` plus member/photo inputs |

---

## 15. Handoff and hallucination-control protocol

### 15.1 Start of every new chat/session

1. Open this document from the active branch.
2. Open the latest repo tree.
3. Check active branch/PR state.
4. Check the status tracker and current packet.
5. Check the latest change-log entry.
6. Do not assume previous planned files exist; verify them.

### 15.2 End of every packet

After each packet, update this file with:

- packet status;
- files changed;
- workflow run ID if any;
- S3 keys written;
- raw GitHub sample URL if any;
- row counts and DQ status;
- next packet recommendation;
- any blocker.

Then send the user a short message:

```text
Packet <id> complete. Changed: <files>. Evidence: <links/run IDs>. Next: <packet id>.
```

### 15.3 Hard stop triggers

Stop and check in with the user when:

- a packet is complete;
- a workflow/test fails twice with different fixes attempted;
- S3/GitHub/AWS permissions block progress;
- API response shape contradicts the planned schema;
- a destructive/cutover change is proposed;
- current chat has become long enough that context loss is likely.

### 15.4 Migration handoff block

At the end of each packet, add a short handoff note in the packet status like:

```text
Handoff: Continue from branch <branch>. Last successful run <workflow_run_id>. Review sample <raw_url>. Next action <exact command/workflow>.
```

---

## 16. Bounded implementation packets

Each packet is intended to be one autonomous build/test/review unit. Do not combine packets unless the user explicitly asks to continue.

Status values:

- `not_started`
- `in_progress`
- `blocked`
- `tested`
- `confirmed`
- `deprecated`

### F00 — Plan and branch hygiene

**Goal:** Ensure this plan branch is current and safe to continue.

**Inputs:** this document, repo tree, branch list.  
**Files likely changed:** `docs/oireachtas_unified_data_model_plan.md`.  
**Actions:** verify branch exists, compare to `main`, optionally open PR if user asks.  
**Acceptance:** document exists on active branch and status tracker is current.  
**Stop/check-in:** after document update or branch/PR state is known.  
**Status:** confirmed.  
**Evidence:** branch `gpt/docs-oireachtas_unified_data_model_plan.md-a4926f5c`, file `docs/oireachtas_unified_data_model_plan.md`.

### F01 — Foundation package skeleton

**Goal:** Create importable package and config skeleton with no external calls.

**Build files:**

- `extract/oireachtas/__init__.py`
- `extract/oireachtas/normalize.py`
- `extract/oireachtas/schemas.py`
- `extract/oireachtas/build_table.py`
- `configs/oireachtas/tables.yml`
- `configs/oireachtas/api_params.yml`

**Test command:**

```bash
python -m extract.oireachtas.build_table --help
```

**Acceptance:** help command works; old pipeline files untouched.  
**Review evidence:** command output in workflow/log or local test.  
**Stop/check-in:** yes.  
**Status:** not_started.

### F02 — S3 + review-branch smoke test

**Goal:** Prove the assistant-visible review loop and unified S3 prefix before table builds.

**Build files:**

- `extract/oireachtas/io_s3.py`
- `extract/oireachtas/review.py`
- `.github/workflows/oireachtas_table_test.yml`

**Workflow mode:** `table=_smoke`, `mode=test`.  
**S3 write:** `processed/oireachtas_unified/review/_smoke/latest/manifest.json`.  
**Review branch:** `oireachtas-review-output`.  
**Acceptance:** S3 Put/Get succeeds; review branch raw URL opens; workflow summary prints raw URL.  
**Stop/check-in:** yes, especially if permissions fail.  
**Status:** not_started.

### F03 — API discovery client

**Goal:** Build shared Oireachtas API client and discover payload shapes.

**Build files:**

- `extract/oireachtas/client.py`
- `extract/oireachtas/discovery.py`
- updates to `extract/oireachtas/build_table.py`

**Endpoints:** `/houses`, `/members`, `/debates`, `/divisions`, `/questions`, `/legislation`, `/parties`, `/constituencies`, plus `/votes` fallback test.  
**Mode:** `discover`.  
**Acceptance:** each endpoint tested with tiny limit; payload-shape summaries written to review branch; `/divisions` vs `/votes` behaviour documented.  
**Stop/check-in:** yes.  
**Status:** not_started.

### T01 — `silver_houses`

**Goal:** First tiny real silver table.

**Depends on:** F01, F02, F03.  
**Endpoint:** `/houses`.  
**Build files:** table builder/parser in `extract/oireachtas/build_table.py` or table-specific module if needed; `configs/oireachtas/tables.yml`.  
**Primary key:** `house_uri` or generated `house_id` fallback.  
**Test:** `mode=test`, small limit.  
**Acceptance:** CSV + Parquet + schema + manifest + DQ; primary key unique; sample reviewed.  
**Stop/check-in:** yes.  
**Status:** not_started.

### T02 — `silver_constituencies`

**Goal:** Constituency dimension.

**Depends on:** T01.  
**Endpoint:** `/constituencies`.  
**Primary key:** `constituency_uri` or generated fallback.  
**Acceptance:** rows join to house where possible; date ranges parsed; sample reviewed.  
**Stop/check-in:** yes.  
**Status:** not_started.

### T03 — `silver_parties`

**Goal:** Party dimension.

**Depends on:** F03.  
**Endpoint:** `/parties`.  
**Primary key:** `party_uri` or generated fallback.  
**Acceptance:** party names normalized; date ranges parsed where present; sample reviewed.  
**Stop/check-in:** yes.  
**Status:** not_started.

### T04 — `silver_members`

**Goal:** Stable member identity table.

**Depends on:** T01, T02, T03 preferred; can start after F03 if needed.  
**Endpoint:** `/members`.  
**Reference old script:** `extract/monthly_members_extract.py`.  
**Primary key:** `member_code` preferred; `member_uri`/hash fallback.  
**Acceptance:** compare Dáil 34 row count to `raw/members/oireachtas_members_34th_dail.csv`; no null primary keys after fallback; sample reviewed.  
**Stop/check-in:** yes.  
**Status:** not_started.

### T05 — `silver_member_memberships`

**Goal:** Time-aware member-to-house bridge.

**Depends on:** T04 and T01.  
**Source:** nested memberships from `/members`.  
**Primary key:** deterministic `membership_id`.  
**Acceptance:** all rows join to `silver_members`; house joins where available; current flag derived; sample reviewed.  
**Stop/check-in:** yes.  
**Status:** not_started.

### T06 — `silver_member_parties`

**Goal:** Time-aware member-to-party bridge.

**Depends on:** T05 and T03.  
**Source:** nested parties in membership payload.  
**Primary key:** `member_party_id`.  
**Acceptance:** joins to members; joins to party where URI/code available; current flag derived; sample reviewed.  
**Stop/check-in:** yes.  
**Status:** not_started.

### T07 — `silver_member_constituencies`

**Goal:** Time-aware member-to-constituency bridge.

**Depends on:** T05 and T02.  
**Source:** nested represents/constituency fields in membership payload.  
**Primary key:** `member_constituency_id`.  
**Acceptance:** joins to members; constituency names/URIs preserved; current flag derived; sample reviewed.  
**Stop/check-in:** yes.  
**Status:** not_started.

### T08 — `silver_member_offices`

**Goal:** Office/ministerial/chair role bridge where exposed.

**Depends on:** T05.  
**Source:** office-like nested fields in member/membership payload.  
**Primary key:** `member_office_id`.  
**Acceptance:** if fields exist, rows parsed and reviewed; if fields absent, document absence and decide whether table remains empty/optional.  
**Stop/check-in:** yes if table is absent/empty because that is a design decision.  
**Status:** not_started.

### T09 — `silver_source_files`

**Goal:** Inventory source XML/PDF files from `formats` fields.

**Depends on:** F03; later receives links from debates, questions, bills.  
**Source:** `formats` fields in API payloads.  
**Primary key:** `source_file_id`.  
**Acceptance:** source URL normalized to `data.oireachtas.ie`; S3 key generated; download status tracked; sample reviewed.  
**Stop/check-in:** yes.  
**Status:** not_started.

### T10 — `silver_debate_records`

**Goal:** Debate metadata table.

**Depends on:** T01 and T09 preferred.  
**Endpoint:** `/debates`.  
**Reference old script:** `extract/monthly_extract.py`.  
**Primary key:** `debate_id`/`debate_uri`/hash fallback.  
**Acceptance:** XML URI and normalized URL preserved; date/chamber/house fields parsed; sample reviewed.  
**Stop/check-in:** yes.  
**Status:** not_started.

### T11 — Debate XML downloader + `silver_source_files` update

**Goal:** Download selected debate XML files and connect them to source-file inventory.

**Depends on:** T09 and T10.  
**Reference old script:** `extract/monthly_extract.py`.  
**Acceptance:** small test downloads XML to unified raw file prefix; source file rows show success; failed downloads are retained with status.  
**Stop/check-in:** yes.  
**Status:** not_started.

### T12 — `silver_debate_sections`

**Goal:** Parse section metadata from debate XML/API.

**Depends on:** T10 and T11.  
**Primary key:** `debate_section_id`.  
**Acceptance:** section ordering stable; sections join to debates; headings preserved; sample reviewed.  
**Stop/check-in:** yes.  
**Status:** not_started.

### T13 — `silver_speeches`

**Goal:** Parse atomic speeches from debate XML.

**Depends on:** T04, T10, T11, T12.  
**Reference old script:** `extract/debates_xml_to_csv_s3.py`.  
**Primary key:** `speech_id`.  
**Acceptance:** speech text extracted with `itertext()`; speaker match method recorded; unmatched retained; sample compared to old parser for same XML; sample reviewed.  
**Stop/check-in:** yes.  
**Status:** not_started.

### T14 — Division endpoint decision

**Goal:** Resolve `/divisions` vs `/votes` before building vote tables.

**Depends on:** F03.  
**Actions:** test both endpoints with same tiny date window; document result.  
**Acceptance:** chosen endpoint and fallback are recorded in config and this document.  
**Stop/check-in:** yes because this affects architecture.  
**Status:** not_started.

### T15 — `silver_divisions`

**Goal:** One row per division/vote event.

**Depends on:** T14.  
**Reference old script:** `process/build_dail_votes_member_records.py`.  
**Primary key:** `division_id`.  
**Acceptance:** committee rows retained; old Dáil-only output comparison possible via filter; sample reviewed.  
**Stop/check-in:** yes.  
**Status:** not_started.

### T16 — `silver_division_tallies`

**Goal:** One row per division and tally type.

**Depends on:** T15.  
**Primary key:** `division_tally_id`.  
**Acceptance:** member counts match nested API tally lengths; joins to divisions; sample reviewed.  
**Stop/check-in:** yes.  
**Status:** not_started.

### T17 — `silver_member_votes`

**Goal:** One row per member vote.

**Depends on:** T04, T15, T16.  
**Reference old script:** `process/build_dail_votes_member_records.py`.  
**Primary key:** `member_vote_id`.  
**Acceptance:** joins to members at acceptable rate; vote labels normalized; old output comparison completed; sample reviewed.  
**Stop/check-in:** yes.  
**Status:** not_started.

### T18 — `silver_questions`

**Goal:** Parliamentary questions metadata and answer/question text where available.

**Depends on:** T04 and T09 preferred.  
**Endpoint:** `/questions`.  
**Primary key:** `question_id`.  
**Acceptance:** payload shape documented; questioner/member fields preserved; source files linked where available; sample reviewed.  
**Stop/check-in:** yes.  
**Status:** not_started.

### T19 — Question XML parser, if required

**Goal:** Parse question/answer text from XML when API metadata is insufficient.

**Depends on:** T18 and T09.  
**Acceptance:** if XML is needed, parser produces reliable sample; if not needed, document why skipped.  
**Stop/check-in:** yes.  
**Status:** not_started.

### T20 — `silver_bills`

**Goal:** Bill/legislation item table.

**Depends on:** T01 and T09 preferred.  
**Endpoint:** `/legislation`.  
**Primary key:** `bill_id`.  
**Acceptance:** bill IDs/titles/status/dates parsed; sample reviewed.  
**Stop/check-in:** yes.  
**Status:** not_started.

### T21 — `silver_bill_versions`

**Goal:** Bill document/version table with XML/PDF links.

**Depends on:** T20 and T09.  
**Primary key:** `bill_version_id`.  
**Acceptance:** file links normalized; S3 keys generated; skip-existing strategy works; sample reviewed.  
**Stop/check-in:** yes.  
**Status:** not_started.

### T22 — `silver_bill_stages`

**Goal:** Bill progress/stage history.

**Depends on:** T20.  
**Primary key:** `bill_stage_id`.  
**Acceptance:** stages ordered; dates parsed; joins to bills; sample reviewed.  
**Stop/check-in:** yes.  
**Status:** not_started.

### G01 — `gold_current_members`

**Goal:** Current roster mart for Instagram/data apps.

**Depends on:** T04-T08.  
**Acceptance:** one row per current member; party/constituency/office fields populated where possible; sample reviewed.  
**Stop/check-in:** yes.  
**Status:** not_started.

### G02 — `gold_member_activity_yearly`

**Goal:** Year-level member activity metrics.

**Depends on:** T13 and T17, plus G01.  
**Acceptance:** speech counts, vote participation, ranks produced; comparison to `processed/members/member_profile_metrics_2025.csv` documented.  
**Stop/check-in:** yes.  
**Status:** not_started.

### G03 — `gold_member_activity_monthly`

**Goal:** Month-level member activity trends.

**Depends on:** T13 and T17.  
**Acceptance:** one row per member/month; current-year sample reviewed.  
**Stop/check-in:** yes.  
**Status:** not_started.

### G04 — `gold_constituency_activity_yearly`

**Goal:** Constituency-level annual activity.

**Depends on:** T07, G02.  
**Acceptance:** constituency joins work; ranks/aggregates reviewed.  
**Stop/check-in:** yes.  
**Status:** not_started.

### G05 — `gold_content_fact_pool`

**Goal:** Deterministic pool of candidate Instagram facts.

**Depends on:** G01-G04, T18/T20 optional.  
**Acceptance:** facts are deterministic and traceable to source table/key; no LLM output; sample reviewed.  
**Stop/check-in:** yes.  
**Status:** not_started.

### W01 — Weekly workflow

**Goal:** Add scheduled weekly refresh only after dependent packets are confirmed.

**Depends on:** at least core dims, debates/speeches, votes, and desired gold tables.  
**Acceptance:** manual run succeeds before schedule is trusted; document run ID and outputs.  
**Stop/check-in:** yes.  
**Status:** not_started.

### W02 — Monthly workflow

**Goal:** Add scheduled monthly refresh.

**Depends on:** constituencies, parties, legislation, reconciliation logic.  
**Acceptance:** manual run succeeds before schedule is trusted.  
**Stop/check-in:** yes.  
**Status:** not_started.

### W03 — Yearly workflow

**Goal:** Add yearly full refresh/freeze workflow.

**Depends on:** stable full/backfill logic.  
**Acceptance:** manual limited dry-run succeeds; schedule committed.  
**Stop/check-in:** yes.  
**Status:** not_started.

### C01 — Cutover comparison

**Goal:** Compare old and new outputs without replacing anything.

**Depends on:** relevant silver/gold tables confirmed.  
**Acceptance:** comparison report written to review branch; differences documented.  
**Stop/check-in:** yes.  
**Status:** not_started.

### C02 — Downstream cutover

**Goal:** Point Instagram deterministic data dependencies at new gold/silver outputs.

**Depends on:** C01 and user approval/documented decision.  
**Acceptance:** old pipelines preserved for at least one successful scheduled cycle; rollback path documented.  
**Stop/check-in:** yes and requires explicit approval.  
**Status:** not_started.

---

## 17. Current packet pointer

**Current packet:** F01 — Foundation package skeleton.  
**Next action:** create package/config skeleton, then test `python -m extract.oireachtas.build_table --help`.  
**Do not start:** F02 or any table packet until F01 is complete and documented.

---

## 18. Change log

| Date | Change | Files affected | Status |
|---|---|---|---|
| 2026-06-02 | Created source-of-truth technical plan for unified Oireachtas data model/pipeline workstream. | `docs/oireachtas_unified_data_model_plan.md` | Done |
| 2026-06-02 | Reviewed plan against user request, uploaded handoff, current repo, and public Oireachtas docs. Strengthened with branch status, proven review-output mechanics, S3 permission caveat, source-file inventory, client/discovery requirements, stricter validation/cutover gates, and initial implementation tickets. | `docs/oireachtas_unified_data_model_plan.md` | Done |
| 2026-06-02 | Reworked implementation strategy into bounded autonomous packets with one-table/one-step checkpoints, handoff rules, stop triggers, current packet pointer, and per-packet evidence requirements. | `docs/oireachtas_unified_data_model_plan.md` | Done |

---

## 19. Status tracker

| Component/table | Status | Active packet | Notes |
|---|---|---|---|
| Plan document | confirmed | F00 | This file on branch `gpt/docs-oireachtas_unified_data_model_plan.md-a4926f5c`. |
| Foundation package | not_started | F01 | Current packet. |
| S3 smoke test | not_started | F02 | Must verify unified-prefix permissions. |
| Manual test workflow | not_started | F02 | Required before table iteration. |
| Review output branch loop | not_started | F02 | Must create `oireachtas-review-output`. |
| Endpoint discovery | not_started | F03 | Needed before table builds. |
| `silver_houses` | not_started | T01 | First recommended proof table. |
| `silver_constituencies` | not_started | T02 | Core dimension. |
| `silver_parties` | not_started | T03 | Core dimension. |
| `silver_members` | not_started | T04 | Existing extractor available. |
| `silver_member_memberships` | not_started | T05 | Required for time-aware joins. |
| `silver_member_parties` | not_started | T06 | Required for party-at-date joins. |
| `silver_member_constituencies` | not_started | T07 | Required for constituency-at-date joins. |
| `silver_member_offices` | not_started | T08 | Useful if API exposes offices reliably. |
| `silver_source_files` | not_started | T09 | Source XML/PDF inventory. |
| `silver_debate_records` | not_started | T10 | Existing old script available. |
| Debate XML downloader | not_started | T11 | Updates source files. |
| `silver_debate_sections` | not_started | T12 | Needs improved XML/API parser. |
| `silver_speeches` | not_started | T13 | Existing old parser available. |
| Division endpoint decision | not_started | T14 | `/divisions` vs `/votes`. |
| `silver_divisions` | not_started | T15 | Existing old script available. |
| `silver_division_tallies` | not_started | T16 | Derived from division payload. |
| `silver_member_votes` | not_started | T17 | Existing old script available. |
| `silver_questions` | not_started | T18 | Discovery needed. |
| Question XML parser | not_started | T19 | Only if needed. |
| `silver_bills` | not_started | T20 | Discovery needed. |
| `silver_bill_versions` | not_started | T21 | Discovery needed. |
| `silver_bill_stages` | not_started | T22 | Discovery needed. |
| `gold_current_members` | not_started | G01 | Build after member silver tables. |
| `gold_member_activity_yearly` | not_started | G02 | Build after speeches/votes. |
| `gold_member_activity_monthly` | not_started | G03 | Build after speeches/votes. |
| `gold_constituency_activity_yearly` | not_started | G04 | Build after member/constituency gold. |
| `gold_content_fact_pool` | not_started | G05 | Build after gold inputs. |
| Weekly workflow | not_started | W01 | Later. |
| Monthly workflow | not_started | W02 | Later. |
| Yearly workflow | not_started | W03 | Later. |
| Cutover comparison | not_started | C01 | Later. |
| Downstream cutover | not_started | C02 | Requires approval. |

---

## 20. Immediate next action

Start **F01 — Foundation package skeleton** only.

Exact next build scope:

```text
extract/oireachtas/__init__.py
extract/oireachtas/normalize.py
extract/oireachtas/schemas.py
extract/oireachtas/build_table.py
configs/oireachtas/tables.yml
configs/oireachtas/api_params.yml
```

Exact next test:

```bash
python -m extract.oireachtas.build_table --help
```

After F01, stop and report:

```text
Packet F01 complete. Changed: <files>. Evidence: <test result>. Next: F02.
```
