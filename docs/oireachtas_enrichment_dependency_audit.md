# Oireachtas enrichment dependency audit

**Status:** complete  
**Last updated:** 2026-06-30

## Scope

This audit covers enrichment/media dependencies that still sit outside the deterministic unified Oireachtas table model.

## Audited workflows

| Area | Workflow | Script | Type |
|---|---|---|---|
| Classified debate issues | `.github/workflows/speech_issue_classifier.yml` | `process/speech_issue_classifier.py` | LLM classification |
| Member photo URLs | `.github/workflows/member_photo_urls.yml` | `process/members_photo_urls.py` | media/index enrichment |
| Member summaries | `.github/workflows/members_background_summarizer.yml` | `process/members_background_summarizer.py` | LLM summarization |
| Constituency image index | `.github/workflows/constituency_images_index.yml` | `process/constituency_images_indexer.py` | media/index enrichment |

## Classified debate issues

Workflow:

```text
.github/workflows/speech_issue_classifier.yml
```

Inputs:

```text
raw/debates/debate_speeches_extracted.csv
```

Outputs:

```text
processed/debates/debate_speeches_classified.csv
processed/debates/parquets/debate_speeches_classified.parquet
```

Key behavior:

- Uses OpenAI via `OPENAI_API_KEY`.
- Classifies missing `PoliticalIssues` values.
- Preserves existing classifications when present.
- Uses `speech_id` as a deterministic row identity when available or computes it from debate fields.
- Supports limited test runs through `TEST_ROWS`.

Known consumer:

```text
process/build_member_profile_metrics_2025.py
```

Default consumer input:

```text
DEBATE_ISSUES_INPUT_KEY=processed/debates/debate_speeches_classified.csv
```

## Member photo URLs

Workflow:

```text
.github/workflows/member_photo_urls.yml
```

Inputs:

```text
raw/members/oireachtas_members_34th_dail.csv
```

Outputs:

```text
processed/members/member_photos/members_photo_urls.csv
processed/members/member_photos/parquets/members_photo_urls.parquet
```

Known consumer:

```text
process/build_member_profile_metrics_2025.py
```

Default candidate inputs:

```text
processed/members/members_photo_urls.csv
processed/members/member_photos/members_photo_urls.csv
processed/members/members_photo_urls.csv
```

Note: the first and third defaults are the same path because `PHOTO_KEY_CANDIDATES` includes the env default and fallback path.

## Member summaries/backgrounds

Workflow:

```text
.github/workflows/members_background_summarizer.yml
```

Inputs:

```text
raw/members/oireachtas_members_34th_dail.csv
```

Outputs:

```text
processed/members/members_summaries.csv
processed/members/parquets/members_summaries.parquet
```

Key behavior:

- Uses OpenAI via `OPENAI_API_KEY`.
- Supports limited test runs through `TEST_ROWS`.
- Produces enrichment text, not source-of-truth Oireachtas facts.

Known consumers:

- Instagram constituency renderer can read member summaries through `process/instagram_render_post.py` dataset candidates.
- This is not required for deterministic Oireachtas table validity.

## Constituency image index

Workflow:

```text
.github/workflows/constituency_images_index.yml
```

Inputs:

```text
processed/constituencies/images/
```

Outputs:

```text
processed/constituencies/constituency_images.csv
processed/constituencies/parquets/constituency_images.parquet
```

Known consumers:

- Instagram constituency rendering via constituency image dataset candidates.

## Consumer dependency chain

```text
classified debate issues
member photo URLs
member summaries
constituency image index
        ↓
member profile metrics and Instagram renderers
        ↓
Instagram constituency/campaign artifacts
```

## Audit conclusion

Do not merge these enrichment outputs into deterministic silver/gold Oireachtas tables. Keep a separate enrichment namespace and use side-by-side outputs before any replacement.
