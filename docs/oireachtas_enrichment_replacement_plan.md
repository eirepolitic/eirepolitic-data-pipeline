# Oireachtas enrichment replacement plan

**Status:** planning  
**Last updated:** 2026-06-30

## Purpose

The unified Oireachtas model now covers deterministic parliamentary source data and the first downstream compatibility outputs. This document separates the remaining enrichment dependencies from the deterministic Oireachtas table model.

## Remaining enrichment dependencies

| Dependency | Current role | Deterministic replacement status | Recommendation |
|---|---|---|---|
| Classified debate issues | Provides issue labels for speeches/member profile cards. | not replaced | Keep existing classified output until a separate issue-classification pipeline is designed. |
| Member photo URLs | Provides Instagram/member profile images. | not replaced | Keep existing photo index workflow; consider a normalized `enrichment_member_photos` table later. |
| Member summaries/backgrounds | Provides biography/context text. | not replaced | Keep existing summarizer output; consider explicit provenance and review fields before cutover. |
| Constituency image index | Provides constituency imagery. | not replaced | Keep existing image index; treat as media metadata, not Oireachtas source data. |

## Current known inputs to keep

```text
processed/debates/debate_speeches_classified.csv
processed/members/member_photos/members_photo_urls.csv
processed/members/members_photo_urls.csv
processed/members/members_summaries.csv
processed/constituencies/constituency_images.csv
```

## Recommended future tables

If these are brought into the unified pipeline, use a separate enrichment namespace rather than mixing them with deterministic silver/gold Oireachtas outputs.

Candidate tables:

```text
enrichment_speech_issue_labels
enrichment_member_photo_urls
enrichment_member_summaries
enrichment_constituency_images
gold_member_profile_cards
```

## Design rules

1. Keep raw Oireachtas API facts separate from enrichment/classification outputs.
2. Include provenance fields on every enrichment output.
3. Include review status fields where humans or LLMs are involved.
4. Do not overwrite existing legacy enrichment keys until the replacement table has a side-by-side trial and comparison report.
5. Prefer compatibility adapters for consumers that expect the old file shape.

## First recommended enrichment packet

Start with classified debate issues because member-profile cards currently depend on issue labels.

Proposed first packet:

```text
E01 — classified issue dependency audit
```

Goal:

- inspect `speech_issue_classifier.yml` and the scripts that build `processed/debates/debate_speeches_classified.csv`;
- identify input keys, output keys, required columns, and review steps;
- decide whether the output should remain legacy-only or become `enrichment_speech_issue_labels`.
