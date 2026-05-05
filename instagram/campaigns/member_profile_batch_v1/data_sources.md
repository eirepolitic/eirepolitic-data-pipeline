# Data Sources

## Inputs

- `processed/members/member_profile_metrics_2025.csv`
- `processed/members/members_summaries.csv` (future optional background slide)
- `processed/debates/debate_speeches_classified.csv` (upstream source for issue metrics)
- `processed/votes/dail_vote_member_records.csv` (upstream source for vote metrics)

## Outputs

- `generated_posts/member_profile_batch_v1/png/<member_slug>.png`
- `generated_posts/member_profile_batch_v1/metadata/*_source_values.json`
- `generated_posts/member_profile_batch_v1/metadata/*_render_manifest.json`
- `generated_posts/member_profile_batch_v1/review/review_table.csv`
- `generated_posts/member_profile_batch_v1/review/review_index.html`

## Notes

The first campaign renderer reads the metrics CSV directly from S3 by default, then writes local GitHub Actions artifacts. S3 output upload can be added after the visual review loop is approved.
