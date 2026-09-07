# IPI raw polling carousel — 6 September 2026

Post ID: `2026-09-06-ipi-polling-carousel`
Repository: `Eirepolitic-data-pipeline`
Created by: ChatGPT execution agent
Created at: 2026-09-06T18:50:00-07:00

## Tools used
- GitHub repository file/branch/PR/workflow actions
- GitHub Actions
- approved July Instagram recurring factory at commit 386b933
- instagram.projects.ipi_polling_factory_v1 adapter and renderers
- Irish Polling Indicator raw polling feed from S3
- Pillow/matplotlib through the approved factory
- raw.githack browser review branch

## Process
1. Inspected main and retained the pinned July factory architecture and approved corner assets.
2. Separated the one-month same-pollster comparison used by slide 2 from the 183-day same-pollster trend window used by slide 3.
3. Expanded fixture data so CI proved the trend included actual Ireland Thinks rows only and ignored another pollster inside the window.
4. Rendered four 1080×1350 slides through python -m instagram.factory.recurring using the approved factory worktree.
5. Refreshed the production IPI ingestion before the final live render.
6. Validated the live package, published a browser-hosted review page, and received human approval before treating the post as complete.

## Decisions
- Use raw published polls only; never the IPI daily model for this carousel.
- The latest poll determines the pollster used across slides 1–3.
- Keep slide 2 as a roughly one-month comparison and slide 3 as an independent six-month trend.
- Use visible % change labels on slide 2 even though the calculation is the difference between percentages.
- Use compact single-row legend variant B above the trend chart.
- Use the approved July glossary component for methodology instead of a bespoke information layout.

## QA
- Pinned factory commit 386b933 verified in CI.
- Approved four corner PNG assets verified by Git blob SHA in CI.
- Slide 2 comparison remained inside the 28–45 day one-month window.
- Slide 3 used a configured 183-day raw-poll window with actual markers only and single-row legend.
- Other pollsters inside the trend period were excluded.
- No IPI model, smoothing or interpolation data was used.
- All four final slides validated at 1080×1350.
- Human review approved the final browser-hosted render.

## Sources
- s3://eirepolitic-data/processed/polling/irish_polling_indicator/latest/csv/polls.csv
- Irish Polling Indicator raw poll dataset

## Limitations
- The configured trend lookback is 183 days, but the earliest same-pollster observation available inside the final live window was 5 April 2026, so the plotted span was 154 days across six actual waves.
- Sampling and weighting details are not invented from absent fields; readers should check the original pollster release for full methodology.

## Related Workflows
- IPI ingestion run 34069398548
- final polling render run 34071067470
- approved July reference workflow 33894430571

## Related Pull Requests
- #133 six-month raw polling trend
- #134 saved polling methodology and final slide download
