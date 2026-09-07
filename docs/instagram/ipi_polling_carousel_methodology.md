# Approved IPI polling carousel methodology

Status: approved baseline for future EirePolitic polling carousels.

## Factory and visual framework

- Reuse the approved July Instagram factory at commit `386b933`.
- Continue using the pinned worktree/factory execution pattern with `python -m instagram.factory.recurring`.
- Keep the approved four corner PNG assets and their CI blob-hash checks.
- Do not replace this architecture with a bespoke renderer.
- Keep output at 1080×1350 per slide.

## Data source and selection

- Use raw published polls from the Irish Polling Indicator feed.
- Do not use the IPI daily model, smoothing, interpolation, or daily estimates.
- The latest published poll determines the pollster used across slides 1–3.
- All trend points must correspond to actual published poll rows from that same pollster.

## Slide 1 — Latest Polling Percentages

- Show the latest actual published voting-intention poll percentages.
- Display pollster, publication date, fieldwork dates where available, and sample size.

## Slide 2 — Who's Up and Who's Down?

- Compare the latest poll with the same pollster's earlier wave.
- Target a 30-day gap.
- Accept a comparison window of 28–45 days.
- If no poll falls inside that window, use the nearest previous same-pollster wave and record the fallback in metadata.
- Display the exact two publication dates.
- Calculate differences between percentages normally, but display visible change labels with `%` rather than `pp`.
- Keep slide 2 as a one-month comparison; do not expand it to the trend window.

## Slide 3 — Six Months of Polling

- Use a 183-day lookback ending on the latest poll.
- Include every actual same-pollster poll observation inside that window.
- Ignore other pollsters even when their polls fall inside the same period.
- Do not interpolate missing dates.
- Connect actual observations with lines and show a visible marker for every poll point.
- Use the approved theme-compatible distinct party colours.
- Use the compact single-row legend above the chart as the default.
- Show the actual first and last poll dates plotted, not an implied six-month start if no poll exists there.
- Supporting text should state that each marker represents one actual published poll.

## Slide 4 — About This Polling

Use the approved July glossary component. Explain:

- slides 1–3 use actual voting-intention polls;
- the selected pollster;
- latest poll publication date;
- fieldwork dates;
- latest sample size;
- slide 2 compares the same pollster with an earlier wave;
- slide 3 shows the same pollster's polls within the six-month window;
- every trend marker is an actual published poll;
- no IPI model, smoothing, interpolation, or daily estimate is used;
- the IPI feed provides pollster, fieldwork dates, publication date, sample size, and party results;
- full sampling and weighting methodology should be checked in the original pollster release.

Do not invent geography, respondent composition, weighting details, or methodology fields that are not present in the source feed.

## Review and publication

- Human review is required before publication.
- Review output should include the four-slide contact sheet, all four full-resolution PNG slides, caption, manifest, and a ZIP containing only the four finished slides for scheduling.
- The contact sheet is an overview image; final visual approval should use the individual PNG slides because they retain full resolution.
- Do not publish to Instagram automatically from the render/review workflow.
