# Instagram speech-metrics carousel follow-up

Status: **Member coverage certified; carousel metrics ready for editorial selection**  
Date: **6 September 2026**

This refines candidate 2 in `instagram_post_candidates.md` into a multi-slide carousel about who speaks most, longest and most often in the current Dáil transcript history.

## Proposed post

**Working title:** Who talks most in the Dáil?

Potential slides:

1. most recorded transcript interventions;
2. most total words spoken;
3. longest average intervention, with a minimum-intervention threshold;
4. longest single recorded intervention;
5. most debate days with at least one intervention;
6. fewest recorded interventions per eligible debate day, with a minimum eligibility threshold.

Do not use issue labels in this post.

## Member-reference investigation

The earlier exploratory probe incorrectly read physical `processed/oireachtas_unified/latest/...` S3 objects directly. Under the immutable batch architecture, those physical legacy objects can be stale and are **not** the canonical production read surface.

Canonical reads must go through `extract.oireachtas.io_s3.get_bytes()` / `resolve_read_key()`, which resolves logical production keys through the active production batch pointer. Candidate builds similarly resolve logical keys inside the active candidate batch.

The corrected production-pointer read on 6 September 2026 resolves to:

- `silver_members`: **176 rows**;
- `silver_member_memberships`: **176 rows**;
- `gold_current_members`: **174 rows**.

Therefore the earlier 100/98-row observation was a diagnostic error, not an incomplete production roster.

A permanent regression test now guards both behaviours:

- logical reads use the production pointer rather than stale direct objects;
- candidate reads use the current candidate batch rather than production or stale direct objects.

## Event-date membership model

The data model explicitly supports members entering or leaving during a Dáil through dated membership fields:

- `membership_start`;
- `membership_end`;
- `house_no`;
- `chamber`;
- `is_current`.

Member-level historical metrics should test the event date against the relevant membership interval rather than use today's roster as the historical denominator.

**Catherine Connolly is a confirmed example:**

- member code: `Catherine-Connolly.D.2016-10-03`;
- 34th Dáil membership start: **29 November 2024**;
- membership end: **25 October 2025**;
- current-member flag: **false**.

The speech analysis therefore includes her speeches while she was a TD and excludes dates after her membership ended. Within the covered transcript period she had **76 eligible debate days**, **370 recorded interventions**, and spoke on **64 eligible debate days**.

## Certified speech-analysis coverage

The current compatibility speech history spans **18 December 2024 to 28 August 2026**:

- **163 debate days**;
- **66,192 transcript rows**;
- **65,749 rows already carry a native `member_code`**;
- **65,740 interventions** remain after event-date Dáil-membership validation;
- **176 members** overlap the covered Dáil period;
- **all 176** have at least one event-date-valid matched intervention in the covered speech surface.

This is sufficient for the proposed member-level carousel, subject to metric-specific interpretation caveats below.

## Current headline metrics

### Most recorded transcript interventions

1. Micheál Martin — **6,126**;
2. Verona Murphy — **5,161**;
3. Simon Harris — **2,836**.

This is a count of transcript interventions, not prepared speeches. Presiding/procedural roles can generate many short interventions, so this slide should be labelled literally rather than as a performance ranking.

### Most total words

1. Micheál Martin — **637,331 words**;
2. Simon Harris — **532,221**;
3. Jim O'Callaghan — **288,341**.

This is cumulative transcript volume during the covered period, not speaking quality or effectiveness.

### Longest average intervention

Using a minimum of **20 recorded interventions** to avoid tiny-sample winners:

1. Ciarán Ahern — **573 words** average;
2. Cormac Devlin — **541**;
3. Barry Ward — **509**.

The final rendered post should state the minimum-intervention rule.

### Longest single recorded intervention

1. Charlie McConalogue — **5,173 words**;
2. Paschal Donohoe — **4,828**;
3. Kieran O'Donnell — **4,748**.

Before rendering, the exact winning transcript row/section should be frozen into the publication evidence bundle.

### Most debate days with at least one intervention

1. Verona Murphy — **153 of 163 eligible debate days**;
2. Ruairí Ó Murchú — **147 of 163**;
3. Michael Collins — **144 of 163**.

This is a presence-in-transcript measure, not chamber attendance. A TD can be present without producing a recorded intervention, and vice versa this metric does not measure full-day attendance.

### Lowest intervention rate among members with at least 50 eligible debate days

A tenure-adjusted low-end comparison is now possible. Using **recorded interventions per eligible debate day** and requiring at least **50 eligible debate days**:

1. Sean Fleming — **12 interventions / 163 eligible days = 0.07 per day**;
2. Willie O'Dea — **22 / 163 = 0.13**;
3. Eamon Scanlon — **34 / 163 = 0.21**.

For public copy, prefer **"fewest recorded interventions per eligible debate day"** rather than "least talkative TD". The latter overstates what transcript activity measures.

## Interpretation guardrails

- A transcript `speech` is an intervention, not necessarily a prepared standalone speech.
- Intervention counts, words and speaking days do not measure political effectiveness, quality, influence or attendance.
- Office-holder and chair/presiding roles can materially affect intervention counts and average length.
- Raw session totals are valid descriptive totals but reflect time actually served in the Dáil; tenure-adjusted measures should use eligible debate days.
- For averages, use a published minimum sample threshold.
- For the low end, use an eligibility threshold and a rate, not raw counts.
- Do not use the current-member roster alone for historical denominators; use event-date membership intervals.

## Living next-step plan

1. Editorially choose 4–6 of the certified metrics for the carousel.
2. Freeze the production batch, coverage dates and metric definitions used for rendering.
3. Retrieve and verify the exact transcript row for the longest-single-intervention slide.
4. Decide whether the presiding-role effect should be shown as an explanatory slide or simply carried as a methodology note.
5. Keep the wording descriptive: "recorded interventions", "total words", "eligible debate days" and "average intervention length".

No production source data was changed during this investigation. The initial incomplete-roster concern was resolved as a read-path mistake in the temporary diagnostic, not a production data defect.
