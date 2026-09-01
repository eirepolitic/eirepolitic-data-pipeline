# Instagram Visual Commissioning Playbook

## Purpose

This document records the working method that proved effective while commissioning the Instagram Content Factory. It is intended for future agents and maintainers who need to continue the work without re-learning the same visual, data, and collaboration lessons.

The core principle is simple:

> **Generate → return a directly viewable artifact → human reviews → capture the exact requested change → update the reusable implementation → regenerate → compare → approve → scale to the batch.**

The human reviewer is the visual editor. The agent should actively inspect its own output before returning it, but should not substitute its taste for explicit human approval.

The machine-readable companion to this document is:

`instagram/reference/visual_commissioning_protocol.yml`

---

## 1. The preferred commissioning loop

### Step 1 — Establish the semantic contract before styling

Before rendering anything, establish:

- what population is being measured;
- the grouping grain (for example party vs constituency);
- the requested time period;
- the metric definition and baseline;
- whether zero values are included in a baseline;
- the output unit and visible label;
- what the result does **not** mean;
- whether the run is review-only or publication-capable.

Ask the user only about decisions that materially change function, cost, semantics, or architecture. Do not repeatedly ask about already-settled details.

If the requested grain changes from a previously commissioned product, confirm it rather than silently reusing the old grouping logic.

### Step 2 — Commission the smallest useful visual slice

Do not start with the full batch unless the design is already approved.

The most efficient sequence is:

1. one representative entity and one slide;
2. one representative entity with the complete carousel;
3. side-by-side comparisons when spacing or scale is being tuned;
4. full batch only after the reusable visual rules are approved.

This avoids wasting batch renders on changes that can be discovered from one example.

### Step 3 — Return the artifact directly

Every visual review turn should contain a one-click artifact link.

Preferred behavior:

- GitHub-generated images: return a `raw.githubusercontent.com` link to the committed PNG/JPG;
- locally generated temporary review image: return a sandbox file link;
- for layout comparisons: return a same-scale side-by-side comparison image as well as the individual asset when useful.

Do not say that a visual is ready without giving the reviewer a direct way to open it.

Large binary files may cause `ResponseTooLargeError` through the GitHub file API. That does **not** mean the file is absent. Use repository paths, raw GitHub links, a workflow artifact, or a narrower API call instead.

### Step 4 — Self-review before asking the user to review

Before returning a render, inspect it for obvious regressions:

- clipping;
- duplicated text;
- ornaments or other assets contaminated by nearby content;
- underlines crossing glyphs;
- title/corner collisions;
- unreadable wrapping or truncation;
- inconsistent scale between related slides;
- unexpected margins;
- incorrect metric labels;
- stale period labels;
- wrong source or analysis grain.

A render that obviously contains one of these defects should be corrected before being presented as a review candidate.

### Step 5 — Treat user feedback as a reusable design change

When the user requests an edit, identify whether it is:

- a one-off content correction;
- a reusable style rule;
- a reusable presentation mapping;
- a metric/semantic change;
- an architecture change.

Prefer changing the reusable template, renderer, mapping, or project definition rather than manually modifying a single exported image.

A temporary mockup is fine for fast iteration, but once the direction is approved, move the rule into the real factory.

### Step 6 — Regenerate and compare

For spacing, scale, alignment, and ornament changes, produce a comparison against a real generated slide at the **same scale**. Do not compare against an unrelated screenshot if the actual factory output is available.

When the user requests a second iteration, preserve the approved parts and change only the requested dimension unless another defect must be corrected.

### Step 7 — Scale only after approval

Once a representative carousel is approved:

- generate all entities;
- run automated QA;
- generate contact sheets grouped by slide type;
- inspect outliers;
- return batch review links;
- keep publication disabled until explicit approval.

---

## 2. Visual rules established during commissioning

### Use the real reusable corner assets

Do not redraw, approximate, or crop the Celtic corner ornaments from a finished slide.

Use the canonical repo assets:

- `instagram/templates/assets/corner_tl.png`
- `instagram/templates/assets/corner_tr.png`
- `instagram/templates/assets/corner_bl.png`
- `instagram/templates/assets/corner_br.png`

The working source reference is:

`instagram/reference/member_profile_template.png`

A failed iteration extracted a corner from a finished glossary image and accidentally included part of the word `Issues`. The new heading was then drawn on top of the baked-in old text. This is why reusable decorative elements must come from dedicated assets, not screenshots of completed compositions.

### Underlines must use rendered text bounds

Do not estimate an underline from the nominal font size or baseline.

Standard method:

1. render or measure the exact heading using the same font and anchor;
2. obtain the actual rendered bounding box;
3. set the underline position to `bbox_bottom + fixed_gap`;
4. use a consistent gap (the approved glossary review used approximately 8 px at the commissioning canvas size).

This prevents the underline from intersecting descenders or the body of the glyphs.

### Main titles use one divider

The main title should have one clear yellow/gold divider beneath it. Do not add small secondary line fragments that make the title appear double-underlined.

### Title bands must accommodate two lines

The title band was moved down because long/two-line titles were colliding with the top corner ornaments.

Current shared layout on the commissioning branch was adjusted to approximately:

- title y: `32`
- title height: `136`
- title rule y: `174`
- main media y: `190`
- main media height: `1126`

These values are implementation context, not eternal design law. Validate visually if the renderer/canvas changes.

### Chart-to-title spacing

The user requested the visual/chart area to sit approximately **half as far from the yellow title divider** as it did in the current January render, and to grow into the recovered space.

This is a desired visual rule for chart slides. It was agreed during commissioning and should be implemented/validated in the reusable visual-slide layout rather than by manually moving individual PNGs.

### Presentation labels are separate from taxonomy

Long classification labels should be shortened only in the presentation layer.

Use:

`instagram/reference/issue_presentation_labels.yml`

Derived data should preserve the canonical classifier label and expose a separate display label. Never rewrite the production taxonomy simply to make a chart fit.

### Consumer titles should explain the question, not the statistical method

Approved plain-English slide titles:

- `Most Discussed Issues`
- `Issues Discussed More Than Average`
- `Issues Discussed More Per TD`

Avoid prominent consumer wording such as:

- normalized;
- over-index;
- issue-share over-index;
- technical formula names.

The visible title should tell a layperson what the slide answers.

### Cover slides should use concrete descriptive metrics

The approved cover direction uses:

- `CLASSIFIED SPEECHES`
- `AVG SPEECHES PER TD`

For the commissioned Fianna Fáil January example, this was:

- `770` classified speeches;
- `16.0` average speeches per TD (`770 / 48`).

Do not use vague cover labels such as `VS AVG / ISSUE SHARE`. Comparisons belong on the analytical slides.

Use `per TD`, not `per speaker`, when the denominator is the party/group TD count.

### Glossary is a standard final slide

The user's normal post format includes a glossary. The repeatable monthly carousel should therefore end with one.

Approved glossary style:

- same green background as the post;
- exact canonical corner assets;
- centered `Glossary` title;
- one yellow/gold divider under the main title;
- glossary term in bold with a safely spaced underline;
- white body text;
- generous vertical spacing;
- no cards or boxes.

Approved party-oriented glossary concepts:

- Issues;
- Classified Speeches;
- Average Party;
- Per TD;
- Points vs Average.

If the analytical grain changes, update the glossary terminology to match the actual baseline (for example `Average Constituency` rather than `Average Party`).

---

## 3. Data and semantic rules

### Political speech frequency is descriptive

The charts describe classified Dáil speech activity. They do **not** establish:

- party policy;
- party position;
- priority;
- sentiment;
- endorsement;
- intent.

This distinction should remain explicit in metadata/caption/methodology where appropriate.

### Current commissioned party metrics

#### Raw counts

`Most Discussed Issues`

For each issue, count classified speech segments for the entity during the period.

Visible values: `N speeches`.

#### Share vs average

`Issues Discussed More Than Average`

For each entity and issue:

- calculate the entity's share of classified speech activity on that issue;
- calculate the unweighted mean issue share across the comparison entities, including zero shares;
- subtract the baseline from the entity share;
- show positive differences only;
- rank and show the top entries (currently top 7).

Visible values: `+X.X pts vs avg`.

For party mode, the baseline is the average party, not a speech-weighted average party.

#### Per-TD vs average

`Issues Discussed More Per TD`

For each entity and issue:

- divide issue speech count by the entity TD count;
- calculate the unweighted mean of those entity-level rates, including zeroes;
- subtract the baseline rate;
- show positive differences only;
- rank and show the top entries (currently top 7).

Visible values: `+X.XX per TD vs avg`.

### Historical denominator caution

A historical/monthly post should ideally use membership/affiliation appropriate to the requested period. Using today's roster to reconstruct an older month can change the denominator or grouping after defections, vacancies, or membership changes.

Before calling a historical run publication-quality, inspect whether a period-correct membership source exists and record which snapshot was used.

---

## 4. Repeatable monthly architecture

A repeatable monthly format should be one reusable project, not a new project YAML per calendar month.

Preferred model:

- project owns slide structure, calculations, wording, templates, mappings, QA, and review policy;
- runtime owns the requested period and source snapshot;
- automation may resolve `last_completed_month`;
- manual runs may override the period for reproduction/backfill.

Changing July to August is a new **run**, not a new **project**.

A new project/version is appropriate only when the actual product changes materially.

### Readiness gate

Before a monthly render:

- verify the period is complete;
- verify classifier coverage reaches the period end;
- verify required membership/grouping data exists;
- verify join quality;
- freeze/record the source snapshot used for the run.

Fail rather than silently generating a partial period.

### Suggested output lineage

Each run should retain enough information to reproduce it:

- project/version;
- resolved period;
- data source bucket/key;
- source batch/coverage;
- membership snapshot;
- grouping grain;
- metric definitions;
- canonical and presentation labels;
- derived values;
- rendered slides;
- QA results;
- review/approval state.

---

## 5. Review states and versioning

Treat visual commissioning as a stateful process rather than a sequence of untracked files.

Recommended states:

1. `draft`
2. `rendered`
3. `review_requested`
4. `changes_requested`
5. `visual_approved`
6. `batch_generated`
7. `batch_qa_passed`
8. `publication_ready`

`publication_ready` is not the same as `published`.

Instagram publication remains a separate action and must not happen automatically unless separately designed and explicitly approved.

For each review iteration, retain when practical:

- iteration number;
- parent iteration/artifact;
- requested change;
- implementation location;
- rendered artifact path;
- QA result;
- approval status.

Do not overwrite the meaning of an approved version with an unrelated experiment.

---

## 6. Artifact review contract

A useful review artifact should answer four questions immediately:

1. **What am I looking at?** — entity, period, slide/metric.
2. **Is this the real factory output or a temporary mockup?**
3. **What changed since the previous version?**
4. **Where do I click to see it at full size?**

When comparing layouts, use same-size canvases and label each side clearly. A comparison image should not rescale one design differently from the other.

Contact sheets are best used after the representative design is stable, to find batch outliers rather than to commission basic typography.

---

## 7. Important failure modes learned

### Do not claim tools are unavailable without checking

A prior response incorrectly claimed repository execution tools were unavailable even though they remained available. Always inspect the available tools/current repo state before stating a limitation.

### Do not use finished slides as reusable asset sources

This caused text contamination in a corner ornament. Use dedicated source assets.

### Do not return an obviously broken review artifact

Self-review first. The reviewer should be deciding taste and intent, not discovering preventable rendering mistakes.

### Do not create month-specific project copies for a repeatable product

Period is runtime configuration. Avoid `...jul2026`, `...aug2026`, etc. as separate project definitions when the content product is the same.

### Do not silently change the analysis grain

Party, constituency, member, and other grains change joins, baselines, wording, and glossary definitions. Confirm an ambiguous grain change.

### Do not modify production taxonomy for presentation

Use a display-label mapping.

### Do not interpret speech frequency as policy priority or position

Keep the semantic limitation explicit.

### Do not treat GitHub `ResponseTooLargeError` as evidence a file is missing

Narrow the tree/query or use raw links/workflow artifacts.

### Do not accumulate temporary review logic in the production workflow

During commissioning, `.github/workflows/instagram_factory_party_project.yml` was temporarily repurposed for glossary and cover review jobs. Future work should inspect the current workflow contents and restore/refactor a reusable idempotent workflow rather than layering more one-off patches on top.

---

## 8. Fast-start checklist for a future agent

Before continuing this project:

1. inspect the current branch and workflow file rather than assuming the handoff reflects the latest commit;
2. read this document;
3. read `instagram/reference/visual_commissioning_protocol.yml`;
4. inspect `instagram/reference/issue_presentation_labels.yml`;
5. use the canonical corner assets under `instagram/templates/assets/`;
6. verify the requested grouping grain and period;
7. verify production data readiness;
8. render one representative entity first if visual rules have changed;
9. return a direct artifact link;
10. apply user feedback to reusable code/config where possible;
11. only then generate the full batch/contact sheets;
12. keep publication disabled until explicit approval.

---

## 9. Current reference artifacts from commissioning

These repository paths are useful reference points on the commissioning branch:

- approved-ish/latest cover review:
  `instagram/commissioning/output/cover-review/v1/fianna-fail-cover.png`
- approved glossary review:
  `instagram/commissioning/output/glossary-review/v5/glossary.png`
- glossary vs actual issue-slide comparison:
  `instagram/commissioning/output/glossary-review/v5/comparison_with_actual_issue_slide.jpg`
- January normalized outputs:
  `instagram/commissioning/output/january-overindex-2026/`

These are visual references, not substitutes for the reusable source templates/renderers.

---

## 10. Definition of success

The commissioning process is working when:

- the user can open every review artifact immediately;
- feedback can be expressed in visual/plain language rather than code;
- the agent can translate that feedback into reusable implementation changes;
- iterations preserve previously approved decisions;
- automated QA catches mechanical failures;
- human review remains the final visual/semantic gate;
- a repeatable format can be rerun for another period without cloning the project;
- the resulting package is reproducible and reviewable before publication.
