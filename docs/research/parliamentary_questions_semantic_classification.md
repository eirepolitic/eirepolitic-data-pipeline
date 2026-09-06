# Written Parliamentary Question semantic classification research

## Status

Research-only investigation completed through a 300-section V2 benchmark and a 25-section lower-token comparison on 2026-09-05/06.

No semantic classification table has been approved or published to production.

## Objective

The production Written Parliamentary Question foundation now gives a reliable relationship:

`submitted question(s) -> Written-answer section -> official ministerial response`

The next problem is semantic: turning free-text questions and answers into a reusable routing/index layer that can select subsets for later analysis such as funding, statistics, commitments, legislation or policy-position extraction.

The first semantic layer is intended to describe and route text. It is not an effectiveness, truthfulness, quality or political-performance score.

## Recommended semantic architecture

### Preserve three views

The system should expose:

1. **Question view** — what each TD question is about and what it asks for.
2. **Answer view** — what the Government answer itself discusses and what kind of response it provides.
3. **Combined exchange view** — the overall subjects represented by the question and answer together.

The question and answer should not be collapsed into one model classification.

In the V2 25-section pilot, 18 of 25 sections had different question and answer topic sets. This supports retaining separate question and answer semantics.

The combined exchange topic set does not need a separate model classification. In the same pilot it was always the union of question and answer topic tags, so the preferred rule is:

`combined_topics = union(question_topics, answer_topics)`

This preserves the requested third view without spending model output on a redundant classification.

### Model-call grain

One model call should operate at **Written-answer-section grain**.

A call receives all questions linked to that section plus the one official answer. This avoids repeatedly sending the same grouped answer when several questions were answered together.

### Keep known metadata deterministic

Do not ask the model to rediscover fields already known from the certified source data, including:

- date/year;
- submitting TD;
- recipient minister/department;
- party/constituency when joined from historical foundations;
- section ID;
- grouped-answer status;
- referral/direct-reply structural status;
- deterministic answer status.

These remain normal structured filters around the semantic layer.

## First-pass semantic dimensions

### Question

Candidate first-pass fields are:

- multi-label topic tags;
- question intent;
- proposed new topic tag when the controlled taxonomy genuinely lacks a useful recurring concept.

Question-intent labels tested include:

- statistics/data request;
- funding/cost request;
- policy-position request;
- policy-status request;
- implementation-status request;
- timeline/deadline request;
- explanation/rationale request;
- action/intervention request;
- case/local-service update;
- legislative status;
- staffing/capacity;
- eligibility/rules;
- review/investigation.

### Answer

Candidate first-pass fields are:

- multi-label topic tags;
- answer characteristics;
- proposed new topic tag.

Answer characteristics tested include:

- factual information supplied;
- statistics/figures supplied;
- funding/cost figures supplied;
- policy position stated;
- policy explanation;
- implementation-status update;
- future action/commitment;
- timeline/deadline stated;
- legislation/regulation discussed;
- information unavailable/not held;
- referred to another body;
- direct reply promised by another body;
- table/structured data present;
- previous PQ/answer referenced;
- no substantive answer.

These are descriptive attributes. For example, `future_action_or_commitment` identifies an answer as potentially relevant to a later commitment extractor; it does not judge whether the commitment is good, credible or effective.

## Topic taxonomy

The preferred design is a controlled hierarchical taxonomy.

Examples:

- Health -> hospitals / waiting_lists / health_staffing / mental_health / ...
- Housing -> social_housing / homelessness / rental_market / planning / ...
- Transport -> public_transport / transport_fares / rail / roads / ...
- Foreign affairs and defence -> international_relations / european_union / defence_forces / ...

The model should normally select specific leaf tags. Broad categories are derived from the hierarchy rather than freely selected by the model.

A `*_general` tag may exist where a category is clearly relevant but no specific child fits.

The taxonomy should not expand automatically. The model may propose a missing tag, but proposed tags require review before entering the controlled vocabulary.

Pilot errors already produced useful taxonomy corrections. For example, public-transport fares require `transport_fares`; they should not be forced into the nearby but incorrect `energy_prices` category.

## Scope separation

Question semantics must be based only on the relevant question text.

Answer semantics must be based only on the answer text.

A short referral answer must not automatically inherit the subject of the question merely because it is responding to that question.

The mere appearance of a department or public body is not itself a `government_public_service` or `state_agencies` topic. Those tags should apply only where administration, governance, agency structure or public-service delivery is substantively discussed.

## Rich V2 evidence-grounded variant

The richer V2 experiment also returned explicit entities and short verbatim evidence for topic tags.

Deterministic validation checked:

- exact question-ID reconciliation;
- approved topic vocabulary;
- topic/evidence coverage;
- evidence quote appearing in the correct question or answer text;
- entity evidence appearing in the correct scope;
- no semantic answer output from empty answer text.

One bounded repair call was permitted when these checks failed.

This implements the principle:

**LLM interprets; deterministic code verifies what can be verified.**

## Benchmark evidence

### V2 25-section pilot

Run `34005953314`:

- 25/25 calls succeeded;
- 0 final validation-error sections;
- question and answer topic sets differed in 18/25 sections;
- all combined topic sets equalled the question+answer union;
- average usage was about 6,189 total tokens per section.

### V2 300-section benchmark

Run `34006240512` completed all 300 attempted model calls before the research job returned failure.

Recovered aggregate results:

- attempted sections: **300**;
- parseable classifications: **299**;
- malformed/unparseable model responses: **1**;
- final deterministic validation-error sections: **7**;
- proposed-new-tag rows: **3**;
- input tokens: **1,715,710**;
- output tokens: **181,847**;
- total tokens: **1,897,557**;
- average total tokens per successful section: **6,346.3**;
- production changed: **false**.

The one malformed response was truncated/unterminated JSON for `2026-01-15` / `dbsect_490`. This is an operational reliability issue: a production-quality runner needs a bounded retry for malformed structured output and must preserve already-successful rows rather than failing an entire backfill.

The seven surviving validation failures mean the rich V2 design is promising but not ready to be silently published as certified semantic metadata.

A production design would need explicit states such as:

- valid;
- repaired-and-valid;
- malformed response awaiting retry;
- unresolved validation failure/quarantine.

### Descriptive patterns from the 300-section sample

These counts describe the stratified research sample only. They are not full-corpus prevalence estimates.

Frequent question intents included:

- `request_statistics_or_data`: 198;
- `request_policy_status`: 118;
- `request_action_or_intervention`: 114;
- `request_funding_or_cost`: 90;
- `request_policy_position`: 72.

Frequent answer characteristics included:

- `factual_information_supplied`: 188;
- `referred_to_another_body`: 135;
- `policy_explanation`: 133;
- `direct_reply_promised_by_another_body`: 124;
- `implementation_status_update`: 99;
- `no_substantive_answer`: 92;
- `future_action_or_commitment`: 88;
- `legislation_or_regulation_discussed`: 80;
- `statistics_or_figures_supplied`: 74;
- `timeline_or_deadline_stated`: 63;
- `funding_or_cost_figures_supplied`: 61.

This supports the idea that the routing layer should contain more than issue tags. Intent and response characteristics can directly select useful subsets for specialised later passes.

## Lower-token routing experiment

A second variant removed first-pass entity extraction and per-topic verbatim evidence, keeping only:

- question topic tags;
- question intents;
- answer topic tags;
- answer characteristics;
- proposed new tags.

It was run on the same deterministic 25-section sample as the full V2 comparison.

Run `34007851573` completed successfully with:

- 25/25 successful sections;
- 0 mechanical validation errors;
- average usage: **3,485.6 tokens per section**;
- total usage: **87,141 tokens** versus **154,712** for the rich V2 sample;
- token reduction: **43.68%**.

Model-vs-model Jaccard agreement with rich V2 was:

- question topics: **0.7460**;
- answer topics: **0.7187**;
- combined topics: **0.7227**;
- question intents: **0.8693**;
- answer characteristics: **0.8700**.

These are agreement measures, **not accuracy measures**. Rich V2 is not a human gold standard.

The experiment suggests that question-intent and answer-characteristic routing may be relatively stable when evidence/entity extraction is removed. Topic tagging changes more materially between the two variants.

That does not yet tell us which topic output is better. Human review is now the necessary next step.

## Scale implication

The production corpus contains approximately 96,675 Written-answer sections.

A simple linear extrapolation from the 300-section rich V2 average is roughly **614 million total tokens** for a full first pass, before retry/repair overhead or differences in full-corpus answer length.

The cheaper 25-section routing variant would imply roughly **337 million total tokens** at its observed average.

These are planning estimates, not billing estimates or guarantees of future usage.

The results strongly argue against treating the current rich V2 call as the obvious production first pass. Model usage should be reserved for semantic work that materially improves routing or later extraction.

Future production design should also consider:

- reusing classifications for unchanged source-section hashes;
- model calls only for new/changed sections after initial backfill;
- bounded targeted retry rather than rerunning whole batches;
- quarantine for unresolved output;
- concurrency/batch execution for throughput;
- prompt/caching efficiency where appropriate;
- moving entities/evidence into later specialised passes if human evaluation shows the cheap router is adequate.

## What is not being classified

This research does not attempt to label:

- whether a TD supports/opposes a policy unless explicitly defined and evidenced in separate research;
- evasiveness;
- truthfulness;
- answer quality;
- ministerial effectiveness;
- TD effectiveness;
- whether one side won an exchange.

These are either subjective, much harder to validate, or analytically distinct from the factual routing layer.

## Relationship to later claim extraction

The semantic router is intended to select text for more specialised passes.

Examples:

- `future_action_or_commitment` -> commitment/action extraction;
- `statistics_or_figures_supplied` -> structured statistics extraction;
- `funding_or_cost_figures_supplied` -> money/allocation extraction;
- `legislation_or_regulation_discussed` -> legislation/policy-state extraction;
- referrals -> external-body follow-up research.

A later claim layer could represent atomic statements with actor, action/claim type, subject, value, date/time period, geography and source evidence.

## Living next-step plan

The next research phase is **human-reviewed semantic evaluation**, not production deployment.

1. Build a bounded gold-review set of roughly 75–100 answer sections. It should deliberately include:
   - grouped answers;
   - referral-only/no-substantive answers;
   - long answers;
   - tables/figures;
   - commitments/timelines;
   - legislation;
   - funding questions;
   - health/housing/transport and smaller topic areas;
   - taxonomy edge cases;
   - examples where rich V2 and the cheap router disagree.
2. For each review item, collect human labels for:
   - question topics;
   - question intents;
   - answer topics;
   - answer characteristics;
   - missing/proposed taxonomy concepts.
3. Compare both the rich V2 and cheaper router against that gold set. Measure agreement/error by output family rather than one overall score.
4. Decide independently which first-pass fields justify their model cost. A plausible outcome is a hybrid design—for example, cheap intents/answer-characteristics with a different topic strategy—but this must be decided from gold-set evidence, not model-vs-model agreement.
5. Tighten taxonomy definitions only where human review shows systematic ambiguity or missing concepts.
6. Define acceptable production thresholds and quarantine rules before proposing semantic production tables.
7. After the router is validated, run a separate bounded second-pass claim-extraction pilot on selected answers tagged for commitments, statistics, funding and legislation.
8. Only then propose the recurring production model workflow and initial historical backfill strategy.

No production semantic-classification dataset is approved yet.
