# Written Parliamentary Question semantic classification research

## Status

Research pilot completed through the first 300-section V2 benchmark on 2026-09-05/06.

This work is **research-only**. It does not publish semantic classifications to production and does not change the production pointer.

The purpose is to determine how EirePolitic should use LLMs to create a reusable semantic routing/index layer over the certified Written Parliamentary Question question-and-answer corpus.

## Why this research exists

The production Written PQ foundations provide a reliable source relationship:

`submitted question(s) -> Written-answer section -> official ministerial response`

The answer corpus is rich natural-language text. Structural XML parsing can reliably isolate the question and answer, but it does not by itself tell us what political/policy subjects are discussed, what the TD is asking for, or what type of response the Government provides.

The intended first semantic layer is therefore an **index/routing layer**, not a final interpretation or performance score.

The first-pass layer should make later targeted extraction possible without repeatedly sending the entire corpus through more expensive specialised analysis.

## Core design decision: three semantic views

The research preserves three views:

1. **Question view** — what each submitted question is about and what information/action it requests.
2. **Answer view** — what the official answer itself actually discusses and what kind of response it provides.
3. **Combined exchange view** — the overall subjects represented by the question/answer exchange.

The question and answer must not be collapsed into one classification.

In the V2 25-section pilot, **18 of 25 sections had different question and answer topic-tag sets**. This is strong evidence that the distinction is analytically meaningful.

### Combined view should be derived

In the same 25-section pilot, every combined topic set was equal to the union of question and answer topic tags.

Therefore the preferred design is:

`combined_topics = union(question_topics, answer_topics)`

rather than spending additional model-output complexity on independently classifying the combined exchange.

This still provides all three requested views while reducing model work and eliminating a redundant source of inconsistency.

## Model-call grain

The current pilot uses **one model call per Written-answer section**.

A section may contain one or several linked submitted questions.

The call receives:

- all linked question texts, each with its source question ID;
- the one official answer text;
- limited source metadata for context only.

The call returns:

- a separate classification for every linked question;
- one answer classification.

The combined exchange topic view is derived afterward.

This avoids paying repeatedly for the same answer where several questions are answered together.

## What source metadata is not delegated to the LLM

Known facts should remain normal structured metadata rather than model-generated semantic tags.

Examples:

- question date/year;
- submitting TD;
- recipient minister/department;
- party/constituency when joined from certified historical foundations;
- Written-answer section ID;
- grouped-answer status;
- referral/direct-reply structural status;
- answer-status fields already derived from XML.

The semantic classifier should not spend tokens rediscovering facts that the source already provides deterministically.

## Semantic output dimensions

### Question output

Per submitted question:

- `topic_tags` — zero or more approved topic leaf tags;
- `question_intents` — one or more controlled intent labels;
- `entities` — named entities explicitly present in the question text;
- `proposed_new_tags` — concepts not adequately covered by the controlled taxonomy;
- `topic_evidence` — short verbatim evidence for every selected topic tag.

Current question-intent examples include:

- statistics/data request;
- funding/cost request;
- policy-position request;
- policy-status request;
- implementation-status request;
- timeline/deadline request;
- explanation/rationale request;
- action/intervention request;
- local/service/case update;
- legislative status;
- staffing/capacity;
- eligibility/rules;
- review/investigation.

### Answer output

Per Written-answer section:

- `topic_tags` — zero or more approved topic leaf tags based only on answer text;
- `answer_characteristics` — controlled descriptive response attributes;
- `entities` — named entities explicitly present in answer text;
- `proposed_new_tags`;
- `topic_evidence`.

Current answer-characteristic examples include:

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

These are descriptive features, not answer-quality scores.

## Topic taxonomy design

### Controlled vocabulary, not free-form production tags

The model selects from an approved taxonomy.

If no approved tag fits a genuinely useful recurring concept, it may return a `proposed_new_tag` with evidence and a reason.

Proposed tags do **not** automatically become permanent taxonomy members.

They must be reviewed and either:

- added;
- mapped to an existing tag;
- made a child of an existing concept;
- rejected as too specific/noisy.

This avoids uncontrolled drift such as multiple near-synonyms for the same political issue.

### Hierarchical taxonomy

Broad categories organise the vocabulary, but the model should normally choose more specific leaf tags.

Examples:

- Health -> hospitals / waiting_lists / health_staffing / mental_health / ...
- Housing -> social_housing / homelessness / rental_market / planning / ...
- Transport -> public_transport / transport_fares / rail / roads / ...
- Foreign affairs and defence -> international_relations / european_union / defence_forces / ...

The V1 pilot showed that allowing broad category IDs to be selected directly could encourage semantically loose labels.

The V2 design therefore makes broad categories **derived hierarchy metadata**, not normal model-selected tags. A `*_general` leaf is available only when the category is clearly relevant but no more specific leaf fits.

### Taxonomy refinement already observed

The pilot produced concrete evidence for refinement.

For example, a public-transport fare question was initially vulnerable to the incorrect nearby tag `energy_prices`. The taxonomy now contains the explicit leaf `transport_fares`, with prompt rules distinguishing transport fares from electricity/gas/fuel energy prices.

Likewise:

- international diplomacy should use `international_relations`, not a broad defence label;
- accessibility/equality issues should not automatically become criminal-justice topics;
- the mere presence of a department or public body must not trigger `government_public_service` or `state_agencies`.

## Strict scope separation

A central guardrail is that every semantic view is grounded only in its own text.

### Question scope

Question topics/entities/evidence may use only that question's text.

### Answer scope

Answer topics/entities/evidence may use only the official answer text.

The classifier must not transfer a question's topic into a short referral-only answer merely because that answer responds to the question.

For example, an answer saying only that the HSE has been asked to reply directly should be classified primarily as a referral/no-substantive-answer response unless it independently contains substantive health-policy content.

### Combined scope

Combined topics are derived from the already-grounded question and answer topic sets.

## Entity extraction guardrails

The first-pass pilot permits lightweight entity extraction, but entities must be explicit in the scoped source text.

The model must not turn generic expressions such as:

- `my Department`;
- `the Department`;
- `the Minister`;

into a named entity using recipient metadata.

More sophisticated entity resolution/linking should be a separate deterministic/semantic stage later.

## Evidence grounding

Every selected topic tag must be accompanied by a short verbatim supporting quote from the same source scope.

Entity and proposed-tag outputs also carry evidence quotes.

Deterministic validation checks:

- expected question IDs exactly reconcile;
- only approved topic tags are used;
- every topic has matching evidence;
- evidence quote occurs in the correct source text;
- entity evidence occurs in the correct source text;
- proposed-tag evidence occurs in the correct source text;
- empty answer text does not receive semantic answer topics/entities.

This is a key design principle:

**LLM interprets; deterministic code verifies what can be verified.**

## Automatic repair

V2 permits one repair attempt when deterministic validation fails.

The repair call is told which structural/evidence checks failed and is asked to correct those errors using the same source text.

No unlimited retry loop is allowed.

The final record retains pilot metadata indicating whether repair was attempted and whether any deterministic validation errors remain.

## Pilot history and findings

### Initial schema failure

The first live attempt failed before inference because OpenAI strict structured output did not accept the JSON Schema keyword `uniqueItems` in this response schema.

No model tokens were used in that failed attempt.

The constraint was removed from the API schema and output normalisation is performed locally instead.

### First successful five-section pilot

Five of five calls succeeded.

However, two responses duplicated the same question object inside the output question array.

This established that strict JSON schema alone is not sufficient to guarantee semantic uniqueness of array members.

The runner now normalises to one output per known source question ID and validates exact source-ID reconciliation.

### V1 25-section pilot

The first 25-section benchmark completed mechanically with:

- 25/25 successful model calls;
- zero API failures;
- no proposed new tags.

Manual/deterministic review exposed two important issues:

1. broad `government_public_service` was being selected too often merely because Government/public bodies appeared in the exchange;
2. some short referral answers inherited the question's subject instead of being classified strictly from their own text.

The prompt and validator were tightened rather than accepting this noise.

### Tightened 25-section pilot

After strict scope separation:

- `government_public_service` question usage fell substantially;
- it disappeared from answer topic classifications in that sample;
- question and answer topic sets differed in 18/25 sections;
- three sections still failed strict evidence validation, mainly evidence-quote exactness/coverage rather than API/schema failure.

Those findings motivated V2:

- leaf/general tags only;
- broad categories derived in code;
- combined topics derived in code;
- one deterministic-validation repair attempt;
- explicit semantic distinctions discovered from errors.

### V2 25-section pilot

Run `34005953314` completed successfully.

Results:

- 25/25 calls succeeded;
- **0 final validation-error sections**;
- zero proposed-new-tag rows in the sample;
- 18/25 question/answer topic sets differed;
- all 25 combined topic sets equalled the question+answer union;
- average model usage was approximately **6,189 total tokens per answer section** in this configuration.

This justified the broader 300-section research benchmark.

### V2 300-section benchmark

Run `34006240512` attempted the full stratified 300-section benchmark.

The model-call loop itself reached all 300 target sections. Aggregate results recovered from the GitHub Actions log were:

- attempted sections: **300**;
- successful parseable sections: **299**;
- malformed/unparseable model responses: **1**;
- sections with final deterministic validation errors: **7**;
- proposed-new-tag rows: **3**;
- input tokens: **1,715,710**;
- output tokens: **181,847**;
- total tokens: **1,897,557**;
- average total tokens per successful section: **6,346.3**;
- production changed: **false**.

The one malformed response was for `2026-01-15` / `dbsect_490` and failed JSON parsing because the returned JSON string was truncated/unterminated. This is an operational robustness issue rather than evidence that the taxonomy itself failed. A production-quality runner must retry malformed structured output and preserve successful records even if one section fails.

The benchmark job returned failure because the research runner was intentionally fail-closed when any model call failed. A separate artifact-persist step then also failed because the long-running workflow attempted to push to a branch that had advanced during execution. The final aggregate benchmark summary was subsequently recovered from the run logs without making additional model calls.

#### What the 300-section benchmark supports

The first-pass semantic routing architecture is technically viable, but **V2 is not yet approved for production**.

The 299/300 structured-response success rate is encouraging, but 7/299 surviving deterministic validation failures are too many to silently accept in a production backfill. The correct production design must distinguish at least:

- valid classification;
- repaired-and-valid classification;
- malformed response requiring retry;
- unresolved validation failure requiring quarantine/review.

No invalid row should be silently published as certified semantic metadata.

The benchmark also confirms that the richer first-pass schema is token-heavy. Nearly 1.9 million tokens were used for only 300 answer sections. With roughly 96,675 production Written-answer sections, naïvely scaling this exact schema would imply hundreds of millions of tokens before retries/repairs.

#### Descriptive benchmark patterns

These counts describe the stratified 300-section research sample only; they are **not estimates of full-corpus prevalence**.

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

These results support the original idea that first-pass routing can do more than issue tagging. Intent and response-characteristic dimensions appear capable of selecting useful subsets for later specialised extraction—for example, commitments, statistics, funding, referrals and legislation.

The three proposed-new-tag rows show that the controlled taxonomy still needs an escape hatch, but the low count does not by itself prove the taxonomy is complete because the proposed-tag contents were lost when the original benchmark artifacts failed to persist.

## Cost and scale implications

The certified production corpus contains approximately 96,675 Written-answer sections.

At the measured V2 300-section average of **6,346.3 total tokens per successful section**, a simple linear extrapolation is roughly **614 million total tokens** for 96,675 sections before accounting for retry/repair overhead or differences in full-corpus answer lengths.

This is a planning estimate, not a billing estimate and not a statement of exact future usage.

Therefore production design must consider:

- reducing repeated static prompt/taxonomy tokens;
- prompt caching where supported and economically useful;
- batching/parallel execution rather than sequential calls;
- separating cheap routing classification from richer entity/evidence extraction;
- reusing classifications for unchanged source-section hashes;
- calling the model only for new/changed sections after initial backfill;
- evaluating whether every first-pass field is worth its token/output cost;
- retrying malformed structured output narrowly rather than rerunning whole batches;
- quarantining unresolved validation failures rather than failing or republishing the entire corpus.

No full-corpus model backfill should be approved solely because this 300-section research benchmark is broadly successful.

## What this first pass is not

The pilot does **not** attempt to decide:

- whether a TD supports/opposes a policy unless separately and explicitly evidenced;
- whether an answer is evasive;
- whether an answer is truthful;
- whether a minister performed well;
- whether a question is effective;
- whether Government policy is good/bad;
- which side 'won' an exchange.

These would require separate definitions, evidence models and validation work and may be unsuitable for EirePolitic metrics altogether.

## Relationship to later claim extraction

Topic routing is only the first semantic layer.

A later specialised pass may extract atomic claims such as:

- monetary allocations;
- reported statistics;
- programme/scheme status;
- stated policy positions;
- future commitments/actions;
- dates/deadlines;
- named organisations and their roles;
- references to earlier PQs;
- geographic scope.

The routing layer should make those passes cheaper and more targeted. For example, a future commitment extractor could run only on answers whose first-pass characteristics include `future_action_or_commitment`.

## Living next-step plan

1. **Do not rerun the 300-section V2 benchmark simply to recreate lost artifacts.** The aggregate evidence has been recovered and additional identical calls would add cost without answering a new research question.
2. Harden the research runner so every successful classification is checkpointed/preserved even when another record fails, and so malformed JSON gets a bounded targeted retry.
3. Preserve unresolved validation failures as explicit quarantine records instead of treating them as publishable classifications.
4. Create a **lower-token routing variant** on the same deterministic sample design. Move entities and per-topic verbatim evidence out of the cheapest first pass where possible, retaining only fields needed to route later analysis.
5. Compare the lower-token variant with V2 on a bounded sample for:
   - topic agreement;
   - question-intent agreement;
   - answer-characteristic agreement;
   - validation failure rate;
   - proposed-tag behavior;
   - token reduction.
6. Build a small human-reviewed gold set (including grouped answers, referral-only answers, long answers, empty/no-reply cases, and taxonomy edge cases).
7. Measure classification agreement/error against the gold set. Deterministic validator success alone is not semantic accuracy.
8. Review the three proposed-tag cases when a future targeted benchmark captures them durably; do not expand the production taxonomy from aggregate counts alone.
9. Only after the cheaper routing benchmark and gold-set review, propose production semantic tables and incremental model-call workflows.
10. Separately pilot second-pass structured claim extraction using subsets selected by answer characteristics such as `future_action_or_commitment`, `statistics_or_figures_supplied`, `funding_or_cost_figures_supplied`, and `legislation_or_regulation_discussed`.

No production semantic-classification dataset is approved yet.
