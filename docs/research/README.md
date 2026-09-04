# Research documentation

This directory is the durable research record for investigations into EirePolitic/Oireachtas data and derived political metrics.

The purpose is to preserve not only final metrics, but also what was investigated, what the source data actually means, methodological caveats, data-quality discoveries, decisions taken, and promising future work.

## Working process

For each research topic:

1. Investigate the source and derived data.
2. Record confirmed findings and corrections in the topic document.
3. Update the living next-steps plan in that document.
4. Only then summarize the result externally.

Research documents should distinguish clearly between:

- confirmed source/data structure;
- derived descriptive findings;
- methodological cautions;
- pipeline changes already implemented;
- open questions;
- proposed enrichments or classifications;
- ideas deliberately deferred.

Do not promote exploratory conclusions into production metrics until their denominator, attribution rules, reliability and public interpretation have been reviewed.

## Topics

- [Parliamentary questions](parliamentary_questions_investigation.md) — main durable record covering submitted question records, written vs oral questions, oral-question sections/exchanges, transcript interventions, grouped questions, participant roles, substitution/proxy-taking, batching, recipient patterns, and future metrics/enrichments.
- [Parliamentary question-taking certification](parliamentary_questions_question_taking.md) — submitted-by vs actually-taken-by attribution, transcript ordering, explicit substitution evidence, bounded unresolved cases, and the conservative evidence model.
- [Oral-question exchange participant metrics](parliamentary_questions_exchange_metrics.md) — certified exchange word/intervention components, submitter participation, minister/chair/ordinary-member roles, mixed-role and anonymous-speaker edge cases, recommended participant foundation, and production implementation plan.
- [Oral-question exchange participant implementation](parliamentary_questions_exchange_metrics_implementation.md) — implemented schema, regression coverage, production deployment, live counts, permanent audits, and deployment evidence.
- [Oral-question section headings](parliamentary_questions_section_headings.md) — heading reuse, stability, normalization tests, recipient drift, public-filter interpretation, and the decision not to add a normalization dataset.
- [Oral versus Written parliamentary questions](parliamentary_questions_oral_vs_written.md) — channel volumes, recipient concentration, Taoiseach/department differences, member/party/constituency channel profiles, safe reusable measures, and the decision that existing foundations are sufficient.
- [Broader deterministic speech context](broader_speech_context.md) — deterministic context profiling beyond Oral questions, certified Leaders' Questions headings, Bill/debate/section relationship diagnostics, rejected debate-wide Bill joins, motion/statement/procedural caveats, precedence guardrails, production implications, and the living path into legislation research.
- [Legislation relationships](legislation_investigation.md) — Bill identifiers, stages, sponsors, debate-section relationships, speech and division linkage, conservative certification rules, source anomalies, coverage limits, production implications, and the living implementation plan.
- [Certified Bill-section bridge implementation plan](legislation_bridge_implementation_plan.md) — approved design for the additive production foundation, exact certified grain, duplicate-source handling, audits, downstream compatibility and deployment sequence.
- [Certified Bill-section bridge implementation](legislation_bridge_implementation.md) — deployed production foundation, live batch and counts, validation/deployment run IDs, permanent audits, production semantics, guardrails and the current living next-step plan.
- [Instagram post candidates](instagram_post_candidates.md) — researched and ranked cross-pipeline shortlist of ten defensible Instagram post ideas spanning issue-labelled speeches, speech structure, divisions, voting, legislation, Leaders' Questions and Parliamentary Questions, with exact evidence, denominators, rejected ideas, unsafe-but-promising areas, and the living editorial next-step plan.
- [Irish political data sources](irish_political_data_sources.md) — completed research-only investigation of Irish polling and other political datasets, including access, licensing, pricing, methodology, automation feasibility, rejected sources, rankings, and top future ingestion candidates.

Future investigations should get their own topic documents here, for example voting behaviour, speech structure, issue analysis, constituency representation and cross-metric profiles.
