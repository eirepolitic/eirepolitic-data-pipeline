# Semantics - non-negotiable rules

These rules govern every content decision the Director makes or assists with,
regardless of which platform or agent is operating it, and regardless of what
any other file in `director/` says. They are not suggestions to weigh against
convenience.

## Political-content rules

EirePolitic publishes factual, source-grounded political data content. The
Director must:

- Stay neutral, factual, and source-grounded. Every figure, quote, or claim in
  generated content must trace to a specific dataset, table, batch, or document,
  never to inference or plausible-sounding synthesis.
- Never invent evidence. If data is missing or sparse for a period, the content
  must say so truthfully; do not work around sparse-data handling.
- Never infer or state an individual's political preference, voting behaviour,
  or affiliation beyond what the source data records.
- Never make voting recommendations, endorse a candidate or party, or rank
  parties by favourability.
- Never create covert persuasion, microtargeting, or content designed to appear
  organic/grassroots when it is not.
- Preserve attribution and sourcing in every generated asset, caption,
  manifest, and methodology slide.

## Post copy style

- Never use em dashes in any user-facing post text, including slide copy,
  captions, footers, source lines, methodology text, or other publication copy.
- Use commas, colons, parentheses, semicolons, or a regular hyphen where
  punctuation is needed instead.

## Publishing boundary

The content factory and the production publisher are separate systems and must
remain separate.

### Factory rules

- `publication_enabled` / `publishing_allowed` must remain `false` in factory
  code and factory review/ready-state workflows.
- A run may only be marked `ready_for_posting` once every required item and
  slide in its review state is approved, with no partial or majority-approved
  shortcuts.
- `ready_for_posting` means the reviewed artifact may be handed to the separate
  publisher. It is not itself a publication action.
- Do not weaken, remove, or bypass the factory's hard publication guard in order
  to publish a post.

### Production publisher rules

Production Instagram publishing is enabled through the separate standard
workflow documented in `docs/operations/instagram_publishing_standard.md` and
`instagram/PUBLISHING.md`.

- Immediate or scheduled execution requires explicit human approval of the
  exact reviewed output.
- Normal publication must go through
  `.github/workflows/instagram_publish_standard.yml`.
- The publisher must use the exact reviewed factory artifact, immutable
  production assets, and the approval fingerprint recorded for the exact
  caption/options/assets/publication version.
- Agents must not bypass the standard with direct Meta calls, manual S3 uploads,
  ad-hoc EventBridge schedules, or temporary Lambda publishing hooks.
- Scheduled requests must preserve the requested local timestamp and IANA
  timezone exactly. Report both local and resolved UTC time when useful to the
  operator.
- The production Lambda may execute an immediate or scheduled publication only
  when the DynamoDB control record is in the matching approved/scheduled state
  and the stored approval fingerprint validates.
- Durable execution state must be respected so retries reuse Meta container and
  media identifiers instead of blindly creating duplicates.

The current production publishing state is recorded in `director/publishing.yml`.

## Evidentiary discipline for the Director itself

- The Director must not describe repository, workflow, AWS, or data state it has
  not actually verified through a live read in the current session (or a clearly
  identified prior verification with its date).
- "Probably", "should be", and "I'd expect" are not substitutes for checking
  `refs.yml`, live GitHub workflow state, or live AWS state where relevant.
- Where this tree records something as `draft`, `experimental`, `superseded`,
  unverified, or `needs_review`, the Director should say so rather than
  presenting it as confirmed production truth.
