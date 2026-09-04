# Step 6 — Direct Meta Publishing Architecture

Status: **complete**

Research date: 2026-09-03

Scope: design and assess the direct Meta/Instagram publishing option using the findings from Steps 1–5 and the current repository structure. This step does **not** select the final scheduler/storage implementation; that comparison is reserved for Step 9.

No publishing code, Meta credentials, live account connection, scheduler, Lambda, database, or production infrastructure was created.

---

## Short conclusion

A direct Meta integration is technically viable and fits Eirepolitic's desired control model very well.

The preferred direct-Meta shape is:

```text
High Director / Overlord
        ↓
Publication intent + explicit human approval
        ↓
Eirepolitic-owned publication record/ledger
        ↓
Eirepolitic-owned scheduler
        ↓
Deterministic publisher worker
        ↓
Meta Instagram API with Facebook Login
        ↓
Instagram
```

The existing post-generation/review pipeline remains upstream of this system.

The direct option gives Eirepolitic the strongest control over:

- exact caption/media/tags;
- approvals;
- cancellation/rescheduling;
- idempotency;
- failure recovery;
- audit history;
- future multi-platform abstraction.

Its main disadvantage is that Eirepolitic must own Meta authentication/token maintenance and API failure handling.

This option should remain a leading candidate, but it should **not** be selected finally until Steps 7–9 compare third-party, hybrid and scheduler alternatives.

---

## 1. Proposed responsibility boundaries

### A. Existing generation pipeline

Existing repository components continue to:

- generate Instagram assets;
- generate explicit caption/alt-text outputs;
- record review state;
- block non-ready content;
- optionally place review assets in S3.

They should **not** gain direct Instagram API credentials or become the scheduler.

### B. High Director / Overlord

High Director controls intent conversationally.

Responsibilities:

- identify the requested project/period/post;
- retrieve approved asset/caption information;
- discuss edits with the human;
- resolve missing publication decisions;
- present a precise final publication summary;
- obtain explicit human approval;
- create/update/cancel deterministic publication instructions;
- query publication history/status.

High Director should not:

- hold Meta access tokens;
- call `sleep()` until publication time;
- act as a cron scheduler;
- regenerate an approved caption at execution time;
- silently alter approved tags/media/timing;
- publish without an approved publication version.

### C. Publication control layer

This is the deterministic boundary between conversation and execution.

It should own records such as:

- publication request;
- approval record;
- schedule record;
- execution attempt;
- published-media result.

Detailed schemas are deferred to Step 10.

### D. Scheduler

The scheduler's responsibility is simply:

```text
At approved execution time → trigger publication_id
```

The scheduler should not carry the complete caption/media configuration as its authoritative payload.

The detailed AWS/GitHub/Power Automate comparison is Step 9.

### E. Publisher worker

The deterministic worker performs the mechanical Meta API work.

Conceptual flow:

```text
receive publication_id
      ↓
load approved publication version
      ↓
verify approval is still valid
      ↓
verify assets/checksums/readiness
      ↓
acquire idempotency/execution lock
      ↓
retrieve Meta credential from secret store
      ↓
create temporary media retrieval URL(s)
      ↓
create Meta media container(s)
      ↓
wait/check container processing as required
      ↓
publish container
      ↓
record Instagram Media ID/result
      ↓
perform configured post-publish actions (e.g. first comment)
      ↓
reconcile/store final state
```

No conversational/LLM decisions occur inside this worker.

### F. Meta Instagram API

Meta performs only the actual Instagram platform operations:

- container creation;
- media processing;
- publication;
- comments/tagging/collaborator operations where supported;
- returning media/status identifiers.

### G. Publication ledger

The ledger records the truth of what Eirepolitic intended, approved, attempted and actually published.

It is distinct from GitHub Actions logs and Meta itself.

---

## 2. Preferred Meta route for this option

Based on Steps 4–5, the current direct option should be designed around:

```text
existing Instagram account
   ↓ future explicit approval
switch same account to Professional
   ↓
link appropriate Facebook Page
   ↓
Instagram API with Facebook Login
```

Reason:

Eirepolitic explicitly wants conversational control over media tags and collaborators. Meta explicitly says the Instagram Login route cannot access tagging, while the current Page-linked API surface exposes `user_tags`, `collaborators`, `location_id`, `alt_text` and comment actions.

This remains an architecture finding only; no account conversion/Page linkage should occur until later approval.

---

## 3. Existing repo components that fit the direct option

### Reuse conceptually

From Step 1:

- `process/instagram_render_campaign.py`
- `process/instagram_build_copy_pack.py`
- `process/instagram_build_publish_queue.py`
- S3 preview integration
- existing review-status fields
- existing `publish_ready` gate

These already establish the correct principle:

```text
generate → review → explicitly mark ready → separate publication step
```

### Do not reuse blindly

The current queue CSV should not automatically become the production publication ledger/scheduler.

The current S3 `latest`/preview paths should not automatically become canonical production media locations.

The current GitHub AWS secret pattern should not automatically be copied into the publication runtime.

Those topics are handled in later steps.

---

## 4. Repo infrastructure finding

Targeted repository search found no existing publishing-oriented:

- AWS Lambda code;
- DynamoDB integration;
- EventBridge Scheduler implementation.

Therefore a direct-Meta runtime would be **new infrastructure**, not something that already exists hidden elsewhere in the repo.

This is important: later architecture selection must justify any new AWS components rather than claiming they are already part of the project.

The repo does already interact with AWS/S3 through the Instagram preview workflow, so AWS is an existing platform dependency even though serverless publication infrastructure is not.

---

## 5. Direct publication lifecycle

Recommended conceptual lifecycle:

```text
1. Generation produces candidate asset package
2. Content review approves immutable asset version
3. High Director builds publication request
4. Human edits/discusses until satisfied
5. High Director presents exact final summary
6. Human explicitly approves exact publication version
7. System records approval fingerprint
8. Scheduler records execution time
9. Scheduler triggers publication ID at the approved time
10. Worker validates current approval/state/assets
11. Worker publishes through Meta
12. Worker reconciles media ID/permalink/status
13. Optional post-publication actions run
14. Ledger records final result
```

This preserves the requested split:

```text
LLM = intent
code = execution
```

---

## 6. What the scheduler should receive

Avoid copying all publication data into the scheduler configuration.

Preferred trigger payload:

```json
{
  "publication_id": "pub_..."
}
```

At execution time, the worker loads the exact approved immutable version from the publication store.

Benefits:

- cancellation/rescheduling is simpler;
- secrets never enter scheduler payloads;
- caption/media duplication is avoided;
- the ledger remains authoritative;
- approval fingerprint can be rechecked immediately before execution.

If a publication changes after approval, it should become a new version and require reapproval before its scheduler record is active.

---

## 7. Meta container timing

Do not create Meta containers when the human schedules a post several days in advance.

Meta states unpublished containers expire after 24 hours.

Therefore:

```text
Tuesday: user schedules Friday publication
        ↓
store Eirepolitic publication/schedule only
        ↓
Friday near execution time
        ↓
create Meta container(s)
        ↓
publish
```

not:

```text
Tuesday: create Meta container
        ↓
wait until Friday
        ↓
container may have expired
```

This further supports keeping scheduling outside Meta's container lifecycle.

Primary Meta source:

- https://www.postman.com/meta/instagram/documentation/6yqw8pt/instagram-api?entity=request-23987686-ab559ffb-8e2c-4b0a-b43a-5737b6d2f672

---

## 8. Direct image/carousel execution

For Eirepolitic's existing static post type, the first direct implementation should target image/carousel publishing.

Conceptual carousel flow:

```text
approved ordered assets
   ↓
validate count/order/files/checksums
   ↓
create child container for slide 1
create child container for slide 2
...
   ↓
confirm all required children are valid/ready
   ↓
create CAROUSEL parent with ordered children
   ↓
publish parent
   ↓
store returned Instagram Media ID
```

The child order must come from the approved publication request, not filesystem sorting.

Reels/Stories should be separate later capabilities because they add more processing/validation complexity.

---

## 9. Asset delivery to Meta

Meta retrieves media by URL.

The direct architecture therefore needs a runtime path from approved canonical assets to a Meta-retrievable HTTPS URL.

The likely pattern, to be evaluated in Step 12, is:

```text
private approved S3 object
       ↓
short-lived execution-time retrieval URL
       ↓
Meta downloads media
```

The canonical asset remains private/immutable; the URL is merely a delivery mechanism.

Do not use temporary GitHub Actions artifact URLs as the publication source.

---

## 10. Authentication/secrets boundary

The worker, not High Director, retrieves the active Meta credential.

Conceptually:

```text
High Director
   ↓
account_ref = eirepolitic

Publisher worker
   ↓
secure secret lookup
   ↓
Meta Page access token
```

High Director may be allowed to know non-secret health metadata such as:

```text
auth_status: valid
auth_expires_at: ...
```

but never token values.

Production logs must redact/omit authorization headers and token-bearing URLs.

Exact secrets/storage choices are Step 15.

---

## 11. Idempotency requirement

Direct Meta publishing has an important distributed-systems risk:

```text
/media_publish succeeds at Meta
        ↓
network response is lost
        ↓
our worker sees timeout
```

If the worker simply retries by creating/publishing a new post, Eirepolitic could post a duplicate.

Therefore idempotency is not optional.

At minimum the architecture needs:

- unique publication identity/version;
- atomic execution lock/state transition;
- storage of every created Meta container ID;
- reconciliation after uncertain outcomes;
- separate idempotency for post-publication actions such as first comments.

AWS explicitly recommends idempotent Lambda/application code because retries/duplicate delivery can occur in serverless/event systems.

AWS references:

- https://docs.aws.amazon.com/lambda/latest/dg/best-practices.html
- https://docs.aws.amazon.com/us_en/lambda/latest/dg/concepts-application-design.html

The detailed mechanism is deferred to Step 16.

---

## 12. Failure categories in the direct option

The direct architecture must classify failures rather than retry everything equally.

### Retryable/transient examples

- network timeout before a known-safe operation;
- Meta 5xx;
- temporary Meta rate limiting;
- media-processing delay;
- temporary asset retrieval failure where the canonical asset is healthy.

### Non-retryable/operator-action examples

- revoked/invalid token;
- wrong permissions;
- account no longer Professional;
- invalid tag/collaborator/location;
- rejected media specification;
- approval fingerprint mismatch;
- asset checksum mismatch;
- incomplete carousel.

### Uncertain-outcome examples

- publish request timed out after Meta may already have accepted it.

Uncertain outcomes require reconciliation rather than blind retry.

The detailed state/retry design is Step 16.

---

## 13. Cancellation/rescheduling under direct Meta

Because our system owns the future schedule and Meta is called only near execution time:

### Cancel Friday's post

```text
scheduled publication
   ↓
cancel our scheduler record
   ↓
ledger → cancelled
```

No Meta content needs to exist yet.

### Move tomorrow's post to 8pm

```text
update/recreate our schedule
   ↓
store new approved schedule/version as required
```

### Edit caption before publishing

```text
edit publication request
   ↓
approval fingerprint changes
   ↓
require fresh publication approval
   ↓
reschedule/activate approved version
```

This is one of the strongest advantages of owning the scheduling/control layer ourselves.

---

## 14. Published-post operations

Based on Step 5:

- current Meta API surface supports deleting an Instagram media object;
- current update surface reviewed does not expose caption editing;
- comments can be created/managed where permissions allow;
- collaborator state can be queried through current collaboration edges.

For safety, destructive operations such as deleting a published Instagram post should be considered a **separate future High Director action with its own explicit confirmation**, not part of initial publishing automation.

No deletion capability should be implemented in the first publishing phase unless separately approved.

---

## 15. Direct option advantages

### Strong conversational-control fit

High Director remains entirely focused on human intent and policy.

The direct integration does not force the conversation to conform to a third-party scheduler's internal data model.

### Exact approval snapshot

Eirepolitic controls precisely which fields invalidate approval.

### Strong auditability

Our ledger can record:

- proposed values;
- approved values;
- approver;
- exact timestamps/timezone;
- attempts;
- Meta IDs/errors;
- final resulting post.

### Cancellation/rescheduling control

Future jobs remain inside Eirepolitic infrastructure until execution.

### Lowest application-level vendor lock-in

Meta itself is unavoidable for Instagram delivery, but no scheduling vendor sits between Eirepolitic and Meta.

### Full control of idempotency/reconciliation

Critical safety behaviour is ours rather than being hidden behind another vendor.

### Good future multi-platform boundary

A future architecture can expose a platform adapter interface such as:

```text
PublisherAdapter
   ├── InstagramMetaPublisher
   ├── FacebookPublisher
   ├── BlueskyPublisher
   └── ...
```

without changing how High Director creates publication intent.

---

## 16. Direct option disadvantages

### Meta authentication lifecycle becomes our responsibility

We must monitor/handle:

- token expiry/revocation;
- Page/account linkage;
- permissions;
- app access/review changes.

### API maintenance

Meta versions/features evolve. Eirepolitic must maintain its adapter.

### Failure recovery must be engineered carefully

The application must understand uncertain publish outcomes, media-processing states and duplicate prevention.

### New runtime infrastructure is required

The repo does not currently contain Lambda/EventBridge/DynamoDB-style publishing infrastructure.

Even if the final solution is small, it introduces a new operational component.

### Meta setup is more involved than authorizing a social-scheduling SaaS

The initial app/Page/token configuration needs controlled setup and testing.

---

## 17. Cost profile

### Meta

No separate Instagram Content Publishing API licence fee was identified in the current Meta documentation.

### Eirepolitic infrastructure

The likely runtime is very low-volume, so serverless execution/storage costs should be small.

However, exact cost should not be finalized in Step 6 because the scheduler/database/secrets architecture has not yet been selected.

Step 9 and Step 19 will estimate cost once the infrastructure choice is justified.

---

## 18. Complexity / risk score

| Dimension | Direct Meta assessment |
|---|---|
| Initial implementation complexity | **Medium** |
| Ongoing operational complexity | **Medium** |
| Scheduling control | **Excellent** |
| Approval/audit control | **Excellent** |
| Tagging/metadata control | **Excellent on Page-linked route, subject to canary validation** |
| Idempotency control | **Excellent if we implement it correctly** |
| Vendor lock-in | **Low beyond Meta itself** |
| Authentication burden | **Medium** |
| Multi-platform extensibility | **Good** |
| Conversational High Director fit | **Excellent** |
| Risk of accidental publish if correctly gated | **Low by design** |
| Risk if retries/state are implemented poorly | **High consequence — must be engineered carefully** |

---

## 19. Direct option safety requirements

A direct implementation should not be considered production-ready unless all of these are true:

```text
✓ exact approved asset version
✓ exact approved caption
✓ exact approved tags/collaborators/location/comment configuration
✓ exact target account
✓ explicit timezone-aware schedule
✓ approval fingerprint matches current request
✓ publication is enabled
✓ execution lock acquired
✓ assets revalidated
✓ current auth valid
✓ Meta container IDs recorded before subsequent operations
✓ uncertain API outcomes reconciled
✓ retries are idempotent
✓ final Meta result written to ledger
```

This is more important than minimizing the number of AWS components.

---

## 20. Direct option verdict at this stage

**Viable and strong candidate.**

It currently best matches the preferred design principle:

```text
High Director / Overlord
→ conversational decisions and human approval

Publication records
→ deterministic approved intent

Scheduler / publisher
→ deterministic timed execution

Meta Instagram API
→ actual delivery

Publication ledger
→ authoritative history/result
```

But do **not** choose it finally yet.

Remaining comparison work:

- Step 7: third-party schedulers/APIs;
- Step 8: hybrid model;
- Step 9: scheduler/runtime infrastructure.

Those steps may reveal that outsourcing some delivery mechanics creates enough operational benefit to outweigh the additional dependency.

---

## Sources

### Meta

- Current Instagram API documentation: https://www.postman.com/meta/instagram/documentation/6yqw8pt/instagram-api
- Content Publishing guide: https://www.postman.com/meta/instagram/documentation/6yqw8pt/instagram-api?entity=request-23987686-ab559ffb-8e2c-4b0a-b43a-5737b6d2f672
- Instagram API with Facebook Login: https://www.postman.com/meta/instagram/folder/u4g5a2a/instagram-api-with-facebook-login
- Current generated Meta Business SDK `IGUser`: https://github.com/facebook/facebook-php-business-sdk/blob/main/src/FacebookAds/Object/IGUser.php
- Current generated Meta Business SDK `IGMedia`: https://github.com/facebook/facebook-php-business-sdk/blob/main/src/FacebookAds/Object/IGMedia.php

### AWS design principles referenced, without selecting the final scheduler yet

- AWS Lambda best practices — idempotency: https://docs.aws.amazon.com/lambda/latest/dg/best-practices.html
- AWS Lambda application design — duplicate-event/idempotency guidance: https://docs.aws.amazon.com/us_en/lambda/latest/dg/concepts-application-design.html
- EventBridge Scheduler overview, for later Step 9 comparison: https://aws.amazon.com/eventbridge/scheduler/

---

## Confidence / unresolved items

**High confidence:**

- direct Meta architecture is viable;
- Page-linked route is the better current direct route for Eirepolitic's tagging requirements;
- Meta containers should be created near execution time rather than when a future publication is approved;
- direct integration requires explicit idempotency/reconciliation;
- current repo has generation/review/S3 components but no existing Lambda/DynamoDB/EventBridge publishing implementation.

**Intentionally unresolved:**

- final scheduler selection;
- final ledger/database selection;
- exact secrets implementation;
- exact retry/idempotency algorithm;
- infrastructure-as-code choice;
- final direct-versus-third-party recommendation.

**Next research step:**

Step 7 will evaluate third-party social scheduling platforms and whether their APIs actually simplify this problem enough to justify the added vendor dependency.
