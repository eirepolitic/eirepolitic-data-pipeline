# Step 16 — Idempotency, Failure Recovery and Retries

Status: **complete**

Research date: 2026-09-04

Scope: define how the publication runtime prevents duplicate Instagram posts, handles duplicate scheduler invocations, recovers from Lambda/network failures, classifies Meta errors, reconciles uncertain `/media_publish` outcomes, and retries post-publication actions safely.

No retry code, DynamoDB table, Lambda, scheduler, Meta call, account connection, or live publication was created.

---

## Short conclusion

The publication system must be **idempotent at the Eirepolitic publication level and at each external side-effect operation**.

The critical rule is:

> Never create a new Instagram publication merely because a prior publish request timed out.

Recommended recovery model:

```text
EventBridge may invoke more than once
        ↓
atomic publication execution lock
        ↓
load immutable approved publication version
        ↓
resume recorded operation state
        ↓
create/reuse Meta containers
        ↓
/media_publish
        ↓ timeout / uncertain response
DO NOT create another parent/post
        ↓
query existing parent container status
        ↓
PUBLISHED → treat as already published; reconcile result
FINISHED → retry same container only under proven safe policy
IN_PROGRESS → wait/reconcile
ERROR/EXPIRED → fail/recovery decision
```

Meta explicitly documents that when `/media_publish` does not return the published media ID, the existing container's status can be queried and may report `PUBLISHED`. That is the primary duplicate-prevention mechanism at the dangerous publication boundary.

EventBridge Scheduler itself uses at-least-once delivery, so duplicate Lambda invocation is expected architecture, not an exceptional bug.

---

# 1. Idempotency goal

For one approved publication version:

```text
one intended publication
```

must never become:

```text
two visible Instagram posts
```

because of:

- duplicate scheduler delivery;
- Lambda retry;
- network timeout;
- process crash;
- Meta 5xx;
- operator retry;
- reconciliation job;
- a second High Director command targeting the same execution.

The system should allow the same execution command to be received many times while producing no additional external side effect after the intended side effect has occurred.

---

# 2. AWS delivery semantics

AWS EventBridge Scheduler currently uses **at-least-once** delivery to targets.

AWS Lambda guidance likewise says functions must tolerate duplicate events and recommends idempotent application logic.

Sources:

- EventBridge Scheduler overview: https://docs.aws.amazon.com/scheduler/latest/UserGuide/what-is-scheduler.html
- Lambda retry behaviour: https://docs.aws.amazon.com/lambda/latest/dg/invocation-retries.html
- Lambda application design / idempotency: https://docs.aws.amazon.com/lambda/latest/dg/concepts-application-design.html

### Consequence

The publisher must assume this can happen:

```text
EventBridge → Lambda A
EventBridge retry/duplicate → Lambda B
```

Both may receive:

```json
{"publication_id":"pub_123","expected_version":3}
```

Only one may acquire the right to execute side effects.

---

# 3. Publication-level idempotency identity

Recommended root idempotency identity:

```text
publication_id + publication_version
```

Example:

```text
pub_01JABC...:v3
```

This identifies one immutable approved publication intent.

Do not use:

- scheduler invocation ID;
- Lambda request ID;
- Meta container ID;
- timestamp alone;

as the publication idempotency identity.

Those values identify attempts, not intent.

---

# 4. Atomic execution claim

Before any Meta side effect, the worker must perform an atomic state transition such as:

```text
scheduled → publishing
```

only if all required conditions still match:

```text
publication version == expected version
approval fingerprint == expected fingerprint
schedule active
state == scheduled
no published result exists
```

A second worker attempting the same claim must fail the conditional update and then inspect the existing state rather than continue publishing.

---

# 5. DynamoDB conditional writes if DynamoDB is selected

AWS DynamoDB supports conditional writes and documents them as useful for idempotent state changes.

Example concept:

```text
Update publication
SET state = publishing,
    execution_owner = attempt_123
WHERE state = scheduled
  AND version = 3
  AND published_media does not exist
```

AWS documents that conditional writes can safely prevent an operation from applying when the expected prior state is no longer true.

Sources:

- DynamoDB conditional writes: https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/WorkingWithItems.html
- Condition expressions: https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/Expressions.ConditionExpressions.html

### Important note

Step 10 intentionally deferred the physical ledger/database choice.

If the final architecture selects a relational database instead, use the equivalent transaction/conditional-update/unique-constraint mechanism.

The requirement is **atomic compare-and-set semantics**, not DynamoDB specifically.

---

# 6. Execution lease

A simple permanent lock is insufficient because Lambda can crash after acquiring it.

Recommended execution claim includes:

```yaml
execution_owner: attempt_01J...
execution_started_at: ...
execution_lease_expires_at: ...
```

A second invocation encountering an active lease exits/reconciles.

A recovery invocation encountering an expired lease may take over **only after inspecting existing recorded Meta operation state**.

It must not assume nothing happened merely because the first Lambda stopped reporting.

---

# 7. Do not use DynamoDB TTL as the lock timer

If DynamoDB is used, TTL is useful for eventual cleanup of disposable idempotency records, but it should not be the real-time lock-release mechanism.

Lock logic should compare the explicit `execution_lease_expires_at` timestamp in a conditional transaction/update.

Reason:

DynamoDB TTL deletion is asynchronous/eventual and is not designed to remove items at an exact second.

The idempotency decision must therefore depend on stored timestamps/state, not whether the TTL background process has deleted a record.

---

# 8. Operation-level idempotency

One publication consists of several operations.

Example carousel:

```text
validate asset package
create child container 1
create child container 2
...
create parent container
publish parent container
create first comment (optional)
reconcile media result
```

Each operation should have a stable logical key.

Examples:

```text
pub_123:v3:create_child:slide_01
pub_123:v3:create_child:slide_02
pub_123:v3:create_parent
pub_123:v3:publish_parent
pub_123:v3:first_comment
```

The execution ledger records whether each operation:

- never started;
- started;
- succeeded;
- failed safely;
- has uncertain outcome;
- was reconciled.

---

# 9. Persist external IDs immediately

When Meta returns a child or parent container ID, store it **before** moving to the next external operation.

Example:

```text
POST /media
   ↓
Meta returns container_ABC
   ↓
write container_ABC to ExecutionAttempt
   ↓
only then create next child
```

On retry/resume:

```text
container ID already recorded
   ↓
reuse/query it
```

not:

```text
create another container because this is a new Lambda invocation
```

---

# 10. Child-container creation failure

Creating a media container is not itself a visible Instagram publication.

If child creation succeeds but the response is lost before its ID is recorded, the system may be unable to address that orphaned container again.

Because no `/media_publish` side effect has occurred, a bounded retry that creates a replacement child is **far safer than retrying a publication**, although it can leave an unused container that later expires.

Meta documents that unpublished containers expire after 24 hours.

Source:

- Meta Content Publishing: https://www.postman.com/meta/instagram/documentation/6yqw8pt/instagram-api?entity=request-23987686-ab559ffb-8e2c-4b0a-b43a-5737b6d2f672

### Policy

For container creation:

- if a container ID is recorded: reuse/query it;
- if no ID was ever obtained and the request outcome is uncertain: a bounded replacement-container retry is acceptable before publication, provided no parent has been published;
- record the uncertainty/orphan possibility for diagnostics.

---

# 11. Partial carousel creation

Example failure:

```text
child 1 succeeded
child 2 succeeded
child 3 timeout
child 4 not attempted
```

Recovery:

```text
load ExecutionAttempt
reuse child 1 ID
reuse child 2 ID
reconcile/recreate child 3 as necessary
create remaining children
only create parent after all approved children are ready
```

Do not recreate successful children on every retry.

The final parent children list must exactly match the approved asset order.

---

# 12. Parent-container creation

The carousel parent is still not visible until `/media_publish` succeeds.

If its container ID is recorded, always reuse/query that parent.

If parent creation outcome is uncertain and no ID was received, a replacement parent can be created before publication using the already recorded child containers, subject to current Meta container rules.

Again, this is acceptable because the visible-publication boundary has not yet been crossed.

---

# 13. The dangerous boundary: `/media_publish`

`POST /{ig_user_id}/media_publish` converts the prepared container into a visible Instagram media object.

Meta normally returns:

```json
{"id":"<instagram_media_id>"}
```

Source:

- Meta Content Publishing: https://www.postman.com/meta/instagram/documentation/6yqw8pt/instagram-api?entity=request-23987686-ab559ffb-8e2c-4b0a-b43a-5737b6d2f672

The dangerous failure case is:

```text
send /media_publish
      ↓
Meta publishes successfully
      ↓
network response lost
      ↓
worker sees timeout
```

A naive retry that creates another container/post can cause a duplicate public post.

---

# 14. Rule after an uncertain `/media_publish`

Immediately mark the publish operation:

```text
outcome = uncertain
```

and **stop all create-new-publication behaviour**.

Then query the **same existing parent container**.

Meta explicitly documents this troubleshooting procedure when `/media_publish` does not return the published media ID.

Container status can be:

```text
EXPIRED
ERROR
FINISHED
IN_PROGRESS
PUBLISHED
```

Meta recommends polling once per minute for no more than five minutes in the normal troubleshooting flow.

Source:

- Meta Content Publishing troubleshooting: https://www.postman.com/meta/instagram/documentation/6yqw8pt/instagram-api?entity=request-23987686-ab559ffb-8e2c-4b0a-b43a-5737b6d2f672

---

# 15. Reconciliation decisions by Meta container status

## `PUBLISHED`

Meaning from Meta: the container's media object has been published.

Required response:

```text
DO NOT call /media_publish again
DO NOT create a replacement parent
DO NOT create a new publication
```

Set a durable publication guard such as:

```text
platform_publication_confirmed = true
```

Then reconcile the Instagram Media ID/permalink separately if they were lost with the original HTTP response.

The absence of the media ID in our ledger is **not** permission to republish.

---

## `IN_PROGRESS`

Wait/requery within the bounded Meta-documented polling window.

Do not publish again while the original operation may still be completing.

---

## `FINISHED`

Meta defines this as ready to be published, not published.

However, immediately after a network timeout the safest automatic policy is still to reconcile for a bounded interval rather than instantly issuing another publish request.

A future canary must verify the behaviour of retrying `/media_publish` against the **same container** after an uncertain request.

Recommended initial production policy:

- poll/reconcile first;
- if the container remains conclusively `FINISHED` and canary testing has proven same-container retry behaviour safe, retry the same parent container under a tightly bounded policy;
- otherwise transition to `needs_attention` rather than risking a duplicate.

Never create a fresh parent/publication merely because status is `FINISHED` after an uncertain publish.

---

## `ERROR`

The container failed.

Record the provider error/status and classify the failure.

A new execution/container may be permitted only if there is no evidence the publication succeeded and the retry policy/approval remains valid.

---

## `EXPIRED`

The unpublished container has expired.

If no evidence of publication exists and the publication remains approved/current, a controlled new-container execution may be created.

Do not reuse expired IDs.

---

# 16. Missing media ID after `PUBLISHED`

Meta's current troubleshooting guide proves the container can tell us that it is `PUBLISHED`, but the documented status response reviewed in this research does not itself guarantee recovery of the lost published media ID.

Therefore represent this accurately:

```text
platform publication confirmed
published media ID not yet reconciled
```

Example state:

```text
published_result_pending_reconciliation
```

This state blocks all republishing.

A future canary should test the best deterministic way to recover the resulting media ID, for example through the current account/media read API and available fields.

Do not match a post solely by caption text if several posts could share the same caption.

---

# 17. Publication guard

Once **any authoritative evidence** indicates the publication succeeded, set a durable guard that makes creation of another public post impossible without an explicit new publication/republication record.

Examples of success evidence:

```text
/media_publish returned Instagram Media ID
OR
parent container status == PUBLISHED
```

The guard should be checked before every operation capable of creating a visible post.

---

# 18. Retry classification

Do not use one generic `retryable=true` rule for every error.

Classify errors as:

```text
transient_safe_retry
transient_reconcile_first
permanent_input
permanent_auth
rate_limited
provider_unknown
success_response_lost
operator_action_required
```

---

# 19. Safe transient retries

Examples where normal bounded retry can be appropriate:

- read-only Meta GET timeout;
- S3 HEAD/metadata transient failure;
- database transient error before a side effect;
- child-container creation when no visible publication is possible and no usable ID was obtained;
- known Meta 5xx on a non-publication operation;
- rate limiting when the operation is known safe to retry.

Use exponential backoff with jitter and bounded attempts.

AWS recommends explicit retry strategies, conditional retries and exponential backoff rather than retrying every error blindly.

Source:

- AWS Lambda retry guidance: https://docs.aws.amazon.com/lambda/latest/dg/invocation-retries.html

---

# 20. Reconcile-first failures

These must not be retried immediately:

```text
/media_publish timeout
/media_publish connection reset after request send
Lambda process killed while awaiting publish response
first-comment POST timeout
Buffer createPost timeout in hybrid architecture
```

Reason:

The external side effect may already have occurred.

Required sequence:

```text
uncertain side effect
    ↓
reconcile provider state
    ↓
only retry if absence of prior effect can be established under proven rules
```

---

# 21. Permanent input failures

Examples:

- invalid media dimensions/format;
- unsupported tag/collaborator/location;
- content/package mismatch;
- missing approved media;
- caption/platform constraint violation;
- invalid carousel count/order.

These should not be retried automatically.

Set:

```text
needs_attention / failed
```

with a sanitized explanation.

The publication request may need a new approved version if content changes are required.

---

# 22. Authentication failures

Examples:

- invalid/revoked token;
- missing permission;
- Page/Instagram account disconnected;
- account changed back to Personal.

Do not hammer Meta with repeated publish retries.

Set:

```text
auth_blocked
```

and notify the operator.

After authentication is repaired, the same approved publication may resume only after reconciling that no prior publish succeeded.

---

# 23. Rate limiting

When Meta signals rate limiting:

- preserve the existing publication/container state;
- respect provider retry timing where documented;
- use exponential backoff;
- do not create new duplicate container trees unnecessarily;
- check the content publishing quota endpoint where useful;
- fail/alert if retry would move unacceptably beyond the intended publication window.

Step 3 identified `/content_publishing_limit` as the live source for account publishing usage/quota.

---

# 24. Publication lateness policy

A retry should not continue forever merely because an approved publication once had a schedule.

Recommended future policy field:

```yaml
execution_policy:
  latest_acceptable_start: ...
```

or a configured grace period such as:

```text
scheduled time + N minutes
```

The exact value is a product/editorial decision for Step 19/implementation.

If the failure persists beyond the allowed publication window:

```text
needs_attention
```

rather than publishing unexpectedly hours later.

EventBridge itself supports event age/retry limits up to 24 hours, but Eirepolitic should choose a much more deliberate publication-specific policy rather than using the AWS maximum automatically.

Source:

- EventBridge Scheduler retry configuration: https://docs.aws.amazon.com/scheduler/latest/UserGuide/getting-started.html

---

# 25. Lambda timeout / process crash

A Lambda may stop after any line of code.

Therefore recovery must not depend on an in-memory sequence like:

```text
I know step 3 completed because my variable says so
```

Persistent operation state must be written between external side effects.

On the next invocation:

```text
load publication + ExecutionAttempt
inspect recorded external IDs/status
reconcile uncertain operation
resume from durable state
```

This is why the execution ledger is required even at very low publication volume.

---

# 26. First-comment idempotency

A first comment is a separate public side effect after the Instagram media ID exists.

Use its own operation key:

```text
pub_123:v3:first_comment
```

Normal sequence:

```text
published media ID known
   ↓
create first comment
   ↓
store returned comment ID/result
```

If the comment request times out after sending:

```text
outcome = uncertain
```

Do not blindly post the same comment again.

The reconciler should query the media's comments where current permissions/API allow and establish whether the expected comment already exists before retrying.

A duplicate comment is less severe than a duplicate Instagram post, but it is still a visible error and should be prevented.

---

# 27. Secondary-action failure must not undo publication success

Example:

```text
Instagram post succeeds
first comment fails
```

Correct result:

```text
media = published
first_comment = failed/needs_attention
```

Incorrect result:

```text
publication = failed
→ republish entire post
```

A failure in a secondary action must never trigger recreation of the already-published media.

---

# 28. Collaborator/location/tag result handling

Fields supplied during container creation are part of the primary approved publication.

If Meta rejects one of those parameters before publication, fail before `/media_publish` rather than silently dropping the unsupported field.

Do not recover by saying:

```text
"tag failed, publish without it"
```

because that would produce content different from the approved fingerprint.

A changed configuration requires a new publication version/approval.

---

# 29. Buffer hybrid idempotency

If Buffer is selected instead of direct Meta, the same principle applies.

Dangerous case:

```text
POST create Buffer scheduled post
   ↓
Buffer accepts it
   ↓
response lost
```

Eirepolitic must reconcile Buffer scheduled posts/provider state before issuing another create.

Store:

```text
provider = buffer
provider_post_id = ...
```

as soon as known.

A Buffer API timeout is not permission to create a second scheduled post.

The Eirepolitic `publication_id` remains the idempotency root.

---

# 30. Operator retry command

High Director may eventually support:

```text
"Retry tonight's failed post."
```

That must not map directly to:

```text
call publisher again from scratch
```

It should invoke a deterministic recovery decision:

1. load publication and latest attempt;
2. verify approval still valid;
3. reconcile any uncertain provider operation;
4. determine whether visible publication already occurred;
5. reuse valid container/provider state where possible;
6. create a new `ExecutionAttempt` only if permitted;
7. never create a new publication identity unless the user explicitly intends a repost/republication.

---

# 31. Retry versus republish

These are different concepts.

## Retry

```text
Same publication_id
Same publication_version
Same approved content
Recovery from execution failure
```

## Republish / repost

```text
New intentional public post
New publication_id (or explicit child/republication relationship)
Fresh human confirmation/approval
```

A user saying:

```text
"Post that again tomorrow"
```

is a republish, not a technical retry.

---

# 32. Recommended execution states

Useful `ExecutionAttempt` states:

```text
started
validation_complete
creating_containers
containers_ready
publish_request_sending
publish_outcome_uncertain
reconciling_publish
platform_publication_confirmed
published_result_pending_reconciliation
post_publish_actions
succeeded
failed
needs_attention
```

These are operational states and need not all be exposed verbatim in normal High Director conversations.

---

# 33. Recommended operation record

Conceptual example:

```yaml
operation_id: pub_123:v3:publish_parent
operation_type: media_publish
attempt_number: 1
container_id: "..."
state: uncertain
started_at: ...
last_checked_at: ...
provider_status: IN_PROGRESS
provider_result_id: null
retry_policy:
  class: reconcile_first
next_action: query_container_status
```

This is far safer than a single publication row containing only:

```text
retry_count = 2
```

because it records **what** was retried and whether the external side effect may already exist.

---

# 34. Recommended retry limits

Exact numbers should be tuned in implementation/canary testing, but the principles are:

- few bounded retries;
- exponential backoff + jitter;
- separate limits by operation type;
- no automatic retries for permanent validation/auth errors;
- reconcile-before-retry for public side effects;
- stop when the latest acceptable publication time is exceeded.

Do not configure EventBridge's maximum 185 retries merely because the service permits it.

---

# 35. DLQ semantics

The Step 9 SQS DLQ is for **scheduler target-delivery failure** after EventBridge retries are exhausted.

A DLQ message means:

```text
EventBridge could not successfully deliver/invoke the target under its policy
```

It does not necessarily mean:

```text
Instagram publication failed
```

The DLQ handler/operator must load the Eirepolitic publication state before deciding what happened.

Sources:

- EventBridge Scheduler DLQ: https://docs.aws.amazon.com/scheduler/latest/UserGuide/configuring-schedule-dlq.html
- Scheduler management/retries: https://docs.aws.amazon.com/scheduler/latest/UserGuide/managing-schedule.html

---

# 36. Exactly-once claim

Do not describe the overall system as technically "exactly once" end-to-end.

The honest goal is:

```text
at-least-once delivery infrastructure
+
application idempotency
+
provider reconciliation
→ effectively one intended visible publication
```

AWS explicitly uses at-least-once semantics in several relevant event paths, and Meta's HTTP side effects can have uncertain outcomes.

The application must enforce the business invariant.

---

# 37. Failure matrix

| Failure | Automatic retry? | Required first action | Duplicate risk |
|---|---|---|---|
| Duplicate EventBridge invocation | No extra side effect | Conditional execution claim | High if lock absent |
| DB read before side effect fails | Yes, bounded | Retry read | None |
| S3 validation transient failure | Yes, bounded | Retry validation | None |
| Child container response lost | Bounded replacement possible | Check recorded ID/state | Low visible-post risk |
| Parent container response lost before publish | Bounded recovery possible | Check recorded ID/state | Low visible-post risk |
| `/media_publish` timeout | **No blind retry** | Query same container | **Critical** |
| Container reports `PUBLISHED` | **Never republish** | Set publication guard/reconcile ID | **Critical prevented** |
| Container remains `IN_PROGRESS` | Poll boundedly | Query status | Critical if republished |
| Permanent media/tag validation error | No | Needs attention/new approval if changed | None if blocked |
| Auth revoked | No | `auth_blocked` | None if blocked |
| Rate limit | Bounded/backoff | Preserve state; retry appropriately | Depends on operation |
| First-comment timeout | **No blind retry** | Reconcile comments | Visible duplicate comment |
| Secondary comment failure after media success | Do not republish media | Retry/reconcile comment only | Critical if modeled wrongly |

---

# 38. Step 16 verdict

Recommended safety model:

```text
one immutable approved publication version
        ↓
atomic execution claim
        ↓
durable operation ledger
        ↓
record/reuse external IDs
        ↓
reconcile uncertain side effects
        ↓
retry only proven-safe operations
        ↓
permanent publication guard once Meta success is known
```

Key rules:

1. EventBridge duplicate delivery is expected; only one worker may execute side effects.
2. Use atomic conditional state transitions/transactions for execution claims.
3. Persist every Meta container ID immediately.
4. Resume successful container work rather than recreating it.
5. Container creation and publication have different risk levels.
6. Never create a replacement public post after an uncertain `/media_publish` without first reconciling the original parent container.
7. Meta `PUBLISHED` is sufficient to block republishing even when the media ID response was lost.
8. Treat missing published media ID as reconciliation work, not publication failure.
9. First-comment creation has its own idempotency/reconciliation path.
10. Secondary-action failure must never cause recreation of successfully published media.
11. Permanent validation/auth failures are not automatically retried.
12. Bound retries by operation and an acceptable publication-lateness window.
13. A technical retry keeps the same publication identity; an intentional repost creates a new publication identity and approval.
14. Do not claim true provider-level exactly-once delivery; enforce the business invariant through idempotency + reconciliation.

---

## Sources

### Meta

- Current Instagram Content Publishing and troubleshooting/status behaviour: https://www.postman.com/meta/instagram/documentation/6yqw8pt/instagram-api?entity=request-23987686-ab559ffb-8e2c-4b0a-b43a-5737b6d2f672
- Current container status/publish examples: https://www.postman.com/meta/instagram/documentation/6yqw8pt/instagram-api?entity=request-23987686-00dab679-d4c0-4957-bed0-0bcda896be09

### AWS

- EventBridge Scheduler at-least-once delivery: https://docs.aws.amazon.com/scheduler/latest/UserGuide/what-is-scheduler.html
- EventBridge Scheduler retries/DLQ: https://docs.aws.amazon.com/scheduler/latest/UserGuide/managing-schedule.html
- EventBridge Scheduler DLQ: https://docs.aws.amazon.com/scheduler/latest/UserGuide/configuring-schedule-dlq.html
- Lambda retry behaviour: https://docs.aws.amazon.com/lambda/latest/dg/invocation-retries.html
- Lambda application idempotency guidance: https://docs.aws.amazon.com/lambda/latest/dg/concepts-application-design.html
- DynamoDB conditional writes: https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/WorkingWithItems.html
- DynamoDB condition expressions: https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/Expressions.ConditionExpressions.html

---

## Confidence / unresolved items

**High confidence:**

- EventBridge Scheduler is at-least-once and the worker must be idempotent;
- Meta exposes container status including `PUBLISHED` specifically for troubleshooting when `/media_publish` does not return the media ID;
- a `PUBLISHED` container must block any attempt to recreate the visible publication;
- durable operation state/external IDs are necessary for crash recovery;
- conditional database writes are appropriate for atomic execution claims;
- secondary actions need separate idempotency.

**Must be proven during future Meta canary testing:**

- exact behaviour of retrying `/media_publish` against the same container after an uncertain response that later remains `FINISHED`;
- best deterministic method to recover the Instagram Media ID when container status is `PUBLISHED` but the original `/media_publish` response was lost;
- exact reconciliation fields available through the selected Graph API version;
- first-comment reconciliation behaviour and returned comment IDs;
- practical operation-specific retry counts/backoff timings.

**Still to design later:**

- exact physical ledger/lock implementation;
- exact publication lateness/grace-period policy;
- alarms/operator UX for `publishing_unknown` / `needs_attention`;
- final Step 19 choice of direct Meta versus Buffer.

**Next research step:**

Step 17 will define monitoring, auditability, operational alerts and the queries High Director needs to answer about scheduled, published, failed and uncertain posts.
