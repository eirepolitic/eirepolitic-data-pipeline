# Step 11 — High Director Conversational Control and Approval Model

Status: **complete**

Research date: 2026-09-04

Scope: define how High Director / Overlord should translate conversation into deterministic publication intent, what must be confirmed before scheduling/publishing, which changes invalidate approval, and how state transitions should protect against accidental publication.

No publishing code, scheduler, account connection, credential, or live publication was created.

---

## Short conclusion

High Director should be allowed to discuss and edit a proposed publication freely while it is a draft.

A publication becomes schedulable only after the human explicitly confirms an exact publication snapshot containing at least:

- target account;
- project/post/period;
- final ordered media;
- exact final caption;
- hashtags/caption mentions;
- media tags;
- collaborators;
- location;
- alt text;
- first-comment configuration;
- publication date;
- publication time;
- timezone;
- resolved UTC execution instant;
- notification/approval behaviour.

The approval must bind to a fingerprint of that exact publication version.

Any change to a material content/account/metadata field after approval invalidates the approval and returns the publication to `pending_publication_approval`.

Schedule-only changes can use a lighter re-confirmation path, but the new date/time/timezone must still be explicitly confirmed before the scheduler is updated.

---

# 1. Design principle

The conversational model should optimize for:

```text
ask only genuinely missing decisions
```

but never at the expense of publication safety.

The safe pattern is:

```text
conversation
   ↓
mutable draft
   ↓
complete deterministic proposal
   ↓
explicit human confirmation
   ↓
immutable approved publication version
   ↓
scheduler/publisher
```

High Director is responsible for intent clarification and human-readable confirmation.

Deterministic code is responsible for validating and enforcing the resulting state.

---

# 2. Recommended human-facing states

Use these primary publication states:

```text
draft
pending_content_review
pending_publication_approval
approved
scheduled
publishing
published
failed
cancelled
```

Add operational exception states where useful:

```text
needs_attention
auth_blocked
publishing_unknown
```

These should remain exceptional rather than becoming normal workflow states.

---

# 3. State meanings

## `draft`

High Director and the human are still discussing or editing the publication.

Allowed:

- change caption;
- choose assets;
- add/remove tags;
- change account;
- change schedule;
- load caption templates;
- decide whether there is a first comment;
- abandon the draft.

Not allowed:

- create an active scheduler job;
- publish.

---

## `pending_content_review`

The selected `AssetPackage` has not yet passed the generation/content review gate.

Typical causes:

- visual review not complete;
- `publish_ready` false;
- QA warnings unresolved;
- safety notes present;
- asset count mismatch.

High Director may discuss the post, but the system must not treat it as publication-ready.

---

## `pending_publication_approval`

The asset package is content-ready and the publication request is complete enough to show for final confirmation, but the human has not yet approved that exact version.

This is the normal state immediately before scheduling.

---

## `approved`

A human approved the exact publication fingerprint.

At this point the publication request version is immutable.

The system may create a schedule if schedule details are already approved and valid.

---

## `scheduled`

An approved publication has an active confirmed scheduler/provider job.

Important:

```text
ledger says scheduled
```

only after the scheduler/provider confirms creation/update.

---

## `publishing`

The deterministic execution service has acquired the publication lock and is actively performing provider operations.

High Director should not allow ordinary edits/rescheduling once publishing has begun.

---

## `published`

Publication success has been reconciled and the resulting media ID/result has been recorded.

---

## `failed`

A known terminal publication attempt failure occurred.

The stored execution record should explain why.

Retry/republication should follow explicit recovery rules rather than simply resetting this state.

---

## `cancelled`

The publication was intentionally cancelled before successful publication.

Cancellation should remain auditable.

Do not delete the publication record from history.

---

# 4. Content review versus publication approval

This distinction is essential.

Current repo fields such as:

```text
publish_ready=yes
review_status=approved
```

mean the generated content can be considered for publication.

They must not authorize:

```text
publish to Instagram account X at time Y with caption Z
```

A second, separate `PublicationApproval` is required.

Therefore:

```text
content review passed
        ≠
publication approved
```

Both gates must pass before scheduling.

---

# 5. High Director's conversational resolution flow

Example request:

```text
"Schedule the August Party Speech Breakdown for Instagram next Tuesday at 7:30pm."
```

High Director should perform these logical steps:

1. resolve `project_id = party_speech_breakdown`;
2. resolve `period = 2026-08`;
3. locate the latest eligible approved `AssetPackage`;
4. determine the intended Instagram `account_ref` from defaults/context;
5. load the normal caption template/prior approved structure if relevant;
6. resolve `next Tuesday 19:30` in `Europe/Dublin`;
7. identify genuinely unresolved choices;
8. ask only for those choices;
9. build the final `PublicationRequest`;
10. present the exact final confirmation snapshot;
11. obtain explicit approval;
12. write `PublicationApproval`;
13. only then create/update the scheduler.

---

# 6. What counts as genuinely missing

High Director should distinguish:

```text
unknown
```

from:

```text
known default
```

and from:

```text
explicitly none
```

Example:

```yaml
collaborators: null
```

means unresolved.

```yaml
collaborators: []
```

means the human/system policy has explicitly selected none.

Likewise:

```yaml
location: null
```

can mean unresolved if location is a required decision for this post type, or explicitly absent if the schema/policy records that decision separately.

The final implementation should make these states unambiguous so High Director does not repeatedly ask already-answered questions.

---

# 7. Required final confirmation

Before the publication becomes approved/schedulable, High Director must show a concise but complete final snapshot containing at least:

```text
Account
Project/post/period
Publication version
Ordered media
Exact caption
Hashtags
Caption mentions
Media tags and their media/slide
Collaborators
Location
Alt text
First comment / none
Date
Time
Timezone
Resolved UTC instant
Immediate vs scheduled
Additional approval/notification behaviour
```

The human must affirmatively approve that snapshot.

Do not treat vague earlier discussion such as:

```text
"that looks fine"
```

as final publication approval unless it is clearly responding to the final confirmation snapshot.

---

# 8. Recommended confirmation style

High Director should not dump the full internal manifest into chat unless requested.

Instead present a compact human-readable confirmation such as:

```text
Ready to schedule:
- Account: @eirepolitic
- Post: Party Speech Breakdown — August 2026
- Media: 8 approved slides, asset package asset_...
- Caption: [exact final caption shown]
- Media tags: @example on slide 3
- Collaborators: none
- Location: none
- First comment: none
- Schedule: Tuesday 8 Sep 2026, 19:30 Europe/Dublin (18:30 UTC)

Confirm scheduling this exact version?
```

The internal deterministic service still validates every underlying field independently.

---

# 9. Approval-binding fields

These fields should always invalidate approval when changed:

```text
target platform
account_ref
project_id
period
asset_package_id
publication version
ordered media list
any asset hash
caption text
alt text
caption mentions
hashtags, if tracked as approved structured metadata
media tags
collaborators
location
first-comment text/configuration
post type / Reel/Story/carousel mode
```

These are material publication-content decisions.

Changing any of them means:

```text
approved/scheduled
      ↓
new publication version
      ↓
pending_publication_approval
```

If a scheduler job already exists, it must be disabled/cancelled or replaced only after the new version is approved.

---

# 10. Schedule changes

A schedule change is different from a content change.

Example:

```text
"Move tomorrow's post from 7:30pm to 8pm."
```

The content has not changed, but the human is changing a material delivery instruction.

Recommended policy:

- do **not** require the human to re-approve the entire caption/media snapshot from scratch;
- do require explicit confirmation of the new date/time/timezone and target account/post identity;
- create a new `PublicationSchedule` version/reference;
- preserve the same approved `PublicationRequest` version;
- update the scheduler only after the schedule change is confirmed.

This gives safety without unnecessary friction.

---

# 11. Schedule-change confirmation

Example:

```text
Move Party Speech Breakdown — August 2026 on @eirepolitic
from Tuesday 8 Sep 19:30 Europe/Dublin
 to Tuesday 8 Sep 20:00 Europe/Dublin?
```

An affirmative response authorizes the schedule update.

If the target account or publication identity also changes, treat it as a new publication approval, not a simple reschedule.

---

# 12. Cancellation confirmation

Cancellation should require confirmation whenever there is any ambiguity about which scheduled publication is affected.

Example:

```text
"Cancel Friday's post."
```

If exactly one scheduled Eirepolitic publication exists on Friday, High Director can resolve it and confirm:

```text
Cancel Party Speech Breakdown — August 2026
scheduled for Friday 11 Sep at 19:30 Europe/Dublin on @eirepolitic?
```

If more than one exists, High Director must identify the intended one before taking action.

After confirmation:

1. cancel/delete/disable scheduler job;
2. verify scheduler result;
3. mark publication `cancelled`;
4. retain audit history.

---

# 13. Immediate publishing

`Publish this now` must not bypass approval.

Recommended flow:

```text
complete PublicationRequest
   ↓
show exact final snapshot
   ↓
human approves immediate publication
   ↓
record approval
   ↓
invoke deterministic publisher immediately
```

The confirmation should explicitly say:

```text
Publication mode: immediate
```

and identify the target account.

Immediate publication is more dangerous than future scheduling because there is less recovery time, so the approval requirement should be at least as strict.

---

# 14. Wrong-month / stale-content protection

High Director must resolve human-friendly references to immutable IDs before approval.

Example:

```text
"August Party Speech Breakdown"
```

should resolve to:

```text
project_id = party_speech_breakdown
period = 2026-08
asset_package_id = asset_...
```

Final confirmation must display the period clearly.

The deterministic validator should reject a request when:

- selected asset package period differs from publication period;
- requested project differs from asset package project;
- asset package is superseded by policy and not explicitly selected;
- required media count differs from approved package.

This directly protects against publishing the wrong month's post.

---

# 15. Draft/unapproved asset protection

A publication must be blocked if any required content-readiness field fails.

Examples:

```text
human_visual_review_status != approved
publication_ready != true
qa_status != passed
safety_notes non-empty
asset hashes missing
expected media count mismatch
```

High Director may explain the block, but it cannot override it conversationally unless there is a separate deterministic authorized override mechanism defined later.

Do not accept:

```text
"publish it anyway"
```

as an override of a failed content-readiness gate by default.

---

# 16. Incomplete carousel protection

The approved `AssetPackage` records expected ordered media.

At final approval High Director shows the expected media count.

At execution the publisher verifies:

```text
actual media count == approved expected count
```

and hashes/order match the approved request.

If one slide is missing:

```text
scheduled
   ↓
validation failure
   ↓
needs_attention / failed-before-publish
```

The system must not silently publish a seven-slide subset of an eight-slide approved carousel.

---

# 17. Wrong-account protection

Accounts should use stable internal references:

```text
account_ref: eirepolitic
```

with deterministic provider mappings.

Final confirmation always shows the human-readable account name/handle.

The publisher checks that:

- requested `account_ref` exists;
- it is enabled for publishing;
- connected provider account ID matches configuration;
- approval fingerprint includes the target account.

Changing account always invalidates approval.

---

# 18. Old-caption protection

The exact caption belongs to the immutable approved publication version.

The scheduled worker loads that exact version.

It must not load:

- "latest caption";
- newest template;
- latest copy-pack file by mutable path;
- fresh LLM output.

This prevents an approved August post from picking up a September or subsequently edited caption.

---

# 19. External/provider drift protection

This matters especially in a Buffer hybrid architecture but can also matter for manually altered Meta/Business Suite state.

If an external scheduled job is editable outside Eirepolitic and its content/time differs from the approved Eirepolitic record:

```text
provider drift detected
       ↓
needs_attention
```

Do not silently update the Eirepolitic approval to match external changes.

Eirepolitic remains authoritative for approved intent.

---

# 20. Notification behaviour

The user requested notification behaviour to be part of final confirmation where applicable.

Recommended model:

```yaml
notifications:
  on_scheduled: false
  on_published: false
  on_failed: true
  channel: operator_default
```

For the first implementation, avoid overcomplicated preferences.

At minimum the publication confirmation should indicate whether the operator will be notified on failure and whether any extra manual notification action is requested.

Detailed monitoring channels are Step 17.

---

# 21. Human approver identity

The approval record should distinguish:

```text
approved_by.type = human
```

from:

```text
created_by = high_director
```

High Director may propose and record the approval action, but it must not mark itself as the human approver.

Where the environment exposes an authenticated user identity, store a stable actor reference rather than free-text names where possible.

Do not store sensitive authentication material as approver identity.

---

# 22. High Director permissions model

High Director should be allowed to:

- create/edit draft `PublicationRequest` records;
- create a `PublicationApproval` only after receiving clear human confirmation;
- create/update/cancel `PublicationSchedule` records after required confirmation;
- query records and execution history;
- explain failures from stored execution state.

High Director should not be allowed to:

- mutate an already-approved publication version in place;
- set content-review gates to approved without the proper review workflow;
- edit `ExecutionAttempt` provider IDs/status as though they were conversational fields;
- fabricate a `PublishedMedia` success record;
- bypass deterministic validation;
- expose or request live Meta tokens in conversation.

---

# 23. Recommended state transition rules

```text
draft
 ├─ content not ready ───────────────→ pending_content_review
 └─ complete + content ready ───────→ pending_publication_approval

pending_content_review
 └─ content becomes ready ──────────→ pending_publication_approval

pending_publication_approval
 └─ explicit approval ──────────────→ approved

approved
 ├─ scheduler confirmed ────────────→ scheduled
 ├─ material content change ────────→ pending_publication_approval (new version)
 └─ cancellation ───────────────────→ cancelled

scheduled
 ├─ confirmed schedule change ──────→ scheduled (new schedule version)
 ├─ material content change ────────→ pending_publication_approval (and disable old schedule)
 ├─ cancellation confirmed ─────────→ cancelled
 └─ worker acquires execution ──────→ publishing

publishing
 ├─ reconciled success ─────────────→ published
 ├─ known terminal failure ─────────→ failed
 └─ outcome uncertain ──────────────→ publishing_unknown / needs_attention

failed
 └─ recovery decision creates a controlled retry/republication path

published
 └─ immutable publication result; destructive post-publish actions are separate commands
```

---

# 24. Retry versus new approval

A retry of the same publication attempt should **not** require a new human approval if:

- approved publication content is unchanged;
- target account is unchanged;
- approved schedule intent is still applicable;
- retry is part of the deterministic recovery policy;
- there is no evidence the post already succeeded.

An intentional revised post does require a new approval.

This distinction is developed fully in Step 16.

---

# 25. Post-publication destructive actions

Commands such as:

```text
"Delete yesterday's post."
```

must be treated as separate destructive operations with their own explicit confirmation.

They are not simply another publication-state transition.

Initial publishing implementation should not automatically include delete capability even though Meta's current API surface exposes deletion.

---

# 26. Example conversation — normal schedule

Conceptual flow:

```text
User:
Schedule the August Party Speech Breakdown next Tuesday at 7:30pm.

High Director:
[resolves asset package, account default, timezone, caption template]

High Director:
The August package is approved. The normal caption is loaded.
One unresolved item: should @example be tagged on slide 3 or only mentioned in the caption?

User:
Mention only, no media tag.

High Director:
[builds final request]

High Director:
Ready to schedule:
Account: @eirepolitic
Post: Party Speech Breakdown — August 2026
Media: 8 approved slides
Caption: <exact final caption>
Caption mention: @example
Media tags: none
Collaborators: none
Location: none
First comment: none
When: Tue 8 Sep 2026, 19:30 Europe/Dublin
Confirm this exact publication?

User:
Yes.

System:
record approval → create scheduler → verify → state scheduled
```

The exact UI wording can vary, but the control semantics should not.

---

# 27. Example conversation — reschedule

```text
User:
Move tomorrow's post to 8pm.

High Director:
Move Party Speech Breakdown — August 2026 on @eirepolitic
from 19:30 to 20:00 Europe/Dublin tomorrow?

User:
Yes.

System:
update schedule → verify provider/scheduler → persist new schedule
```

No full content reapproval is required because the approved content is unchanged.

---

# 28. Example conversation — caption edit after scheduling

```text
User:
Change tomorrow's caption to mention the unusually low Dáil activity.

High Director:
[edits draft of new publication version]

System:
old approved version remains immutable
old schedule is disabled before replacement becomes active
new version → pending_publication_approval

High Director:
[shows full exact final publication snapshot again]

User:
Yes.

System:
new approval → new/updated schedule
```

This prevents an unapproved caption from inheriting an old approval.

---

# 29. Step 11 verdict

The recommended approval model is:

```text
free conversational drafting
        ↓
deterministic complete PublicationRequest
        ↓
explicit final human confirmation
        ↓
immutable approval fingerprint
        ↓
verified scheduler state
        ↓
deterministic execution
```

Key policy decisions:

1. content review and publication approval are separate gates;
2. material content/account/metadata changes always require fresh approval;
3. schedule-only changes require explicit schedule confirmation but not full content reapproval;
4. immediate publishing requires the same full publication approval as scheduled publishing;
5. High Director never changes runtime/provider state by conversation alone;
6. scheduler/provider changes must be verified before the ledger is updated as complete;
7. external drift becomes `needs_attention` rather than silently changing approved intent;
8. failed readiness checks cannot be overridden by casual conversational commands;
9. cancelled publications remain auditable;
10. post-publication destructive operations require separate explicit confirmation.

This gives the conversational experience requested while making accidental wrong-account, wrong-period, wrong-caption and incomplete-carousel publication difficult.

---

## Repository/design dependencies

Builds on:

- `01_repo_review.md`
- `10_publication_data_model.md`

Existing repo review fields remain content-readiness inputs, not final publication authorization.

---

## Confidence / unresolved items

**High confidence:**

- approval should bind to an immutable publication version/fingerprint;
- target account/media/caption/tags must be included in final confirmation;
- content edits after approval must invalidate approval;
- schedule-only changes can safely use a narrower explicit confirmation path;
- High Director should not be able to fabricate execution success/state;
- cancellation must update the scheduler first and retain audit history.

**Still to design later:**

- exact canonical fingerprint serialization;
- exact authenticated human actor-reference implementation;
- operational timeouts/locks while publishing;
- failure-retry state transitions;
- monitoring/notification channel implementation.

**Next research step:**

Step 12 will define asset readiness and media hosting: immutable approved S3 assets, hashes, dimensions, media URL generation and Meta retrieval requirements.
