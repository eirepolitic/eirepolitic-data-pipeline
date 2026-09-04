# Step 8 — Hybrid Architecture: Eirepolitic Intent/Ledger + Buffer Delivery

Status: **complete**

Research date: 2026-09-03

Scope: assess a hybrid architecture where Eirepolitic/High Director owns publication intent, approval and audit history, while Buffer performs final scheduled delivery to Instagram.

No Buffer account, Meta account, credentials, scheduler or live publication was created.

---

## Short conclusion

The hybrid model is **viable and materially simpler than direct Meta in some areas**, but it does not eliminate the core safety/control infrastructure Eirepolitic needs.

Buffer can take responsibility for:

- holding the future scheduled job;
- triggering publication at the requested time;
- maintaining the Instagram account connection/token relationship;
- translating its post object into the final Instagram delivery call;
- exposing scheduled/sending/sent/error state;
- allowing scheduled posts to be edited/deleted/rescheduled through its API.

Eirepolitic would still need to own:

- approved asset identity/version;
- exact final caption;
- tags/metadata intent;
- human approval record;
- schedule intent/timezone;
- idempotency between our publication and Buffer's post ID;
- reconciliation of Buffer state back into our ledger;
- failure explanations;
- historical publication records;
- multi-platform conversational abstraction.

Therefore Buffer removes a meaningful amount of **execution plumbing**, but it does **not** replace the publication-control layer.

The largest hybrid-specific downside is media hosting: Buffer's current API requires media URLs to remain publicly reachable until the scheduled publication time and explicitly advises against expiring S3 presigned URLs. Direct Meta would allow Eirepolitic to keep canonical assets private and create retrieval URLs only at execution time.

The hybrid option should remain in the final comparison, but current evidence still favors direct Meta if maximum Instagram capability/control is more important than reducing initial implementation effort.

---

# 1. Hybrid architecture

Recommended hybrid shape:

```text
High Director / Overlord
        ↓
Eirepolitic PublicationRequest
        ↓
Eirepolitic PublicationApproval
        ↓
Eirepolitic PublicationLedger
        ↓
BufferPublisher adapter
        ↓
Buffer scheduled Post
        ↓
Instagram
```

Buffer is a **delivery provider**, not Eirepolitic's source of truth.

---

# 2. What High Director should own

The conversational model should behave the same whether the eventual provider is direct Meta or Buffer.

High Director should resolve and confirm:

- project/post/period;
- exact media order;
- exact final caption;
- hashtags;
- caption mentions;
- media tags;
- collaborators where supported;
- location;
- alt text;
- first-comment intent;
- Instagram account;
- publication date/time/timezone;
- publication approval.

It then creates an approved Eirepolitic publication version.

Only after approval does an adapter translate that version into Buffer-specific API fields.

This prevents Buffer's current limitations from contaminating the conversational data model.

---

# 3. What Buffer would remove from Eirepolitic

## A. Publication clock

Buffer supports exact scheduled times using `customScheduled` + `dueAt`.

Eirepolitic would not need an EventBridge-style timed trigger for Instagram delivery itself.

Buffer would hold the scheduled job and send it at the appropriate time.

Source:

- https://developers.buffer.com/examples/create-scheduled-post.html

## B. Instagram account-token lifecycle

Eirepolitic would authorize its Instagram account through Buffer rather than maintaining Meta Page access tokens directly in our publishing runtime.

Our secret would be a Buffer API credential rather than a Meta publication credential.

This removes some Meta-specific token lifecycle/permission handling from our code.

It does not eliminate account-connection failures; those simply appear as Buffer/channel problems rather than direct Meta authentication problems.

## C. Some Meta API adaptation

Buffer owns the translation from its schema to the relevant social-network API.

Eirepolitic would not need to maintain all of Meta's media-container calls directly.

## D. Multi-platform delivery foundation

Buffer already exposes multiple connected channels/networks through one API.

If Eirepolitic later wants the same approved content sent to several platforms, a Buffer-based adapter could potentially reduce the number of direct platform integrations needed.

This is an advantage, though it also increases Buffer lock-in.

---

# 4. What Buffer does NOT remove

## A. Human approval

Buffer approval workflows do not replace Eirepolitic's required High Director approval.

We still need to know exactly what the human approved and when.

## B. Publication ledger

High Director must be able to answer:

```text
What's scheduled this week?
What went out yesterday?
Why did tonight's post fail?
```

Those queries should come from Eirepolitic's ledger, not depend entirely on Buffer's current API/history retention or product UI.

## C. Idempotency

A failure during `createPost` can create the same distributed-systems question as direct Meta:

```text
Buffer accepted post
   ↓
network response lost
   ↓
Eirepolitic sees timeout
```

Our system must not blindly call `createPost` again and create two scheduled Buffer jobs.

We therefore still need:

- publication identity/version;
- external provider ID;
- create-attempt records;
- reconciliation before retry where outcome is uncertain.

## D. State reconciliation

Buffer exposes post states including:

- draft;
- needs_approval;
- scheduled;
- sending;
- sent;
- error.

Eirepolitic must map those provider states into its own publication states and periodically reconcile them.

Sources:

- https://developers.buffer.com/types/PostStatus.html
- https://developers.buffer.com/examples/get-scheduled-posts.html

## E. Failure explanations

High Director still needs stored, sanitized provider errors and state transitions to explain failures.

## F. Asset validation

Before a Buffer post is created, Eirepolitic still needs to verify:

- approved asset package;
- correct slide count/order;
- dimensions/formats;
- hashes;
- QA/human review state.

Buffer cannot know whether `slide_03.jpg` is the correct approved August slide.

---

# 5. Scheduling lifecycle under the hybrid model

Recommended lifecycle:

```text
1. High Director builds PublicationRequest
2. Human approves exact version
3. Eirepolitic records PublicationApproval
4. Buffer adapter validates provider capability
5. Buffer adapter creates scheduled Buffer Post
6. Eirepolitic stores Buffer post ID + dueAt
7. Eirepolitic state → scheduled
8. Buffer handles timed delivery
9. Reconciler observes sending/sent/error
10. Eirepolitic ledger stores final result
```

Important difference from direct Meta:

**The external vendor post exists days before publication.**

With direct Meta, only our schedule exists until execution time.

---

# 6. Cancellation and rescheduling

Buffer's API supports editing/deleting posts and updating `dueAt`/schedule-related fields.

Sources:

- https://developers.buffer.com/types/EditPostInput.html
- https://developers.buffer.com/guides/posts-and-scheduling.html

Therefore:

## Cancel Friday's post

```text
High Director resolves Eirepolitic publication
       ↓
confirmation if required
       ↓
delete/cancel corresponding Buffer post
       ↓
verify provider state
       ↓
Eirepolitic ledger → cancelled
```

## Move tomorrow's post to 8pm

```text
update approved schedule intent
       ↓
update Buffer dueAt
       ↓
verify returned provider state/time
       ↓
ledger records change
```

### Important rule

Do not update only our ledger and assume Buffer changed.

A schedule change is complete only after the Buffer update is confirmed.

---

# 7. Editing approved content before publication

Buffer supports editing scheduled post text/assets/metadata.

But Eirepolitic approval rules should remain stricter than Buffer's raw API.

Example:

```text
approved publication
   ↓
caption changed conversationally
   ↓
approval invalidated
   ↓
pending_publication_approval
```

Only after new approval should Eirepolitic update the Buffer job.

This avoids the dangerous state:

```text
Buffer post changed
but old human approval remains marked valid
```

---

# 8. Media-hosting problem

This is the largest technical disadvantage of the Buffer hybrid model.

Buffer currently has no ordinary media-upload endpoint for scheduled posts. The application supplies asset URLs.

Buffer explicitly states that the URL must:

- be public without authentication;
- be direct;
- use HTTPS;
- remain reachable **until the scheduled post publishes**;
- not be an expiring signed URL.

It specifically warns against S3 presigned URLs for scheduled posts because they can expire before publication.

Source:

- https://developers.buffer.com/guides/hosting-media.html

## Consequence

A Buffer architecture cannot simply use:

```text
private S3 object
   ↓
15-minute presigned URL
   ↓
Buffer job scheduled for Friday
```

That URL could expire days before Buffer fetches it.

### Viable approaches

If Buffer were chosen, Eirepolitic would need a stable media-delivery layer, for example:

- an S3/CloudFront public read path limited to immutable publication assets;
- another public object/CDN path;
- Buffer-recommended stable public media hosting.

Detailed asset-storage design remains Step 12.

### Comparison with direct Meta

Direct Meta is cleaner for private assets:

```text
private S3 canonical asset
   ↓
our worker fires at publication time
   ↓
generate short-lived retrieval URL
   ↓
Meta retrieves immediately
```

This is an architectural advantage for direct Meta.

---

# 9. Capability translation

A hybrid adapter should validate Eirepolitic intent against Buffer's current capabilities.

Conceptually:

```text
PublicationRequest
       ↓
Buffer capability validator
       ↓
Buffer CreatePostInput
```

## Known good mappings

| Eirepolitic intent | Buffer API |
|---|---|
| Exact caption | `text` |
| Ordered media | `assets` |
| Scheduled instant | `dueAt` |
| Automatic publish | `schedulingType: automatic` |
| Image alt text | image metadata `altText` |
| Image user tags | image metadata `userTags` |
| Instagram location | Instagram `geolocation` metadata |
| Post/Reel/Story | Instagram `type` metadata |

## Known gap

Current Buffer Instagram input does not expose collaborator/co-author input.

Therefore a publication containing:

```yaml
collaborators:
  - username: example
```

must fail provider validation rather than silently drop the field.

High Director should be told clearly that the selected delivery provider cannot execute that approved configuration.

## Known problem

Buffer's API roadmap currently reports that Instagram `firstComment` may not persist when supplied via API.

Until resolved, the hybrid adapter should disable that capability rather than silently accepting it.

---

# 10. Provider capability matrix

The Eirepolitic data model should allow capabilities to be provider-specific.

Example concept:

```yaml
provider_capabilities:
  buffer:
    instagram:
      caption: true
      image_tags: true
      alt_text: true
      location: true
      collaborators: false
      first_comment: false  # disabled while provider bug is active
```

This prevents High Director from promising a feature that the selected provider cannot deliver.

The capability record should be maintained/configured deterministically, not inferred by the LLM at publication time.

---

# 11. Reconciliation design

Because no suitable Buffer post-status webhook was identified during Step 7, design the hybrid option assuming **polling/reconciliation** is required.

Buffer exposes retrieval/filtering of scheduled posts and post lifecycle status.

A small reconciler could periodically query outstanding Buffer jobs and update the Eirepolitic ledger.

Example:

```text
Eirepolitic publication = scheduled
Buffer post = sent
       ↓
ledger reconciliation
       ↓
Eirepolitic publication = published
```

or:

```text
Eirepolitic publication = scheduled
Buffer post = error
       ↓
ledger = failed / needs_attention
       ↓
store sanitized provider error
```

If Buffer introduces reliable webhooks later, they could reduce polling, but the ledger should still perform periodic reconciliation as a safety mechanism.

---

# 12. Source of truth

The single most important hybrid rule is:

**Buffer must not become the authoritative publication database.**

Use:

```text
Eirepolitic publication_id
        ↓
external_delivery
    provider: buffer
    provider_post_id: ...
```

not:

```text
publication identity = Buffer post ID
```

This preserves portability.

---

# 13. Failure model under Buffer

Failure categories include:

### Before Buffer job creation

- asset not ready;
- invalid Eirepolitic approval;
- provider capability mismatch;
- inaccessible media URL;
- Buffer API auth failure.

### After Buffer job creation, before publication

- Buffer job deleted externally;
- Buffer channel disconnected;
- media URL becomes unavailable;
- account permission changes;
- schedule edited externally;
- provider bug.

### During publication

- Buffer state becomes `sending` for longer than expected;
- Buffer returns `error`;
- Meta rejects the post through Buffer;
- provider outcome becomes uncertain.

### After publication

- Eirepolitic reconciliation delayed;
- provider says `sent` but external platform result metadata is incomplete;
- unsupported secondary action such as first comment fails.

Hybrid architecture therefore reduces Meta-specific execution work but adds **cross-system consistency** work.

---

# 14. Operational visibility

Buffer provides a useful secondary UI/calendar where a human can inspect scheduled posts.

That is a genuine operational advantage.

However, direct edits made in Buffer's UI introduce a governance question:

```text
What if somebody changes the caption/time in Buffer after High Director approval?
```

Recommended policy if hybrid is chosen:

- treat Eirepolitic as authoritative;
- reconciliation detects material differences;
- external modification invalidates the Eirepolitic approval/state or raises `needs_attention`;
- do not silently accept drift.

The final implementation could optionally restrict who has edit access to the Buffer workspace/channel.

---

# 15. Security comparison

## Direct Meta

Eirepolitic stores Meta credentials.

## Buffer hybrid

Eirepolitic stores Buffer API credentials; Buffer stores/manages the social connection.

This is simpler from our perspective but transfers trust to Buffer.

### Asset exposure

Buffer hybrid has the less attractive asset-security model because scheduled media URLs must remain public/reachable until publication.

Direct Meta allows publication-time URL generation from private storage.

---

# 16. Cost

At one Instagram channel, Buffer's current cost is small:

- Free may be usable for limited scheduled volume;
- Essentials is approximately $5/channel/month billed annually.

Therefore cost alone does not disqualify the hybrid option.

Eirepolitic would still incur minimal storage/ledger/reconciliation infrastructure costs.

Hybrid does not mean "zero AWS" because we still need an authoritative publication store and a way for High Director/runtime services to query/update it.

---

# 17. Complexity comparison with direct Meta

| Area | Direct Meta | Buffer hybrid |
|---|---|---|
| Meta token lifecycle | **Our responsibility** | **Buffer responsibility** |
| Timed publication trigger | **Our responsibility** | **Buffer responsibility** |
| Meta container flow | **Our responsibility** | **Buffer responsibility** |
| Publication ledger | Our responsibility | Our responsibility |
| Human approval | Our responsibility | Our responsibility |
| Idempotency | Our responsibility | Our responsibility |
| Reconciliation | Meta reconciliation | **Buffer + downstream reconciliation** |
| Stable media hosting before scheduled time | Not necessarily | **Required by Buffer** |
| Instagram collaborator support | Better direct API evidence | **Current Buffer API gap** |
| Vendor lock-in | Low beyond Meta | **Medium** |
| Extra operator calendar UI | No, unless we build one | **Yes** |
| Initial engineering effort | Higher | **Lower** |
| Maximum platform control | **Higher** | Lower |

---

# 18. Hybrid advantages

- Lower initial publishing implementation effort.
- No Eirepolitic-owned publication clock for Instagram.
- Less direct Meta token/permission handling.
- Mature social-management UI as an operational backup.
- Straightforward API editing/deleting/rescheduling of scheduled jobs.
- Low cost at current scale.
- Potentially accelerates later multi-platform delivery.
- High Director can still remain the conversational control layer if Eirepolitic retains its own ledger.

---

# 19. Hybrid disadvantages

- Buffer becomes another production dependency between Eirepolitic and Meta.
- Current public API is relatively new/evolving.
- Current collaborator API gap.
- Current Instagram first-comment API issue.
- Stable publicly accessible media URLs required until publication.
- Eirepolitic and Buffer state can drift.
- Need polling/reconciliation unless suitable webhooks become available.
- Buffer/API subscription terms and feature availability can change.
- Errors may be abstracted/normalized rather than exposing the exact Meta operation.
- Future migration requires replacing the provider adapter and reconciling outstanding Buffer jobs.

---

# 20. Operational risk

### Hybrid risk rating: medium-low for initial delivery, medium for long-term dependency

Buffer reduces risk arising from incorrectly implementing Meta container/token mechanics ourselves.

But it introduces risk from:

- external service availability;
- external API evolution;
- state drift;
- feature gaps;
- public media hosting requirements.

The first-comment issue in Buffer's current API roadmap is a concrete reminder that a third-party scheduler does not eliminate integration defects; it changes where they can occur.

---

# 21. Conversational-control fit

### Rating: very good

The hybrid model can support the desired conversation well **provided High Director talks to Eirepolitic's publication model, not directly to Buffer's model**.

Example:

```text
User:
"Move tomorrow's Party Speech Breakdown to 8pm."

High Director:
resolve Eirepolitic publication
→ update/reapprove schedule if required
→ Buffer adapter updates provider job
→ verify returned dueAt
→ ledger records change
```

This is nearly as clean conversationally as direct Meta.

The important limitation is capability mismatch. If the user asks for collaborators and the current Buffer API cannot represent them, High Director must say so rather than dropping the instruction.

---

# 22. Hybrid option verdict

**Viable and worth carrying into the final architecture comparison.**

Buffer genuinely removes:

- scheduled trigger infrastructure for delivery;
- direct Meta publishing/container implementation;
- much of the social-account credential lifecycle.

But Eirepolitic still needs almost all of the **control-plane** work:

```text
publication request
approval
ledger
idempotency
provider reconciliation
monitoring
asset readiness
High Director commands
```

Therefore the architectural decision is not:

```text
Build everything ourselves
vs
Buffer builds everything
```

It is:

```text
Direct Meta:
Eirepolitic owns control plane + execution plane

Buffer hybrid:
Eirepolitic owns control plane
Buffer owns much of the final delivery execution plane
```

Current preference after Step 8:

- **Direct Meta** if maximum feature control, private asset delivery and low vendor lock-in are priorities.
- **Buffer hybrid** if minimizing initial Meta/scheduler implementation effort is the priority and collaborator/first-comment gaps are acceptable.

Do not make the final selection yet. Step 9 must still determine how difficult/reliable the direct scheduling infrastructure actually is; if EventBridge-style direct scheduling is extremely small and inexpensive, Buffer's execution-simplicity advantage becomes less compelling.

---

## Sources

Primary Buffer sources used for this step:

- Buffer API introduction: https://developers.buffer.com/guides/introduction.html
- Posts and Scheduling: https://developers.buffer.com/guides/posts-and-scheduling.html
- Create Scheduled Post: https://developers.buffer.com/examples/create-scheduled-post.html
- Get Scheduled Posts: https://developers.buffer.com/examples/get-scheduled-posts.html
- EditPostInput: https://developers.buffer.com/types/EditPostInput.html
- PostStatus: https://developers.buffer.com/types/PostStatus.html
- Hosting Media: https://developers.buffer.com/guides/hosting-media.html
- Content Items: https://developers.buffer.com/guides/content-items.html
- Instagram metadata schema / feature findings from Step 7: https://developers.buffer.com/types/InstagramPostMetadataInput.html
- Buffer API roadmap / current first-comment issue: https://developers.buffer.com/roadmap.html

---

## Confidence / unresolved items

**High confidence:**

- Buffer can own exact scheduled delivery and supports editing/deleting scheduled post objects;
- Eirepolitic must still retain its own ledger/approval/idempotency model;
- Buffer media URLs must remain publicly reachable until publication;
- current Buffer state model supports scheduled/sending/sent/error reconciliation;
- current Buffer Instagram input lacks collaborator configuration;
- current first-comment API problem makes that capability unsafe to rely on today.

**Unresolved until later steps:**

- final cost/complexity difference after evaluating direct AWS scheduling;
- exact ledger/database implementation;
- exact stable-media-hosting design if Buffer is selected;
- whether Buffer gains collaborator/webhook/first-comment improvements before implementation.

**Next research step:**

Step 9 will compare scheduler/runtime options for direct Meta, especially EventBridge Scheduler, Lambda, Step Functions, SQS, GitHub Actions cron and Power Automate, and determine whether owning the direct scheduling layer is actually burdensome enough to justify Buffer.
