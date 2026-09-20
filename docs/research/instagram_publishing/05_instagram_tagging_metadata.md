# Step 5 — Instagram Tagging and Publication Metadata

Status: **complete**

Research date: 2026-09-03

Scope: determine current API support for caption mentions, media/user tags, carousel tags, collaborators, location, alt text, first comments and product tags.

No Meta app/account/token was created and nothing was published.

---

## Short conclusion

The Page-linked **Instagram API with Facebook Login** currently exposes the richer metadata surface Eirepolitic wants.

Current Meta-generated API/SDK surfaces expose:

- `user_tags` on media creation;
- `collaborators` on media creation;
- `location_id` on media creation;
- `alt_text` on media creation;
- creation of comments on published Instagram media;
- collaborator read/invite edges.

This materially strengthens the case for the Facebook Login/Page-linked route.

The simpler Instagram Login route remains attractive operationally, but Meta explicitly states that it **cannot access tagging**, so it is not a safe choice if conversational tagging is a firm requirement.

Product/shopping tags and branded-content/paid-partnership fields are currently inconsistent between Meta's general Content Publishing guide and Meta's generated Business SDK. They should be treated as **restricted/conditional and excluded from v1** until tested against the exact app/account/API version.

---

## 1. Caption `@mentions`

### Status: supported as caption text

The publishing API accepts a complete caption string. A caption may contain normal Instagram text such as `@username` and hashtags.

This is different from a media tag.

Recommended representation:

```yaml
caption:
  text: |
    Final approved caption containing @example if desired.

caption_entities:
  mentions:
    - example
```

The structured mention list is useful for validation/audit, but the **exact caption text is canonical**.

High Director should never reconstruct the mention placement at publication time.

### Important distinction

```text
@username in caption
```

is not equivalent to:

```text
user_tags on the image/media
```

The final publication manifest should keep them separate.

---

## 2. Media/user tagging

### Status: API supported on the richer Page-linked API surface

Meta's current generated Business SDK exposes the following parameter on Instagram media creation:

```text
user_tags: list<map>
```

The same generated `IGUser.createMedia()` surface also exposes caption, image/video URL, collaborators, location, alt text and other publication parameters.

Meta's current Instagram Login documentation separately states:

> This API setup cannot access ads or tagging.

Therefore the safe architectural interpretation is:

- **Facebook Login/Page-linked route:** supports `user_tags`;
- **Instagram Login/no-Page route:** do not rely on media tagging.

Sources:

- Meta current generated Business SDK `IGUser.createMedia()`: https://github.com/facebook/facebook-php-business-sdk/blob/main/src/FacebookAds/Object/IGUser.php
- Meta official Instagram Login limitation: https://www.postman.com/meta/instagram/folder/6raa77c/instagram-api-with-instagram-login

### Tag position

The historic/current Graph API model uses username plus image coordinates (`x`, `y`) for image tags. Meta's current generated SDK deliberately types `user_tags` as a generic map rather than publishing the full validation schema there.

Do not hard-code undocumented assumptions beyond what we verify during the future canary test.

The manifest should still be capable of representing coordinates:

```yaml
media_tags:
  - asset_id: slide_02
    username: example
    position:
      x: 0.50
      y: 0.40
```

Validation against the current API version should occur before scheduling/publishing.

---

## 3. User tags in carousels

### Status: supported in the container model; represent tags per child media

A carousel is built from child media containers. Each child uses the same media-creation API surface before the parent carousel container is created.

Therefore the publication model should attach user tags to the **specific child asset**, not to the carousel globally.

Recommended shape:

```yaml
media:
  - asset_id: slide_01
    user_tags: []

  - asset_id: slide_02
    user_tags:
      - username: example
        x: 0.50
        y: 0.40
```

This prevents ambiguity such as "tag this account somewhere in the carousel".

The exact supported combination of image/video child tagging should be confirmed in the eventual Meta canary tests before the capability is enabled for production.

---

## 4. Collaborator posts

### Status: current Meta API surface supports collaborators; acceptance remains external

Meta's current generated Business SDK exposes:

```text
collaborators: list<string>
```

on `IGUser.createMedia()`.

It also exposes:

- `IGMedia.getCollaborators()`;
- `IGUser.getCollaborationInvites()`;
- `IGUser.createCollaborationInvite()`;
- `IGUser.getCollaborativeMedia()`.

This is strong current evidence that Instagram collaboration is represented in the Graph API.

Sources:

- Meta generated `IGUser`: https://github.com/facebook/facebook-php-business-sdk/blob/main/src/FacebookAds/Object/IGUser.php
- Meta generated `IGMedia`: https://github.com/facebook/facebook-php-business-sdk/blob/main/src/FacebookAds/Object/IGMedia.php

### Architecture implication

High Director may eventually support:

```yaml
collaborators:
  - username: example
```

but must distinguish:

```text
collaboration requested
```

from:

```text
collaboration accepted/active
```

The invited account controls acceptance. Eirepolitic cannot guarantee that a requested collaborator becomes an active co-author.

### Route limitation

Because Meta explicitly excludes tagging from Instagram Login, collaborators should be treated as a **Facebook Login/Page-linked capability unless the exact Instagram Login endpoint is later documented otherwise**.

### Do not hard-code limits yet

Do not hard-code a maximum number of collaborators from secondary documentation. Confirm the live/current Meta API limit during the canary phase before enabling this field.

---

## 5. Location tagging

### Status: current Meta API surface exposes `location_id`; enable only after location-ID validation is proven

Meta's current generated `IGUser.createMedia()` exposes:

```text
location_id: string
```

Historically, Instagram Graph publishing uses a Facebook Page/location object ID that contains valid geographic location data.

The current high-level Meta Postman publishing guide does not spell out the complete current location lookup/eligibility rules as clearly as older reference documentation did.

### Conservative design

Represent location as a structured optional field:

```yaml
location:
  requested_name: Leinster House
  meta_location_id: null
```

The publication request should not accept a guessed numeric ID.

A future capability should:

1. resolve/validate the location through Meta;
2. show the selected location to the human at final approval;
3. store the resolved stable Meta ID;
4. fail validation before scheduling if the location is not valid for publishing.

### Route implication

Treat location tagging as part of the richer Page-linked feature set until Meta explicitly documents equivalent support under Instagram Login.

Source:

- Meta generated `IGUser.createMedia()`: https://github.com/facebook/facebook-php-business-sdk/blob/main/src/FacebookAds/Object/IGUser.php

---

## 6. Accessibility / alt text

### Status: API supported for media creation; v1 should use it for images

Meta's current generated `IGUser.createMedia()` exposes:

```text
alt_text: string
```

This means Eirepolitic can preserve the existing generated accessibility text and send it deterministically at publication time.

### Recommended scope

For the first implementation:

- single image: store one explicit alt-text value;
- carousel: store alt text per image child;
- video/Reels/Stories: do not assume the same accessibility parameter behaves identically until separately tested/documented.

Example:

```yaml
media:
  - asset_id: slide_01
    alt_text: "..."
  - asset_id: slide_02
    alt_text: "..."
```

This maps well to the repo's existing deterministic alt-text generation.

Source:

- Meta generated `IGUser.createMedia()`: https://github.com/facebook/facebook-php-business-sdk/blob/main/src/FacebookAds/Object/IGUser.php

---

## 7. First comment

### Status: API supported as a **separate post-publication action**, not an atomic publish field

Meta's current generated `IGMedia` API surface exposes:

```text
POST /{ig_media_id}/comments
message=<text>
```

through `IGMedia.createComment()`.

The Page-linked API permission set includes:

```text
instagram_manage_comments
```

and Meta's current Instagram API documentation explicitly includes comment management/replies among supported capabilities.

Sources:

- Meta generated `IGMedia.createComment()`: https://github.com/facebook/facebook-php-business-sdk/blob/main/src/FacebookAds/Object/IGMedia.php
- Meta official Facebook Login API overview: https://www.postman.com/meta/instagram/folder/u4g5a2a/instagram-api-with-facebook-login

### Architecture implication

A "first comment" should **not** be modeled as part of the atomic media publish request.

Instead:

```text
publish media
   ↓
receive Instagram Media ID
   ↓
create configured first comment
   ↓
record comment ID/result
```

The publication may therefore have states such as:

```text
media_published_comment_pending
published
```

or equivalent execution steps.

### Idempotency requirement

The first-comment action needs its own idempotency key/result record.

If the media publishes successfully and the comment call times out, a retry must not blindly create duplicate comments.

The exact final comment text must be approved/stored before publication, just like the caption.

---

## 8. Product / shopping tags

### Status: **restricted/inconsistent — exclude from v1**

There is a real inconsistency in Meta's current public surfaces.

Meta's current general Content Publishing guide says:

- shopping tags are not supported.

However, Meta's current generated Business SDK exposes:

```text
product_tags: list<map>
```

on media creation, and current `IGMedia` also exposes:

- `GET /product_tags`;
- `POST /product_tags`.

Sources:

- Meta Content Publishing guide: https://www.postman.com/meta/instagram/documentation/6yqw8pt/instagram-api?entity=request-23987686-ab559ffb-8e2c-4b0a-b43a-5737b6d2f672
- Meta generated `IGUser.createMedia()`: https://github.com/facebook/facebook-php-business-sdk/blob/main/src/FacebookAds/Object/IGUser.php
- Meta generated `IGMedia`: https://github.com/facebook/facebook-php-business-sdk/blob/main/src/FacebookAds/Object/IGMedia.php

### Interpretation

The SDK surface suggests product tagging exists for some Graph/commerce configurations, while the general organic Content Publishing guide does not guarantee it.

That is not sufficient evidence to make it a normal Eirepolitic publishing feature.

Recommendation:

```text
product_tags = capability disabled
```

for v1.

Only enable later if Meta's exact app/account/catalog configuration explicitly documents and successfully tests it.

Eirepolitic currently has no stated need for shopping/product tagging, so there is no reason to absorb this complexity now.

---

## 9. Branded-content / paid-partnership fields

### Status: API surface exists, but general Content Publishing documentation remains restrictive — exclude from v1

Meta's current generated media-creation surface also exposes fields including:

```text
branded_content_sponsor_ids
is_paid_partnership
```

and the generated Instagram objects expose branded-content-related edges.

At the same time, the current general Content Publishing guide says branded-content tags are not supported.

Therefore use the same conservative policy as product tags:

- do not include them in the initial manifest;
- do not promise them conversationally;
- add only if a future concrete Eirepolitic requirement justifies a dedicated Meta capability proof.

---

## 10. Capability matrix

| Feature | Facebook Login / Page-linked | Instagram Login / no Page | Eirepolitic v1 recommendation |
|---|---|---|---|
| Caption text | Supported | Supported | **Enable** |
| Hashtags in caption | Supported as text | Supported as text | **Enable** |
| `@mentions` in caption | Supported as caption text | Supported as caption text | **Enable** |
| Media/user tags | **Current API surface supports `user_tags`** | Meta says tagging unavailable | **Design for it; enable after canary** |
| Per-carousel-child user tags | Supported by child-container model/API surface | Do not rely on it | **Design for it; enable after canary** |
| Collaborators | **Current API surface supports collaborators/invite edges** | Do not rely on it | **Design for it; enable after canary** |
| Location | **Current API surface exposes `location_id`** | Do not rely on it | Optional; enable after location validation proof |
| Alt text | **Current media-creation surface exposes `alt_text`** | Content-publishing surface supports media metadata but tagging limitation does not itself block alt text | **Enable for image posts/carousel images** |
| First comment | **Supported as separate comment API action** | Comment API exists with Instagram Login permissions too | Optional Phase 2 action |
| Product tags | SDK surface exists; general guide says unsupported | Not safe to assume | **Disable v1** |
| Branded/paid-partnership tags | SDK surface exists; general guide restrictive | Not safe to assume | **Disable v1** |

---

## 11. Consequence for authentication choice

Step 4 left the authentication route provisional.

Step 5 materially narrows the choice.

The desired High Director experience includes commands such as:

```text
"Tag @example on slide 3."
"Invite @example as a collaborator."
"Don't tag anyone in the media; just mention them in the caption."
```

Meta explicitly states that the Instagram Login route cannot access tagging, while the current Page-linked Business SDK surface exposes the relevant media-tag/collaboration fields and edges.

Therefore the current **preferred direct-Meta authentication route is now:**

```text
Professional Instagram account
        ↓
linked Facebook Page
        ↓
Instagram API with Facebook Login
```

This is still an architecture recommendation only. No account/Page changes should occur during research.

---

## 12. Manifest consequences

The publication-intent layer should keep these separate:

```yaml
caption:
  text: "..."

caption_mentions:
  - username: example

media:
  - asset_id: slide_01
    alt_text: "..."
    user_tags: []

  - asset_id: slide_02
    alt_text: "..."
    user_tags:
      - username: example
        x: 0.50
        y: 0.40

collaborators:
  - username: another_account

location:
  meta_location_id: null
  display_name: null

first_comment:
  text: null
```

Do not collapse these into one generic `tags` list.

Each represents different Instagram behaviour and different failure conditions.

---

## 13. Approval consequences

The final High Director confirmation must explicitly show:

- exact caption, including caption mentions;
- media tags and which slide/image they apply to;
- collaborator invitations;
- resolved location or `none`;
- alt text for each image where used;
- first-comment text or `none`.

A change to any of those fields after approval must invalidate the publication approval fingerprint.

---

## 14. Correction to Step 3 found during this research

The current Meta-generated `IGMedia` object exposes:

```text
DELETE /{ig_media_id}
```

through `IGMedia.deleteSelf()`.

Therefore Step 3's conservative statement that published-media deletion should be treated as unsupported was too restrictive.

**Corrected finding:** the current Page-linked Graph API surface supports deletion of Instagram media objects. This should still be capability-tested before High Director is allowed to delete published content conversationally.

The same generated object exposes `updateSelf()` only with `comment_enabled`; it does **not** expose caption text as an update field. Therefore the Step 3 conclusion that we should not promise programmatic editing of an already-published caption remains valid.

Source:

- Meta generated `IGMedia`: https://github.com/facebook/facebook-php-business-sdk/blob/main/src/FacebookAds/Object/IGMedia.php

### Scheduling note rechecked

Meta's current generated `IGUser` exposes a **read** edge:

```text
GET /scheduled_media
```

but the generated SDK surface reviewed here does not expose a matching create-scheduled-media operation. The documented Content Publishing flow still publishes via `/media_publish`.

Therefore the earlier architecture conclusion remains: Eirepolitic should not rely on Meta as the authoritative future scheduler unless Meta later documents a supported create/update scheduling endpoint for this use case.

---

## Sources

Primary current Meta sources used:

- Meta official Instagram API workspace: https://www.postman.com/meta/instagram/overview
- Meta official Content Publishing guide: https://www.postman.com/meta/instagram/documentation/6yqw8pt/instagram-api?entity=request-23987686-ab559ffb-8e2c-4b0a-b43a-5737b6d2f672
- Meta official Instagram API with Facebook Login: https://www.postman.com/meta/instagram/folder/u4g5a2a/instagram-api-with-facebook-login
- Meta official Instagram API with Instagram Login: https://www.postman.com/meta/instagram/folder/6raa77c/instagram-api-with-instagram-login
- Meta current generated Business SDK `IGUser`: https://github.com/facebook/facebook-php-business-sdk/blob/main/src/FacebookAds/Object/IGUser.php
- Meta current generated Business SDK `IGMedia`: https://github.com/facebook/facebook-php-business-sdk/blob/main/src/FacebookAds/Object/IGMedia.php

The Meta Business SDK is generated from Meta's API schema and is used here as authoritative evidence of the current Graph API surface where the high-level Postman guide does not expose every media parameter.

---

## Confidence / unresolved items

**High confidence:**

- Instagram Login explicitly excludes tagging;
- Page-linked current API surface exposes `user_tags`;
- current API surface exposes `collaborators` and collaboration edges;
- current API surface exposes `location_id`;
- current API surface exposes `alt_text`;
- current IG Media API surface can create comments;
- current IG Media API surface supports deletion;
- caption editing is not exposed by the current IG Media update surface reviewed.

**Must be proven in a future non-production/canary phase before enabling:**

- exact username eligibility for media tags;
- exact coordinate validation/rules;
- exact carousel/video tag behaviour;
- collaborator limits and eligibility;
- exact location lookup/eligibility rules;
- whether alt text should be sent for any non-image format;
- product/shopping tagging under Eirepolitic's exact app/account configuration;
- branded/paid-partnership functionality.

**Next research step:**

Step 6 will design and assess the direct Meta publishing architecture using the capability/authentication findings from Steps 2–5.
