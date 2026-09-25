# Step 3 — Current Meta Instagram Publishing API Capabilities

Status: **complete**

Research date: 2026-09-03

Scope: establish what Meta's current Instagram publishing API can and cannot do at a high level. Authentication details are intentionally deferred to Step 4. Detailed tagging/mentions/collaborators/location/alt-text/first-comment analysis is intentionally deferred to Step 5.

No Meta account, app, token, or live publication was created.

---

## Short conclusion

Meta currently provides a real server-side publishing API for Instagram **Professional accounts** (Business and Creator).

It directly supports:

- single images;
- videos;
- Reels;
- Stories;
- carousels containing up to 10 images/videos in combination;
- captions;
- media-processing/container status checks;
- published Instagram media IDs;
- a live content-publishing quota endpoint.

The API does **not** expose a documented `publish_at`/future-schedule operation in the current Content Publishing flow. The application creates media containers and later calls `/media_publish`; therefore Eirepolitic should assume that future scheduling must be handled by our own scheduler or a third-party scheduling service.

Meta requires publishing media to be retrievable from a publicly accessible URL at publication/container-creation time.

---

## 1. Supported account types

Meta's current official Instagram API workspace states that Instagram Professionals are supported:

- Business;
- Creator.

Personal/consumer Instagram accounts are not supported for the publishing API.

This reinforces Step 2: the existing Eirepolitic account can remain Personal during research, but real API publishing will eventually require switching that same account to a professional type.

For the Facebook Login API route, Meta states that publishing is available to all professional accounts **except Stories publishing, which is Business-only on that route**.

Source:

- Meta official Instagram API workspace: https://www.postman.com/meta/instagram/documentation/6yqw8pt/instagram-api
- Meta official Facebook Login folder: https://www.postman.com/meta/instagram/folder/u4g5a2a/instagram-api-with-facebook-login

---

## 2. Current publishing flow

The current Meta Content Publishing flow is container based:

```text
Approved media URL
      ↓
POST /{ig_user_id}/media
      ↓
Instagram media container ID
      ↓
wait/check processing when needed
      ↓
POST /{ig_user_id}/media_publish
      ↓
published Instagram Media ID
```

For video/Reels processing, Meta exposes container status retrieval:

```text
GET /{container_id}?fields=status_code,status
```

The documented status values include:

- `EXPIRED`
- `ERROR`
- `FINISHED`
- `IN_PROGRESS`
- `PUBLISHED`

Meta states that an unpublished container expires after 24 hours.

Meta recommends polling container status approximately once per minute, for no more than five minutes in the ordinary processing case.

Source:

- Meta Content Publishing guide: https://www.postman.com/meta/instagram/documentation/6yqw8pt/instagram-api?entity=request-23987686-ab559ffb-8e2c-4b0a-b43a-5737b6d2f672

### Architecture implication

Do **not** create Meta media containers days in advance when a post is conversationally scheduled.

Store our own approved publication request, then create the Meta container shortly before/at execution time.

Meta container IDs belong to execution state, not human content intent.

---

## 3. Images

Current Meta publishing documentation supports single-image publishing.

The current general limitation says **JPEG is the supported image format** for the standard image publishing path; extended JPEG variants such as MPO/JPS are not supported.

Meta retrieves the image from the supplied `image_url`.

### Eirepolitic implication

Before live publication is ever enabled, the approved-asset package must validate the final media format against the current API requirements. Existing generated PNG assets cannot simply be assumed publishable through the API if the active endpoint/version requires JPEG.

This needs to be reflected later in the asset-readiness design.

Source:

- Meta Content Publishing guide: https://www.postman.com/meta/instagram/documentation/6yqw8pt/instagram-api?entity=request-23987686-ab559ffb-8e2c-4b0a-b43a-5737b6d2f672

---

## 4. Carousels

Meta currently supports carousels containing:

- images;
- videos;
- or a mixture of both.

A carousel is built by:

1. creating each child media container;
2. collecting the child container IDs;
3. creating a parent `CAROUSEL` container;
4. supplying the ordered `children` list;
5. publishing the parent container.

Current documented maximum: **10 children** per API carousel.

The ordering of the `children` list therefore matters and must be deterministic.

Meta also states that carousel images are cropped according to the first image's orientation/aspect behaviour.

### Eirepolitic implication

Our future publication request must store ordered media explicitly. It cannot rely on directory/file listing order at execution time.

The publisher must not publish a carousel parent until all expected child containers are ready.

Source:

- Meta Content Publishing guide: https://www.postman.com/meta/instagram/documentation/6yqw8pt/instagram-api?entity=request-23987686-ab559ffb-8e2c-4b0a-b43a-5737b6d2f672

---

## 5. Reels

Meta currently supports Reel publishing.

The official current example uses:

```text
media_type=REELS
video_url=...
caption=...
share_to_feed=true|false
```

The current Meta Reel guide lists, among other requirements:

- MOV or MP4 container;
- AAC audio;
- HEVC or H.264 video;
- 23–60 FPS;
- recommended 9:16 aspect ratio;
- maximum 1920 horizontal pixels;
- maximum 25 Mbps video bitrate;
- maximum 128 kbps audio bitrate;
- duration 3 seconds to 15 minutes;
- maximum file size 1 GB.

The Reel must be processed into a finished container before `/media_publish` is called.

Source:

- Meta Reels Publishing: https://www.postman.com/meta/instagram/folder/830j7my/reels-publishing

### Eirepolitic implication

Reels are technically possible, but they introduce asynchronous media processing and broader asset validation than static carousel posts. They do not need to be part of the first implementation if Eirepolitic's current generated posts are static images.

---

## 6. Stories

The current Content Publishing API supports `media_type=STORIES`.

On Meta's Facebook Login route, Meta explicitly states that Stories publishing is available only to **Business** Instagram accounts, while general feed content publishing supports Business and Creator.

This is one reason the Creator-versus-Business decision should remain open until the complete feature/authentication analysis is finished.

Source:

- Meta Instagram API documentation: https://www.postman.com/meta/instagram/documentation/6yqw8pt/instagram-api

---

## 7. Captions

Caption text is a supported media/container parameter. Meta's official Reel and carousel examples both include `caption`.

This supports the proposed Eirepolitic principle:

```text
conversation/template generation
       ↓
exact final caption stored
       ↓
publication API receives that exact caption
```

The publisher should never ask an LLM to regenerate the caption when the scheduled job fires.

### Hashtags and caption mentions

At this high-level step they should be regarded as text contained in the caption. Detailed rules around account mentions versus media tagging are intentionally deferred to Step 5.

---

## 8. Media hosting

Meta's current publishing guide is explicit that it **cURLs media supplied by URL** and the media must be on a publicly accessible server at the time of the publishing attempt/container creation.

This does not mean the underlying canonical S3 object must be permanently public.

A likely later design is:

```text
private immutable S3 asset
        ↓
execution-time HTTPS/presigned URL
        ↓
Meta retrieves asset
```

The exact approach and URL lifetime will be investigated in Step 12.

### Important restriction

Temporary GitHub Actions artifact URLs are not a suitable production asset source.

Source:

- Meta Content Publishing guide: https://www.postman.com/meta/instagram/documentation/6yqw8pt/instagram-api?entity=request-23987686-ab559ffb-8e2c-4b0a-b43a-5737b6d2f672

---

## 9. Native scheduling

### Finding

The current Content Publishing flow documents:

- `/media` — create/upload a container;
- `/?fields=status_code` — check readiness/state;
- `/media_publish` — publish the ready container;
- `/content_publishing_limit` — inspect quota.

It does **not** document a `scheduled_publish_time`, `publish_at`, or equivalent future-time parameter for Instagram content publishing.

Meta's guide does refer to applications that allow users to "schedule posts to be published in the future" when advising apps to enforce quota limits. In context, the actual publication endpoint remains `/media_publish`; the scheduling layer therefore belongs to the application/service, not to a native deferred Instagram publishing object.

### Architecture conclusion

Treat Meta as an **execute-publication-now** API.

Eirepolitic must own the future scheduling decision, either:

- directly in our infrastructure; or
- through a third-party scheduler API.

This will be compared in Steps 7–9.

---

## 10. Cancelling/rescheduling future posts

Because the recommended interpretation is that Meta does not create a native future-scheduled Instagram publication object, there is no Meta future job that Eirepolitic should depend on cancelling/rescheduling.

Instead, if Eirepolitic owns scheduling:

```text
approved publication request
      ↓
our scheduler job
      ↓
Meta called only at execution time
```

then:

- `Cancel Friday's post` means cancel/disable our scheduler record;
- `Move tomorrow's post to 8pm` means update our own schedule and approval rules;
- no Meta post exists yet.

This is desirable because conversational changes remain under Eirepolitic's deterministic control until publication time.

---

## 11. Published media ID and status

A successful `/media_publish` operation returns an Instagram Media ID.

That ID must be stored in the future publication ledger.

Current Meta APIs also allow professional accounts' media to be read back and expose media metadata such as captions/media information. Meta's current Insights documentation explicitly references `permalink` as a media field (with the limitation that some fields such as permalink are not available on carousel child photos).

Therefore the future publisher should reconcile the result after publication and store, where available:

- Instagram Media ID;
- permalink;
- media type/product type;
- actual timestamp;
- final caption retrieved from Instagram when useful for verification.

Sources:

- Meta publish response example: https://www.postman.com/meta/instagram/documentation/6yqw8pt/instagram-api
- Meta Insights documentation: https://www.postman.com/meta/instagram/folder/23987686-f659d7d1-d74c-44e4-9192-9b1e8694c511

---

## 12. Quotas and rate limits

Meta's current general Content Publishing section states:

- **100 API-published posts per moving 24-hour period**;
- a carousel counts as one post;
- usage can be queried using `/content_publishing_limit`.

However, the carousel subsection in the same current Meta documentation still contains a statement saying accounts are limited to **50 published posts within 24 hours**.

That is an internal inconsistency in the current documentation.

### Architecture conclusion

Do not hard-code `100` or `50` as the only safety control.

The eventual publisher should query:

```text
GET /{ig_user_id}/content_publishing_limit
```

and use the live quota information for the connected account/API version.

Eirepolitic's expected publishing volume is far below either value, so this discrepancy is not currently a practical capacity risk.

Source:

- Meta Content Publishing guide: https://www.postman.com/meta/instagram/documentation/6yqw8pt/instagram-api?entity=request-23987686-ab559ffb-8e2c-4b0a-b43a-5737b6d2f672

---

## 13. Current general limitations explicitly documented by Meta

The current general Content Publishing guide lists these limitations:

- JPEG is the only image format supported in the documented image publishing flow;
- shopping tags are not supported by that general publishing guide;
- branded-content tags are not supported by that general publishing guide;
- filters are not supported.

Detailed tagging-related findings will be revisited in Step 5 because Meta has multiple API surfaces/login modes and some newer generated SDK fields need to be reconciled against the general guide before Eirepolitic relies on them.

Source:

- Meta Content Publishing guide: https://www.postman.com/meta/instagram/documentation/6yqw8pt/instagram-api?entity=request-23987686-ab559ffb-8e2c-4b0a-b43a-5737b6d2f672

---

## 14. Editing/deleting published posts

The current official Content Publishing flow reviewed for this step documents creation, publishing, status inspection and quota inspection. It does **not** document an operation for editing the caption of an already-published Instagram media object, nor a publication-management operation for deleting a published media object.

Therefore the architecture should currently treat:

- **editing an already-published caption through the publishing API** as unsupported unless a later authoritative endpoint is found;
- **deleting already-published Instagram content through the publishing API** as unsupported unless a later authoritative endpoint is found.

This is deliberately conservative. We should not design conversational commands that promise API capabilities Meta does not currently document.

This is different from changing or cancelling an Eirepolitic job **before** it is published, which remains fully controllable in our own system.

---

## 15. Capability matrix at the end of Step 3

| Capability | Current finding | Notes |
|---|---|---|
| Personal account publishing | **Unsupported** | Professional account required |
| Business feed publishing | **Supported** | Direct API |
| Creator feed publishing | **Supported** | Direct API |
| Single image | **Supported** | Current guide specifies JPEG |
| Video | **Supported** | Container/processing flow |
| Carousel | **Supported** | Up to 10 children |
| Mixed image/video carousel | **Supported** | Up to 10 children |
| Reels | **Supported** | Async processing; documented specs |
| Stories | **Supported with restriction** | Facebook Login route: Business-only |
| Caption | **Supported** | Explicit container parameter |
| Hashtags in caption | **Supported as caption text** | Detailed semantics later |
| `@mentions` in caption | **Caption-level capability; detailed rules later** | Step 5 |
| Media/user tags | **Deferred** | Step 5 |
| Collaborators | **Deferred** | Step 5 |
| Location | **Deferred** | Step 5 |
| Alt text | **Deferred** | Step 5 |
| First comment | **Deferred** | Step 5 |
| Product tags | **General guide says unsupported** | Reconcile in Step 5 |
| Branded-content tags | **General guide says unsupported** | Reconcile in Step 5 |
| Filters | **Unsupported** | Explicit Meta limitation |
| Public/retrievable media URL | **Required** | Meta cURLs media |
| Media container status | **Supported** | EXPIRED/ERROR/FINISHED/IN_PROGRESS/PUBLISHED |
| Published Media ID | **Supported** | Returned by `/media_publish` |
| Permalink readback | **Available on media data where applicable** | Not every child object supports it |
| Native future scheduling | **No documented native deferred scheduling operation** | Scheduler must live elsewhere |
| Cancel/reschedule pre-publication job | **Our infrastructure** | No Meta post exists yet |
| Edit published caption | **Not documented in current publishing flow; treat as unsupported** | Do not promise |
| Delete published media | **Not documented in current publishing flow; treat as unsupported** | Do not promise |
| Publishing quota inspection | **Supported** | `/content_publishing_limit` |

---

## 16. Step 3 architecture consequences

The following design assumptions are now safe enough to carry forward:

1. The existing Eirepolitic account can eventually be converted to a professional account and used for direct API publishing.
2. Static image/carousel posts are a strong first implementation target.
3. The future scheduler should store our own intent and call Meta only when publication time arrives.
4. Meta container IDs belong to execution records, not human-approved content records.
5. Approved publication assets need stable, Meta-retrievable hosting at execution time.
6. Ordered carousel assets must be explicit and validated.
7. The publication ledger must store Meta's returned media ID and reconcile resulting media metadata.
8. Live quota inspection is preferable to hard-coding Meta's conflicting published limit documentation.
9. Reels/Stories can be added later without changing the conversational control model.
10. We should not promise post-publication edit/delete operations that Meta's current publishing documentation does not expose.

---

## Sources

Primary authoritative sources used for this step:

- Meta official Instagram API workspace: https://www.postman.com/meta/instagram/overview
- Meta official Instagram API documentation: https://www.postman.com/meta/instagram/documentation/6yqw8pt/instagram-api
- Meta official Content Publishing guide: https://www.postman.com/meta/instagram/documentation/6yqw8pt/instagram-api?entity=request-23987686-ab559ffb-8e2c-4b0a-b43a-5737b6d2f672
- Meta official Instagram API with Facebook Login: https://www.postman.com/meta/instagram/folder/u4g5a2a/instagram-api-with-facebook-login
- Meta official Reels Publishing folder: https://www.postman.com/meta/instagram/folder/830j7my/reels-publishing
- Meta official Insights folder: https://www.postman.com/meta/instagram/folder/23987686-f659d7d1-d74c-44e4-9192-9b1e8694c511

## Confidence / unresolved items

**High confidence:**

- professional-account requirement;
- images/carousels/Reels/Stories support;
- carousel maximum of 10;
- container-based publish flow;
- status codes and 24-hour expiration;
- URL-retrieval requirement;
- no native deferred scheduling parameter in the documented Content Publishing flow;
- `/content_publishing_limit` availability.

**Intentionally unresolved until later steps:**

- exact tagging behaviour by login route;
- collaborators;
- location tagging;
- alt/accessibility-text scope;
- first-comment support;
- product/branded-content tagging inconsistencies;
- exact authentication/review/token requirements;
- final Creator-versus-Business recommendation.
