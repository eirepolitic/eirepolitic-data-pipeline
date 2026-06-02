# Instagram visual generation layer

Review-only reusable visual asset generation for the Eirepolitic Instagram content factory.

This layer is separate from `instagram/templates/layouts/`, which defines complete post layouts. Visuals generated here are standalone PNG assets that can be placed inside those post layouts.

Current draft visual:

- `horizontal_bar_draft_v1`

Preview outputs are published by the manual review workflow to:

```text
branch: instagram-preview-output
preview/visuals/png/
preview/visuals/metadata/
preview/visuals/manifests/
```

This system is review-only. It does not publish, schedule, or approve Instagram posts.
