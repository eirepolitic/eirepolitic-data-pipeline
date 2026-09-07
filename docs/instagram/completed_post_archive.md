# Completed post archive

Purpose: preserve approved finished social posts and the agent-authored creation record so future agents can inspect both the visual target and the process that produced it.

## S3 location

This repository owns:

`s3://eirepolitic-data/instagram/previews/completed-posts/eirepolitic-data-pipeline/`

The archive sits beneath the existing private Instagram S3 namespace because that is the repository's established least-privilege write boundary. The `completed-posts` subtree is durable reference storage, not an approval or publishing queue.

Each approved post is immutable once archived:

`<post_id>/assets/...`
`<post_id>/agent-summary.json`
`<post_id>/agent-summary.md`
`<post_id>/provenance.json`

The repo-level catalog is `index.json`.

## Agent summary requirement

The creating agent should write the summary after human approval and before archival. It must record the stable post ID, title, platform, repository and completion timestamp; every final asset; repo/workflow/rendering tools actually used; ordered creation process; important design/data decisions; QA and approvals; and sources, limitations, related workflows and PRs when applicable.

Do not claim tools, checks or methodology that were not actually used.

## Archive behavior

Use `.github/workflows/completed_post_archive.yml`. It accepts an asset Git ref/directory and a summary Git ref/path separately, allowing generated preview branches to provide the finished files while `main` stores the durable agent summary.

The workflow validates metadata, copies only declared assets, generates SHA-256 provenance, uploads recoverable content, updates `index.json` idempotently, and writes `provenance.json` last as the completion marker. A matching completed archive can repair a missing catalog entry without modifying the finished assets; a conflicting summary fails closed.

## Reference workflow

Use `.github/workflows/completed_post_reference.yml` to list the catalog or fetch one archived post as a temporary GitHub artifact for inspection. S3 remains the durable source of truth.
