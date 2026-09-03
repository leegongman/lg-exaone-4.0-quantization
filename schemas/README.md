# Schemas

This directory holds versioned public schemas for sanitized experiment manifests, benchmark summaries, and configuration examples.

## Included Schema

- [`experiment-manifest.schema.json`](experiment-manifest.schema.json): schema for sanitized method and evidence records. It intentionally omits competition-score, data-path, checkpoint-path, and credential fields.

Schemas should describe metadata, not embed private dataset fields, credentials, raw checkpoint paths, or competition submission payloads.
