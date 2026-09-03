# Experiment Manifests

This directory holds sanitized experiment manifests and result summaries. It is not a location for raw logs, checkpoints, notebook outputs, or private leaderboard exports.

## Included Example

- [`gptq-w4a16.example.yaml`](gptq-w4a16.example.yaml): a score-free documentation record for an attempted GPTQ track. It conforms to [`schemas/experiment-manifest.schema.json`](../schemas/experiment-manifest.schema.json).

Every future result table must label values as `official`, `unverified-internal`, `local-benchmark`, `failed`, `partial`, or `exploratory`. `official` is permitted only after organizer-confirmed evidence is attached.
