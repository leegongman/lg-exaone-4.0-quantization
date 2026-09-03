# Competition Overview

## Context

This portfolio documents an LG Aimers AI Hackathon project centered on lightweight optimization of `EXAONE-4.0-1.2B`. The publicly available [Phase 2](https://dacon.io/competitions/official/236673/overview/description) and [Phase 3](https://dacon.io/competitions/official/236689/overview/description) pages define different submission and evaluation conditions.

The project narrative is supported by reviewed local materials, internal notes, and public source repositories. Official leaderboard artifacts are not stored in this repository, so numeric ranking and score claims remain unverified internal records.

## Phase 2: Model Optimization Under Organizer Runtime

The Phase 2 page frames the task as optimizing `EXAONE-4.0-1.2B` for inference quality and lightweight deployment, then evaluating final model weights against a hidden test set. The published rules allow use of publicly disclosed datasets and preprocessing, while the organizer controls the private evaluation setup.

For this project, Phase 2 is documented as the checkpoint-focused path:

- The main deliverable was an optimized model-weight artifact.
- Compatibility with the organizer's vLLM-based runtime was a practical constraint from project records.
- Quantization, calibration, model-size, local quality, and latency trade-offs were investigated together.
- A participant-supplied custom vLLM wheel is not treated as part of the Phase 2 submission surface in this documentation.

## Phase 3: Model Engine and Wheel-Level Scope

The Phase 3 page is restricted to the Phase 2 award-candidate/finalist path and adds a code-review period. Crucially, it permits a model-engine artifact alongside model weights, explicitly including a vLLM wheel. This changed the feasible optimization surface.

For this project, Phase 3 is documented as the runtime-customization path:

- A custom vLLM build could be evaluated with the model artifact.
- EXAONE-specific model registration, loading behavior, and SmoothQuant-style identity parameter handling became relevant implementation concerns.
- Wheel packaging, installation, and compatibility were part of the engineering risk, not merely a model-quality issue.

## Phase Progression and Result Boundary

Internal project records describe the team as passing the earlier stages and participating in the Phase 3 final-stage track. This is a progression record, not a prize, score, or verified leaderboard claim.

Internal note fields also contain Phase 2 and Phase 3 rank values. They are retained only as **unverified internal records** in [project-status.md](project-status.md) until organizer-confirmed evidence can be attached. The README intentionally does not foreground those numbers.

## Result Classification

| Label | Meaning | Allowed public use |
|---|---|---|
| Official competition result | Organizer-confirmed leaderboard, award notice, or result page | Quote only with direct evidence and context |
| Unverified internal record | Notion, PDF, submission memo, or team record without attached official proof | State as internal only; do not use promotional wording |
| Local benchmark | Local script, public benchmark task, or experiment log | Report only with environment and task context |
| Failed, partial, or exploratory experiment | Attempted method without a conclusive competition outcome | Include as a lesson, not as a result |

## Implication for the Technical Story

The Phase 2/3 difference is why the repository documents both model-side and runtime-side work. A quantization scheme had to preserve useful quality and fit the model-size/latency budget; in Phase 3 it also had to remain loadable through a custom wheel path. The project therefore tracks checkpoint configuration, layer/module targeting, quantization format, calibration, local benchmarks, and vLLM compatibility as separate evidence streams.
