# Project Status

## Current Public Status

This repository is a clean public documentation draft. It is not yet a verified final release.

The project should be presented as a technical portfolio and postmortem, not as a complete reproduction of the competition submission package.

## Completed

- EXAONE 4.0 1.2B lightweight optimization study; Phase 1/2 passage and Phase 3 final-stage participation are recorded internally but await organizer evidence for public verification
- Quantization method exploration across GPTQ, AWQ, SmoothQuant-style approaches, OmniQuant, and lower-precision variants
- Technical inventory expansion from full Notion export, local notebooks/scripts, and internal submission memo labels
- EXAONE structure, hidden-layer targeting, and compatibility analysis
- vLLM customization investigation
- Local benchmark and latency-oriented evaluation work
- Source cleanup plan and public documentation draft

## Partially Completed

- OmniQuant adaptation to EXAONE/vLLM
- SmoothQuant-style model integration
- LoRA fine-tuning path
- Clean reproducibility packaging
- Mapping internal records to official public evidence

## Failed or Inconclusive

- Some low-bit quantization paths produced poor local quality.
- Some local benchmark results did not predict private competition performance.
- Some custom runtime or wheel submission paths did not translate into strong competition outcomes.
- Calibration and benchmark selection require further validation before public performance claims.

## Not Reviewed

The following repositories were not reviewed because they were not publicly accessible during the cleanup pass:

- `leegongman/vLLM_OmniQuant`
- `leegongman/vllm-exaone-sq`

They should not be used as evidence in this public repository unless they become accessible and are reviewed separately.

## Security and Publication Actions

Original DACON helper files were found to contain credential-like token material. Those files are excluded from this clean repository.

Before publishing, the relevant DACON token should be revoked or rotated, even if it is believed to be expired.

The public repository must not include:

- DACON token files
- API credential scripts
- Raw secret pages
- Private competition data
- Submission artifacts

## Public Claims Table

| Claim | Evidence source | Status | Public wording |
|---|---|---|---|
| Phase 2 rank | Notion/PDF; official leaderboard not yet attached | unverified internal record | Internal note fields record a Phase 2 ranking; this is pending official public verification. |
| Phase 3 rank | Notion/PDF; official leaderboard not yet attached | unverified internal record | Internal note fields record a Phase 3 ranking; this is pending official public verification. |
| Phase progression | Internal records plus official phase descriptions; official outcome page not yet attached | unverified internal record | Internal records describe progression through earlier phases and participation in the Phase 3 final-stage track. This is not a prize or verified leaderboard claim. |
| Phase 2 score weighting | Internal postmortem PDF; organizer formula not attached | unverified internal record | The postmortem records `5 x accuracy + 5 x speed`; this is included as an internal evaluation record, not an official formula. |
| Phase 3 score weighting | Internal postmortem PDF; organizer formula not attached | unverified internal record | The postmortem records `60 x accuracy + 20 x speed + 20 x model size`; this is included as an internal evaluation record, not an official formula. |
| EXAONE quantization experiments | Local folder, reviewed GitHub repos, full Notion export, PDF | verified as attempted | Explored multiple quantization approaches for EXAONE 4.0 1.2B under vLLM-related constraints. |
| EXAONE architecture and hidden-layer targeting | Reviewed local configuration, local model-analysis notebooks, full Notion export | reviewed evidence | Used layer-specific Q/K/V and MLP projection targeting with mixed-precision/protected-module variants; no base-architecture rewrite is claimed. |
| Broad GPTQ/AWQ schema sweeps | Full Notion export, local notebooks, internal submission memo labels, user clarification | verified as attempted | Broad GPTQ and AWQ schema, internal-parameter, target-module, calibration, and hyperparameter sweeps were attempted. |
| vLLM customization | Public vLLM fork commits, reviewed docs | implementation evidence reviewed | Reviewed public evidence documents EXAONE/vLLM integration experiments, including model-registration and quantization-compatibility work. |
| Local benchmark | Full Notion export, local scripts, GitHub docs, experiment notes | partial | Local benchmark results were used for experiment comparison and are not reported as official competition scores. |
| LoRA fine-tuning | Local fine-tuning folder, full Notion export, PDF | partial | Explored LoRA fine-tuning, data preprocessing, and dataset-format experiments as auxiliary optimization paths; raw datasets are excluded. |
| Knowledge distillation and layer drop | Full Notion export, local notebooks, checkpoint labels | partial | Explored block distillation, KD on layer-dropped models, and drop-last variants as compression paths; raw notebooks/data are excluded. |
| OmniQuant integration | Reviewed `scheme_vLLM_omniquant` documentation and full Notion export | implementation evidence reviewed | Documented OmniQuant adaptation attempts for EXAONE/vLLM; upstream OmniQuant code is attributed separately. |
| Upstream vLLM usage | vLLM upstream and fork history | reviewed external dependency | vLLM was used as an external inference framework; upstream code is not claimed as original work. |
| Upstream OmniQuant usage | OmniQuant-derived public repos and documentation | reviewed external dependency | OmniQuant was used as an external research baseline; upstream code and method ownership remain with the original authors. |
| Inaccessible repos | `vLLM_OmniQuant`, `vllm-exaone-sq` | not reviewed | These repositories were not reviewed and are not used as evidence in this public cleanup. |
