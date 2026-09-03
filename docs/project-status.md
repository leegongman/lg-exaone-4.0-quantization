# Project Status

## Current Public Status

This repository is a clean public documentation draft. It is not yet a verified final release.

The project should be presented as a technical portfolio and postmortem, not as a complete reproduction of the competition submission package.

## Completed

- EXAONE 4.0 1.2B lightweight optimization study
- Quantization method exploration across GPTQ, AWQ, SmoothQuant-style approaches, OmniQuant, and lower-precision variants
- Technical inventory expansion from Notion taxonomy, local notebooks/scripts, and internal submission memo labels
- EXAONE structure and compatibility analysis
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
| EXAONE quantization experiments | Local folder, reviewed GitHub repos, Notion/PDF | verified as attempted | Explored multiple quantization approaches for EXAONE 4.0 1.2B under vLLM-related constraints. |
| Broad GPTQ/AWQ schema sweeps | Local notebooks, internal submission memo labels, user clarification | verified as attempted | Broad GPTQ and AWQ schema, target-module, calibration, and hyperparameter sweeps were attempted. |
| vLLM customization | Public vLLM fork commits, reviewed docs | implementation evidence reviewed | Reviewed public evidence documents EXAONE/vLLM integration experiments, including model-registration and quantization-compatibility work. |
| Local benchmark | Local scripts, GitHub docs, experiment notes | partial | Local benchmark results were used for experiment comparison and are not reported as official competition scores. |
| LoRA fine-tuning | Local fine-tuning folder, Notion/PDF | partial | Explored LoRA fine-tuning and data preprocessing as an auxiliary optimization path; raw datasets are excluded. |
| OmniQuant integration | Reviewed `scheme_vLLM_omniquant` documentation | verified documentation evidence | Documented OmniQuant adaptation attempts for EXAONE/vLLM; upstream OmniQuant code is attributed separately. |
| Upstream vLLM usage | vLLM upstream and fork history | reviewed external dependency | vLLM was used as an external inference framework; upstream code is not claimed as original work. |
| Upstream OmniQuant usage | OmniQuant-derived public repos and documentation | reviewed external dependency | OmniQuant was used as an external research baseline; upstream code and method ownership remain with the original authors. |
| Inaccessible repos | `vLLM_OmniQuant`, `vllm-exaone-sq` | not reviewed | These repositories were not reviewed and are not used as evidence in this public cleanup. |
