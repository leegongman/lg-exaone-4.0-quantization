# EXAONE OmniQuant Adaptation Patch

## Purpose

This patch preserves the EXAONE-specific changes extracted from the former OmniQuant work snapshots without vendoring the full OpenGVLab OmniQuant repository.

## Source and Base

| Item | Value |
|---|---|
| Upstream base | [OpenGVLab/OmniQuant](https://github.com/OpenGVLab/OmniQuant) commit `feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14` |
| Consolidated source snapshot | `leegongman/CP` commit `ea6bcaba99703fbe82c09ddb84b408001b43eadb` |
| Duplicate evidence snapshots | `scheme_vLLM_omniquant` commit `5178a0e5a1c359dd39b8e2457b16036303ab570d`; `EXAONE_Quantization_method` commit `216e1b2f256a640cdd4b967aeaa6539258ae006a` |
| License | MIT; see [`../licenses/OmniQuant-MIT.txt`](../licenses/OmniQuant-MIT.txt) |

## Included Changes

[`0001-exaone-v4-omniquant-adaptation.patch`](0001-exaone-v4-omniquant-adaptation.patch) contains the EXAONE adaptation delta only:

- EXAONE 4 quantized decoder-layer support.
- EXAONE-aware model loading and transformation changes.
- Real activation-quantization export and checkpoint verification support.
- OmniQuant optimization-loop stability changes.
- EXAONE/vLLM readiness, smoke, and evaluation helper scripts.

It excludes raw `config.json`, datasets, notebook files, logs, checkpoints, model weights, and evaluation output.

## Legacy Variant

[`legacy/0000-vllm-speed-exaone4-layer-v1.patch`](legacy/0000-vllm-speed-exaone4-layer-v1.patch) preserves the earlier EXAONE layer implementation from `vLLM_Speed` commit `f70db19b276ab8ca4707c6825b7d440dd39bb9b4`. It is retained as historical implementation evidence and is not presented as a newer or preferred runtime path.

## Applying

Apply only after checking out the stated OmniQuant base revision:

```bash
patch -p1 < patches/omniquant-exaone/0001-exaone-v4-omniquant-adaptation.patch
```

The patch has not been presented as a one-command reproducible environment. It requires separately installed compatible dependencies and a user-provided EXAONE checkpoint.
