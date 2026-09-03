# vLLM Omni Activation-Real Runtime Patch

## Purpose

This patch preserves the focused vLLM runtime work required to load the EXAONE OmniQuant activation-real format. It is not a vendored vLLM distribution and does not include a wheel.

## Source and Base

| Item | Value |
|---|---|
| Upstream base | [vLLM `v0.14.1`](https://github.com/vllm-project/vllm/tree/v0.14.1), commit `d7de043d55d1dd629554467e23874097e1c48993` |
| Consolidated source snapshot | `leegongman/CP` commit `ea6bcaba99703fbe82c09ddb84b408001b43eadb` |
| Related wheel archive | `leegongman/vLLM_OmniQuant` commit `7c96a51b71d62c6a19fd99b8a06c59d932df6566`; wheel parts are excluded |
| License | Apache-2.0; see [`../licenses/vLLM-Apache-2.0.txt`](../licenses/vLLM-Apache-2.0.txt) |

## Included Files

[`0001-v0.14.1-omni-activation-real-runtime.patch`](0001-v0.14.1-omni-activation-real-runtime.patch) modifies or adds only these vLLM files:

- `vllm/model_executor/layers/quantization/__init__.py`
- `vllm/model_executor/layers/quantization/omni_activation_real.py`
- `vllm/model_executor/layers/quantization/utils/omni_cutlass_utils.py`
- `vllm/model_executor/layers/quantization/utils/omni_triton_utils.py`

The delta covers quantization registration, packed 4-bit/6-bit handling, EXAONE linear-module loading, and runtime helper paths.

## Applying

```bash
patch -p1 < patches/vllm-omni-activation-real/0001-v0.14.1-omni-activation-real-runtime.patch
```

Use an independently built compatible vLLM environment. The former split wheel, model checkpoint, calibration data, and local performance results are not distributed here.
