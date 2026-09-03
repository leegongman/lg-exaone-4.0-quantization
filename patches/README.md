# Patches and References

This directory preserves the project's focused EXAONE/vLLM/OmniQuant code deltas. Filtered source-only snapshots are separately retained under [`../sources/`](../sources/README.md).

| Directory | Scope | Base and license |
|---|---|---|
| [`omniquant-exaone/`](omniquant-exaone/) | EXAONE OmniQuant adaptation and legacy initial layer revision | OpenGVLab OmniQuant, MIT |
| [`vllm-omni-activation-real/`](vllm-omni-activation-real/) | vLLM loader/runtime for the Omni activation-real format | vLLM v0.14.1, Apache-2.0 |
| [`vllm-exaone-sq/`](vllm-exaone-sq/) | EXAONE SmoothQuant model path and registry entry | vLLM, Apache-2.0 |
| [`vllm-fp16-skip/`](vllm-fp16-skip/) | Consolidation record and earlier Omni activation-real runtime source | vLLM, Apache-2.0 |

Each patch README names the exact base revision, source snapshot, included files, and exclusions. The license texts are retained under [`licenses/`](licenses/). None of these patches is a complete buildable vLLM or OmniQuant distribution.
