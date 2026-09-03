# Experiments

## Experiment Categories

| Category | Evidence | Status | Public handling |
|---|---|---|---|
| GPTQ baseline | Local workspace, notes, public repo artifacts | Verified as attempted | Describe method and setup; do not imply official score unless mapped |
| AWQ baseline | Local workspace, notes | Verified as attempted | Describe as baseline exploration |
| SmoothQuant-style EXAONE adaptation | Public vLLM fork commits, notes | Verified implementation evidence | Attribute vLLM upstream separately |
| OmniQuant adaptation | Reviewed public docs and local context | Verified documentation evidence | Keep patch/reference level only |
| NVFP4 / lower precision | Notes and local artifacts | Partial | Describe as exploratory |
| FP16 skip / mixed precision | Public repo reference, notes | Partial | Describe as attempted variant |
| LoRA fine-tuning | Local folder, notes | Partial | Document workflow; exclude raw data |
| Local benchmark | Scripts, docs, logs where available | Partial | Use only for local comparison |
| Failed or low-score paths | Internal notes/PDF | Internal record | Include as lessons learned |

## Local Benchmark Policy

Local benchmark results are useful for comparing variants within the same environment. They should not be presented as official competition scores.

If local numbers are included later, each table should specify:

- Model or checkpoint variant
- Quantization method
- Calibration data, if public and redistributable
- Benchmark task
- Hardware
- vLLM/PyTorch/Transformers versions
- Whether the result is local, internal, or official

## Failed and Partial Experiments

Failed experiments are part of the public technical story because they explain why the project moved between methods.

Examples of useful failure documentation:

- Quantization methods that were vLLM-compatible but accuracy-limited
- Methods that looked promising locally but did not map to competition performance
- Runtime customization paths that were difficult to package reliably
- Benchmark choices that did not predict private evaluation behavior

These should be labeled as failed, partial, or inconclusive rather than reframed as successful outcomes.

