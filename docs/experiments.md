# Experiments

## Experiment Categories

| Category | Evidence | Status | Public handling |
|---|---|---|---|
| GPTQ schema sweep | Local workspace, DACON memo exports, notes, public repo artifacts | Verified as attempted | Describe broad W4/W8, activation, block/group, calibration, KV, and target-layer sweeps; do not imply official score unless mapped |
| AWQ schema sweep | Local workspace, DACON memo exports, notes | Verified as attempted | Describe W4/W8, tensor, sequence-length, ignore-layer, and target-module variants |
| SmoothQuant combinations | Local notebooks, public vLLM fork commits, notes | Verified as attempted | Attribute vLLM upstream separately; describe SQ+GPTQ and SQ+AWQ combinations as experiments |
| OmniQuant adaptation | Reviewed public docs and local context | Implementation evidence reviewed | Keep patch/reference level only; distinguish LET/LWC and runtime adaptation from upstream OmniQuant ownership |
| AutoRound / RTN / HQQ / GGUF-GGML | Local notebooks and notes | Verified as attempted | Describe as alternative quantization tracks |
| FP8 / NVFP4 / INT4 / INT8 | Local scripts, notebooks, submission memos | Verified as attempted | Describe as advanced-format and integer quantization exploration |
| FP16 skip / mixed precision | Public repo reference, notes | Partial | Describe as attempted variant |
| LoRA fine-tuning | Local folder, notes | Partial | Document workflow; exclude raw data |
| Data construction and formatting | Local fine-tuning scripts, generated dataset filenames, notes | Partial | Document categories and transformations; exclude raw JSONL/ZIP data |
| Local benchmark | Scripts, docs, logs where available | Partial | Use only for local comparison |
| Failed or low-score paths | Internal notes/PDF | Internal record | Include as lessons learned |

## Observed Experiment Variants

The following variants were observed in local notebooks, scripts, and internal submission memo exports. They are not official competition-score claims.

| Track | Observed variants |
|---|---|
| GPTQ | W4A16, W8A16, W8A8, W4A8, block-size variants such as BLK32/BLK64, calibration-size variants such as CS256/CS512, KV-related variants, front-layer and late-layer mixed precision, module-specific precision such as `down_proj`/`o_proj` protection |
| AWQ | W4A16, W8A16, W4A8, W8A8, tensor variant, sequence-length 256/512 variants, prime variants, `lm_head`/`embed_tokens` exclusion, target-module mapping experiments |
| SmoothQuant combinations | SQ+GPTQ W4A16/W8A16/W8A8, SQ+AWQ W4A16/W8A16-style notebooks, EXAONE layer-map compatibility investigation |
| OmniQuant | W4A16-style local notebook, W4A4/W4A8/W6A6-style public documentation, LET/LWC, group size, learning-rate, AMP, activation scale/shift, calibration dataset, and packed-format/runtime compatibility investigation |
| Other quantization methods | AutoRound W4A16 and KV variants, RTN W4A16, RTN-XK W4A16, HQQ W4A16/W8A16, GGUF/GGML Q4, SpinQuant+INT4, INT4/INT8, FP8, FP8 block/dynamic, NVFP4 |
| Fine-tuning | LoRA notebooks, baseline fine-tuning notebook, data collection/merge scripts, DeepSeek-style conversion scripts, GSM8K/MMLU/KMMLU-oriented data scripts, token-length filtering, phase-based dataset construction |
| Evaluation | `lm-eval` task runs, normalized speed/performance scoring scripts, throughput and token-latency scripts, HF and vLLM evaluation paths |

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
