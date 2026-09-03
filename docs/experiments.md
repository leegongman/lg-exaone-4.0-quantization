# Experiments

## Evidence Update

The full Notion export adds a broader experiment record than the initial cleanup pass. The exported `Checkpoints` database contains 69 checkpoint/submission labels: 57 submission-marked records, 11 failed records, and 1 pending/waiting record. The labels are listed in `notion-export-inventory.md`.

The same export includes 104 technical-note rows, 30 meeting-note rows, AWQ tracking tables, and attached notebooks/scripts/config files. These are used to classify experiment scope, not to publish raw competition scores.

## Experiment Categories

| Category | Evidence | Status | Public handling |
|---|---|---|---|
| GPTQ schema sweep | Full Notion export, local workspace, DACON memo exports, notes, public repo artifacts | Verified as attempted | Describe broad W4/W8, activation, block/group, calibration, sparse, KV, and target-layer sweeps; do not imply official score unless mapped |
| AWQ schema sweep | Full Notion export, local workspace, DACON memo exports, notes | Verified as attempted | Describe W4/W8, tensor, sequence-length, config-group, ignore-layer, target-module, and internal-parameter variants |
| SmoothQuant combinations | Full Notion export, local notebooks, public vLLM fork commits, notes | Verified as attempted | Attribute vLLM upstream separately; describe SQ+GPTQ, SQ+AWQ, SQ+INT8/FP8, and Pre-Identity combinations as experiments |
| OmniQuant adaptation | Reviewed public docs and local context | Implementation evidence reviewed | Keep patch/reference level only; distinguish LET/LWC and runtime adaptation from upstream OmniQuant ownership |
| AutoRound / RTN / HQQ / GGUF-GGML | Full Notion export, local notebooks and notes | Verified as attempted or exploratory | Describe as alternative quantization tracks |
| FP8 / NVFP4 / INT4 / INT8 | Full Notion export, local scripts, notebooks, submission memos | Verified as attempted | Describe as advanced-format and integer quantization exploration |
| SqueezeLLM / TurboQuant / ZeroQuant / OWQ / SpQR / Slim-LLM | Full Notion export | Reviewed or exploratory | Describe as reviewed/benchmarked only where supported; avoid claiming all were implemented |
| FP16 skip / mixed precision | Public repo reference, notes, checkpoint labels | Partial | Describe as attempted variant |
| LoRA fine-tuning | Full Notion export, local folder, notes | Partial | Document workflow; exclude raw data |
| Knowledge distillation / layer drop | Full Notion export, local notebooks, checkpoint labels | Partial | Document as attempted compression path; exclude raw notebooks/data |
| Data construction and formatting | Full Notion export, local fine-tuning scripts, generated dataset filenames, notes | Partial | Document categories and transformations; exclude raw JSONL/ZIP data |
| Local benchmark | Scripts, docs, logs where available | Partial | Use only for local comparison |
| Failed or low-score paths | Internal notes/PDF | Internal record | Include as lessons learned |

## Observed Experiment Variants

The following variants were observed in local notebooks, scripts, and internal submission memo exports. They are not official competition-score claims.

| Track | Observed variants |
|---|---|
| GPTQ | W4A16, W8A16, W8A8, W4A8, BLK32/BLK64, weight group-size variants, input-activation tensor/channel/token variants, static/dynamic activation handling, calibration-size variants such as CS2/CS256/CS512, calibration-set variants, sparse 2:4, KV-related variants, front-layer and late-layer mixed precision, module-specific precision such as `down_proj`/`o_proj` protection |
| AWQ | W4A16, W8A16, W4A8, W8A4, W8A8, tensor variant, sequence-length 256/512 variants, prime variants, W4/W8 mixed recipes, W4+FP16 recipes, config-group recipes, `lm_head`/`embed_tokens` exclusion, attention/MLP target-module mapping experiments |
| SmoothQuant combinations | SQ+GPTQ W4A16/W8A16/W8A8, SQ+AWQ W4A16/W8A16-style notebooks, SQ+INT8/FP8, Pre-Identity plus SmoothQuant, EXAONE layer-map compatibility investigation |
| OmniQuant | W4A16-style local notebook, W4A4/W4A8/W6A6-style public documentation, LET/LWC, group size, learning-rate, AMP, activation scale/shift, calibration dataset, and packed-format/runtime compatibility investigation |
| Other quantization methods | AutoRound W4A16 and KV variants, RTN W4A16, RTN-XK W4A16, HQQ W4A16/W8A16, GGUF/GGML Q4, SpinQuant+INT4, QuaRot, SqueezeLLM, TurboQuant, ZeroQuant, OWQ, SpQR, Slim-LLM, INT4/INT8, FP8, FP8 block/dynamic, NVFP4 |
| Fine-tuning | LoRA notebooks, baseline fine-tuning notebook, short dataset, stacked-train attempts, deep dataset variants, CoT generation, phase-set variants, data collection/merge scripts, DeepSeek-style conversion scripts, GSM8K/MMLU/KMMLU-oriented data scripts, token-length filtering, phase-based dataset construction |
| Knowledge distillation | Block distillation, teacher/student training setup, KD on layer-dropped model, drop-last/layer-drop variants |
| Evaluation | `lm-eval` task runs, normalized speed/performance scoring scripts, throughput and token-latency scripts, HF and vLLM evaluation paths |

## Internal Checkpoint Status Policy

The checkpoint labels include `Status`, `Score`, runtime, work-date, and submission-date fields in the Notion export. In this public repo:

- checkpoint names may be listed as internal evidence
- submitted, failed, and pending statuses may be summarized
- numeric score/runtime fields should remain internal until cleaned and contextualized
- competition-score claims require official leaderboard or organizer evidence
- local benchmark tables require environment, task, hardware, and script context

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
