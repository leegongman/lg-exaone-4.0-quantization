# Technical Inventory

This document expands the technical classification behind the project. It separates broad literature review, project-attempted methods, local evaluation, and excluded artifacts.

## Evidence Boundary

| Source | Use in this file | Public handling |
|---|---|---|
| Full local Notion export | Quantization taxonomy, checkpoint labels, technical notes, meeting timeline, attached code/notebook/config inventory | Summarized; raw export, assets, data, scores, and notebook outputs are not committed |
| Notion export: `Categories of Quantization Techniques` | Quantization taxonomy and initial filtering categories | Summarized as one part of the larger Notion export |
| Local workspace file inventory | Evidence of attempted notebooks/scripts | File names and method categories only |
| Internal DACON submission memo exports | Evidence of attempted schema variants | Memo labels only; raw JSON, scores, files, and tokens are excluded |
| Reviewed GitHub repositories | vLLM/OmniQuant adaptation evidence | Linked in `source-map.md` |
| User clarification | Broad GPTQ/AWQ/OmniQuant schema and hyperparameter sweep scope | Treated as internal record unless tied to reviewed code |

## Full Notion Export Coverage

The later full Notion export materially expands the evidence base beyond the small quantization-taxonomy export. Its public handling is documented in `notion-export-inventory.md`.

| Evidence group | Observed coverage | Public handling |
|---|---:|---|
| Total files in inner export | 593 | Inventory only |
| Markdown pages | 237 | Summarized as source evidence |
| CSV database exports | 18 | Metadata summarized; raw rows excluded |
| Code/config files scanned | 119 | Method coverage only |
| Notebooks scanned | 48 notebooks, 560 cells | Raw notebooks and outputs excluded |
| `Checkpoints` rows | 69 | Labels/status categories only; score fields are internal records |
| `자료` rows | 104 | Technical coverage and page-title evidence |
| `회의록` rows | 30 | Timeline and phase context |

This review adds explicit evidence for calibration-set evolution, GPTQ internal-parameter comparisons, AWQ recipe/config-group work, SmoothQuant and Pre-Identity integration, LoRA and knowledge-distillation attempts, vLLM build/wheel work, and exploratory methods such as SqueezeLLM and TurboQuant.

## Quantization Taxonomy Reviewed

The attached Notion export classifies quantization methods into the following major families. These are reviewed research categories, not all project-implemented methods.

| Family | Subcategories reviewed | Project relevance |
|---|---|---|
| Better scale and zero-point | quantization loss, task loss | Used as background for scale/zero-point and calibration decisions |
| Metric and mechanism | oscillation reduction, loss function design, Hessian/closed-form mechanisms | Relevant to GPTQ, ZeroQuant-style reasoning, and layer-wise reconstruction |
| Mixed precision | rough allocation, adaptive allocation, search algorithms, alternative search | Relevant to FP16 skip, layer/module-specific precision, W4/W8 combinations |
| Redistribution | bias, distribution uniformization, outlier processing, rounding | Directly relevant to SmoothQuant, OmniQuant, AWQ, AutoRound, SpinQuant, QuaRot-style reasoning |
| Data-free quantization | truly data-free, generated calibration data, adversarial generation | Reviewed but not a primary public claim for this project |
| Advanced formats | float-based formats, fixed-point formats, other formats | Relevant to FP8, NVFP4, GGUF/GGML, QLoRA/LoftQ context |
| Diffusion model quantization | calibration-set, activation distribution, quantization error methods | Reviewed as out-of-scope for this LLM project |
| Other | pruning-aware quantization, sharded/distributed quantization, stochastic bit precision | Background only unless explicitly tied to local files |

Internal notes also indicate a first-pass filter that removed CNN, Vision Transformer, Diffusion Model, and QAT-focused methods from the main LLM competition path.

## Project-Attempted Technical Areas

| Area | Techniques and variants | Claim level |
|---|---|---|
| EXAONE model analysis | Post-LN behavior, QK RMSNorm, tied embedding, layer/module mapping, activation outlier inspection, min/max/mean and layer-flow analysis | Internal notes and local notebooks |
| GPTQ | W4A16, W8A16, W8A8, W4A8, block-size variants, weight group-size variants, input-activation quantization variants, calibration-size variants, calibration-set variants, sparse 2:4 variants, KV-related variants, front-layer variants, late-layer mixed precision, selective module precision such as `down_proj`/`o_proj`, Marlin-kernel-related submission memo | Verified as attempted; not official score claim |
| AWQ | W4A16, W8A16, W4A8, W8A8, W8A4, tensor variant, sequence-length 256/512 variants, prime variants, W4/W8 mixed recipes, W4+FP16 recipes, `lm_head` and `embed_tokens` exclusions, target-module mapping, config-group planning, internal-parameter adjustments | Verified as attempted; not official score claim |
| SmoothQuant combinations | SQ+GPTQ, SQ+AWQ, SQ+INT8/FP8, W4A16/W8A16/W8A8-style variants, Pre-Identity integration, EXAONE layer-map compatibility investigation | Verified as attempted |
| OmniQuant | LET, LWC, learnable scaling/offset framing, W4A16 local notebook, W4A4/W4A8/W6A6 public-doc variants, group-size experiments, learning-rate settings, AMP/deactivated AMP, activation scale/shift generation, calibration dataset changes, packed runtime compatibility, config/checkpoint preparation notes | Implementation evidence reviewed; upstream code attributed separately |
| Integer and low-bit quantization | INT8 baseline, INT4 notebook, W4/W8 activation/weight combinations | Verified as attempted |
| Advanced numeric formats | FP8, FP8 block, FP8 dynamic, NVFP4, HQQ W4A16/W8A16, GGUF/GGML Q4, RTN W4A16, RTN-XK W4A16, SpinQuant+INT4, QuaRot, SqueezeLLM, TurboQuant, ZeroQuant, OWQ, SpQR, Slim-LLM | Verified as attempted, reviewed, or exploratory depending on source |
| AutoRound and rounding methods | AutoRound W4A16, AutoRound W4A16-KV, rounding-method review from Notion taxonomy | Verified as attempted |
| Mixed precision and skip strategies | FP16 skip variant, W4/W8 layer and module mixes, W4+FP16 recipes, protected modules, late-layer higher precision, layer-drop and drop-last variants | Partial / implementation evidence reviewed |
| Fine-tuning | LoRA notebooks, baseline fine-tuning, short dataset, stacked train, phase datasets, DeepSeek-style formatting, deep dataset variants, CoT data generation, GSM8K/MMLU/KMMLU-oriented data scripts, Korean/English data categories, token-length filtering, data merging | Partial; raw data excluded |
| Knowledge distillation | Block distillation, teacher/student setup, KD on layer-dropped model, layer-drop compression path | Partial; raw notebooks and generated data excluded |
| Evaluation and scoring scripts | `lm-eval`, GSM8K, TruthfulQA, AIME, MMLU/KMMLU variants, HellaSwag, ARC Challenge, normalized speed/performance, throughput, token latency, HF/vLLM evaluation paths | Partial; local benchmark only |
| vLLM customization | EXAONE custom model registration, SmoothQuant-style identity/smooth-factor integration, OmniQuant/vLLM runtime compatibility, loader/config investigation, custom wheel path | Implementation evidence reviewed |
| Submission packaging | vLLM-compatible checkpoint packaging, `save_compressed` paths, custom wheel packaging, DACON submission memo tracking | Internal record; raw submission files excluded |

## Technique Classification From Export

| Classification | Included technical items | Evidence examples |
|---|---|---|
| Baseline and evaluation pipeline | baseline checkpoint, `lm-eval`, KMMLU-Redux task integration, normalized speed, runtime columns, score-estimation notes | `00-baseline.zip`, evaluation-system pages, `norm_speed.py` notes |
| Weight-only and weight-activation quantization | W4A16, W8A16, W8A8, W4A8, W8A4, W2/W4/W8 mixes | GPTQ/AWQ checkpoint labels and recipe pages |
| Calibration design | calibset v1.0, v1.1, v2.x, calibration statistics, benchmark-derived calibration, sample count, sequence length, token-length filtering | calibration pages, GPTQ W8A8 calibset comparison, AWQ recipes |
| Module/layer targeting | attention vs MLP targeting, Q/K/V/O projection handling, gate/up/down projection handling, `lm_head`/embedding exclusion, front/back layer mapping | GPTQ/AWQ schema pages and AWQ parameter table |
| Internal quantization hyperparameters | block size, group size, input activation granularity, static/dynamic activation handling, min/max and tensor/channel/token axes | GPTQ block-size/input-activation/weight-group YAML pages |
| Runtime compatibility | vLLM quantization config parsing, custom model registration, EXAONE SmoothQuant model class, loader compatibility, wheel build/install | vLLM analysis/build pages and reviewed public commits/docs |
| Fine-tuning and data | LoRA, PEFT-style workflow, dataset merge, DeepSeek-style conversion, short/deep/phase datasets, CoT generation | LoRA FT, KD/data pages, checkpoint labels |
| Compression beyond PTQ | layer drop, block distillation, KD on dropped layers, sparse 2:4 | block distillation and sparse comparison pages |
| Exploratory research survey | OmniQuant, ZeroQuant, OWQ, SpQR, SqueezeLLM, Slim-LLM, TurboQuant, SpinQuant, QuaRot | survey pages and selected experiment notes |

## GPTQ Sweep Details

The GPTQ track should be described as a broad sweep rather than a single baseline. Evidence includes local notebooks, calibration files, and internal submission memo labels.

Observed axes:

- Bit schemes: W4A16, W8A16, W8A8, W4A8
- Block/group-style variants: BLK32, BLK64, group-size-style runs
- Calibration-size variants: CS256, CS512
- KV-related variants
- Front-layer and late-layer mixed-precision variants
- Selective module precision, including `down_proj` and `o_proj`
- Protected or ignored modules such as `lm_head` and embedding modules
- vLLM/kernel compatibility concerns, including Marlin-kernel-related notes

Public wording should remain: "Broad GPTQ schema and hyperparameter sweeps were attempted." It should not say a GPTQ variant achieved an official score unless a leaderboard-backed mapping is added.

## AWQ Sweep Details

The AWQ track also involved broad schema and internal-parameter changes.

Observed axes:

- Bit schemes: W4A16, W8A16, W4A8, W8A8
- Tensor-level variant
- Sequence-length variants, including 256 and 512
- Prime variants
- All-layer versus selective-layer recipes
- `lm_head` and `embed_tokens` exclusion/protection
- Target-module mapping for attention and MLP projections
- Calibration sample count and calibration data changes

Public wording should remain: "Broad AWQ schema, target-module, and calibration sweeps were attempted." It should not imply that upstream AWQ itself is original work.

## OmniQuant Details

The OmniQuant track should be framed as adaptation and runtime-compatibility work, not ownership of the upstream method.

Observed axes:

- LET and LWC enablement
- Weight/activation bit combinations such as W4A16 locally and W4A4/W4A8/W6A6 in reviewed public docs
- Group-size and optimization hyperparameters
- Learning-rate settings for LET/LWC
- AMP stability and deactivated-AMP runs
- Activation scale/shift generation
- Calibration dataset choices such as wikitext2 and other benchmark-derived data
- Packed low-bit representation and vLLM runtime compatibility investigation
- Custom vLLM wheel path for Phase 3-style submissions

Public wording should remain: "OmniQuant adaptation attempts and vLLM compatibility work were documented and partially implemented, with upstream OmniQuant attributed separately."

## Fine-Tuning and Data Work

Fine-tuning was not just a single LoRA notebook. The local workspace indicates a broader data construction and formatting track.

Observed areas:

- LoRA experiments across multiple notebooks
- Baseline fine-tuning notebook
- Data collection and merging scripts
- DeepSeek-style conversion scripts
- Phase-based dataset construction
- Token-length filtering around 400-500 token ranges
- GSM8K-oriented data construction
- MMLU/KMMLU-oriented scripts
- Korean and English category splits, including math, science, engineering, humanities, social science, pure science, reasoning, and other categories

Raw data, generated JSONL files, dataset ZIP files, and notebook outputs remain excluded from the public repo.

## Evaluation and Benchmarking

Evaluation work should be split from official competition scoring.

Observed local evaluation areas:

- `lm-eval` task use
- GSM8K
- TruthfulQA
- AIME
- MMLU and MMLU-style variants
- KMMLU-style variants
- HellaSwag
- ARC Challenge
- normalized performance and speed scoring
- throughput and token-latency scripts
- HF and vLLM evaluation paths

These are local benchmark or internal-evaluation references only unless official competition evidence is attached.

## Not Publicly Included

The following are evidence sources or artifacts, not public repo contents:

- raw Notion export ZIP and image
- DACON JSON exports, pages, tokens, API scripts, and submission artifacts
- raw JSONL datasets and generated training data
- checkpoint, model, tokenizer, wheel, cache, and notebook-output files
- full upstream vLLM, OmniQuant, AWQ, or other source trees
