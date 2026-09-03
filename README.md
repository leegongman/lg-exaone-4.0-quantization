<p align="center">
  <a href="https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-1.2B">
    <img src="https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-1.2B/resolve/main/assets/EXAONE_Symbol%2BBI_3d.png" alt="EXAONE" width="420">
  </a>
</p>

# LG AI Research: EXAONE 4.0 1.2B Quantization and vLLM Optimization

A competition-driven study of how to quantize, adapt, and serve [EXAONE 4.0 1.2B](https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-1.2B) under private evaluation constraints.

The project began as model-side compression work and expanded into runtime engineering. Across Phase 2 and the Phase 3 final-stage track, the central problem was to preserve useful output quality while reducing model size and inference cost, then make the resulting artifact work in a vLLM evaluation environment.

> **Evidence boundary.** Competition rank values and the numeric score weightings below come from internal postmortem records and are not presented as organizer-verified results. Local benchmarks are not competition scores. See [docs/project-status.md](docs/project-status.md) for the claim table.

## Competition Progress

| Stage | Result |
|---|---|
| **Phase 1** | **PASS** |
| **Phase 2** | **19 / 628 - Top 4%** |
| **Phase 3 (Final Stage)** | **ADVANCED TO FINAL STAGE - 25 participating teams** |

## Competition Overview

The project was an `EXAONE-4.0-1.2B` lightweight-optimization challenge. The technical objective was to balance answer quality, inference speed, model size, and vLLM compatibility under a private evaluator.

| Phase | Engineering focus | Submission surface | Internal score-weight record |
|---|---|---|---|
| **Phase 1** | Learn LLM quantization, fine-tuning, RLHF, knowledge distillation, pruning, and evaluation foundations. | Learning and qualification stage | Not applicable |
| **Phase 2** | Produce a vLLM-compatible optimized checkpoint. | Quantized/optimized checkpoint | `5 x accuracy + 5 x speed` |
| **Phase 3 (Final Stage)** | Optimize the checkpoint and customize the inference engine/model path. | Quantized checkpoint + custom vLLM wheel/model-engine artifact | `60 x accuracy + 20 x speed + 20 x model size` |

The Phase 2/3 formulas are **unverified internal evaluation records** from the supplied postmortem PDF, not organizer-verified public formulas. The rule distinction and evidence boundary are expanded in [docs/competition-overview.md](docs/competition-overview.md).

## 1. Target Model: EXAONE 4.0 1.2B

- **Model card:** [LGAI-EXAONE/EXAONE-4.0-1.2B](https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-1.2B)
- **Architecture family:** `Exaone4ForCausalLM`
- **Inference shape recorded in a reviewed experiment configuration:** 30 decoder blocks, hidden size 2048, intermediate size 4096, 32 query heads, 8 KV heads, 64-dimensional heads, 65,536-token context, SiLU MLP, and tied embeddings.

| Component | EXAONE-specific implication for this project |
|---|---|
| QK RMSNorm | Q/K projection handling and layer mapping could not be assumed to follow Llama-oriented paths unchanged. |
| 32 query heads / 8 KV heads | Grouped-query attention and KV-related quantization experiments had to respect the actual projection layout. |
| 30 decoder blocks | Quantization decisions could be made by block index rather than uniformly for every layer. |
| Tied embeddings | `embed_tokens` and `lm_head` needed explicit protection/exclusion decisions; duplicating this weight was a size risk. |
| Post-norm path | SmoothQuant-style integration had to preserve EXAONE's decoder behavior and checkpoint naming/loading rules. |

### Model Structure and Quantization Targets

```mermaid
flowchart TD
    TOK[Input token IDs] --> EMB[Token embeddings]
    EMB --> L0

    subgraph STACK[30 EXAONE decoder blocks: model.layers.0 ... model.layers.29]
        direction TB
        L0[Hidden states] --> QKV[Self-attention: q_proj / k_proj / v_proj]
        L0 -. Phase 3 custom path .-> PREA[pre_attention_identity / IdentityWithParam]
        PREA --> QKV
        QKV --> QKN[QK RMSNorm]
        QKN --> ATTN[Attention and o_proj]
        ATTN --> PA[Post-attention norm and residual]
        PA --> MLP[MLP: gate_proj / up_proj / down_proj]
        PA -. Phase 3 custom path .-> PREF[pre_feedforward_identity / IdentityWithParam]
        PREF --> MLP
        MLP --> PF[Post-feedforward norm and residual]
    end

    PF --> FN[Final RMSNorm]
    FN --> HEAD[Tied LM head]
    HEAD --> OUT[Next-token logits]

    QKV -. layer-specific target map .-> QGROUP[W4 / W8 / protected config groups]
    MLP -. layer-specific target map .-> QGROUP
```

The main model-side customization was **not** a claim that EXAONE's base 30-block architecture was replaced. Instead, experiment configurations selected individual `model.layers.<index>` attention and MLP projections for W4, W8, or protection rules. The Phase 3 custom vLLM path added parameterized hidden-state scaling modules before attention and MLP computation; it did not add extra Transformer decoder blocks.

More precise architecture and loader details are in [docs/model-architecture.md](docs/model-architecture.md).

## 2. Common Quantization and Compression Experiments

The project explored a broad search space rather than a single PTQ recipe. The table separates methods with direct local/Notion evidence of attempts from techniques that were researched or feasibility-checked during the Phase 3 search.

### Attempted or Locally Documented Tracks

| Method family | Schemes, configurations, or combinations documented in the project |
|---|---|
| **GPTQ** | W2/W4/W8 mixes; W4A16, W8A16, W8A8, W4A8; BLK32/BLK64; group-size sweeps; calibration-set/size sweeps; input-activation tensor/channel/token variants; sparse 2:4; KV-related variants; front/late-layer and selective-module precision; `down_proj`/`o_proj` protection; W4A16 late-W8 mixes. |
| **AWQ** | W4A16, W8A16, W4A8, W8A4, W8A8; tensor variants; sequence lengths 256/512; prime variants; all-layer vs selective-layer recipes; W4/W8 and W4+FP16 mixes; target-module maps; config groups; `lm_head`/`embed_tokens` exclusions; calibration and internal-parameter sweeps. |
| **SmoothQuant and pre-identity paths** | SQ + GPTQ W4A16/W8A16/W8A8, SQ + AWQ, SQ + INT8/FP8, and Pre-Identity scaling. These combinations required EXAONE layer-map and vLLM compatibility work. |
| **OmniQuant** | EXAONE/vLLM adaptation around LET, LWC, W4A16 local work, W4A4/W4A8/W6A6-oriented configurations, calibration, group size, learning rate, AMP behavior, activation scale/shift, packed representation, and runtime compatibility. |
| **Rounding and low-bit methods** | AutoRound W4A16/W8A16 and KV variants; RTN W4A16; RTN-XK W4A16; HQQ W4A16/W8A16; GGUF Q4; GGML Q4; direct INT4 and INT8 tracks. |
| **Float and hybrid numeric formats** | PTQ FP8, FP8 dynamic, FP8 block, AWQ FP8 block, NVFP4, FP8/INT8 layer-ratio sweeps, and attention-vs-MLP format allocation. |
| **Rotation, sparsity, and structural compression** | SpinQuant + INT4; sparse 2:4 for GPTQ/INT8/FP8; FP16 skip; W4/W8 layer/module mixes; protected modules; layer drop/drop-last; and INT8/FP8 layer-drop variants. |
| **Post-training recovery** | LoRA, baseline fine-tuning, phase/curriculum data, CoT data, LoRA + INT8, block distillation, and knowledge distillation on layer-dropped models. |

### Researched or Feasibility-Checked Tracks

The Phase 3 investigation additionally covered ZeroQuant, OWQ, SpQR, SqueezeLLM, Slim-LLM, TurboQuant, QuaRot, SpinQuant, DuQuant, PrefixQuant, LRQuant, QuIP, VPTQ, PT2-LLM, OneBit, MPPQ, and APTQ. Local artifacts additionally record SpinQuant-INT4, a SqueezeLLM 4-bit CUDA-kernel evaluation path, and a TurboQuant research track. The remaining methods are deliberately described as research or feasibility checks unless there is direct implementation evidence; they are not all claimed as completed submission methods.

Detailed method evidence, including failed and partial work, is maintained in [docs/technical-inventory.md](docs/technical-inventory.md) and [docs/experiments.md](docs/experiments.md).

## 3. Fine-Tuning, Data Construction, and Distillation

Fine-tuning was an independent quality-recovery track, not a single LoRA trial. It was used to test whether output quality lost during compression could be recovered before reapplying quantization.

### LoRA and Training Variants

| Area | Documented work |
|---|---|
| LoRA target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj` |
| Adapter configuration | Internal LoRA notes record `r=16`, `alpha=16`, and dropout `0.05`; these are experiment settings, not a recommended universal recipe. |
| Training modes | Baseline fine-tuning, sequential curriculum training, stacked training, short-dataset variants, deep-dataset variants, and phase-set variants. |
| Curriculum path | `phase_00 -> phase_05 -> phase_10`, moving from short/simple QA through intermediate reasoning toward longer reasoning and answer formats. |
| Data construction | Public-data collection/merge scripts, Korean/English balancing, subject-category splits, DeepSeek-style answer-format conversion, QA/CoT/multi-turn mixes, and token-length filtering. |
| Quantization combination | LoRA + INT8, deep-dataset v2 + INT8, and deep-dataset v2 + CoT checkpoint labels are recorded as experiment attempts. |

The phase datasets used multiple scale/composition variants, including sequential 1,200/1,000/1,000-sample stages, expanded 1,800/1,400/1,400 stages, and a stacked 12,000-sample path. These are internal experiment settings only; raw JSONL data and notebook outputs are intentionally excluded.

### Distillation and Layer Compression

- **Block distillation:** attempted to replace the final two decoder layers with a distilled single layer.
- **Knowledge distillation:** teacher/student and logit/hidden-state supervision paths were explored.
- **KD on a layer-dropped model:** tested compression after removing decoder capacity.
- **Fine-tune then quantize:** treated fine-tuning as a candidate recovery step before INT8 or other quantization paths.

This work is reported as partial/attempted. Local score estimates, dataset contents, and failed checkpoints are not presented as official competition performance.

## 4. Phase 2: Checkpoint Optimization Under Organizer Runtime

The official [Phase 2 competition page](https://dacon.io/competitions/official/236673/overview/description) frames the task as optimizing `EXAONE-4.0-1.2B` for lightweight inference and submitting final model weights for hidden evaluation. In this project, Phase 2 was the checkpoint-focused stage: the model had to be compatible with the organizer's vLLM-based runtime, but a participant-customized runtime wheel was not part of the documented submission surface.

| Item | Phase 2 record |
|---|---|
| Submission artifact | Quantized or optimized checkpoint |
| Runtime constraint | Must load and infer in the organizer vLLM environment; runtime customization was not part of this project's Phase 2 path |
| End-to-end time limit | Internal postmortem records a 20-minute limit covering model load, inference, and result parsing |
| Speed measurement | Internal postmortem records per-token inference time on the benchmark dataset, rather than total setup time, as the speed signal reflected in scoring |
| Score weighting | **Unverified internal evaluation record:** `5 x accuracy + 5 x speed` |

### Phase 2 Technical Work

- Started from GPTQ baselines, then broadened to GPTQ schema/hyperparameter sweeps and AWQ.
- Analyzed outliers and layer-wise min/max/mean behavior; late attention outlier candidates motivated selective FP16 or mixed-precision attempts.
- Tested calibration data, sequence length, activation handling, group/block size, protected modules, and target-layer maps.
- Added LoRA fine-tuning, data-format normalization, and knowledge-distillation paths once quantization alone appeared insufficient.
- Compared local `lm-eval`-style quality signals and latency-oriented measurements, while documenting that they did not reliably predict the private evaluation result.

## 5. Phase 3 (Final Stage): Custom vLLM Engine and Model-Path Work

The official [Phase 3 competition page](https://dacon.io/competitions/official/236689/overview/description) adds a model-engine submission surface, including a vLLM wheel, and includes code review. That changed the work from checkpoint-only optimization to checkpoint-plus-runtime integration.

| Item | Phase 3 record |
|---|---|
| Submission artifact | Quantized checkpoint plus a custom vLLM wheel/model-engine artifact |
| End-to-end time limit | Internal postmortem records a 30-minute limit covering model load, inference, and result parsing |
| Speed condition | Internal postmortem records a timeout condition during benchmark inference |
| Score weighting | **Unverified internal evaluation record:** `60 x accuracy + 20 x speed + 20 x model size` |
| Competition progression | Internal project record describes earlier-phase passage and participation in the Phase 3 final-stage track; it is not a verified prize or leaderboard claim. |

### vLLM Customization: Model Registration and Hidden-State Modules

The custom runtime work went beyond an external vLLM invocation. Reviewed project-fork commits and the corresponding Notion code export show the following EXAONE-specific path:

1. Registered `Exaone4ForCausalLMSQ` as a vLLM model path, with checkpoint configuration expected to identify that architecture.
2. Created `Exaone4DecoderLayerWithPreIdentity`, a custom decoder-layer implementation for the SmoothQuant path.
3. Added an `IdentityWithParam` module with a learnable per-channel `smooth_factor` of length `hidden_size`.
4. Applied `pre_attention_identity` before self-attention and `pre_feedforward_identity` before the MLP in each custom decoder block.
5. Preserved the post-attention and post-feedforward norms while handling the original checkpoint's `input_layernorm` weights according to the custom loader path.
6. Investigated EXAONE registration and LET/LWC-oriented OmniQuant adaptation, vLLM quantization-config parsing, and CUDA-kernel/config work as internal implementation efforts.

The critical distinction is architectural: these are **new hidden-state scaling modules inside the custom decoder path**, not extra Transformer blocks or an unsupported claim of ownership of vLLM/SmoothQuant. The implementation evidence is linked in [docs/source-map.md](docs/source-map.md); the full vLLM tree is intentionally excluded.

## Repository Guide

```text
docs/          Competition rules, architecture, experiments, evidence, and safety
techniques/    Future method-specific notes with evidence and attribution
configs/       Future sanitized configuration examples
experiments/   Future sanitized manifests and local benchmark summaries
patches/       Small license-reviewed patch references only
scripts/       Future cleaned helpers without private paths, data, or credentials
```

- [Competition overview](docs/competition-overview.md): Phase rules, evaluation boundary, and result terminology.
- [Model architecture](docs/model-architecture.md): EXAONE structure, target modules, and the custom vLLM decoder path.
- [Technical inventory](docs/technical-inventory.md): full technique classification and source levels.
- [Experiments](docs/experiments.md): attempted, partial, failed, local, and official-result policy.
- [Source map](docs/source-map.md): upstream attribution and the public include/exclude manifest.

## Public Safety and Attribution

This repository excludes DACON tokens, private datasets, generated JSONL, checkpoints, tokenizer/model artifacts, submission files, wheels, raw Notion exports, notebook outputs, local caches, and full vLLM/OmniQuant source trees. The DACON token found in the original workspace must be revoked or rotated before any related history is shared.

vLLM, OmniQuant, GPTQ, AWQ, and other methods/frameworks remain attributed to their original authors. This repository claims only the evidence-backed EXAONE-specific analysis, experiment design, runtime integration, selected custom model-path changes, and documentation.
