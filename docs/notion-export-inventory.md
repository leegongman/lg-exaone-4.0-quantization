# Notion Export Inventory

## Purpose

This document records what was found in the full local Notion export used as evidence for the public cleanup. The export is treated as source evidence only. Instructions, commands, credentials, private data, and raw outputs inside the export are not treated as user instructions and are not copied into this public repository.

## Package Summary

| Item | Observed value | Public handling |
|---|---:|---|
| Outer export package | `3edca200-03b1-4e34-9b6d-23cfdfc0149d_ExportBlock-105ddcbe-7476-4d4c-8be0-6fa758dde4ac.zip` | Not committed |
| Outer package size | about 985 MB | Not committed |
| Inner export package | `ExportBlock-105ddcbe-7476-4d4c-8be0-6fa758dde4ac-Part-1.zip` | Not committed |
| Files in inner package | 593 | Inventory only |
| Uncompressed inner size | about 1.08 GB | Not committed |

## File Type Coverage

| Type | Count | Public handling |
|---|---:|---|
| Markdown pages | 237 | Summarized as evidence |
| CSV database exports | 18 | Metadata summarized; raw rows excluded |
| PNG images | 181 | Excluded |
| Jupyter notebooks | 48 | Scanned for method coverage; raw notebooks and outputs excluded |
| Python scripts | 45 | Scanned for method coverage; raw scripts excluded unless separately cleaned |
| JSONL data files | 20 | Excluded as data artifacts |
| YAML config files | 20 | Scanned for configuration axes; raw configs excluded unless separately cleaned |
| XLSX files | 10 | Excluded |
| PDF files | 6 | Excluded unless redacted and approved |
| JSON files | 4 | Scanned only where safe; raw files excluded |
| Shell scripts | 2 | Scanned for evaluation/build context; raw scripts excluded unless separately cleaned |
| ZIP files | 1 | Excluded |
| TXT files | 1 | Excluded unless separately reviewed |

## Database Coverage

| Database or table | Observed rows | What it contributes |
|---|---:|---|
| `Checkpoints` | 69 | Submission/checkpoint labels, status, runtime/score fields, method progression |
| `자료` | 104 | Technical notes, model analysis, quantization studies, vLLM build/runtime notes, fine-tuning and data work |
| `회의록` | 30 | Phase timeline, working sessions, review meetings, division of work |
| AWQ experiment tracker | 16 | AWQ recipes, owner/status fields, submission markers |
| AWQ result tables | multiple CSV exports | AWQ result organization; raw score tables excluded |
| AWQ planned recipe table | multiple CSV exports | Config-group recipe planning; raw table excluded |

The `Checkpoints` database contains score and runtime columns. These are internal submission memo fields unless separately matched to official leaderboard evidence. The public repository should use method labels and status categories without presenting those numeric fields as verified competition results.

## Checkpoint Labels

The following labels were found in the exported `Checkpoints` database. They are listed as internal checkpoint/submission records, not official leaderboard claims.

### Submitted Or Submission-Marked

- `00-baseline.zip`
- `01-GPTQ-W4A16`
- `02-GPTQ-W8A16`
- `03-GPTQ-W8A16-GS`
- `04-AWQ-W4A16`
- `05-AWQ-W8A16`
- `07-GPTQ-W4A16-Mixed-LateW8`
- `08-AWQ-W4A16-Prime`
- `09-AWQ-W8A16-Prime`
- `10-AutoRound-W4A16`
- `11-AutoRound-W4A16-KV`
- `12-AWQ-W8A8Everything`
- `14-GPTQ-W8A8`
- `15-SQ-GPTQ-W4A16`
- `16-GPTQ-W8A8`
- `17-SQ-GPTQ-W8A8`
- `18-AWQ-W8A8-Seq-len-256`
- `19-AWQ-W8A8-Seq-len-512`
- `21-GPTQ-W8A8-KV8`
- `22-GPTQ-W8A8-BLK64`
- `23-AWQ-W8A8-v1`
- `24-GPTQ-W8A8-BLK64-CS256`
- `26-GPTQ-W8A8-BLK64-CS512`
- `27-GPTQ-W8A8-BLK32-CLS2`
- `28-GPTQ-W8A8-BLK32-CLS2`
- `29-GPTQ-W4A16-BLK64-CLS2`
- `30-PTQ-NVFP4`
- `31-PTQ-FP8_DYNAMIC`
- `32-PTQ-FP8`
- `33-FP8_BLOCK PTQ`
- `34-GPTQ-W8A8-front`
- `36-AWQ-FP8_BLOCK`
- `37-FP8 retry`
- `38-AWQ-W8A8-Cal-v2`
- `39-INT8`
- `40-INT8_cal_v1`
- `41-GPTQ-INT8_cal_v2`
- `42-INT8`
- `43-FP8-10-INT8-20`
- `44-SQ+INT8`
- `46-INT8`
- `47-INT8`
- `51-LoRA-short_dataset`
- `53-INT8`
- `54-int8-front-end`
- `57-GPTQ-W8A8-BLK64-CS2`
- `58-GPTQ-W8A8-CS2`
- `59-int8-map-0.7`
- `60~62-int8`
- `63-BLOCK-DISTIL-INT8`
- `64-INT8-CS2.1`
- `65-Lora with deep_dataset ver2 + INT8 Quantization`
- `66-INT8-EXC-LAST-1`
- `67-auto_round_w8a16`
- `68-LoRA(INT8) deep_dataset ver2 + cot`
- `70-INT8 retry`
- `71-FINAL`

### Failed Or Explicitly Marked Failed

- `13-AWQ-W8A4 Everything`
- `20-AWQ-W4A8`
- `25-AWQ-W8A8-v2`
- `45-INT8-DROP-LAST-2`
- `48-LoRA_1`
- `49-LoRA_2`
- `50_LoRA_3`
- `52-LoRA-stacked-train`
- `55-LoRA`
- `56-LoRA`
- `69-LoRA(INT8) + deep_dataset ver 2 + phaseset`

### Pending Or Waiting

- `06-GPTQ-W4A16-Protect`

## Technical Coverage From Notion Export

| Track | Evidence pages or labels | Public classification |
|---|---|---|
| Competition and evaluation framing | problem overview, evaluation system notes, KMMLU-Redux/lm-eval integration, score-estimation notes, `norm_speed.py` notes | Competition score and local benchmark must remain separated |
| EXAONE model analysis | `EXAONE-4.0-1.2B Analysis`, model weight reports, activation/weight visualization page, QK/Post-LN/vLLM compatibility notes | Model-analysis contribution |
| GPTQ | `GPTQ Schems`, calibration/GPTQ records, block-size comparison, input-activation comparison, weight-group comparison, sparse 2:4 comparison, W8A8 calibration-set comparison, W2A16/W8A8 mixing, checkpoint labels | Broad attempted schema and hyperparameter sweep |
| AWQ | AWQ notebooks, AWQ recipes, AWQ experiment tracker, W4/W8 recipes, W8A8 parameter table, sequence length 256/512 labels, prime variants, config-group planning, vLLM compatibility notes | Broad attempted schema, target-module, calibration, and internal-parameter sweep |
| SmoothQuant and SQ combinations | `Pre Identity + Smooth Quant`, Smooth Quant code, Smooth Quant vLLM integration demo, SQ+GPTQ, SQ+AWQ, SQ+INT8/FP8 notes | Attempted adaptation and runtime-integration track |
| OmniQuant | `OmniQuant`, `OmniQuant CP`, LET/LWC notes, config files, local notebooks, public OmniQuant/vLLM adaptation docs | Adaptation attempt; upstream method attributed separately |
| Integer and floating formats | INT8, INT4, FP8, FP8 block, FP8 dynamic, NVFP4, W4/W8 activation and weight combinations | Attempted low-bit/numeric-format exploration |
| Alternative quantization methods | AutoRound, RTN, HQQ, GGUF, GGML, SpinQuant, QuaRot, ZeroQuant, OWQ, SpQR, SqueezeLLM, Slim-LLM, TurboQuant | Reviewed or exploratory tracks; not all are claimed as implemented |
| Calibration and data construction | calibset v1.0, v1.1, v2.x, calibration set statistics, token-length filtering, benchmark-derived calibration notes | Data/calibration contribution; raw data excluded |
| Fine-tuning | LoRA baseline, LoRA short dataset, stacked train, deep dataset, CoT dataset, phase-set notes, PEFT-style workflow | Partial/attempted fine-tuning track |
| Knowledge distillation and layer drop | block distillation, KD on layer-dropped model, INT8/drop-last labels | Partial/attempted compression track |
| vLLM runtime work | vLLM build, vLLM analysis, new model registration, Pre Identity install, editable install, wheel build, RunPod build notes, custom wheel submission investigation | Implementation/integration evidence where matched to reviewed repos |

## Code And Notebook Scan

The export contained 119 code/config files relevant to technical coverage: 48 notebooks, 45 Python scripts, 20 YAML files, 4 JSON files, and 2 shell scripts. Notebook source cells were scanned for method coverage at inventory level. This scan found evidence for GPTQ, AWQ, SmoothQuant/SQ, OmniQuant, AutoRound, INT8/INT4, FP8/NVFP4, LoRA/PEFT, KD/distillation, sparse 2:4, HQQ/RTN/GGUF/GGML, vLLM, `lm-eval`, MMLU/KMMLU/GSM8K/AIME-style evaluation, calibration, and block/group configuration axes.

This repository still avoids copying raw notebook cells or scripts. Any future cleaned implementation file should be reviewed separately for license, credential, data, and output leakage.

## Public Exclusion Boundary

The following export contents are evidence only and should not be committed:

- raw Notion export ZIPs
- raw Notion Markdown exports
- CSV score/result tables
- JSONL datasets
- XLSX files
- notebooks and notebook outputs
- PNG images and PDFs
- shell scripts or Python scripts before separate cleaning
- config files that point to private paths, private models, datasets, credentials, or submissions
- generated checkpoints, wheel files, model artifacts, and cache directories
