# Methodology

## Model Analysis

The project began with structural analysis of `EXAONE-4.0-1.2B`. Internal notes indicate that the model did not map cleanly to several Llama-oriented quantization assumptions.

Important model-level considerations included:

- Post-LN style behavior
- QK RMSNorm-related handling
- Layer-wise outlier and activation behavior
- Tied embedding considerations
- Compatibility with vLLM model loading and quantized checkpoint formats

## Quantization Tracks

The explored quantization tracks included broad schema and hyperparameter sweeps, not only one-off baseline runs.

- GPTQ baseline experiments
- AWQ baseline experiments
- SmoothQuant-style scaling attempts
- OmniQuant adaptation attempts
- AutoRound, INT, FP8, NVFP4, RTN, HQQ, GGUF/GGML, and lower-precision exploration
- FP16 skip or mixed-precision ideas

These tracks should be described as experiments unless there is direct evidence linking a specific method to an official competition score.

Detailed classification is maintained in `technical-inventory.md`.

## Sweep Axes

The full Notion export indicates that the main quantization work varied multiple axes at the same time:

- weight and activation bit-widths, including W4A16, W8A16, W8A8, W4A8, W8A4, INT8, FP8, FP8 block/dynamic, and NVFP4
- block size, group size, calibration size, calibration-set version, sequence length, and input-activation quantization granularity
- attention and MLP target modules, including Q/K/V/O and gate/up/down projections
- protected or excluded modules such as `lm_head`, embeddings, and selected late/front layers
- mixed-precision recipes such as W4/W8 module mixes, FP16 skip, and drop-last/layer-drop variants
- sparse 2:4, rounding, and alternative low-bit method reviews

This repository should describe those axes as explored design space. It should not rank individual variants by score unless the score source is labeled and safe to publish.

## Calibration And Evaluation Loop

The project used calibration and local-evaluation loops to decide which variants were worth submitting or investigating further. Internal notes mention multiple calibration-set versions, benchmark-derived calibration data, token-length filtering, sample-count changes, and local score-estimation notebooks.

The public documentation separates:

- calibration data construction from private/raw dataset release
- local benchmark estimates from competition scores
- failed or inconclusive variants from successful submission claims

## Fine-Tuning Track

LoRA fine-tuning, data preprocessing, and knowledge-distillation-style compression were explored as auxiliary paths. Internal records mention open data preparation, dataset-format conversion, DeepSeek-style formatting, CoT data generation, phase-specific datasets, block distillation, and KD on layer-dropped models.

The public repository should document the workflow and lessons, but should not redistribute:

- Raw training datasets
- Generated JSONL files
- Private or competition-derived data
- Notebook outputs

## vLLM Customization

The project included vLLM-side customization experiments for EXAONE compatibility and quantization-aware runtime behavior.

Reviewed public evidence includes:

- EXAONE custom model registration work in a vLLM fork
- SmoothQuant-style model integration experiments
- OmniQuant/vLLM adaptation documentation
- Runtime and wheel-build notes in reviewed repositories
- Custom wheel packaging and submission-path investigation
- editable-install and build-environment notes from local/Notion records

Full upstream vLLM and OmniQuant source trees are intentionally excluded from the clean repository. Where useful, the public repo should keep patch-level references or links to reviewed commits/docs.

## Attribution Boundary

Original project claims should be limited to:

- Analysis and experiment design
- EXAONE-specific adaptation work
- Integration and compatibility investigation
- Selected implementation changes supported by reviewed evidence
- Reproducibility and postmortem documentation

The repository should not claim ownership of upstream vLLM, OmniQuant, or other external research implementations.
