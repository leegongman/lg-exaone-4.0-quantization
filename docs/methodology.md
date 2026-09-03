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

The explored quantization tracks included:

- GPTQ baseline experiments
- AWQ baseline experiments
- SmoothQuant-style scaling attempts
- OmniQuant adaptation attempts
- NVFP4 and lower-precision exploration
- FP16 skip or mixed-precision ideas

These tracks should be described as experiments unless there is direct evidence linking a specific method to an official competition score.

## Fine-Tuning Track

LoRA fine-tuning and data preprocessing were explored as an auxiliary path. Internal records mention open data preparation and dataset-format conversion experiments.

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

Full upstream vLLM and OmniQuant source trees are intentionally excluded from the clean repository. Where useful, the public repo should keep patch-level references or links to reviewed commits/docs.

## Attribution Boundary

Original project claims should be limited to:

- Analysis and experiment design
- EXAONE-specific adaptation work
- Integration and compatibility investigation
- Selected implementation changes supported by reviewed evidence
- Reproducibility and postmortem documentation

The repository should not claim ownership of upstream vLLM, OmniQuant, or other external research implementations.

