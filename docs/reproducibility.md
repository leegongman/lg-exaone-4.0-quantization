# Reproducibility

## Scope

This clean repository is designed to reproduce the project narrative, selected methodology, and documentation structure. It is not intended to fully reproduce the private competition submission environment.

## Reproducible Items

The repository may provide:

- Environment notes
- Public source references
- A cleaned synthetic-prompt local vLLM timing utility: [`scripts/benchmark_vllm_tpt.py`](../scripts/benchmark_vllm_tpt.py)
- Experiment tables
- Focused, license-attributed patches for reviewed EXAONE OmniQuant adaptation and vLLM customization
- Methodology and postmortem notes

## Non-Reproducible Items

The following cannot be reproduced from this public repository alone:

- Official competition scoring
- Private evaluation dataset behavior
- Organizer runtime behavior
- Exact final submission artifacts
- Private checkpoint or wheel submissions

## Excluded Artifacts

The following are intentionally excluded:

- DACON tokens and credential-bearing files
- Private data
- Generated JSONL datasets
- Dataset archives
- Model checkpoints
- Quantized checkpoints
- Tokenizer/model artifact directories
- Wheel files
- Full vLLM source tree
- Full OmniQuant source tree
- Notebook outputs
- Raw Notion exports
- Large caches and logs

## Environment Notes

Exact package versions should be documented only when verified from reviewed setup files or reproducible environment records.

Known tool families involved in the project include:

- PyTorch
- Transformers
- vLLM
- lm-eval
- quantization tooling for GPTQ/AWQ-style experiments
- LoRA/fine-tuning utilities

The `environment.md` file in this draft is a reference note only. It is not an install lockfile and should be refined with verified pinned versions before claiming one-command reproducibility.

## Patch Reproduction Boundary

The patch packages under [`patches/`](../patches/README.md) are intended to preserve implementation deltas, not to provide an installable runtime. Each adjacent README names its exact upstream base revision, source snapshot, applicable upstream license, and excluded artifacts. Applying a patch requires a separately prepared compatible upstream checkout, dependencies, and a user-provided EXAONE model artifact.

## Local vLLM Timing

The included timing utility measures one model at a time with deterministic synthetic prompts after model load and warmup. Its JSON report is explicitly marked `local-benchmark`. It does not use competition data, organizer runtime logic, or an official score formula.

For a meaningful local comparison, run both models separately with the same GPU, vLLM version, prompt-length settings, maximum generation length, and seed. Save reports under `outputs/`, which is excluded from Git.
