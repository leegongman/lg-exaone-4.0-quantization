# Reproducibility

## Scope

This clean repository is designed to reproduce the project narrative, selected methodology, and documentation structure. It is not intended to fully reproduce the private competition submission environment.

## Reproducible Items

The repository may provide:

- Environment notes
- Public source references
- Selected cleaned evaluation script structure
- Experiment tables
- Patch-level references to reviewed vLLM customization
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
