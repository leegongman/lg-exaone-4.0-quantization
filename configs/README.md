# Public Configuration Examples

This directory contains small, sanitized configuration examples that demonstrate a method or layer-targeting strategy without shipping model artifacts.

## Included Example

- [`quantization-target-schema.example.yaml`](quantization-target-schema.example.yaml): an illustrative EXAONE projection-target and bit-width schema reconstructed from public-safe fields. It is not a checkpoint configuration, final recipe, or verified best-performing setting.

Each future file must identify its source, target model version, method, evidence status, and any omitted private fields. Do not add raw checkpoint `config.json` files, tokenizer files, or submission configurations without a separate safety review.
