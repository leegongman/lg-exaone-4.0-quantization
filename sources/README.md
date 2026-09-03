# Consolidated Source Archive

This directory retains the public-safe source, configuration, and documentation files from the former project repositories so that their code does not disappear when those repositories are retired. Each subdirectory preserves the original repository-relative file paths.

| Source repository | Reviewed commit | Imported files | Imported size |
|---|---|---:|---:|
| `scheme_vLLM_omniquant` | `5178a0e5a1c359dd39b8e2457b16036303ab570d` | 4,548 | 38.9 MiB |
| `CP` | `ea6bcaba99703fbe82c09ddb84b408001b43eadb` | 3,887 | 32.1 MiB |
| `EXAONE_Quantization_method` | `216e1b2f256a640cdd4b967aeaa6539258ae006a` | 4,541 | 38.8 MiB |
| `vLLM_Speed` | `f70db19b276ab8ca4707c6825b7d440dd39bb9b4` | 240 | 0.7 MiB |
| `vLLM_FP16_skip` | `fd6f373b17aadfeaadb027beecff4c7294c850ca` | 4,266 | 37.9 MiB |
| `vLLM_OmniQuant` | `7c96a51b71d62c6a19fd99b8a06c59d932df6566` | 6 | less than 0.1 MiB |
| `vllm-exaone-sq` | `17411cee9275e317b6674c8190c2a88cd4a56e46` | 4,115 | 36.7 MiB |

The standalone `leegongman/vllm` fork is not archived here. Its two EXAONE-specific commits are preserved as focused patches under [`../patches/vllm-exaone-sq/`](../patches/vllm-exaone-sq/).

## Filtering Rule

Imported material is limited to tracked source code, build/configuration files, tests, examples, and text documentation smaller than 1 MiB. The archive excludes:

- Credentials, DACON tokens, private paths, and private competition payloads.
- Datasets and calibration JSONL/CSV/Parquet files.
- Checkpoints, tokenizer/model files, weights, wheels, split wheel parts, and generated artifacts.
- Notebooks, notebook outputs, images, PDFs, HTML exports, caches, and logs.

Some upstream documentation, examples, and tests contain literal placeholder values such as `EMPTY`, `dummy`, or `sk-fake-key`. These are public examples, not credentials. No credential-value pattern was found in the imported files.

## Use

These are historical source snapshots, not a single buildable monorepo. The focused patch packages remain the preferred reading path for the EXAONE-specific modifications. See [`../docs/repository-consolidation.md`](../docs/repository-consolidation.md) for the source-to-destination map and [`../THIRD_PARTY_LICENSES.md`](../THIRD_PARTY_LICENSES.md) for attribution.
