# vLLM EXAONE SQ Reference

## Scope

This file documents the focused EXAONE/vLLM customization evidence without vendoring the complete vLLM tree. The corresponding source delta is retained in [`vllm-exaone-sq/`](vllm-exaone-sq/); it is not an installable vLLM extension.

## Reviewed Source Commits

| Item | Source | Observed purpose |
|---|---|---|
| EXAONE SQ model path | [`d0d38db`](https://github.com/leegongman/vllm/commit/d0d38db63090185dfb8ca131def8d04146de9e5e) | Adds an `Exaone4ForCausalLMSQ` model path, identity modules for smooth-factor handling, and model-registry support. |
| Checkpoint-key remap | [`a57dca9`](https://github.com/leegongman/vllm/commit/a57dca9a83cf25d14be6a78f72d2ea028e0ccb1b) | Remaps `pre_*_identity.weight` keys to the preceding `smooth_factor` parameter name. |
| Checkpoint-name compatibility fix | [`d8fe813`](https://github.com/leegongman/vllm/commit/d8fe813d6715869fa85471e5a3d3631c599b7e72) | Aligns the identity-module parameter name with the checkpoint naming expectation. |

## Technical Boundary

The reviewed implementation evidence describes a custom EXAONE model path derived from the vLLM EXAONE 4 model implementation. Its documented elements are:

- `Exaone4ForCausalLMSQ` registration for a distinct architecture name.
- An `IdentityWithParam` module used to hold a learnable smooth factor.
- Pre-attention and pre-feedforward identity insertion points in the decoder path.
- Loading and registry work needed for checkpoint/runtime compatibility.

These are implementation-evidence statements, not a claim of ownership of vLLM or SmoothQuant. The upstream vLLM framework and the underlying quantization methods remain external dependencies.

## License and Distribution Rule

The reviewed fork is based on vLLM, whose relevant source is Apache-2.0 licensed. This repository retains only the focused source delta under [`vllm-exaone-sq/`](vllm-exaone-sq/) with the applicable license text; it does not contain the full vLLM tree or a build artifact.

The retained patch includes all of the following:

- The exact upstream base revision.
- The source commit link.
- The Apache-2.0 license and required notices.
- A concise explanation of the EXAONE-specific modification.
- Confirmation that no unrelated upstream files, model artifacts, or credentials are included.

## Reproduction Boundary

Applying the focused patch still requires an independently prepared compatible vLLM environment and a compatible checkpoint. Those components are not distributed here. Any runtime measurement made with such an environment is a local benchmark, not an official competition result.
