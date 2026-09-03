# vLLM EXAONE SmoothQuant Model Path

## Purpose

This patch preserves the distinct `Exaone4ForCausalLMSQ` model path and its model-registry registration. It is the code-level evidence for the SmoothQuant-style hidden-state identity modules described in the project documentation.

## Source and Base

| Item | Value |
|---|---|
| Upstream base | vLLM commit `600a039f572ac28128750f0463af428c5a260f1a` |
| Change commit | [`d0d38db63090185dfb8ca131def8d04146de9e5e`](https://github.com/leegongman/vllm/commit/d0d38db63090185dfb8ca131def8d04146de9e5e) |
| Remap commit | [`a57dca9a83cf25d14be6a78f72d2ea028e0ccb1b`](https://github.com/leegongman/vllm/commit/a57dca9a83cf25d14be6a78f72d2ea028e0ccb1b) |
| Compatibility commit | [`d8fe813d6715869fa85471e5a3d3631c599b7e72`](https://github.com/leegongman/vllm/commit/d8fe813d6715869fa85471e5a3d3631c599b7e72) |
| Remap patch base | `d0d38db63090185dfb8ca131def8d04146de9e5e` |
| Compatibility patch base | `a57dca9a83cf25d14be6a78f72d2ea028e0ccb1b` |
| Preserved private snapshot | `leegongman/vllm-exaone-sq` commit `17411cee9275e317b6674c8190c2a88cd4a56e46` |
| License | Apache-2.0; see [`../licenses/vLLM-Apache-2.0.txt`](../licenses/vLLM-Apache-2.0.txt) |

## Included Changes

[`0001-exaone-sq-model-path.patch`](0001-exaone-sq-model-path.patch) adds:

- `exaone_refactoring.py`, derived from the EXAONE 4 vLLM model path.
- `IdentityWithParam` for per-channel smooth factors.
- Pre-attention and pre-feedforward hidden-state scaling insertion points.
- `Exaone4ForCausalLMSQ` registry registration.

[`0002-weight-name-remap.patch`](0002-weight-name-remap.patch) remaps checkpoint keys ending in `pre_*_identity.weight` to the prior `smooth_factor` parameter name.

[`0003-smooth-factor-to-weight.patch`](0003-smooth-factor-to-weight.patch) changes `IdentityWithParam` from `smooth_factor` to `weight`, aligning the module parameter directly with the checkpoint expectation.

It deliberately excludes the remainder of the vLLM tree, build system, wheel, and checkpoint.

## Applying

```bash
git apply patches/vllm-exaone-sq/0001-exaone-sq-model-path.patch
```

This patch requires an independently prepared compatible vLLM environment and a checkpoint whose architecture is configured for `Exaone4ForCausalLMSQ`.
