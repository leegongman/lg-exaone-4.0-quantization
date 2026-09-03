# OmniQuant and SmoothQuant Paths

## Status

**Adaptation and implementation evidence reviewed.** OmniQuant and SmoothQuant are external methods. This project records EXAONE-specific experiments and vLLM compatibility work; it does not claim ownership of the original methods. Source-only historical snapshots are retained with their upstream licenses under [`sources/`](../sources/README.md).

## OmniQuant Adaptation

The reviewed evidence covers an OmniQuant path with the following technical dimensions:

| Area | Recorded work |
|---|---|
| Learnable transforms | LET and LWC enablement; learnable scale/offset framing |
| Quantization regimes | Local W4A16 work; W4A4, W4A8, and W6A6 variants in reviewed project documentation |
| Optimization | Group-size, learning-rate, AMP-stability, and deactivated-AMP variants |
| Activation handling | Activation scale/shift generation and calibration-dataset changes |
| Runtime | Packed low-bit representation, config/checkpoint preparation, vLLM loading compatibility, and custom-wheel investigation |

The original local OmniQuant directory identifies [OpenGVLab OmniQuant](https://github.com/OpenGVLab/OmniQuant) as its upstream source. Filtered source-only snapshots are retained under [`sources/`](../sources/README.md); notebooks, checkpoints, example data, images, and upstream Git history remain excluded.

## SmoothQuant-Style Work

The local work included SmoothQuant combinations with GPTQ, AWQ, and integer/FP8-style paths. A separate runtime path added hidden-state scaling modules before attention and MLP computation in a custom EXAONE decoder model path.

The public implementation evidence includes a focused, Apache-2.0-attributed [EXAONE SmoothQuant model-path patch](../patches/vllm-exaone-sq/README.md), its companion [commit reference](../patches/vllm-exaone-sq-reference.md), the MIT-attributed [EXAONE OmniQuant adaptation patch](../patches/omniquant-exaone/README.md), and filtered source snapshots under [`sources/`](../sources/README.md). The patches remain the clearest view of project-specific changes.

## Claim Boundary

- A documented combination or feasibility check is not necessarily a completed submission implementation.
- LET/LWC results and local benchmarks are not official competition measurements.
- The runtime customization is described as implementation evidence; upstream vLLM and OmniQuant remain explicitly attributed external dependencies.

See [`docs/model-architecture.md`](../docs/model-architecture.md) for the EXAONE decoder-path explanation and [`docs/source-map.md`](../docs/source-map.md) for attribution.
