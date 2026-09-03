# Third-Party Licenses

This repository's top-level [MIT license](LICENSE) applies to the project-owned documentation, schemas, and cleaned helper scripts. The patch files listed below contain modifications to upstream projects and retain their applicable upstream licenses.

| Path | Upstream project | License | License text |
|---|---|---|---|
| `patches/omniquant-exaone/` | [OpenGVLab OmniQuant](https://github.com/OpenGVLab/OmniQuant) | MIT | [`patches/licenses/OmniQuant-MIT.txt`](patches/licenses/OmniQuant-MIT.txt) |
| `patches/vllm-omni-activation-real/` | [vLLM](https://github.com/vllm-project/vllm) | Apache-2.0 | [`patches/licenses/vLLM-Apache-2.0.txt`](patches/licenses/vLLM-Apache-2.0.txt) |
| `patches/vllm-exaone-sq/` | [vLLM](https://github.com/vllm-project/vllm) | Apache-2.0 | [`patches/licenses/vLLM-Apache-2.0.txt`](patches/licenses/vLLM-Apache-2.0.txt) |
| `patches/vllm-fp16-skip/` | [vLLM](https://github.com/vllm-project/vllm) | Apache-2.0 | [`patches/licenses/vLLM-Apache-2.0.txt`](patches/licenses/vLLM-Apache-2.0.txt) |

The patches are intentionally distributed without the full upstream source trees. Apply them only to the exact base revision named in the adjacent patch README.
