# vLLM FP16-Skip Snapshot Record

The former `vLLM_FP16_skip` repository was reviewed at commit `fd6f373b17aadfeaadb027beecff4c7294c850ca`. Its prior source-snapshot commit is `c49b2a951ae985c93f6ec3fd903bb6270b2036a7`. It is a full Apache-2.0 vLLM source snapshot, not a minimal patch repository.

No isolated implementation named `fp16_skip`, `skip_fp16`, or equivalent was found in its non-test source paths. Its EXAONE/Omni runtime material overlaps the broader Omni activation-real work preserved in [`../vllm-omni-activation-real/`](../vllm-omni-activation-real/).

The selected legacy runtime source is retained at [`legacy/omni_activation_real.py`](legacy/omni_activation_real.py). It is a 1,661-line earlier implementation of the `omni_activation_real` quantization method and has SHA-1 `3adaaf527482e198a159fed477df6cf95d7a2b41`. It is a source-preservation record, not a claimed FP16-skip implementation or an independently applicable patch.

Accordingly, this clean repository does not duplicate the full snapshot. The selected file carries the vLLM Apache-2.0 header; the full license text is at [`../licenses/vLLM-Apache-2.0.txt`](../licenses/vLLM-Apache-2.0.txt). Upstream vLLM remains available from [vllm-project/vllm](https://github.com/vllm-project/vllm).
