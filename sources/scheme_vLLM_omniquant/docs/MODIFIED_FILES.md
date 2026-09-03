# Modified Files

## OmniQuant side

- `OmniQuant_EXAONE_v4/quantize/omniquant.py`
  - 저비트(`w4*`) 안정화 로직
  - best-state 복구
  - grad clip
  - non-finite 대응

- `OmniQuant_EXAONE_v4/quantize/activation_real_quant.py`
  - omni-act export
  - `w4*` packed 4bit 저장
  - `w6a6` packed 6bit 저장
  - repack helper

## vLLM side

- `vllm/vllm/model_executor/layers/quantization/omni_activation_real.py`
  - omni-act loader/runtime
  - packed 4bit / packed 6bit 지원
  - `w4a4/w4a8`가 `w6a6` fast path를 잘못 타지 않게 제한

- `vllm/vllm/model_executor/layers/quantization/utils/omni_triton_utils.py`
  - Omni Triton helper

- `vllm/vllm/model_executor/layers/quantization/utils/omni_cutlass_utils.py`
  - Omni CUTLASS helper

- `vllm/vllm/model_executor/layers/quantization/__init__.py`
  - quantization registry 연결

## Active runtime note

실험할 때는 source만 바꾸면 안 되고, 실제 사용 중인 `.venv` runtime에도 sync가 필요했다.

활성 runtime 경로:

- `/home/ubuntu/vLLM_OmniQuant/.venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/omni_activation_real.py`

실험 방식:

1. source 수정
2. source `py_compile`
3. site-packages로 복사
4. site-packages `py_compile`
5. smoke / eval
