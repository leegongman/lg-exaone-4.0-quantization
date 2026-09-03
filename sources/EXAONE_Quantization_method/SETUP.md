# RunPod + VSCode Remote Setup Guide

이 문서는 `RunPod SSH + VSCode Remote SSH` 환경에서
`vLLM_OmniQuant` 저장소를 처음부터 다시 받아서,

1. 환경 설치
2. vLLM 네이티브 빌드
3. 체크포인트 준비
4. 노트북 실행
5. 평가 재실행

까지 재현하는 절차를 정리한 문서입니다.

## 1. 접속

RunPod 인스턴스가 켜진 상태에서:

1. RunPod에서 SSH 접속 정보를 확인합니다.
2. 로컬 PC의 VSCode에서 `Remote - SSH`로 접속합니다.
3. 원격 홈 디렉터리 `/home/ubuntu`를 엽니다.

이후 모든 작업은 원격 서버 기준으로 진행합니다.

## 2. 저장소 받기

```bash
cd /home/ubuntu

git lfs install
git clone https://github.com/leegongman/vLLM_OmniQuant.git
cd vLLM_OmniQuant
git lfs pull
```

## 3. Python 가상환경

성공했던 실제 버전 조합은 아래입니다.

- `torch==2.8.0+cu128`
- `transformers==4.57.5`
- `tokenizers==0.22.0`
- `huggingface_hub==0.36.2`
- `accelerate==1.13.0`
- `optimum==2.1.0`
- `auto-gptq==0.7.1`
- `lm-eval==0.4.11`

설치:

```bash
cd /home/ubuntu/vLLM_OmniQuant

python3.12 -m venv .venv
source .venv/bin/activate

pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

pip install \
  transformers==4.57.5 tokenizers==0.22.0 huggingface_hub==0.36.2 \
  accelerate==1.13.0 safetensors sentencepiece datasets pandas packaging \
  ninja py7zr gekko pycountry optimum==2.1.0

pip install \
  omegaconf sacrebleu scikit-learn rouge-score sqlitedict pytablewriter \
  texttable termcolor wonderwords nltk

pip install \
  cloudpickle msgspec blake3 cachetools fastapi openai pydantic tiktoken \
  gguf setuptools_scm cbor2 ijson pybase64 watchfiles partial-json-parser \
  lm-format-enforcer==0.11.3 llguidance==1.3.0 outlines_core==0.2.11 \
  diskcache==5.6.3 lark==1.2.2 setproctitle uvloop py-cpuinfo \
  openai-harmony einops

BUILD_CUDA_EXT=0 pip install --no-cache-dir 'git+https://github.com/AutoGPTQ/AutoGPTQ.git@v0.7.1'
pip install --no-cache-dir --no-build-isolation 'git+https://github.com/AutoGPTQ/AutoGPTQ.git@v0.7.1'

pip install jupyter ipykernel nbconvert
python -m ipykernel install --user --name runpod-omni --display-name "Python (runpod-omni)"
```

## 4. 주요 디렉터리 구조

저장소 루트:

- `/home/ubuntu/vLLM_OmniQuant`

주요 경로:

- OmniQuant 코드: `/home/ubuntu/vLLM_OmniQuant/OmniQuant_EXAONE_v4`
- vLLM 소스: `/home/ubuntu/vLLM_OmniQuant/vllm`
- 가상환경: `/home/ubuntu/vLLM_OmniQuant/.venv`
- 노트북: `/home/ubuntu/vLLM_OmniQuant/omni_quant_lwc_4_1_omni-act.ipynb`
- 실행된 노트북 예시: `/home/ubuntu/vLLM_OmniQuant/omni_quant_lwc_4_1_omni-act.executed.ipynb`
- 체크포인트 artifact: `/home/ubuntu/vLLM_OmniQuant/artifacts/checkpoints`
- 평가 요약: `/home/ubuntu/vLLM_OmniQuant/artifacts/eval/final_summary_core.json`

## 5. 체크포인트 artifact

현재 리포에 들어 있는 최종 1 epoch 체크포인트:

- `artifacts/checkpoints/w4a4_1epoch_alpha0.75_let1e-3_lwc1e-2_aug_loss`
- `artifacts/checkpoints/w4a8_1epoch_let1e-2_lwc5e-3`
- `artifacts/checkpoints/w6a6_1epoch_let1e-2_lwc5e-3`

정리 파일:

- `artifacts/checkpoints/README.md`

각 설정 요약:

- `w4a4`: `1 epoch`, `alpha=0.75`, `let_lr=1e-3`, `lwc_lr=1e-2`, `aug_loss=yes`
- `w4a8`: `1 epoch`, `let_lr=1e-2`, `lwc_lr=5e-3`
- `w6a6`: `1 epoch`, `let_lr=1e-2`, `lwc_lr=5e-3`

## 6. 체크포인트 런타임 위치로 복사

노트북과 평가 스크립트는 런타임 경로 `/home/ubuntu/checkpoints/exaone4_omni`를 기준으로 동작합니다.
clone 후 아래처럼 한 번 복사해 두면 됩니다.

```bash
mkdir -p /home/ubuntu/checkpoints/exaone4_omni

cp -a /home/ubuntu/vLLM_OmniQuant/artifacts/checkpoints/w4a4_1epoch_alpha0.75_let1e-3_lwc1e-2_aug_loss \
  /home/ubuntu/checkpoints/exaone4_omni/w4a4_augloss

cp -a /home/ubuntu/vLLM_OmniQuant/artifacts/checkpoints/w4a8_1epoch_let1e-2_lwc5e-3 \
  /home/ubuntu/checkpoints/exaone4_omni/w4a8

cp -a /home/ubuntu/vLLM_OmniQuant/artifacts/checkpoints/w6a6_1epoch_let1e-2_lwc5e-3 \
  /home/ubuntu/checkpoints/exaone4_omni/w6a6
```

## 7. vLLM 네이티브 빌드

이 저장소는 patched `vllm` 소스를 사용합니다.
fresh clone 뒤에는 네이티브 확장을 한 번 빌드해야 합니다.

### 7.1 보조 소스 준비

```bash
mkdir -p /tmp/cutlass-src
curl -L https://github.com/nvidia/cutlass/archive/refs/tags/v4.2.1.tar.gz -o /tmp/cutlass-v4.2.1.tar.gz
tar -xzf /tmp/cutlass-v4.2.1.tar.gz -C /tmp/cutlass-src

mkdir -p /tmp/flashmla-min/flash_mla
cp /home/ubuntu/vLLM_OmniQuant/vllm/vllm/third_party/flashmla/flash_mla_interface.py \
  /tmp/flashmla-min/flash_mla/flash_mla_interface.py
```

### 7.2 configure + build

```bash
cd /home/ubuntu/vLLM_OmniQuant/vllm
source /home/ubuntu/vLLM_OmniQuant/.venv/bin/activate

rm -rf build_clean3
VLLM_CUTLASS_SRC_DIR=/tmp/cutlass-src/cutlass-4.2.1 \
TRITON_KERNELS_SRC_DIR=/home/ubuntu/vLLM_OmniQuant/vllm/vllm/third_party/triton_kernels \
FLASH_MLA_SRC_DIR=/tmp/flashmla-min \
cmake -G Ninja -B build_clean3 -S . \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DVLLM_TARGET_DEVICE=cuda \
  -DVLLM_BUILD_FLASH_ATTN=OFF \
  -DVLLM_PYTHON_EXECUTABLE=/home/ubuntu/vLLM_OmniQuant/.venv/bin/python \
  -DCMAKE_INSTALL_PREFIX=/home/ubuntu/vLLM_OmniQuant/vllm

cmake --build build_clean3 --target _C cumem_allocator --parallel 4
cmake --install build_clean3 --component _C
cmake --install build_clean3 --component cumem_allocator
```

### 7.3 빌드 산출물 위치

- configure/build dir: `/home/ubuntu/vLLM_OmniQuant/vllm/build_clean3`
- 설치된 `_C`: `/home/ubuntu/vLLM_OmniQuant/vllm/vllm/_C.abi3.so`
- 설치된 allocator: `/home/ubuntu/vLLM_OmniQuant/vllm/vllm/cumem_allocator.abi3.so`

## 8. 노트북 실행

메인 노트북:

- `/home/ubuntu/vLLM_OmniQuant/omni_quant_lwc_4_1_omni-act.ipynb`

VSCode에서:

1. 노트북 열기
2. 커널을 `Python (runpod-omni)`로 선택
3. `Run All`

터미널에서 실행할 경우:

```bash
cd /home/ubuntu/vLLM_OmniQuant
source .venv/bin/activate
python -m nbconvert --to notebook --execute omni_quant_lwc_4_1_omni-act.ipynb --output omni_quant_lwc_4_1_omni-act.executed.ipynb
```

현재 노트북 기본 동작:

- `w4a4`는 repaired checkpoint인 `w4a4_augloss`를 사용
- 태스크는 `aime25,gsm8k,truthfulqa_mc1`
- `FORCE_QUANT=False`
- `FORCE_EVAL=False`

처음부터 양자화를 다시 돌리려면 노트북 첫 코드 셀에서 `FORCE_QUANT=True`로 바꾸면 됩니다.

## 9. 평가 스크립트

주요 스크립트:

- readiness 체크:
  `/home/ubuntu/vLLM_OmniQuant/OmniQuant_EXAONE_v4/scripts/check_vllm_omni_activation_real_readiness.py`
- lm-eval 진입 래퍼:
  `/home/ubuntu/vLLM_OmniQuant/OmniQuant_EXAONE_v4/scripts/lm_eval_entry_with_omni_quant.py`
- smoke:
  `/home/ubuntu/vLLM_OmniQuant/OmniQuant_EXAONE_v4/scripts/run_vllm_omni_activation_real_smoke.py`

## 10. eval만 다시 돌리는 방법

예시: `w4a8`

```bash
cd /home/ubuntu/vLLM_OmniQuant
source .venv/bin/activate

PYTHONPATH=/home/ubuntu/vLLM_OmniQuant/OmniQuant_EXAONE_v4:/home/ubuntu/vLLM_OmniQuant/vllm \
HF_HUB_ENABLE_HF_TRANSFER=0 CUDA_VISIBLE_DEVICES=0 \
python /home/ubuntu/vLLM_OmniQuant/OmniQuant_EXAONE_v4/scripts/lm_eval_entry_with_omni_quant.py \
  --model vllm \
  --model_args pretrained=/home/ubuntu/checkpoints/exaone4_omni/w4a8,gpu_memory_utilization=0.85,enable_thinking=False,enforce_eager=True,dtype=float16,quantization=omni_activation_real,attention_backend=TRITON_ATTN \
  --tasks aime25,gsm8k,truthfulqa_mc1 \
  --output_path /home/ubuntu/lm_eval_results_exaone4_omni/w4a8_core \
  --limit 100 \
  --batch_size auto \
  --apply_chat_template \
  --gen_kwargs max_gen_toks=256
```

## 11. 최종 결과 위치

런타임 평가 결과:

- `/home/ubuntu/lm_eval_results_exaone4_omni/w4a4_core`
- `/home/ubuntu/lm_eval_results_exaone4_omni/w4a8_core`
- `/home/ubuntu/lm_eval_results_exaone4_omni/w6a6_core`

리포에 저장된 요약:

- `/home/ubuntu/vLLM_OmniQuant/artifacts/eval/final_summary_core.json`

최종 핵심 수치:

- `w4a4`: `gsm8k flexible 0.03`, `truthfulqa_mc1 0.28`
- `w4a8`: `gsm8k flexible 0.59`, `truthfulqa_mc1 0.30`
- `w6a6`: `gsm8k flexible 0.61`, `truthfulqa_mc1 0.27`

## 12. 주의사항

- `ruler`는 grouped task 수가 많아서 현재 최종 노트북 흐름에서는 제외했습니다.
- `aime25`는 이번 최종 100-sample 기준에서 세 설정 모두 `0.00`이었습니다.
- clone 직후 바로 실행되지 않으면 대부분 원인은 아래 셋 중 하나입니다.
  - `git lfs pull`을 안 해서 체크포인트가 pointer 상태인 경우
  - `vllm` 네이티브 `.so`를 아직 빌드하지 않은 경우
  - 체크포인트를 `/home/ubuntu/checkpoints/exaone4_omni`로 복사하지 않은 경우
