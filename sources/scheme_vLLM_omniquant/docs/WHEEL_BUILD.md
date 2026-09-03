# Wheel Build

목표:

- `vllm==0.19.0`
- `torch==2.10.0+cu129`
- 현재 포함된 OmniQuant / vLLM quant 수정 유지

clean source 준비:

```bash
cd /path/to/clone/w6a6_vllm_omniquant/vllm
```

이미 이 저장소의 `vllm/`에는 현재 수정본이 반영돼 있다.

만약 fresh clone 위에 다시 덮어쓰고 싶으면:

```bash
cp /path/to/patched_sources/vllm/vllm/model_executor/layers/quantization/omni_activation_real.py \
  vllm/model_executor/layers/quantization/
cp /path/to/patched_sources/vllm/vllm/model_executor/layers/quantization/utils/omni_triton_utils.py \
  vllm/model_executor/layers/quantization/utils/
cp /path/to/patched_sources/vllm/vllm/model_executor/layers/quantization/utils/omni_cutlass_utils.py \
  vllm/model_executor/layers/quantization/utils/
cp /path/to/patched_sources/vllm/vllm/model_executor/layers/quantization/__init__.py \
  vllm/model_executor/layers/quantization/
```

현재 pin:

- `requirements/build.txt`: `torch==2.10.0`
- `requirements/cuda.txt`:
  - `torch==2.10.0`
  - `torchaudio==2.10.0`
  - `torchvision==0.25.0`
- `pyproject.toml`: `torch == 2.10.0`

빌드:

```bash
source $HOME/.local/bin/env
uv venv --python 3.11 --seed --managed-python
source .venv/bin/activate

uv pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu129
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$VIRTUAL_ENV/lib/python3.11/site-packages/nvidia/cusparse/lib:${LD_LIBRARY_PATH}"
grep -v '^torch==' requirements/build.txt | uv pip install -r -

export VLLM_VERSION_OVERRIDE=0.19.0
MAX_JOBS=10 uv build --wheel --no-build-isolation
```

산출물:

```bash
ls dist/*.whl
```

메타데이터 확인:

```bash
unzip -p dist/*.whl '*/METADATA' | rg '^(Version|Requires-Dist: torch|Requires-Dist: torchvision|Requires-Dist: torchaudio)'
```

기대값:

- `Version: 0.19.0`
- `Requires-Dist: torch==2.10.0`
- `Requires-Dist: torchvision==0.25.0`
- `Requires-Dist: torchaudio==2.10.0`

설치 확인:

```bash
uv venv --python 3.11 --seed --managed-python
source .venv/bin/activate
uv pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu129
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$VIRTUAL_ENV/lib/python3.11/site-packages/nvidia/cusparse/lib:${LD_LIBRARY_PATH}"
uv pip install --no-deps dist/*.whl
python - <<'PY'
import sys, torch, vllm
print(sys.version.split()[0], torch.__version__, torch.version.cuda, vllm.__version__)
PY
```
