# Wheel Build

목표:

- `vllm==0.14.1`
- `torch==2.9.0+cu128`
- 현재 고정된 OmniQuant / vLLM quant 수정 포함

clean source 준비:

```bash
cd /path/to/clone/sources/vllm_submit_v0141
```

이미 이 저장소의 `sources/vllm_submit_v0141/`에는 수정본이 반영돼 있다.

만약 fresh clone 위에 다시 덮어쓰고 싶으면:

```bash
cp /path/to/patched_sources/OmniQuant_EXAONE_v4/quantize/omniquant.py /tmp/ignore  # 참고용
cp /path/to/patched_sources/vllm/vllm/model_executor/layers/quantization/omni_activation_real.py \
  vllm/model_executor/layers/quantization/
cp /path/to/patched_sources/vllm/vllm/model_executor/layers/quantization/utils/omni_triton_utils.py \
  vllm/model_executor/layers/quantization/utils/
cp /path/to/patched_sources/vllm/vllm/model_executor/layers/quantization/utils/omni_cutlass_utils.py \
  vllm/model_executor/layers/quantization/utils/
cp /path/to/patched_sources/vllm/vllm/model_executor/layers/quantization/__init__.py \
  vllm/model_executor/layers/quantization/
```

pin 수정 포인트:

- `requirements/build.txt`: `torch==2.9.0`
- `requirements/cuda.txt`:
  - `torch==2.9.0`
  - `torchaudio==2.9.0`
  - `torchvision==0.24.0`
- `pyproject.toml`: `torch == 2.9.0`

빌드:

```bash
source $HOME/.local/bin/env
uv venv --python 3.11 --seed --managed-python
source .venv/bin/activate

uv pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
  --index-url https://download.pytorch.org/whl/cu128
grep -v '^torch==' requirements/build.txt | uv pip install -r -

export VLLM_VERSION_OVERRIDE=0.14.1
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

- `Version: 0.14.1`
- `Requires-Dist: torch==2.9.0`
- `Requires-Dist: torchvision==0.24.0`
- `Requires-Dist: torchaudio==2.9.0`

설치 확인:

```bash
uv venv --python 3.11 --seed --managed-python
source .venv/bin/activate
uv pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --no-deps dist/*.whl
python - <<'PY'
import sys, torch, vllm
print(sys.version.split()[0], torch.__version__, torch.version.cuda, vllm.__version__)
PY
```
