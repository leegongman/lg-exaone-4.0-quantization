# Runpod Setup

기준 환경:

- `python 3.11`
- `torch==2.9.0+cu128`
- `vllm==0.14.1`

apt 패키지:

```bash
sudo apt update
sudo apt install -y python3.11 python3.11-dev python3.11-distutils python3-pip git tmux
```

uv 설치:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

작업용 가상환경:

```bash
uv venv --python 3.11 --seed --managed-python
source .venv/bin/activate
uv pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

버전 확인:

```bash
python - <<'PY'
import sys, torch
print("python:", sys.version.split()[0])
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
PY
```

기대값:

- `python: 3.11.x`
- `torch: 2.9.0+cu128`
- `cuda: 12.8`

권장 작업 디렉터리:

- 소스 작업: `/home/ubuntu/vLLM_OmniQuant`
- wheel 빌드: 별도 clean clone
- 장시간 작업: `tmux`

권장 tmux:

```bash
tmux new -s omni
```
