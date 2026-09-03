# Restore From Clone

Clone and enter the repo:

```bash
git clone https://github.com/leegongman/vLLM_OmniQuant.git
cd vLLM_OmniQuant
```

Install the local patched `vllm/` source into the virtualenv you use for runs:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -e ./vllm
```

If you need the extra packages used in the current workflow, install them again in
the same environment:

```bash
pip install mistral_common xgrammar==0.1.29 numba==0.61.2
BUILD_CUDA_EXT=0 pip install --no-cache-dir 'git+https://github.com/AutoGPTQ/AutoGPTQ.git@v0.7.1'
```

First checks after the GPU session comes up:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
python OmniQuant_EXAONE_v4/scripts/run_vllm_omni_activation_real_smoke.py \
  --model_dir /home/ubuntu/checkpoints/exaone4_omni/w6a6_e3 \
  --max_tokens 8 \
  --max_model_len 256
```
