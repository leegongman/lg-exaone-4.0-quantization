# Checkpoint Generation

기준 소스:

- OmniQuant: `OmniQuant_EXAONE_v4`
- vLLM: `v0.14.1` 기반 custom runtime

공통 옵션:

- 모델: `LGAI-EXAONE/EXAONE-4.0-1.2B`
- calibset: `wikitext2`
- `nsamples=128`
- `batch_size=1`
- `group_size=128`
- `save_format=omni-act`
- `deactive_amp`

## W4A4

현재 안정화 기준:

- `alpha=0.75`
- `let_lr=1e-3`
- `lwc_lr=1e-2`
- `aug_loss=False`

예시:

```bash
python OmniQuant_EXAONE_v4/main.py \
  --model LGAI-EXAONE/EXAONE-4.0-1.2B \
  --net EXAONE-4.0-1.2B \
  --cache_dir OmniQuant_EXAONE_v4/cache \
  --calib_dataset wikitext2 \
  --nsamples 128 \
  --batch_size 1 \
  --wbits 4 \
  --abits 4 \
  --group_size 128 \
  --epochs 1 \
  --alpha 0.75 \
  --let --lwc \
  --let_lr 1e-3 \
  --lwc_lr 1e-2 \
  --real_quant \
  --save_format omni-act \
  --deactive_amp \
  --act-scales OmniQuant_EXAONE_v4/act_scales/EXAONE-4.0-1.2B.pt \
  --act-shifts OmniQuant_EXAONE_v4/act_shifts/EXAONE-4.0-1.2B.pt \
  --output_dir /tmp/w4a4_run \
  --save_dir /tmp/w4a4_ckpt
```

## W4A8

현재 best 기준:

- `alpha=0.5`
- `let_lr=1e-2`
- `lwc_lr=5e-3`
- `aug_loss=False`

예시:

```bash
python OmniQuant_EXAONE_v4/main.py \
  --model LGAI-EXAONE/EXAONE-4.0-1.2B \
  --net EXAONE-4.0-1.2B \
  --cache_dir OmniQuant_EXAONE_v4/cache \
  --calib_dataset wikitext2 \
  --nsamples 128 \
  --batch_size 1 \
  --wbits 4 \
  --abits 8 \
  --group_size 128 \
  --epochs 1 \
  --alpha 0.5 \
  --let --lwc \
  --let_lr 1e-2 \
  --lwc_lr 5e-3 \
  --real_quant \
  --save_format omni-act \
  --deactive_amp \
  --act-scales OmniQuant_EXAONE_v4/act_scales/EXAONE-4.0-1.2B.pt \
  --act-shifts OmniQuant_EXAONE_v4/act_shifts/EXAONE-4.0-1.2B.pt \
  --output_dir /tmp/w4a8_run \
  --save_dir /tmp/w4a8_ckpt
```

## W6A6

현재 유지 기준:

- `alpha=0.5`
- `let_lr=1e-2`
- `lwc_lr=5e-3`
- `aug_loss=False`

예시:

```bash
python OmniQuant_EXAONE_v4/main.py \
  --model LGAI-EXAONE/EXAONE-4.0-1.2B \
  --net EXAONE-4.0-1.2B \
  --cache_dir OmniQuant_EXAONE_v4/cache \
  --calib_dataset wikitext2 \
  --nsamples 128 \
  --batch_size 1 \
  --wbits 6 \
  --abits 6 \
  --group_size 128 \
  --epochs 1 \
  --alpha 0.5 \
  --let --lwc \
  --let_lr 1e-2 \
  --lwc_lr 5e-3 \
  --real_quant \
  --save_format omni-act \
  --deactive_amp \
  --act-scales OmniQuant_EXAONE_v4/act_scales/EXAONE-4.0-1.2B.pt \
  --act-shifts OmniQuant_EXAONE_v4/act_shifts/EXAONE-4.0-1.2B.pt \
  --output_dir /tmp/w6a6_run \
  --save_dir /tmp/w6a6_ckpt
```

## 평가

GSM8K 100:

```bash
python OmniQuant_EXAONE_v4/scripts/run_vllm_omni_activation_real_lm_eval.py \
  --model-dir /tmp/w4a8_ckpt \
  --output-root /tmp/w4a8_eval \
  --tasks gsm8k \
  --limit 100 \
  --batch-size 8 \
  --gpu-memory-utilization 0.85 \
  --skip-readiness-check
```

참고 점수:

- `w4a4` best seen: flexible `0.60`, strict `0.60`
- `w4a8` best seen: flexible `0.68`, strict `0.66`
- `w6a6` current packed baseline: flexible `0.63`, strict `0.63`
