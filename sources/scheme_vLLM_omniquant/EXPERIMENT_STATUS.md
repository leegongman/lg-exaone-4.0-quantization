# Experiment Status

Last updated: 2026-04-02 UTC

## Current resumable state

The only backed-up in-progress epoch sweep state is `w6a6_e3`.

- Quantization target: `w6a6`
- Base config: `wbits=6`, `abits=6`, `group_size=128`
- Optimization flags: `--lwc --let`
- Hyperparameters: `alpha=0.5`, `let_lr=1e-2`, `lwc_lr=5e-3`
- Important runtime note: `epoch > 1` diverged with the original AMP path, so the resumable sweep uses `--deactive_amp`

## Backed-up artifacts

- `epoch=3` checkpoint:
  - `artifacts/checkpoints/w6a6_3epoch_deactive_amp_let1e-2_lwc5e-3`
- `epoch=3` resume state:
  - `artifacts/checkpoints/w6a6_3epoch_deactive_amp_let1e-2_lwc5e-3_resume/omni_parameters.pth`
- `epoch=3` eval result:
  - `artifacts/eval/w6a6_epoch_sweep/w6a6_e3_results.json`

## e3 eval result

- `gsm8k`: `0.64` flexible, `0.63` strict
- `aime25`: `0.00`
- `truthfulqa_mc1`: `0.34`

## Intended next steps

Resume from the backed-up `e3` state and continue:

1. `e5`: run `--epochs 2 --resume artifacts/checkpoints/w6a6_3epoch_deactive_amp_let1e-2_lwc5e-3_resume/omni_parameters.pth`
2. `e10`: resume from the new `e5` run state with `--epochs 5`
3. `e20`: resume from the new `e10` run state with `--epochs 10`
4. Evaluate each checkpoint with the existing vLLM + lm-eval wrapper

## Restart handoff

If a future Codex session needs to continue this work, the handoff message can be:

`Use SETUP.md and EXPERIMENT_STATUS.md. Continue the w6a6 epoch sweep from e3 to e5/e10/e20.`
