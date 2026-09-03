# Next Session Notes

고정 조건:

- `torch==2.9.0+cu128`
- `vllm==0.14.1`

현재 best:

- `w4a8`
  - checkpoint family: `w4a8_ep1_fix1_stable`
  - GSM8K 100: flexible `0.68`, strict `0.66`

- `w4a4`
  - checkpoint family: `w4a4_ep1_fix1_noaug_stable`
  - GSM8K 100: flexible `0.60`, strict `0.60`

- `w6a6`
  - current packed baseline: flexible `0.63`, strict `0.63`
  - 아직 목표 `0.67+` 미달

남은 큰 과제:

1. `w6a6` 체크포인트 품질 개선
2. 현재 고정 vLLM 기준 wheel 빌드 완료 확인
3. 필요 시 wheel 설치 smoke / serve / curl 재확인
