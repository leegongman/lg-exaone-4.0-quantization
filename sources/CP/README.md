# CP Notes

이 저장소는 이번 작업에서 쓴 OmniQuant / vLLM 수정 사항과 Runpod 재현 절차를 정리한 보관소다.

현재 넣어둔 내용:

- `docs/RUNPOD_SETUP.md`
  - Runpod 시작 후 apt / uv / venv / pip 세팅
- `docs/WHEEL_BUILD.md`
  - `torch==2.9.0+cu128`, `vllm==0.14.1` 기준 wheel 빌드 절차
- `docs/CHECKPOINT_GENERATION.md`
  - OmniQuant 체크포인트 생성 / 평가 명령
- `docs/MODIFIED_FILES.md`
  - 실제로 수정한 핵심 파일 목록과 역할
- `sources/OmniQuant_EXAONE_v4/`
  - 실제 작업에 쓴 OmniQuant 소스 전체
- `sources/vllm_submit_v0141/`
  - clean `v0.14.1` 기반 custom vLLM 소스 전체
- `patched_sources/`
  - 현재 고정된 참조용 소스 복사본

의도적으로 제외한 것:

- 체크포인트 본체
- `model.safetensors`
- split checkpoint parts

이유:

- GitHub 용량 리스크를 피하기 위해서다.
- 체크포인트는 별도 보관소나 tar/scp 방식으로 다루는 것이 안전하다.

다음 세션에서 가장 빠른 복구 방법:

1. 이 저장소를 clone
2. `docs/RUNPOD_SETUP.md` 순서대로 환경 복구
3. `sources/OmniQuant_EXAONE_v4/` 또는 `sources/vllm_submit_v0141/`에서 바로 작업 시작
4. 필요하면 `patched_sources/`의 파일만 따로 덮어쓰기
5. `docs/WHEEL_BUILD.md`로 wheel 재빌드
6. `docs/CHECKPOINT_GENERATION.md`로 체크포인트 생성 / 평가
