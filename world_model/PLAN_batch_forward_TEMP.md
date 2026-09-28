# R-SSM Batch Forward 구현 계획 (임시 작업 노트)

> 이 파일은 구현 완료 + 테스트 통과 후 삭제한다. 영구 문서 아님.
> 배경: forward가 항상 B=1(샷 1개씩)이라 GPU 활용률이 낮음.
> 여러 샷을 "웨이브프론트" 방식으로 동기화해서 이벤트 단위로 배치 처리한다.

## 핵심 발견 (리스크를 크게 낮추는 사실)

`world_model/rssm_model.py`의 모든 MLP (`msg_mlp`, `upd_mlp`, `single_mlp`,
`dec_mlp`, `type_mlp`, `pocket_mlp`)는 `nn.Linear` + `SiLU`만 쌓은 `nn.Sequential`이다.
`nn.Linear`는 leading batch dim이 몇 개든 그대로 broadcast한다.

→ **모델 구조는 이미 배치를 완전히 지원한다.** 새 수식/새 파라미터 불필요.
   병목은 100% 호출부(`step_ball_ball`, `step_single`, `compute_shot_ss_loss`)가
   `(dim,)` 1-샘플 텐서만 만들어서 넘기고 있다는 것뿐이다.
   → 이번 작업은 순수 "데이터 마샬링" 문제. 수식 중복/드리프트 리스크 없음.

## 설계 원칙 (반드시 지킬 것)

1. **모든 rolling 상태는 Python list로 유지한다.**
   `h`, loss 누적값, `pred_rvws` 전부 텐서 index-assignment(in-place) 금지.
   MLP 호출 시점에만 `torch.stack`으로 묶고, 결과는 다시 list 원소로 되돌린다.
   (이미 검증된 `h.clone()` 제거 패턴의 일반화 — [[project_spr_mdn_state]] 이전 작업과 동일 원리)

2. **기존 per-shot loss 정규화 의미론을 그대로 보존한다.**
   지금 코드는 `(loss / cfg.accum_steps).backward()`를 accum_steps번 호출 →
   "accum_steps개 샷의 정규화된 loss 평균"과 수학적으로 동일.
   새 코드는 batch_size개 샷의 정규화된 loss를 평균낸 뒤 1번 backward().
   → **batch_size = 기존 accum_steps로 맞추면 gradient가 거의 동일해야 한다.**
   (teacher forcing 구간(ss_prob=1.0)에서는 RNG 소비 순서가 달라도 분기 결과가
   항상 True라서 **정확히 일치**해야 함 — 강력한 회귀 테스트 포인트)

3. **n_balls는 배치 내 모든 샷이 동일해야 한다** (기존 TrainConfig 전제와 동일).
   배치 함수에 assert로 명시 — 조용히 깨지는 것 방지.

4. **make_node/make_edge(numpy)는 배치화하지 않는다.**
   이미 마이크로초 단위라 병목 아님 (pure_physics 분석과 동일 결론).
   배치화 대상은 오직 신경망 forward 호출.

5. **ball_ball의 i→j / j→i 방향은 2번의 개별 배치 호출로 처리한다** (합쳐서 2B' 한번에
   부르는 최적화는 하지 않음). 인덱싱 버그 리스크 대비 단순성을 우선한다.

## Phase 순서

### Phase 0 — Wavefront 스케줄러
- 신규 파일: `world_model/rssm_batch.py`
- `iter_wavefronts(shots: list[ShotData]) -> Iterator[list[(shot_idx, local_event_idx)]]`
  아직 이벤트가 남은 샷들의 "현재 포인터가 가리키는 이벤트"를 매 스텝마다 모아서 반환.
  호출부에서 event_type == BALL_BALL 그룹 / 나머지(single) 그룹으로 분리.

### Phase 1 — 배치 모델 메서드 (rssm_model.py에 추가, 기존 함수 안 건드림)
얇은 오케스트레이션 래퍼. 내부적으로 같은 `self.msg_mlp` 등을 재사용 (수식 중복 없음).

- `step_ball_ball_batch(h_i, h_j, node_i, node_j, edge) -> (h_i_new, h_j_new, delta_i, delta_j, type_i, type_j)`
  모든 인자/리턴 `(B', ...)` 텐서. list/shot 인덱싱은 전혀 모름 (순수 함수).
- `step_single_batch(h_i, node_i, normal) -> (h_i_new, delta_i, type_i)` — 동일 원칙.
- `predict_pocket_batch` : **불필요할 가능성 높음.** 기존 `predict_pocket`이 이미
  `(B', N, H)` 텐서를 그대로 받아 동작한다 (Linear가 임의 leading dim을 지원하므로).
  → 새 함수 추가 전에 테스트로 먼저 확인, 되면 기존 함수 재사용.

### Phase 2 — 배치 손실 함수 (train_rssm.py)
- `compute_batch_ss_loss(model, shots: list[ShotData], ss_prob, params, device, weights...)
  -> (batch_loss 구성요소, per_shot_n_events)`
- 내부 상태(전부 Python list, 원칙 1):
  `h_list[s][bi]`, `pred_rvws_list[s]: dict`, `vel_sum_list[s]`, `type_sum_list[s]`,
  `pocket_sum_list[s]`, `n_ev_list[s]`
- 매 wavefront:
  1. 그룹별(ball_ball / single)로 활성 아이템의 h/node/edge를 `torch.stack`으로 묶음
  2. Phase 1 배치 함수 1회 호출
  3. 결과를 `unbind`해서 `h_list[s][bi] = ...` 로 되돌림 (list reassignment)
  4. per-shot loss 누적도 list reassignment로 갱신 (`vel_sum_list[s] = vel_sum_list[s] + x`)
- 마지막에 원칙 2에 따라 "샷별 정규화 평균의 평균"으로 최종 batch loss 구성.

### Phase 3 — 학습 루프 통합
- `train()`의 `for shot in train_shots` 루프를 `for batch in chunks(train_shots, batch_size)`로 교체
- `TrainConfig.batch_size` 필드 추가 (기본값 후보: 기존 accum_steps 값과 동일하게 시작)
- `accum_steps` 의미 재정의 문서화: "샷 개수" → "배치 개수" (기본 1로, 필요시 추가 누적)
- (선택, 이번 스코프 아님) length bucketing — 롱테일 샷으로 인한 웨이브프론트 후반부
  배치 축소 완화용. 랜덤성 감소 트레이드오프 있어 기본 OFF로 시작.

### Phase 4 — evaluate() 배치화
- **이번 스코프 제외.** train()이 훨씬 빨라지면 evaluate()가 상대적 병목이 될 수 있음.
  근거(프로파일링) 나오면 후속 작업으로 진행.

## 테스트 계획 (Phase마다 1:N 매칭)

**[Phase 0]**
- `test_wavefront_all_events_visited_exactly_once`
- `test_wavefront_active_count_nonincreasing`
- `test_wavefront_type_grouping_correct`

**[Phase 1 — step_ball_ball_batch]**
- `test_step_ball_ball_batch_matches_single_call` ← 핵심 회귀 테스트
  (동일 입력으로 기존 `step_ball_ball`을 B'번 루프 vs 배치 1번 호출, 수치 일치 확인)
- `test_step_ball_ball_batch_output_shapes`
- `test_step_ball_ball_batch_gradients_flow`

**[Phase 1 — step_single_batch]** (동일 3종 미러링)
- `test_step_single_batch_matches_single_call`
- `test_step_single_batch_output_shapes`
- `test_step_single_batch_gradients_flow`

**[Phase 1 — predict_pocket 배치 지원 확인]**
- `test_predict_pocket_accepts_batched_BNH_tensor` (새 함수 없이 기존 함수로 충분한지 검증)

**[Phase 2 — compute_batch_ss_loss]**
- `test_batch_loss_size1_matches_compute_shot_ss_loss` ← 핵심 회귀 테스트 (원칙 2 검증)
- `test_batch_loss_teacher_forcing_exact_match_sum_of_per_shot` (ss_prob=1.0, RNG 순서 무관하게 정확 일치해야 함)
- `test_batch_loss_free_running_finite_no_crash` (ss_prob=0.0, 정확 일치는 기대 안 함)
- `test_batch_loss_gradients_flow`
- `test_batch_loss_n_events_matches_sum_of_shot_lengths`
- `test_batch_loss_heterogeneous_lengths_no_crash` (2 vs 30 이벤트 샷 혼합)
- `test_batch_loss_mixed_ball_ball_and_single_in_same_wavefront`
- `test_batch_loss_legacy_pkl_node_i_not_none_still_respected` (fix②/⑤ 하위호환 회귀)
- `test_loss_accumulator_list_no_inplace_error` (원칙 1 일반화 검증, h 테스트와 동일 패턴)

**[Phase 3 — 학습 루프 통합]**
- `test_train_with_batch_size_smoke` (batch_size>1로 train() 완주, result.json 생성)
- `test_train_batch_size_1_equivalent_to_legacy_shot_loop` (batch_size=1 → 기존 결과와 동일 궤적)

## 사이드 이펙트 더블체크

1. **RNG 재현성 (ss_prob ∈ (0,1) 구간)**: wavefront 인터리빙으로 RNG 소비 순서가
   기존 순차 처리와 달라짐 → free-running SS 구간에서는 기존 학습과 bit-for-bit
   재현 불가 (다른 유효한 값은 나옴, 크래시 아님). teacher forcing(ss_prob=1.0)
   구간은 분기 결과가 RNG 값과 무관하게 항상 True라서 **영향 없음**.
   → v5 이후 재학습 시 정확한 값 재현은 포기해야 함, 문서화 필요.

2. **accum_steps 재해석**: 기존 config(`rssm_v4/v5`: accum_steps=32)를 그대로 쓰려면
   `batch_size=32, accum_steps=1`로 변경해야 동등한 gradient가 나옴. 두 개념을
   혼동하면(batch_size=32 AND accum_steps=32 동시 적용) 기존 대비 32배 더 큰
   유효 배치가 되어 LR 재튜닝이 필요해짐 — 학습 루프 문서/주석에 명시 필요.

3. **Python 오버헤드 (stack/unbind 호출 자체)**: wavefront마다 소규모 Python 레벨
   list comprehension + stack 호출 발생. avg_events≈5.8이라 wavefront 수 자체는
   적지만, 실측 프로파일링 전까지는 "이득이 확실하다"고 단정 못 함 → 구현 후
   벤치마크로 검증 필요 (문제 되면 추후 인덱스 텐서 기반 gather로 개선).

4. **evaluate() 미배치화로 인한 train/eval 속도 불균형**: train이 빨라지면 상대적으로
   eval이 더 느리게 느껴질 수 있음. 이번 스코프 제외, 필요시 후속.

5. **메모리**: h_dim=64, n_balls 1~3, batch_size 두 자릿수 수준이면 메모리 영향 무시 가능.

6. **n_balls 이질성 가드**: 배치 함수에 assert 추가 — 현재는 TrainConfig가 학습 1회당
   n_balls 고정이라 위반 안 되지만, 향후 커리큘럼에서 n_balls 혼합 시 조용히
   깨지는 것 방지.

7. **레거시 pkl 하위호환**: `ev.node_i is not None`일 때 재사용하는 fix②/⑤ 로직을
   배치 버전에서도 그대로 유지해야 함 — 새 배치 아이템 구성 코드에서 개별적으로
   같은 분기를 복제해야 하며, 이 분기 자체를 테스트로 고정.

8. **RolloutEngine(rssm_rollout.py)은 영향 없음**: 실제 배포/추론 시나리오는 항상
   "샷 1개를 실시간으로 시뮬레이션"이라 배치할 대상 자체가 없음. 새 배치 함수는
   훈련 전용 추가 기능이며 기존 step_ball_ball/step_single을 대체하지 않음
   → 회귀 리스크 없음.

9. **길이 버킷팅(선택 기능)의 랜덤성 감소**: 도입 시 SGD의 셔플 엔트로피가 줄어듦
   (비슷한 길이 샷끼리 항상 같이 배치됨). 기본 OFF로 시작해 필요성이 실측으로
   확인되면 별도 플래그로 켠다.
