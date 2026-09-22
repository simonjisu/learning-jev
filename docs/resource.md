# Jev 실습 리소스

공유 [대화](https://chatgpt.com/share/6ab225ec-e558-83e8-a7ed-7f5445b2e2d2)의 관심사인 **병렬 후보 평가, decision head, 확률 보정**을 따라가는 학습 순서다. 공개 복제 프로젝트는 Jev의 비공개 내부 구현을 증명하지 않는다. 특히 NanoJev는 독립 구현이다.

## 1. 출력 계약 이해하기

| 리소스 | 확인할 것 | 실습 |
| --- | --- | --- |
| [TypeSafe primitives](https://docs.typesafe.ai/primitives) | Choice, Score, Noul의 입력과 반환 필드; 같은 state를 공유하는 독립 질문 | 동일한 고객 문의에 세 질문을 동시에 작성하고 어떤 필드가 필요한지 표로 적기 |
| [State](https://docs.typesafe.ai/concepts/state) · [Choice](https://docs.typesafe.ai/primitives/choice) · [Score](https://docs.typesafe.ai/primitives/score) · [Noul](https://docs.typesafe.ai/primitives/noul) | 선택지 설명, 순서 있는 level, yes 확률의 의미 | 선택지 순서 변경, Score level 설명 변경이 기대 결과에 주는 영향 예상하기 |
| [TypeSafe ML primer](https://docs.typesafe.ai/introduction/machine-learning-primer) | calibrated decision과 RLCD의 공개 설명 | 예측 확률과 실제 관측 결과를 분리해 기록할 평가 표 설계하기 |

TypeSafe 문서의 `noul`은 NanoJev 로컬 요청의 `boolean`에 대응하지만 필드명과 응답 형식이 같지는 않다. [NanoJev의 계약 비교](https://github.com/TianyuCodings/NanoJev/blob/76fdfc9ecdca45a9bcef17991a07d3041a87685a/docs/TYPESAFE_CONTRACT.md)를 먼저 읽으면 혼동을 줄일 수 있다.

## 2. NanoJev 입력 → 출력 따라가기

이 저장소의 [입력 출력 추적 노트북](../notebooks/nanojev_input_output.ipynb)은 공개 소스 `76fdfc9`를 기준으로 한다. 첫 부분은 Python 표준 라이브러리만으로 실행된다. 원본 저장소를 찾지 못하면 노트북이 해당 커밋을 임시 디렉터리에 복제하므로 최초 실행에는 네트워크가 필요하다.

1. [샘플 요청](https://github.com/TianyuCodings/NanoJev/blob/76fdfc9ecdca45a9bcef17991a07d3041a87685a/research/toy_inference_example.json): `states → questions → criteria` 구조와 각 질문의 후보 수 세기.
2. [`prepare_examples`](https://github.com/TianyuCodings/NanoJev/blob/76fdfc9ecdca45a9bcef17991a07d3041a87685a/scripts/predict_toy_decisions.py): 질문마다 candidate path가 어떻게 문자열과 token으로 바뀌는지 확인. Boolean은 **semantic path 하나**만 만든다.
3. [`DecisionModel.forward`](https://github.com/TianyuCodings/NanoJev/blob/76fdfc9ecdca45a9bcef17991a07d3041a87685a/scripts/train_toy_decisions.py): 모든 path를 batch 축에 펼쳐 backbone을 한 번 호출하고, 마지막 hidden state → LayerNorm → scalar head → Choice에 한해 set attention 보정 순으로 계산한다.
4. [`DecisionPredictor.predict`](https://github.com/TianyuCodings/NanoJev/blob/76fdfc9ecdca45a9bcef17991a07d3041a87685a/scripts/predict_toy_decisions.py): temperature와 softmax를 거쳐 Choice의 `choice`, Boolean의 `p_true`, Score의 확률 가중 평균을 만든다.
5. [`serve_decisions.py`](https://github.com/TianyuCodings/NanoJev/blob/76fdfc9ecdca45a9bcef17991a07d3041a87685a/scripts/serve_decisions.py): 로컬 HTTP `POST /api/evaluate`에서 같은 predictor를 재사용한다.

핵심 구분: **하나의 backbone forward는 여러 candidate sequence의 batch 처리**를 뜻한다. 현재 경로는 반복된 state prefix를 각 후보에 다시 넣는다. 공유 KV cache 최적화와 실제 Jev의 내부 방식은 이 코드로 입증할 수 없다. `softmax` 결과가 합계 1이라는 사실만으로 calibration도 입증되지 않는다.

## 3. 모델 실행과 확률 평가

| 리소스 | 실습 | 준비 |
| --- | --- | --- |
| [NanoJev README와 quick start](https://github.com/TianyuCodings/NanoJev) · [공개 모델](https://huggingface.co/C-Tianyu/NanoJev) | `unified-games-v1` checkpoint로 샘플 요청을 예측하고 `execution.candidate_paths`, `forward_passes`, `autoregressive_decode_steps` 확인 | 모델 파일, PyTorch/Transformers, CUDA GPU. 노트북의 실제 추론 셀은 준비되지 않으면 건너뜀 |
| [질문 계약 오프라인 테스트](https://github.com/TianyuCodings/NanoJev/blob/76fdfc9ecdca45a9bcef17991a07d3041a87685a/scripts/test_question_contract.py) | ID 이름과 무관한 질문을 바꿔도 기존 path token이 그대로인지 검사 | Python 표준 라이브러리만 필요 |
| [훈련 파이프라인](https://github.com/TianyuCodings/NanoJev/blob/76fdfc9ecdca45a9bcef17991a07d3041a87685a/research/pipeline_runbook.md) · [RLCD inspired 실험](https://github.com/TianyuCodings/NanoJev/blob/76fdfc9ecdca45a9bcef17991a07d3041a87685a/docs/RLCD_EXPERIMENT.md) | CE, Brier, temperature 변경 전후 NLL/Brier/ECE를 분리해 비교 | 실제 outcome label과 별도 calibration split 필요 |

현재 공개 unified checkpoint의 설명은 complete-question cross entropy 훈련이다. NanoJev의 RLCD inspired 실험은 TypeSafe의 비공개 RLCD 학습법을 복원한 것이라고 해석하면 안 된다.

## 4. 구조 비교 실습

공유 대화에 나온 대안을 비교하려면 [mini-jev](https://github.com/r-ms/mini-jev)의 answer-token logits, [jevlike](https://github.com/vinnylarouge/jevlike)의 option-conditioned attention, NanoJev의 candidate path + decision head를 같은 `(state, question, options)` 사례에 대입해 본다. 각 방식에서 **여러 토큰으로 된 새 옵션을 추가할 때 무엇을 다시 계산하는지**, **후보 간 상호작용이 어디서 생기는지**, **보정이 어디에 적용되는지**를 기록하면 좋다. 이 비교는 아키텍처 아이디어 비교이며 Jev 내부 구조 판정은 아니다.
