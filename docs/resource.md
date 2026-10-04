# Jev 실습 리소스

공유 [대화](https://chatgpt.com/share/6ab225ec-e558-83e8-a7ed-7f5445b2e2d2)의 관심사인 **병렬 후보 평가, decision head, 확률 보정**을 따라가는 학습 순서다. 공개 복제 프로젝트는 Jev의 비공개 내부 구현을 증명하지 않는다. 특히 NanoJev는 독립 구현이다.

현재 노트북은 [Jev 소개와 API 실습](https://github.com/simonjisu/learning-jev/blob/main/notebooks/jev.ipynb), [Jev-like 방법론과 학습 설계](https://github.com/simonjisu/learning-jev/blob/main/notebooks/jevlike_training.ipynb) 두 개다. Jev-like 노트북의 **1.11.1~1.11.7**에는 Laya·decider의 학습 중 확률 목표, 학습 후 temperature fitting, confidence 정의와 로컬 합성 검증을 비교했다.

## 1. 출력 계약 이해하기

| 리소스 | 확인할 것 | 실습 |
| --- | --- | --- |
| [TypeSafe primitives](https://docs.typesafe.ai/primitives) | Choice, Score, Noul의 입력과 반환 필드; 같은 state를 공유하는 독립 질문 | 동일한 고객 문의에 세 질문을 동시에 작성하고 어떤 필드가 필요한지 표로 적기 |
| [State](https://docs.typesafe.ai/concepts/state) · [Choice](https://docs.typesafe.ai/primitives/choice) · [Score](https://docs.typesafe.ai/primitives/score) · [Noul](https://docs.typesafe.ai/primitives/noul) | 선택지 설명, 순서 있는 level, yes 확률의 의미 | 선택지 순서 변경, Score level 설명 변경이 기대 결과에 주는 영향 예상하기 |
| [TypeSafe ML primer](https://docs.typesafe.ai/introduction/machine-learning-primer) | calibrated decision과 RLCD의 공개 설명 | 예측 확률과 실제 관측 결과를 분리해 기록할 평가 표 설계하기 |

TypeSafe 문서의 `noul`은 NanoJev 로컬 요청의 `boolean`에 대응하지만 필드명과 응답 형식이 같지는 않다. [NanoJev의 계약 비교](https://github.com/TianyuCodings/NanoJev/blob/76fdfc9ecdca45a9bcef17991a07d3041a87685a/docs/TYPESAFE_CONTRACT.md)를 먼저 읽으면 혼동을 줄일 수 있다.

## 2. NanoJev 입력 → 출력 따라가기

아래는 공개 소스 `76fdfc9`를 기준으로 한 입력·출력 추적 순서다. NanoJev 전용 노트북은 제거했으며, 구조 비교 설명은 Jev-like 노트북의 1.5에서 읽을 수 있다.

1. [샘플 요청](https://github.com/TianyuCodings/NanoJev/blob/76fdfc9ecdca45a9bcef17991a07d3041a87685a/research/toy_inference_example.json): `states → questions → criteria` 구조와 각 질문의 후보 수 세기.
2. [`prepare_examples`](https://github.com/TianyuCodings/NanoJev/blob/76fdfc9ecdca45a9bcef17991a07d3041a87685a/scripts/predict_toy_decisions.py): 질문마다 candidate path가 어떻게 문자열과 token으로 바뀌는지 확인. Boolean은 **semantic path 하나**만 만든다.
3. [`DecisionModel.forward`](https://github.com/TianyuCodings/NanoJev/blob/76fdfc9ecdca45a9bcef17991a07d3041a87685a/scripts/train_toy_decisions.py): 모든 path를 batch 축에 펼쳐 backbone을 한 번 호출하고, 마지막 hidden state → LayerNorm → scalar head → Choice에 한해 set attention 보정 순으로 계산한다.
4. [`DecisionPredictor.predict`](https://github.com/TianyuCodings/NanoJev/blob/76fdfc9ecdca45a9bcef17991a07d3041a87685a/scripts/predict_toy_decisions.py): temperature와 softmax를 거쳐 Choice의 `choice`, Boolean의 `p_true`, Score의 확률 가중 평균을 만든다.
5. [`serve_decisions.py`](https://github.com/TianyuCodings/NanoJev/blob/76fdfc9ecdca45a9bcef17991a07d3041a87685a/scripts/serve_decisions.py): 로컬 HTTP `POST /api/evaluate`에서 같은 predictor를 재사용한다.

핵심 구분: **하나의 backbone forward는 여러 candidate sequence의 batch 처리**를 뜻한다. 현재 경로는 반복된 state prefix를 각 후보에 다시 넣는다. 공유 KV cache 최적화와 실제 Jev의 내부 방식은 이 코드로 입증할 수 없다. `softmax` 결과가 합계 1이라는 사실만으로 calibration도 입증되지 않는다.

## 3. 모델 실행과 확률 평가

| 리소스 | 실습 | 준비 |
| --- | --- | --- |
| [NanoJev README와 quick start](https://github.com/TianyuCodings/NanoJev) · [공개 모델](https://huggingface.co/C-Tianyu/NanoJev) | `unified-games-v1` checkpoint로 샘플 요청을 예측하고 `execution.candidate_paths`, `forward_passes`, `autoregressive_decode_steps` 확인 | 원본 저장소의 실행 안내, 모델 파일, PyTorch/Transformers, CUDA GPU |
| [질문 계약 오프라인 테스트](https://github.com/TianyuCodings/NanoJev/blob/76fdfc9ecdca45a9bcef17991a07d3041a87685a/scripts/test_question_contract.py) | ID 이름과 무관한 질문을 바꿔도 기존 path token이 그대로인지 검사 | Python 표준 라이브러리만 필요 |
| [훈련 파이프라인](https://github.com/TianyuCodings/NanoJev/blob/76fdfc9ecdca45a9bcef17991a07d3041a87685a/research/pipeline_runbook.md) · [RLCD inspired 실험](https://github.com/TianyuCodings/NanoJev/blob/76fdfc9ecdca45a9bcef17991a07d3041a87685a/docs/RLCD_EXPERIMENT.md) | CE, Brier, temperature 변경 전후 NLL/Brier/ECE를 분리해 비교 | 실제 outcome label과 별도 calibration split 필요 |

현재 공개 unified checkpoint의 설명은 complete-question cross entropy 훈련이다. NanoJev의 RLCD inspired 실험은 TypeSafe의 비공개 RLCD 학습법을 복원한 것이라고 해석하면 안 된다.

## 4. 구조 비교 실습

공유 대화에 나온 대안을 비교하려면 [mini-jev](https://github.com/r-ms/mini-jev)의 answer-token logits, [jevlike](https://github.com/vinnylarouge/jevlike)의 option-conditioned attention, NanoJev의 candidate path + decision head를 같은 `(state, question, options)` 사례에 대입해 본다. 각 방식에서 **여러 토큰으로 된 새 옵션을 추가할 때 무엇을 다시 계산하는지**, **후보 간 상호작용이 어디서 생기는지**, **보정이 어디에 적용되는지**를 기록하면 좋다. 이 비교는 아키텍처 아이디어 비교이며 Jev 내부 구조 판정은 아니다.

## 5. Contrastive Language Models

[Jev-like 방법론 노트북](https://github.com/simonjisu/learning-jev/blob/main/notebooks/jevlike_training.ipynb)의 1.7은 [technical report](https://contrastive-lm.notion.site/)와 [공개 학습 코드](https://github.com/Contrastive-LM/CLM/tree/bb42c6c5bf914fd449bed2f6ca65be80602cb1f7)를 참고해 dual-encoder와 InfoNCE를 설명한다. CLM 전용 실행 노트북은 제거했으며, 공개 recipe에서 확인할 요소는 다음과 같다.

- frozen state/action embedding 위에서 서로 다른 projection head 두 개를 학습
- 공개 `finetune.py`의 group-masked bidirectional InfoNCE
- pre-training → synthetic hard-negative mid-training → 60/40 agentic/replay post-training의 단계 구분
- candidate dot product가 Choice, Noul, Score 확률로 바뀌는 과정 확인

CLM은 2026-09-23 공개된 technical report다. 공개 저자 성능은 독립 재현으로 확인해야 하며, CLM의 구조가 Jev의 실제 내부 구조와 같다는 증거는 없다.

## 6. Primitive별 학습 데이터셋

[Jev-like 학습 노트북](https://github.com/simonjisu/learning-jev/blob/main/notebooks/jevlike_training.ipynb)의 데이터 구성은 다음과 같다.

| Primitive | 데이터셋 | 학습 정답 |
| --- | --- | --- |
| Choice | [BANKING77](https://github.com/PolyAI-LDN/task-specific-datasets) | 은행 문의의 77개 의도 중 하나 |
| Noul | [GoEmotions simplified](https://huggingface.co/datasets/google-research-datasets/go_emotions) | 감정별 포함 여부 |
| Score | [SST-5 sentence-level](https://huggingface.co/datasets/SetFit/sst5/tree/e51bdcd8cd3a30da231967c1a249ba59361279a3) | 0 매우 부정적 → 1 부정적 → 2 중립 → 3 긍정적 → 4 매우 긍정적 |

SST-5는 영화 리뷰의 **순서형 감성 극성** 실습용이다. train 8,544 / validation 1,101 / test 2,210개이며 감정별 강도나 일반 위험도 정답은 제공하지 않는다. [Socher et al., EMNLP 2013](https://aclanthology.org/D13-1170/), [SetFit dataset card](https://huggingface.co/datasets/SetFit/sst5/blob/e51bdcd8cd3a30da231967c1a249ba59361279a3/README.md). 선택한 SetFit 저장소에는 데이터 라이선스가 명시되어 있지 않으며 manifest에도 이를 기록했다.

`python scripts/download_decision_datasets.py`로 세 데이터셋을 `.cache/jevlike-training/datasets/`에 내려받는다. 고정 revision, 원본 URL, SHA-256은 데이터별 `manifest.json`에 기록한다. 노트북의 SST-5 로딩·균형 표본 추출·Score 학습 레코드 변환 예제는 미실행 상태다.
