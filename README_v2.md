# WM-811K 웨이퍼 불량 패턴 분류: 클래스 불균형 처리와 Knowledge Distillation (실험 결과 최종본)

AI System Architecture Term Project — 반도체 웨이퍼 맵 불량 패턴 분류에서 **클래스 불균형 처리**(데이터 엔지니어링)와 **Knowledge Distillation**(경량 AI/ML) 두 기법을 적용하고, 각각의 Ablation Study와 두 기법의 시너지를 분석한다.

> 이 문서는 Tesla P100-PCIE-16GB GPU 환경에서 `run_all.py`로 실행한 **실제 실험 결과**를 기반으로 작성되었다. 실험 환경 및 추가 인사이트가 기존 README와 다를 수 있다.

---

## 핵심 결과

| 구성 | Params | 압축비 | Macro-F1 | 소수 클래스 평균 F1 |
|---|---|---|---|---|
| Teacher baseline (ResNet-18) | 11.17M | 1× | 0.8868 | 0.8681 |
| Student baseline (KD·불균형처리 없음) | 0.39M | 28.6× | 0.8609 | 0.8370 |
| 불균형 처리 단독 (best: classweight) | 0.39M | 28.6× | 0.8556 | 0.8343 |
| KD 단독 (T=5, α=0.5) | 0.39M | 28.6× | 0.8679 | 0.8431 |
| classweight + KD (T=1) | 0.39M | 28.6× | **0.8690** | **0.8479** |
| **MobileNetV2 + KD** | **2.23M** | **5×** | **0.8872** | **0.8583** |

**주요 발견:**
- **불균형 처리 단독은 전체 F1을 향상시키지 못했다.** 다수 클래스 성능을 희생해 극소수 클래스만 선택적으로 개선하는 trade-off가 일관되게 관찰됐다.
- **KD 단독은 안정적인 향상**(+0.007)을 보였으며, Temperature·α에 민감했다.
- **classweight+KD는 T=1(hard label 위주)에서 시너지**, T=5(soft label)에서는 오히려 KD 단독보다 낮아졌다. soft label과 classweight의 재가중 신호가 충돌하는 것으로 해석된다.
- **MobileNetV2 + KD가 Teacher R18을 초과했다.** (0.8872 > 0.8868) 강력한 backbone과 KD의 결합이 teacher 성능 자체를 넘을 수 있음을 보였다.

---

## 데이터셋

[WM-811K](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map) — 실제 반도체 생산 라인에서 수집된 웨이퍼 맵 데이터.

- 라벨 보유 데이터: **172,950장** (9개 클래스)
- 입력: 웨이퍼 맵을 64×64 단일 채널로 리사이즈 (NEAREST 보간)
- Train / Test = 138,360 / 34,590 (stratified split, seed=42)

| 클래스 | 샘플 수 | 비율 |
|---|---|---|
| none (정상) | 147,431 | 85.24% |
| Edge-Ring | 9,680 | 5.60% |
| Edge-Loc | 5,189 | 3.00% |
| Center | 4,294 | 2.48% |
| Loc | 3,593 | 2.08% |
| Scratch | 1,193 | 0.69% |
| Random | 866 | 0.50% |
| Donut | 555 | 0.32% |
| Near-full | 149 | 0.09% |

최다(none) : 최소(Near-full) = 약 **1,000 : 1**의 심각한 클래스 불균형.

---

## 적용 기법

### 카테고리 1 — 클래스 불균형 처리
| 기법 | 수준 | 설명 |
|---|---|---|
| Random Oversampling | 데이터 | `WeightedRandomSampler`로 소수 클래스 과표집 |
| SMOTE | 데이터 | 합성 샘플 생성 — 138K→1M+ 폭발로 **실험 제외** |
| Focal Loss | 알고리즘 | 어려운 샘플에 손실 가중 (γ=2) |
| Class Weight | 알고리즘 | 역빈도 기반 클래스 가중치 |

### 카테고리 2 — Knowledge Distillation
- Teacher(ResNet-18, 11.17M) → Student(경량 CNN, 0.39M), **약 28.6배 압축**
- KD Loss: `α × KL_div(log_softmax(s/T), softmax(t/T)) × T² + (1−α) × CE(s, y)`

---

## 모델

| 모델 | 파라미터 | 역할 |
|---|---|---|
| ResNet-18 | 11.17M | Teacher |
| ResNet-50 | 23.52M | Teacher 크기 ablation — **R18과 차이 0.0014로 미미, 실험 제외** |
| StudentCNN | 0.39M | 메인 Student (Conv-BN-ReLU-Pool ×4 + GAP) |
| MobileNetV2 | 2.23M | Student 크기 ablation |

모든 모델은 단일 채널 64×64 입력에 맞춰 수정 (`in_channels=1`, ResNet은 `conv1` 3×3 stride=1, maxpool 제거).

---

## 실험 구성

| Step | 내용 | 실험 수 |
|---|---|---|
| Step 1 | Teacher(ResNet-18) baseline 학습 | 1 |
| Step 2 | Student 불균형 처리 Ablation (none / oversample / focal / classweight) | 4 |
| Step 3 | KD Ablation (Temperature ×5, α ×3, Student 크기) | 8 |
| Step 4 | 시너지 분석 (focal+KD, classweight+KD T=5) | 2 |
| Step 5 | 종합 Ablation (oversample+KD, classweight+KD T=1) | 2 |

**설계 노트**: Teacher는 불균형 처리 없이도 Macro-F1 0.8868로 충분히 강해 불균형 처리 효과가 드러나지 않는다. 따라서 불균형 처리 Ablation은 더 약한 **StudentCNN**을 대상으로 삼았다. augmentation(flip)은 불균형 완화 효과가 있으므로 순수 효과 측정을 위해 `augment=False`로 통제했다.

---

## Step 2 — 클래스 불균형 처리 Ablation (StudentCNN 기준)

| 기법 | Macro-F1 | 소수 평균 F1 | Near-full F1 | Scratch F1 |
|---|---|---|---|---|
| **none (기준)** | **0.8609** | **0.8370** | 0.949 | 0.691 |
| oversample | 0.8352 | 0.8096 | 0.892 | 0.668 |
| focal | 0.8517 | 0.8197 | 0.889 | 0.706 |
| classweight | 0.8556 | 0.8343 | 0.892 | 0.729 |

- 전체 Macro-F1은 기준(none)이 가장 높다. 불균형 처리 기법이 단독으로는 전체 성능을 떨어뜨린다.
- Near-full F1은 기준(0.949)이 이미 높아, 불균형 처리가 추가 개선을 만들지 못했다.
- classweight는 Scratch(0.729)에서 다른 기법보다 유리해 특정 소수 클래스 개선에는 효과적이다.

---

## Step 3 — KD Ablation

### Temperature (α=0.5 고정, StudentCNN)

| T | Macro-F1 | 소수 평균 F1 | Near-full F1 | Scratch F1 |
|---|---|---|---|---|
| 1 | 0.8554 | 0.8231 | 0.885 | 0.689 |
| 3 | 0.8675 | 0.8515 | 0.921 | 0.712 |
| **5** | **0.8679** | 0.8431 | 0.892 | 0.701 |
| 10 | 0.8635 | 0.8318 | 0.875 | 0.691 |
| **20** | 0.8686 | 0.8466 | **0.951** | 0.702 |

- 전체 F1 최적: **T=5~20** 구간
- Near-full 최적: **T=20** (0.951) — 높은 Temperature일수록 soft label이 부드러워져 희귀 클래스 학습에 유리

### α (T=5 고정, StudentCNN)

| α | Macro-F1 | 소수 평균 F1 |
|---|---|---|
| 0.1 (hard label 위주) | 0.8572 | 0.8278 |
| **0.5 (균형)** | **0.8679** | **0.8431** |
| 0.9 (soft label 위주) | 0.8608 | 0.8267 |

- α=0.5(균형)이 최적. soft label 과다(α=0.9) 시 소수 클래스 F1이 하락하는 경향.

### Student 크기

| Student | Params | 압축비 | Macro-F1 | 소수 평균 F1 |
|---|---|---|---|---|
| StudentCNN | 0.39M | 28.6× | 0.8679 | 0.8431 |
| **MobileNetV2** | **2.23M** | **5×** | **0.8872** | **0.8583** |

MobileNetV2 + KD는 Teacher(0.8868)를 초과했다. 6배 더 작은 StudentCNN이 0.019 차 이내로 근접한 것도 주목할 만하다.

---

## Step 4 & 5 — 시너지 Ablation

| 조합 | 단독 F1 | KD단독 F1 | 결합 F1 | KD 대비 | 판정 |
|---|---|---|---|---|---|
| focal + KD (T=5) | 0.8517 | 0.8679 | 0.8652 | −0.003 | ⚠️ 약한 상쇄 |
| classweight + KD (T=5) | 0.8556 | 0.8679 | 0.8635 | −0.004 | ⚠️ 상쇄 |
| oversample + KD (T=5) | 0.8352 | 0.8679 | 0.8654 | −0.003 | ⚠️ 약한 상쇄 |
| **classweight + KD (T=1)** | 0.8556 | 0.8554 | **0.8690** | **+0.014** | ✅ 시너지 |

T=5 기준으로 모든 불균형 처리+KD 조합이 KD 단독보다 낮았다. **classweight는 T=1에서만 시너지**를 냈다. T=1은 soft label 비중을 최소화해 classweight의 재가중 신호와 충돌하지 않기 때문으로 해석된다.

---

## 추가 인사이트

### 클래스별 F1 상세 비교

| 클래스 | Teacher R18 | Student (none) | KD T=5 | **MobileNet+KD** | CW+KD T=1 |
|---|---|---|---|---|---|
| none | 0.990 | 0.988 | 0.988 | **0.991** | 0.989 |
| Center | 0.922 | 0.912 | 0.922 | **0.936** | 0.916 |
| Donut | **0.889** | 0.822 | 0.883 | **0.889** | 0.866 |
| Edge-Loc | 0.843 | 0.815 | 0.823 | **0.857** | 0.818 |
| Edge-Ring | 0.981 | 0.974 | 0.979 | **0.983** | 0.980 |
| Loc | 0.771 | 0.711 | 0.727 | **0.785** | 0.726 |
| Near-full | 0.933 | **0.949** | 0.892 | 0.921 | 0.896 |
| Random | **0.911** | 0.886 | 0.897 | 0.867 | 0.873 |
| Scratch | 0.740 | 0.691 | 0.701 | **0.757** | **0.757** |

**주목할 점:**
- **Near-full**: Student baseline(0.949)이 Teacher(0.933)보다 높다. 소수 클래스에서 작은 모델이 더 단순한 결정 경계를 학습해 오히려 유리할 수 있다.
- **MobileNetV2+KD**: Near-full·Random을 제외한 모든 클래스에서 최고. Loc(0.785), Edge-Loc(0.857), Scratch(0.757) 등 중간 희귀 클래스에서 두드러진다.
- **Scratch**: 전 실험에서 가장 낮은 클래스. MobileNet+KD와 CW+KD T=1에서 공동 최고(0.757).

### KD Temperature별 Near-full vs 전체 F1 Trade-off

| T | Macro-F1 | Near-full F1 | 추천 상황 |
|---|---|---|---|
| T=3 | 0.8675 | 0.921 | 전반적 균형 |
| T=5 | 0.8679 | 0.892 | 전체 F1 최적화 |
| T=20 | 0.8686 | **0.951** | Near-full 특화 목표 |

### 기술별 기여도 요약

| 기법 | Macro-F1 Δ | 방향 | 비고 |
|---|---|---|---|
| KD (T=5, α=0.5) | +0.007 | ✅ 상승 | 가장 안정적인 기여 |
| classweight 단독 | −0.005 | ❌ 하락 | 단독 사용 시 역효과 |
| focal 단독 | −0.009 | ❌ 하락 | 단독 사용 시 역효과 |
| oversample 단독 | −0.026 | ❌ 하락 | 단독 사용 시 최악 |
| classweight + KD (T=1) | +0.008 | ✅ 상승 | 조합 시 시너지 |
| MobileNet backbone | +0.026 | ✅ 상승 | Teacher 초과 달성 |

### 목적별 최적 구성

| 목적 | 추천 구성 | Macro-F1 | 압축비 |
|---|---|---|---|
| 최고 성능 | **MobileNetV2 + KD** | **0.8872** | 5× |
| 최경량 균형 | StudentCNN + CW + KD (T=1) | 0.8690 | **28.6×** |
| KD 단독 최적 | StudentCNN + KD (T=20) | 0.8686 | 28.6× |
| Near-full 특화 | StudentCNN + KD (T=20) | Near-full **0.951** | 28.6× |

---

## 프로젝트 구조

```
wm811k_term/
├── data_prep.py                      # LSWMD.pkl → 라벨 필터 → 리사이즈 → npz
├── run_all.py                        # Step 1~5 전체 자동 실행 (재시작 안전)
├── notebooks/
│   ├── Step1_Teacher_Baseline_v2.ipynb
│   ├── Step2_Imbalance_Ablation.ipynb
│   ├── Step3_KD_Ablation.ipynb
│   ├── Step4_Synergy.ipynb
│   └── Step5_Full_Ablation.ipynb     # 종합 ablation + 시각화
├── data/
│   └── wm811k_64.npz                 # 전처리된 데이터 (172,950장, 54.6MB)
├── checkpoints/                      # 학습된 모델 가중치 (.pt)
└── results/                          # 실험 결과 JSON (17개)
```

## 실행 방법

### 1. 데이터 전처리 (1회)
```bash
python data_prep.py --pkl LSWMD.pkl --img-size 64
# → data/wm811k_64.npz 생성 (54.6MB)
```

### 2. 전체 실험 자동 실행
```bash
CUDA_VISIBLE_DEVICES=0 python3 -u run_all.py
# result_exists() 체크로 완료된 실험은 자동 스킵 → 중단 후 재시작 안전
```

### 환경
- Python 3, PyTorch 2.3+, scikit-learn, imbalanced-learn, scikit-image
- GPU: NVIDIA CUDA (mixed precision `torch.amp` 자동 적용)
- 학습 설정: 30 epoch, batch 128, SGD + Cosine Annealing LR (num_workers=0)

## 재현성

- 데이터 분할 및 모델 초기화에 `seed=42` 고정
- 모든 실험은 동일한 `wm811k_64.npz`와 동일 학습 설정으로 통일
- 평가 지표: Accuracy, Macro-F1, 소수 클래스(Near-full / Donut / Random / Scratch) 평균 F1

---

## 결론

1. **클래스 불균형 처리 단독은 전체 성능을 보장하지 않는다.** StudentCNN은 기본 학습만으로도 다수 클래스에서 강한 성능을 보이며, 불균형 기법은 특정 희귀 클래스를 개선하는 대신 전체 F1을 낮추는 trade-off를 만든다.

2. **Knowledge Distillation은 경량 모델의 성능을 안정적으로 끌어올린다.** T=5~20, α=0.5가 최적이며, Teacher 크기(R18 vs R50)보다 Temperature 설정이 더 민감한 하이퍼파라미터다.

3. **시너지는 기법과 T 값의 조합에 따라 달라진다.** classweight+KD는 T=5(soft label 위주)에서 상쇄되지만 T=1(hard label 위주)에서 시너지를 낸다. 불균형 기법의 재가중 신호와 soft label이 겹치는지 여부가 핵심이다.

4. **MobileNetV2 + KD는 Teacher를 초과할 수 있다.** (Macro-F1 0.8872 > Teacher 0.8868) 강력한 backbone과 KD의 결합은 단순한 teacher 모방을 넘어 teacher 성능 자체를 초과하는 것이 가능함을 보였다.

5. **최종 권장 구성:**
   - 극한 경량화 목표: **StudentCNN + classweight + KD (T=1)** — 28.6× 압축, Macro-F1 0.8690
   - 최고 성능 목표: **MobileNetV2 + KD** — 5× 압축, Macro-F1 0.8872 (Teacher 초과)
