# WM-811K 웨이퍼 불량 패턴 분류: 클래스 불균형 처리와 Knowledge Distillation

AI System Architecture Term Project — 반도체 웨이퍼 맵 불량 패턴 분류에서 **클래스 불균형 처리**(데이터 엔지니어링)와 **Knowledge Distillation**(경량 AI/ML) 두 기법을 적용하고, 각각의 Ablation Study와 두 기법의 시너지를 분석한다.

## 핵심 결과

| 구성 | Macro-F1 | 소수 클래스 평균 F1 |
|---|---|---|
| Student baseline (불균형 처리·KD 없음) | 0.8606 | 0.8398 |
| 불균형 처리 단독 (best: focal) | 0.8569 | 0.8334 |
| KD 단독 (T=5, α=0.5) | 0.8732 | 0.8498 |
| **classweight + KD** | **0.8744** | **0.8567** |

- **불균형 처리 단독은 전체 성능을 향상시키지 못했다.** 다수 클래스를 희생해 극소수 클래스(Near-full)만 선택적으로 개선하는 trade-off가 관찰되었다.
- **KD는 단독으로 명확한 향상**(+1.25%p)을 보였으며, Temperature·α에 민감했다.
- **시너지는 불균형 기법의 종류에 따라 달랐다.** `focal + KD`는 상쇄(−0.44%p)였으나, `classweight + KD`는 상승 시너지(+0.92%p)를 보였다.

## 데이터셋

[WM-811K](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map) — 실제 반도체 생산 라인에서 수집된 웨이퍼 맵 데이터.

- 라벨 보유 데이터: **172,950장** (9개 클래스)
- 입력: 웨이퍼 맵을 64×64 단일 채널로 리사이즈 (NEAREST 보간)
- Train/Test = 138,360 / 34,590 (stratified split, seed=42)

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

최다 클래스(none)와 최소 클래스(Near-full)의 비율은 약 **1,000배**로, 심각한 클래스 불균형을 보인다.

## 적용 기법

### 카테고리 1 — 클래스 불균형 처리 (데이터 엔지니어링)
- **Random Oversampling** (데이터 수준): `WeightedRandomSampler`로 소수 클래스 과표집
- **Focal Loss** (알고리즘 수준): 어려운 샘플에 손실 가중
- **Class Weight** (알고리즘 수준): 역빈도 기반 클래스 가중치

### 카테고리 2 — Knowledge Distillation (경량 AI/ML)
- Teacher(ResNet-18, 11.17M) → Student(경량 CNN, 0.39M), **약 28.6배 압축**
- soft label(KL divergence, T² 스케일) + hard label(CE)을 α로 결합

## 모델

| 모델 | 파라미터 | 역할 |
|---|---|---|
| ResNet-18 | 11.17M | Teacher (입력 1채널·소형 이미지용 수정) |
| ResNet-50 | 23.52M | Teacher 크기 ablation |
| StudentCNN | 0.39M | 메인 Student (Conv-BN-ReLU-Pool ×4 + GAP) |
| MobileNetV2 | 2.23M | Student 크기 ablation |

모든 모델은 웨이퍼 맵이 단일 채널이므로 첫 conv를 `in_channels=1`로 수정했다. ResNet 계열은 64×64 입력에 맞춰 `conv1`을 3×3 stride=1로 바꾸고 maxpool을 제거했다.

## 실험 구성

| Step | 내용 |
|---|---|
| Step 1 | Teacher(ResNet-18) baseline 학습 (KD용 선생 모델 확보) |
| Step 2 | Student 기준 클래스 불균형 처리 Ablation (none / oversample / focal / classweight) |
| Step 3 | KD Ablation (Temperature, α, Student 크기, Teacher 크기) |
| Step 4 | 2×2 시너지 분석 (불균형 처리 × KD) |

**설계 노트**: Teacher(ResNet-18)는 불균형 처리 없이도 macro-F1 0.89로 충분히 높아 불균형 처리의 효과가 드러나지 않았다. 따라서 불균형 처리 Ablation은 더 약한 **Student CNN**을 분석 대상으로 삼았다. 또한 augmentation(flip)은 그 자체가 불균형 완화 효과를 가지므로, 불균형 처리 기법의 순수 효과 측정을 위해 통제 변수로 제거(`augment=False`)했다.

## 주요 Ablation 결과

### 클래스 불균형 처리 (Student 기준, Step 2)
| 기법 | Macro-F1 | 소수 평균 F1 | Near-full F1 |
|---|---|---|---|
| none | 0.8606 | 0.8398 | 0.933 |
| oversample | 0.8360 | 0.8137 | 0.921 |
| focal | 0.8569 | 0.8334 | 0.951 |
| classweight | 0.8527 | 0.8406 | **0.968** |

전체 성능은 baseline이 가장 높았으나, 가장 희귀한 Near-full 클래스에서는 classweight·focal이 뚜렷하게 개선했다. → **다수 클래스를 희생해 극소수 클래스를 살리는 trade-off**.

### Knowledge Distillation (Step 3)
- **Temperature** (α=0.5 고정): T=1 → 0.8545, **T=5 → 0.8732 (최적)**, T=20 → 0.8686
- **α** (T=5 고정): α=0.1 → 0.8612, **α=0.5 → 0.8732 (최적)**, α=0.9 → 0.8708
- **Student 크기**: StudentCNN(0.39M) 0.8732 / MobileNetV2(2.23M) 0.8780 — 6배 작은 모델이 근접
- **Teacher 크기**: ResNet-18 → 0.8732 / ResNet-50 → 0.8746 — **거의 차이 없음** (capacity gap: 더 강한 Teacher가 더 나은 Student를 보장하지 않음)

### 2×2 시너지 (Step 4)
| 조합 | 단독 합 | 결합 | 시너지 |
|---|---|---|---|
| focal + KD | (−0.38) + (+1.25) | +0.43%p | **−0.44%p (상쇄)** |
| classweight + KD | (−0.80) + (+1.25) | +1.37%p | **+0.92%p (상승)** |

같은 "불균형 처리 + KD"라도 기법 종류에 따라 상호작용이 달랐다. focal은 KD의 soft label과 "어려운 샘플 강조" 메커니즘이 겹쳐 간섭한 반면, class weight는 단순 재가중이라 KD의 안정적 학습 신호와 보완적으로 작동해 시너지를 냈다.

## 프로젝트 구조

```
wm811k_term/
├── data_prep.py                      # LSWMD.pkl → 라벨 필터 → 리사이즈 → npz
├── models.py                         # Teacher / Student / MobileNetV2 정의
├── utils.py                          # Dataset, FocalLoss, 학습/평가 루프
├── train_teacher.py                  # Teacher 학습 (불균형 옵션 포함)
├── notebooks/
│   ├── Step1_Teacher_Baseline_v2.ipynb   # Teacher baseline (augment on/off)
│   ├── Step2_Imbalance_Ablation.ipynb    # Student 불균형 처리 ablation
│   ├── Step3_KD_Ablation.ipynb           # KD ablation (T / α / 크기)
│   └── Step4_Synergy.ipynb               # 2×2 시너지 분석
├── data/
│   └── wm811k_64.npz                 # 전처리된 데이터 (재현용)
├── checkpoints/                      # 학습된 모델 가중치
└── results/                          # 실험 결과 JSON
```

## 실행 방법

### 1. 데이터 전처리 (로컬, 1회)
```bash
python data_prep.py --pkl LSWMD.pkl --img-size 64
# → data/wm811k_64.npz 생성
```

### 2. 실험 (Google Colab 권장)
전처리된 `wm811k_64.npz`를 Google Drive `MyDrive/wm811k_term/data/`에 업로드한 뒤,
`notebooks/`의 Step1 → Step2 → Step3 → Step4 순서로 실행한다.
각 노트북은 Drive에 체크포인트와 결과 JSON을 저장하므로 세션이 끊겨도 안전하다.

### 환경
- Python 3, PyTorch
- 학습: NVIDIA L4 / A100 (Colab) — mixed precision(`torch.amp`) 적용
- 로컬 검증: Apple M4 Pro (MPS)

## 재현성
- 데이터 분할 및 모델 초기화에 `seed=42` 고정
- 모든 실험은 동일한 전처리 데이터(`wm811k_64.npz`)와 동일 학습 설정(30 epoch, batch 128, SGD + Cosine Annealing)으로 통일
- 평가 지표: Accuracy, Macro-F1, 소수 클래스(Near-full / Donut / Random / Scratch) 평균 F1

## 결론

1. 충분한 학습이 가능한 환경에서 **클래스 불균형 처리는 전체 성능을 보장하지 않으며**, 극소수 클래스를 선택적으로 개선하는 trade-off를 가진다.
2. **Knowledge Distillation은 경량 모델의 성능을 안정적으로 끌어올린다.** 단, Temperature·α에 민감하고 Teacher 크기에는 둔감하다.
3. **두 기법의 시너지는 조합에 따라 달라진다.** class weight + KD는 상승 시너지를, focal + KD는 상쇄를 보였다. → 불균형 기법과 KD의 메커니즘이 겹치는지 보완적인지가 시너지를 좌우한다.
4. 최종 권장 구성: **class weight + KD** (Macro-F1 0.8744, 소수 클래스 평균 F1 0.8567).
