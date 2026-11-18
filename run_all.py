"""
WM-811K 전체 실험 파이프라인 (로컬 실행용)
Steps 1 → 2 → 3 → 4 → 5 순서로 실행
결과는 results/ 체크포인트는 checkpoints/ 에 저장
"""
import os, sys, random, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.models as tvm
from sklearn.metrics import f1_score

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, 'data', 'wm811k_64.npz')
CKPT_DIR    = os.path.join(BASE_DIR, 'checkpoints')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(CKPT_DIR,    exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── 전역 설정 ─────────────────────────────────────────────────────────────────
MINORITY = ['Near-full', 'Donut', 'Random', 'Scratch']

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def get_device():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {dev}', flush=True)
    return dev

# ── 데이터 로드 ───────────────────────────────────────────────────────────────
def load_data():
    assert os.path.exists(DATA_PATH), f'데이터 없음: {DATA_PATH}'
    d = np.load(DATA_PATH, allow_pickle=True)
    X_train, y_train = d['X_train'], d['y_train']
    X_test,  y_test  = d['X_test'],  d['y_test']
    classes = list(d['classes'])
    print(f'Train: {X_train.shape}  Test: {X_test.shape}  Classes: {len(classes)}')
    return X_train, y_train, X_test, y_test, classes

# ── Dataset ──────────────────────────────────────────────────────────────────
class WaferDataset(Dataset):
    def __init__(self, X, y, train=False, augment=False):
        self.X, self.y, self.train, self.augment = X, y, train, augment
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        img = self.X[i].astype(np.float32) / 2.0
        if self.train and self.augment:
            if random.random() < 0.5: img = np.fliplr(img).copy()
            if random.random() < 0.5: img = np.flipud(img).copy()
        return torch.from_numpy(img).unsqueeze(0), int(self.y[i])

def make_loaders(X_tr, y_tr, X_te, y_te, batch=128,
                 oversample=False, augment=False, num_workers=4):
    if oversample:
        nc = len(np.unique(y_tr))
        counts  = np.bincount(y_tr, minlength=nc).astype(np.float64)
        weights = 1.0 / np.maximum(counts[y_tr], 1)
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        tr = DataLoader(WaferDataset(X_tr, y_tr, True, augment),
                        batch_size=batch, sampler=sampler,
                        num_workers=num_workers, pin_memory=True)
    else:
        tr = DataLoader(WaferDataset(X_tr, y_tr, True, augment),
                        batch_size=batch, shuffle=True,
                        num_workers=num_workers, pin_memory=True)
    te = DataLoader(WaferDataset(X_te, y_te, False, False),
                    batch_size=256, shuffle=False,
                    num_workers=num_workers, pin_memory=True)
    return tr, te

# ── 모델 ─────────────────────────────────────────────────────────────────────
def get_teacher_r18(n):
    m = tvm.resnet18(weights=None)
    m.conv1   = nn.Conv2d(1, 64, 3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc      = nn.Linear(m.fc.in_features, n)
    return m

def get_teacher_r50(n):
    m = tvm.resnet50(weights=None)
    m.conv1   = nn.Conv2d(1, 64, 3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc      = nn.Linear(m.fc.in_features, n)
    return m

class StudentCNN(nn.Module):
    def __init__(self, n):
        super().__init__()
        def b(ci, co): return nn.Sequential(
            nn.Conv2d(ci, co, 3, padding=1, bias=False),
            nn.BatchNorm2d(co), nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.features   = nn.Sequential(b(1,32), b(32,64), b(64,128), b(128,256))
        self.gap        = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, n)
    def forward(self, x):
        return self.classifier(self.gap(self.features(x)).flatten(1))

def get_student_cnn(n): return StudentCNN(n)

def get_student_mobilenet(n):
    m = tvm.mobilenet_v2(weights=None)
    old = m.features[0][0]
    m.features[0][0] = nn.Conv2d(1, old.out_channels, 3, stride=2, padding=1, bias=False)
    m.classifier[1]  = nn.Linear(m.classifier[1].in_features, n)
    return m

def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

# ── 손실 ─────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__(); self.gamma = gamma
    def forward(self, logits, targets):
        ce   = F.cross_entropy(logits, targets, reduction='none')
        loss = (1 - torch.exp(-ce)) ** self.gamma * ce
        return loss.mean()

def compute_class_weights(y, nc):
    counts = np.bincount(y, minlength=nc).astype(np.float64)
    counts[counts == 0] = 1.0
    w = 1.0 / counts
    return torch.tensor(w / w.sum() * nc, dtype=torch.float32)

def kd_loss_fn(s_logits, t_logits, targets, T, alpha, hard_crit):
    soft = F.kl_div(F.log_softmax(s_logits / T, dim=1),
                    F.softmax(t_logits / T, dim=1),
                    reduction='batchmean') * (T * T)
    hard = hard_crit(s_logits, targets)
    return alpha * soft + (1 - alpha) * hard

# ── 평가 ─────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, device, classes, minority_idx):
    model.eval(); preds, trues = [], []
    for x, y in loader:
        preds.append(model(x.to(device, non_blocking=True)).argmax(1).cpu().numpy())
        trues.append(y.numpy())
    pred, true = np.concatenate(preds), np.concatenate(trues)
    nc   = len(classes)
    acc  = float((pred == true).mean())
    macro = float(f1_score(true, pred, average='macro', zero_division=0))
    per   = f1_score(true, pred, average=None, labels=list(range(nc)), zero_division=0)
    min_f1 = float(np.array(per)[minority_idx].mean())
    return {'accuracy': acc, 'macro_f1': macro,
            'minority_f1': min_f1, 'per_class_f1': per.tolist()}

def save_result(name, metrics):
    path = os.path.join(RESULTS_DIR, f'{name}.json')
    with open(path, 'w') as f: json.dump(metrics, f, indent=2)

def load_result(name):
    path = os.path.join(RESULTS_DIR, f'{name}.json')
    with open(path) as f: return json.load(f)

def result_exists(name):
    return os.path.exists(os.path.join(RESULTS_DIR, f'{name}.json'))

# ── 범용 학습 함수 ────────────────────────────────────────────────────────────
def train_model(model, tr_loader, te_loader, criterion, device,
                epochs, lr, tag, classes, minority_idx,
                log_every=5):
    """일반 분류 학습 (teacher or student standalone)"""
    ckpt = os.path.join(CKPT_DIR, f'{tag}.pt')
    optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    use_amp   = device.type == 'cuda'
    scaler    = torch.amp.GradScaler('cuda', enabled=use_amp)

    best_f1, best_metrics = 0.0, {}
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        for x, y in tr_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
        scheduler.step()
        if ep % log_every == 0 or ep == epochs:
            m = evaluate(model, te_loader, device, classes, minority_idx)
            elapsed = time.time() - t0
            print(f'  [{tag}] ep{ep:3d}/{epochs}  '
                  f'macro-F1={m["macro_f1"]:.4f}  '
                  f'minority={m["minority_f1"]:.4f}  '
                  f'acc={m["accuracy"]*100:.2f}%  '
                  f'({elapsed:.0f}s)', flush=True)
            if m['macro_f1'] > best_f1:
                best_f1, best_metrics = m['macro_f1'], m
                torch.save(model.state_dict(), ckpt)
    save_result(tag, best_metrics)
    print(f'  → best macro-F1={best_f1:.4f}  saved: results/{tag}.json\n', flush=True)
    return best_metrics

def train_kd(teacher, student, tr_loader, te_loader, hard_crit,
             device, T, alpha, epochs, lr, tag, classes, minority_idx,
             log_every=5):
    """Knowledge Distillation 학습"""
    ckpt      = os.path.join(CKPT_DIR, f'{tag}.pt')
    optimizer = torch.optim.SGD(student.parameters(), lr=lr,
                                momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    use_amp   = device.type == 'cuda'
    scaler    = torch.amp.GradScaler('cuda', enabled=use_amp)
    teacher.eval()

    best_f1, best_metrics = 0.0, {}
    t0 = time.time()
    for ep in range(1, epochs + 1):
        student.train()
        for x, y in tr_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                with torch.no_grad(): t_logits = teacher(x)
                s_logits = student(x)
                loss     = kd_loss_fn(s_logits, t_logits, y, T, alpha, hard_crit)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
        scheduler.step()
        if ep % log_every == 0 or ep == epochs:
            m = evaluate(student, te_loader, device, classes, minority_idx)
            elapsed = time.time() - t0
            print(f'  [{tag}] ep{ep:3d}/{epochs}  '
                  f'macro-F1={m["macro_f1"]:.4f}  '
                  f'minority={m["minority_f1"]:.4f}  '
                  f'acc={m["accuracy"]*100:.2f}%  '
                  f'({elapsed:.0f}s)', flush=True)
            if m['macro_f1'] > best_f1:
                best_f1, best_metrics = m['macro_f1'], m
                torch.save(student.state_dict(), ckpt)
    save_result(tag, best_metrics)
    print(f'  → best macro-F1={best_f1:.4f}  saved: results/{tag}.json\n', flush=True)
    return best_metrics

# ─────────────────────────────────────────────────────────────────────────────
def main():
    set_seed(42)
    device = get_device()
    X_train, y_train, X_test, y_test, classes = load_data()
    NC          = len(classes)
    MIN_IDX     = [classes.index(c) for c in MINORITY if c in classes]
    EPOCHS      = 30
    BATCH       = 128
    LR_TEACHER  = 0.1
    LR_STUDENT  = 0.05
    NW          = 0   # num_workers=0: 단일 스레드 로딩 (containerized 환경에서 더 빠름)

    def tr_te(oversample=False, augment=False, X_tr=None, y_tr=None):
        if X_tr is None: X_tr, y_tr = X_train, y_train
        return make_loaders(X_tr, y_tr, X_test, y_test,
                            BATCH, oversample, augment, NW)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 1: Teacher (ResNet-18) Baseline
    # ══════════════════════════════════════════════════════════════════════════
    print('\n' + '='*60)
    print('STEP 1: Teacher (ResNet-18) Baseline')
    print('='*60)
    TEACHER_CKPT = os.path.join(CKPT_DIR, 'teacher_none_aug0.pt')

    if result_exists('teacher_none_aug0'):
        print('  결과 있음 — 스킵')
        teacher = get_teacher_r18(NC).to(device)
        teacher.load_state_dict(torch.load(TEACHER_CKPT, map_location=device))
    else:
        teacher = get_teacher_r18(NC).to(device)
        print(f'  ResNet-18 파라미터: {count_params(teacher):,}')
        tr_loader, te_loader = tr_te(augment=False)
        crit = nn.CrossEntropyLoss()
        train_model(teacher, tr_loader, te_loader, crit, device,
                    EPOCHS, LR_TEACHER, 'teacher_none_aug0', classes, MIN_IDX)
        teacher.load_state_dict(torch.load(TEACHER_CKPT, map_location=device))
    teacher.eval()

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2: Student 불균형 처리 Ablation (4가지 + SMOTE)
    # ══════════════════════════════════════════════════════════════════════════
    print('\n' + '='*60)
    print('STEP 2: Student 불균형 처리 Ablation')
    print('='*60)

    def run_student(imbalance, tag=None):
        tag = tag or f'student_{imbalance}'
        if result_exists(tag):
            print(f'  [{tag}] 결과 있음 — 스킵')
            return load_result(tag)
        set_seed(42)
        oversample = (imbalance == 'oversample')
        tr_loader, te_loader = tr_te(oversample=oversample)
        if imbalance == 'focal':
            crit = FocalLoss(gamma=2.0)
        elif imbalance == 'classweight':
            crit = nn.CrossEntropyLoss(weight=compute_class_weights(y_train, NC).to(device))
        else:
            crit = nn.CrossEntropyLoss()
        student = get_student_cnn(NC).to(device)
        return train_model(student, tr_loader, te_loader, crit, device,
                           EPOCHS, LR_STUDENT, tag, classes, MIN_IDX)

    for imb in ['none', 'oversample', 'focal', 'classweight']:
        run_student(imb)

    # SMOTE — X_smote/y_smote를 상위 스코프 변수로 보관해 Step 5에서 재사용
    X_smote, y_smote = None, None
    smote_tag = 'student_smote'
    if result_exists(smote_tag):
        print(f'  [{smote_tag}] 결과 있음 — 스킵')
    else:
        try:
            from imblearn.over_sampling import SMOTE
            print('  SMOTE 적용 중...')
            set_seed(42)
            n, h, w = X_train.shape
            X_flat = X_train.reshape(n, -1).astype(np.float32)
            X_smote, y_smote = SMOTE(random_state=42).fit_resample(
                X_flat, y_train)
            X_smote = X_smote.reshape(-1, h, w)
            print(f'  SMOTE: {n:,} → {len(X_smote):,} 샘플')
            tr_loader, te_loader = make_loaders(
                X_smote, y_smote, X_test, y_test, BATCH, False, False, NW)
            student = get_student_cnn(NC).to(device)
            train_model(student, tr_loader, te_loader, nn.CrossEntropyLoss(),
                        device, EPOCHS, LR_STUDENT, smote_tag, classes, MIN_IDX)
        except ImportError:
            print('  imbalanced-learn 없음 — SMOTE 스킵')

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 3: KD Ablation
    # ══════════════════════════════════════════════════════════════════════════
    print('\n' + '='*60)
    print('STEP 3: KD Ablation')
    print('='*60)

    def run_kd_exp(T, alpha, tag, X_tr=None, y_tr=None,
                   imbalance='none', student_fn=None):
        if result_exists(tag):
            print(f'  [{tag}] 결과 있음 — 스킵')
            return load_result(tag)
        set_seed(42)
        if X_tr is None: X_tr, y_tr = X_train, y_train
        oversample = (imbalance == 'oversample')
        tr_loader, te_loader = make_loaders(
            X_tr, y_tr, X_test, y_test, BATCH, oversample, False, NW)
        if imbalance == 'focal':
            hard_crit = FocalLoss(gamma=2.0)
        elif imbalance == 'classweight':
            hard_crit = nn.CrossEntropyLoss(
                weight=compute_class_weights(y_tr, NC).to(device))
        else:
            hard_crit = nn.CrossEntropyLoss()
        fn      = student_fn or get_student_cnn
        student = fn(NC).to(device)
        return train_kd(teacher, student, tr_loader, te_loader, hard_crit,
                        device, T, alpha, EPOCHS, LR_STUDENT, tag, classes, MIN_IDX)

    # Temperature ablation (alpha=0.5 고정)
    for T in [1, 3, 5, 10, 20]:
        run_kd_exp(T=T, alpha=0.5, tag=f'kd_r18_T{T}_a0.5')

    # Alpha ablation (T=5 고정)
    for a in [0.1, 0.9]:   # 0.5는 위에서 이미 실행
        run_kd_exp(T=5, alpha=a, tag=f'kd_r18_T5_a{a}')

    # Student 크기 ablation
    run_kd_exp(T=5, alpha=0.5, tag='kd_r18_mobilenet',
               student_fn=get_student_mobilenet)

    # Teacher 크기 ablation: ResNet-50 학습 후 KD
    r50_ckpt = os.path.join(CKPT_DIR, 'teacher_r50.pt')
    if not result_exists('teacher_r50'):
        teacher_r50 = get_teacher_r50(NC).to(device)
        tr_loader, te_loader = tr_te()
        crit = nn.CrossEntropyLoss()
        train_model(teacher_r50, tr_loader, te_loader, crit, device,
                    EPOCHS, LR_TEACHER, 'teacher_r50', classes, MIN_IDX)
    else:
        print('  [teacher_r50] 결과 있음 — 스킵 (로드)')
        teacher_r50 = get_teacher_r50(NC).to(device)
        teacher_r50.load_state_dict(torch.load(r50_ckpt, map_location=device))
    teacher_r50.eval()

    if not result_exists('kd_r50_studentcnn'):
        set_seed(42)
        tr_loader, te_loader = tr_te()
        student   = get_student_cnn(NC).to(device)
        hard_crit = nn.CrossEntropyLoss()
        train_kd(teacher_r50, student, tr_loader, te_loader, hard_crit,
                 device, T=5, alpha=0.5, epochs=EPOCHS, lr=LR_STUDENT,
                 tag='kd_r50_studentcnn', classes=classes, minority_idx=MIN_IDX)
    else:
        print('  [kd_r50_studentcnn] 결과 있음 — 스킵')

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 4: 2×2 시너지 (focal+KD, classweight+KD)
    # ══════════════════════════════════════════════════════════════════════════
    print('\n' + '='*60)
    print('STEP 4: 시너지 Ablation')
    print('='*60)

    run_kd_exp(T=5, alpha=0.5, tag='kd_focal_T5_a0.5',   imbalance='focal')
    run_kd_exp(T=5, alpha=0.5, tag='kd_classweight_T5_a0.5', imbalance='classweight')

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 5: 전체 Ablation (oversample+KD, SMOTE+KD, CW+KD T=1)
    # ══════════════════════════════════════════════════════════════════════════
    print('\n' + '='*60)
    print('STEP 5: 종합 Ablation')
    print('='*60)

    run_kd_exp(T=5, alpha=0.5, tag='kd_oversample_T5_a0.5', imbalance='oversample')

    # SMOTE+KD — Step 2에서 만든 X_smote 재사용, 없으면 재생성
    smote_kd_tag = 'kd_smote_T5_a0.5'
    if result_exists(smote_kd_tag):
        print(f'  [{smote_kd_tag}] 결과 있음 — 스킵')
    else:
        if X_smote is None:
            try:
                from imblearn.over_sampling import SMOTE
                set_seed(42)
                n, h, w = X_train.shape
                X_flat = X_train.reshape(n, -1).astype(np.float32)
                X_smote, y_smote = SMOTE(random_state=42).fit_resample(
                    X_flat, y_train)
                X_smote = X_smote.reshape(-1, h, w)
                print(f'  SMOTE 재생성: {n:,} → {len(X_smote):,} 샘플')
            except ImportError:
                print('  imbalanced-learn 없음 — SMOTE+KD 스킵')
        if X_smote is not None:
            run_kd_exp(T=5, alpha=0.5, tag=smote_kd_tag,
                       X_tr=X_smote, y_tr=y_smote)

    # Temperature 기여도: CW + KD T=1
    run_kd_exp(T=1, alpha=0.5, tag='kd_classweight_T1_a0.5', imbalance='classweight')

    # ══════════════════════════════════════════════════════════════════════════
    # 최종 결과 요약
    # ══════════════════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print('최종 결과 요약')
    print('='*70)

    ALL = [
        ('student_none',            'Student baseline'),
        ('student_oversample',      'Student + oversample'),
        ('student_smote',           'Student + SMOTE'),
        ('student_focal',           'Student + focal'),
        ('student_classweight',     'Student + classweight'),
        ('kd_r18_T5_a0.5',         'Student + KD only (T=5, α=0.5)'),
        ('kd_oversample_T5_a0.5',  'Student + oversample + KD'),
        ('kd_smote_T5_a0.5',       'Student + SMOTE + KD'),
        ('kd_focal_T5_a0.5',       'Student + focal + KD'),
        ('kd_classweight_T5_a0.5', 'Student + classweight + KD ★'),
        ('kd_classweight_T1_a0.5', 'Student + classweight + KD (T=1)'),
    ]

    print(f"{'구성':<38} {'Macro-F1':>10} {'소수평균':>10} {'Near-full':>10}")
    print('-' * 72)
    nf_idx = classes.index('Near-full') if 'Near-full' in classes else -1
    for tag, desc in ALL:
        if not result_exists(tag):
            print(f'{desc:<38}  (결과 없음)')
            continue
        v = load_result(tag)
        nf = v['per_class_f1'][nf_idx] if nf_idx >= 0 else float('nan')
        print(f'{desc:<38} {v["macro_f1"]:>10.4f} {v["minority_f1"]:>10.4f} {nf:>10.3f}')

    print('='*70)

    # 기술 하나씩 제거 ablation
    if (result_exists('kd_classweight_T5_a0.5') and
        result_exists('kd_r18_T5_a0.5') and
        result_exists('student_classweight') and
        result_exists('kd_classweight_T1_a0.5') and
        result_exists('student_none')):

        best = load_result('kd_classweight_T5_a0.5')['macro_f1']
        contribs = {
            'KD 기여도':           best - load_result('student_classweight')['macro_f1'],
            'classweight 기여도':  best - load_result('kd_r18_T5_a0.5')['macro_f1'],
            'T=5 최적화 기여도':    best - load_result('kd_classweight_T1_a0.5')['macro_f1'],
        }
        print('\n기술별 기여도 (Macro-F1 기준):')
        for rank, (k, v) in enumerate(
                sorted(contribs.items(), key=lambda x: -x[1]), 1):
            bar = '█' * max(1, int(abs(v) * 1000))
            print(f'  {rank}위 {k:<25} Δ={v:+.4f}  {bar}')

    print('\n실험 완료!', flush=True)

if __name__ == '__main__':
    main()
