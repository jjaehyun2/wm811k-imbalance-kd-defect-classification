"""
WM-811K 데이터 전처리: LSWMD.pkl → data/wm811k_64.npz
- 라벨 있는 172,950장만 사용 (9개 클래스)
- 64×64 NEAREST 리사이즈
- Stratified split (80/20, seed=42)
"""
import argparse, os, sys, types, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from skimage.transform import resize   # pip install scikit-image

# pandas.indexes 호환성 패치 (LSWMD.pkl은 구버전 pandas로 저장됨)
def _patch_pandas():
    import pandas.core.indexes as _ci
    for attr in ['base','range','multi','category','interval',
                 'datetimes','timedeltas','period','frozen','accessors']:
        try:
            sys.modules.setdefault(f'pandas.indexes.{attr}', getattr(_ci, attr))
        except AttributeError:
            sys.modules.setdefault(f'pandas.indexes.{attr}',
                                   types.ModuleType(f'pandas.indexes.{attr}'))
    sys.modules.setdefault('pandas.indexes.numeric',
                           types.ModuleType('pandas.indexes.numeric'))
    sys.modules.setdefault('pandas.indexes', _ci)

CLASSES = ['none', 'Center', 'Donut', 'Edge-Loc', 'Edge-Ring',
           'Loc', 'Near-full', 'Random', 'Scratch']

def parse_label(raw):
    """failureType: ndarray([['ClassName']]) → 문자열, 빈 배열 → None"""
    arr = np.array(raw)
    if arr.size == 0:
        return None
    label = str(arr.flat[0]).strip()
    return label if label else None

def resize_map(wmap, size=64):
    """웨이퍼 맵을 float32 [0,2] 값 유지하며 size×size로 리사이즈"""
    arr = np.array(wmap, dtype=np.float32)
    if arr.ndim == 1:
        n = int(len(arr) ** 0.5)
        arr = arr.reshape(n, n)
    out = resize(arr, (size, size), order=0, preserve_range=True,
                 anti_aliasing=False).astype(np.float32)
    return out

def main(pkl_path, img_size, out_dir, seed):
    _patch_pandas()
    print(f'Loading {pkl_path} ...')
    with open(pkl_path, 'rb') as f:
        df = pickle.load(f, encoding='latin1')
    print(f'  전체 샘플: {len(df):,}')

    # 라벨 파싱
    df['label_str'] = df['failureType'].apply(parse_label)
    df = df[df['label_str'].notna()].copy()

    # 클래스 필터 (9개만)
    df = df[df['label_str'].isin(CLASSES)].copy()
    df['label_int'] = df['label_str'].map({c: i for i, c in enumerate(CLASSES)})
    print(f'  라벨 있는 샘플: {len(df):,}')
    print('  클래스 분포:')
    for c in CLASSES:
        n = (df['label_str'] == c).sum()
        print(f'    {c:<14} {n:>7,}')

    # 리사이즈
    print(f'\n  {img_size}×{img_size} 리사이즈 중 ...')
    X = np.stack([resize_map(w, img_size) for w in df['waferMap'].values])
    y = df['label_int'].values.astype(np.int64)
    print(f'  X shape: {X.shape}  dtype: {X.dtype}')

    # Stratified split (80/20)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed)
    print(f'  Train: {X_tr.shape}  Test: {X_te.shape}')

    # 저장
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'wm811k_{img_size}.npz')
    np.savez_compressed(out_path,
                        X_train=X_tr, y_train=y_tr,
                        X_test=X_te,  y_test=y_te,
                        classes=np.array(CLASSES))
    print(f'\n저장 완료: {out_path}  ({os.path.getsize(out_path)/1e6:.1f} MB)')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkl',      default='LSWMD.pkl')
    ap.add_argument('--img-size', type=int, default=64)
    ap.add_argument('--out-dir',  default='data')
    ap.add_argument('--seed',     type=int, default=42)
    args = ap.parse_args()
    main(args.pkl, args.img_size, args.out_dir, args.seed)
