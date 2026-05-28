import numpy as np
import uuid
from datetime import datetime

from backend.continual.config import (
    NORMAL_POOL_DIR,
    ANOMALY_POOL_DIR,
    DATASETS_DIR,
    ANOMALY_DATASETS_DIR,
    DATA_DIR,
    WINDOW_LEN,
    N_FEATURES,
    TARGET_DATASET_SIZE,
    MIN_WINDOWS_FOR_DATASET_BUILD,
)

# ==========================================
# تحميل البيانات (من datasets + fallback normal_pool)
# ==========================================

def load_all_normal_windows():

    # أولاً نحاول من datasets (بعد approve)
    files = sorted(DATASETS_DIR.glob("*.npy"))

    if not files:
        print("No files in datasets → fallback to normal_pool")
        files = sorted(NORMAL_POOL_DIR.glob("*.npy"))

    if not files:
        print("No data found anywhere.")
        return np.zeros((0, WINDOW_LEN, N_FEATURES), dtype=np.float32)

    all_batches = []

    for f in files:
        try:
            X = np.load(f)

            if X.ndim == 3 and X.shape[1] == WINDOW_LEN:
                all_batches.append(X.astype(np.float32))

        except Exception as e:
            print("Skipping file:", f, e)
            continue

    if not all_batches:
        return np.zeros((0, WINDOW_LEN, N_FEATURES), dtype=np.float32)

    return np.concatenate(all_batches, axis=0)


def load_all_anomaly_windows():
    """
    Load approved anomaly windows (anomaly_datasets) with fallback to anomaly_pool.
    Shape expected: (N, WINDOW_LEN, N_FEATURES)
    """
    files = sorted(ANOMALY_DATASETS_DIR.glob("*.npy"))
    if not files:
        print("No files in anomaly_datasets → fallback to anomaly_pool")
        files = sorted(ANOMALY_POOL_DIR.glob("*.npy"))

    if not files:
        print("No anomaly data found.")
        return np.zeros((0, WINDOW_LEN, N_FEATURES), dtype=np.float32)

    all_batches = []
    for f in files:
        try:
            X = np.load(f)
            if X.ndim == 3 and X.shape[1] == WINDOW_LEN:
                all_batches.append(X.astype(np.float32))
        except Exception as e:
            print("Skipping anomaly file:", f, e)
            continue

    if not all_batches:
        return np.zeros((0, WINDOW_LEN, N_FEATURES), dtype=np.float32)
    return np.concatenate(all_batches, axis=0)


# ==========================================
# إزالة التكرار
# ==========================================

def remove_duplicates(X):

    if X.shape[0] == 0:
        return X

    flat = X.reshape(X.shape[0], -1)

    _, unique_indices = np.unique(flat, axis=0, return_index=True)

    return X[np.sort(unique_indices)]


# ==========================================
# خلط البيانات
# ==========================================

def shuffle_windows(X):

    if X.shape[0] == 0:
        return X

    indices = np.random.permutation(X.shape[0])

    return X[indices]


# ==========================================
# بناء Dataset (combined)
# ==========================================

def build_dataset(target_size=TARGET_DATASET_SIZE):

    print("\nLoading approved datasets...")

    Xn = load_all_normal_windows()
    Xa = load_all_anomaly_windows()

    if Xn.shape[0] == 0:
        print("No data available.")
        return None

    print("Total windows before cleaning:", Xn.shape[0])

    # إزالة التكرار
    Xn = remove_duplicates(Xn)
    print("After removing duplicates:", Xn.shape[0])

    # remove duplicates on anomalies too (if any)
    if Xa.shape[0] > 0:
        Xa = remove_duplicates(Xa)
        print("Anomaly windows after removing duplicates:", Xa.shape[0])

    # ==========================================
    # Replay Memory (old + new balance)
    # ==========================================

    split_idx = int(0.7 * Xn.shape[0])

    X_old = Xn[:split_idx]
    X_new = Xn[split_idx:]

    X_old = shuffle_windows(X_old)
    X_new = shuffle_windows(X_new)

    Xn = np.concatenate([X_old, X_new], axis=0)

    print("After replay balance:", Xn.shape[0])

    # خلط نهائي
    Xn = shuffle_windows(Xn)

    # تقليل الحجم
    if Xn.shape[0] > target_size:
        Xn = Xn[:target_size]

    # cap anomaly size relative to normals (keep unsupervised balance)
    if Xa.shape[0] > 0:
        cap_a = max(100, int(0.25 * max(1, Xn.shape[0])))
        if Xa.shape[0] > cap_a:
            Xa = Xa[:cap_a]

    if Xn.shape[0] < MIN_WINDOWS_FOR_DATASET_BUILD:
        print(
            "Dataset extremely small.",
            f"have {Xn.shape[0]} normal windows, need at least {MIN_WINDOWS_FOR_DATASET_BUILD}.",
        )
        return None

    # ==========================================
    # Replay previous combined if exists (anti-forgetting)
    # ==========================================
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    legacy = DATA_DIR / "combined.npy"
    prev_norm = DATA_DIR / "combined_normal.npy"
    prev_npz = DATA_DIR / "combined_dataset.npz"

    X_prev = None
    if prev_npz.exists():
        try:
            z = np.load(prev_npz)
            if "X_normal" in z:
                X_prev = z["X_normal"].astype(np.float32)
        except Exception:
            X_prev = None
    elif prev_norm.exists():
        try:
            X_prev = np.load(prev_norm).astype(np.float32)
        except Exception:
            X_prev = None
    elif legacy.exists():
        try:
            X_prev = np.load(legacy).astype(np.float32)
        except Exception:
            X_prev = None

    if X_prev is not None and X_prev.ndim == 3 and X_prev.shape[1] == WINDOW_LEN:
        # keep a small replay slice
        replay_k = min(int(0.3 * Xn.shape[0]), X_prev.shape[0], 3000)
        if replay_k > 0:
            X_prev = shuffle_windows(X_prev)
            Xn = np.concatenate([Xn, X_prev[:replay_k]], axis=0)
            Xn = shuffle_windows(Xn)
            if Xn.shape[0] > target_size:
                Xn = Xn[:target_size]
            print("After adding replay slice:", Xn.shape[0])

    out_norm = DATA_DIR / "combined_normal.npy"
    out_anom = DATA_DIR / "combined_anomaly.npy"
    out_npz = DATA_DIR / "combined_dataset.npz"

    np.save(out_norm, Xn)
    if Xa.shape[0] > 0:
        np.save(out_anom, Xa)

    np.savez_compressed(out_npz, X_normal=Xn, X_anomaly=Xa)

    print("\nCombined dataset saved:", out_npz)
    print("Final normal windows:", Xn.shape[0])
    print("Final anomaly windows:", Xa.shape[0])

    return str(out_npz)


# ==========================================
# تشغيل مباشر
# ==========================================

if __name__ == "__main__":
    build_dataset()