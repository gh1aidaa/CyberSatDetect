import numpy as np
import uuid
from datetime import datetime

from backend.continual.config import (
    NORMAL_POOL_DIR,
    ANOMALY_POOL_DIR,
    WINDOW_LEN,
    N_FEATURES,
    SAFE_THRESHOLD_RATIO,
    MAX_WINDOWS_PER_RUN
)

# ==========================================
# اختيار النوافذ الطبيعية الآمنة
# ==========================================

def select_safe_windows(series_1d, scores, threshold, spans):
    """
    اختيار النوافذ الطبيعية الآمنة
    """

    series_1d = np.asarray(series_1d)

    if series_1d.ndim != 1:
        raise ValueError("series_1d must be 1D")

    scores = np.asarray(scores, dtype=np.float32)

    if len(scores) != len(spans):
        raise ValueError("scores and spans length mismatch")

    if scores.size == 0:
        print("No scores provided → returning empty dataset")
        return np.zeros((0, WINDOW_LEN, N_FEATURES), dtype=np.float32)

    finite_mask = np.isfinite(scores)
    if not np.all(finite_mask):
        print("Non-finite scores detected → filtering invalid values")
        scores = scores[finite_mask]
        spans = [spans[i] for i in np.where(finite_mask)[0]]

    if scores.size == 0:
        print("No valid scores after filtering → returning empty dataset")
        return np.zeros((0, WINDOW_LEN, N_FEATURES), dtype=np.float32)

    safe_limit = SAFE_THRESHOLD_RATIO * threshold

    # ==========================================
    # تحسين الاختيار (threshold + percentile)
    # ==========================================
    percentile_limit = np.percentile(scores, 30)

    safe_indices = np.where(
        (scores <= safe_limit) &
        (scores <= percentile_limit)
    )[0]

    # 🔥 fallback إذا ما فيه ولا safe window
    if safe_indices.size == 0:
        print("No safe windows → using lowest scores fallback")

        k = min(300, len(scores))
        safe_indices = np.argsort(scores)[:k]

    # تقليل التكرار
    step = max(1, safe_indices.size // MAX_WINDOWS_PER_RUN)

    selected = safe_indices[::step][:MAX_WINDOWS_PER_RUN]

    windows = []

    for idx in selected:

        start, end = spans[idx]

        if end > len(series_1d):
            continue

        window = series_1d[start:end].astype(np.float32)

        if window.shape[0] != WINDOW_LEN:
            continue

        # استبعاد النوافذ الثابتة
        if float(np.std(window)) < 1e-5:
            continue

        windows.append(window)

    # 🔥 fallback ثاني
    if not windows:
        print("All windows filtered → forcing minimal dataset")

        for idx in safe_indices[:100]:

            start, end = spans[idx]

            if end > len(series_1d):
                continue

            window = series_1d[start:end].astype(np.float32)

            if window.shape[0] != WINDOW_LEN:
                continue

            windows.append(window)

    # 🔥 fallback أخير
    if not windows:
        print("Still empty → returning empty dataset")

        return np.zeros((0, WINDOW_LEN, N_FEATURES), dtype=np.float32)

    X = np.asarray(windows, dtype=np.float32)

    if X.ndim == 2:
        X = X[..., None]

    print("Selected windows:", X.shape[0])

    return X


# ==========================================
# اختيار النوافذ الشاذة (hard anomalies)
# ==========================================

def select_anomaly_windows(series_1d, scores, threshold, spans):
    """
    اختيار نوافذ شاذة اعتماداً على scores والـ threshold.
    الهدف: بناء مجموعة شاذة عالية الثقة لاستخدامها كـ replay/negative samples.
    """
    series_1d = np.asarray(series_1d)
    if series_1d.ndim != 1:
        raise ValueError("series_1d must be 1D")

    scores = np.asarray(scores, dtype=np.float32)
    if len(scores) != len(spans):
        raise ValueError("scores and spans length mismatch")

    if scores.size == 0:
        return np.zeros((0, WINDOW_LEN, N_FEATURES), dtype=np.float32)

    finite_mask = np.isfinite(scores)
    if not np.all(finite_mask):
        scores = scores[finite_mask]
        spans = [spans[i] for i in np.where(finite_mask)[0]]
    if scores.size == 0:
        return np.zeros((0, WINDOW_LEN, N_FEATURES), dtype=np.float32)

    # pick strong anomalies: above threshold and above high percentile
    hi = np.percentile(scores, 90)
    anom_indices = np.where((scores >= float(threshold)) & (scores >= float(hi)))[0]

    # fallback: take top-k windows by score
    if anom_indices.size == 0:
        k = min(200, len(scores))
        anom_indices = np.argsort(scores)[-k:]

    # limit to max windows per run
    if anom_indices.size > MAX_WINDOWS_PER_RUN:
        step = max(1, anom_indices.size // MAX_WINDOWS_PER_RUN)
        anom_indices = anom_indices[::step][:MAX_WINDOWS_PER_RUN]

    windows = []
    for idx in anom_indices:
        start, end = spans[int(idx)]
        if end > len(series_1d):
            continue
        window = series_1d[start:end].astype(np.float32)
        if window.shape[0] != WINDOW_LEN:
            continue
        # avoid constant windows
        if float(np.std(window)) < 1e-5:
            continue
        windows.append(window)

    if not windows:
        return np.zeros((0, WINDOW_LEN, N_FEATURES), dtype=np.float32)

    X = np.asarray(windows, dtype=np.float32)
    if X.ndim == 2:
        X = X[..., None]
    return X


# ==========================================
# حفظ batch داخل normal_pool
# ==========================================

def save_normal_batch(X_windows):

    if X_windows.shape[0] == 0:
        print("Nothing to save.")
        return None

    NORMAL_POOL_DIR.mkdir(parents=True, exist_ok=True)

    filename = f"normal_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.npy"

    file_path = NORMAL_POOL_DIR / filename

    np.save(file_path, X_windows)

    print("Saved batch:", file_path)

    return str(file_path)


# ==========================================
# حفظ batch داخل anomaly_pool
# ==========================================

def save_anomaly_batch(X_windows):

    if X_windows.shape[0] == 0:
        return None

    ANOMALY_POOL_DIR.mkdir(parents=True, exist_ok=True)

    filename = f"anomaly_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.npy"
    file_path = ANOMALY_POOL_DIR / filename
    np.save(file_path, X_windows)
    return str(file_path)


# ==========================================
# الدالة الرئيسية بعد التحليل
# ==========================================

def process_and_store(series_1d, scores, threshold, spans):

    X_safe = select_safe_windows(series_1d, scores, threshold, spans)
    X_anom = select_anomaly_windows(series_1d, scores, threshold, spans)

    saved_path = save_normal_batch(X_safe)
    saved_anom_path = save_anomaly_batch(X_anom)

    return {
        "selected_windows": int(X_safe.shape[0]),
        "saved_path": saved_path,
        "selected_anomaly_windows": int(X_anom.shape[0]),
        "saved_anomaly_path": saved_anom_path,
    }