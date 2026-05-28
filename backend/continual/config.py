import os
from pathlib import Path


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return max(1, int(str(raw).strip()))
    except ValueError:
        return default

# ==========================================
# Base Directories
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path("/var/data")

# ==========================================
# Continual Data Structure
# ==========================================

CONTINUAL_DIR = DATA_DIR / "continual"

NORMAL_POOL_DIR = CONTINUAL_DIR / "normal_pool"
ANOMALY_POOL_DIR = CONTINUAL_DIR / "anomaly_pool"
DATASETS_DIR = CONTINUAL_DIR / "datasets"
ANOMALY_DATASETS_DIR = CONTINUAL_DIR / "anomaly_datasets"

# إنشاء المجلدات تلقائياً
NORMAL_POOL_DIR.mkdir(parents=True, exist_ok=True)
ANOMALY_POOL_DIR.mkdir(parents=True, exist_ok=True)
DATASETS_DIR.mkdir(parents=True, exist_ok=True)
ANOMALY_DATASETS_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# Base Model (المودل الأساسي)
# ==========================================

BASE_MODEL_PATH = BASE_DIR / "app" / "best_model_render.keras"
BASE_THRESHOLD_PATH = BASE_DIR / "app" / "thresholds_qc_filtered.json"

# ==========================================
# Versioned Models
# ==========================================

MODELS_DIR = DATA_DIR / "continual" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# Data Parameters
# ==========================================

WINDOW_LEN = 100
N_FEATURES = 1

# ==========================================
# Safe Selection Parameters
# ==========================================

# نعتبر window طبيعي آمن إذا كان أقل من 50% من threshold
SAFE_THRESHOLD_RATIO = 0.5

# أقصى عدد نوافذ من كل تحليل
MAX_WINDOWS_PER_RUN = 500

# حجم dataset المطلوب قبل بدء التدريب
TARGET_DATASET_SIZE = 100

# أقل عدد نوافذ طبيعية بعد التنظيف لقبول "Build dataset" (للإنتاج ارفع CSD_CONTINUAL_MIN_BUILD_WINDOWS)
MIN_WINDOWS_FOR_DATASET_BUILD = _int_env("CSD_CONTINUAL_MIN_BUILD_WINDOWS", 32)

# ==========================================
# Training Hyperparameters
# ==========================================
LEARNING_RATE = 1e-5
EPOCHS = 1
BATCH_SIZE = 4

# ==========================================
# Continual Training Safety
# ==========================================

# لا يبدأ التدريب إلا إذا وصل عدد نوافذ الطبيعي لهذا الحد على الأقل.
# الافتراضي يسمح بالتجريب على مجموعات صغيرة؛ للإنتاج استخدم CSD_CONTINUAL_MIN_WINDOWS=3000 (أو أكبر).
MIN_DATASET_SIZE_FOR_TRAIN = _int_env("CSD_CONTINUAL_MIN_WINDOWS", 32)

# يمكن للأدمن تعطيل التدريب
ADMIN_APPROVAL_REQUIRED = False