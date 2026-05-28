import numpy as np
try:
from backend.continual.buffer_manager import process_and_store
	from backend.continual.config import WINDOW_LEN
except ModuleNotFoundError:
	from buffer_manager import process_and_store
	from config import WINDOW_LEN

# ==========================================

# إنشاء بيانات تجريبية

# ==========================================

series_length = 1000

series = np.random.rand(series_length).astype(np.float32)

# عدد النوافذ الممكنة

num_windows = series_length - WINDOW_LEN + 1

scores = np.random.rand(num_windows).astype(np.float32)

threshold = 0.8

spans = [(i, i + WINDOW_LEN) for i in range(num_windows)]

# ==========================================

# تشغيل buffer_manager

# ==========================================

result = process_and_store(series, scores, threshold, spans)

print("\nTest result:")
print(result)
