import os
import numpy as np
from tensorflow.keras.models import load_model

DATA_FOLDER = "dataset_reduced"

print("📦 تحميل المودل...")
model = load_model("final_model.keras", compile=False)

all_errors = []

print("🔍 حساب Hybrid Error...")

for file in sorted(os.listdir(DATA_FOLDER)):

    if not file.endswith(".npy"):
        continue

    print(f"📂 معالجة: {file}")

    data = np.load(os.path.join(DATA_FOLDER, file))

    # (N,100) → (N,100,1)
    X = data.reshape(data.shape[0], 100, 1)

    # ===== توقعات المودل =====
    recon_pred, forecast_pred = model.predict(X, verbose=0)

    # ===== Reconstruction Error =====
    recon_error = np.mean(np.square(X - recon_pred), axis=(1, 2))

    # ===== Forecast Error =====
    last_value = X[:, -1:, :]  # shape (N,1,1)
    forecast_error = np.mean(np.square(last_value - forecast_pred), axis=(1, 2))

    # ===== Hybrid Error =====
    hybrid_error = recon_error + 0.5 * forecast_error

    all_errors.extend(hybrid_error)

# ===== تحويل إلى numpy =====
all_errors = np.array(all_errors)

print("\n📊 النتائج:")
print("عدد العينات:", len(all_errors))
print("متوسط الخطأ:", np.mean(all_errors))
print("أعلى خطأ:", np.max(all_errors))
print("أقل خطأ:", np.min(all_errors))
print("الانحراف المعياري:", np.std(all_errors))

# ===== Thresholdات =====
mean = np.mean(all_errors)
std = np.std(all_errors)

threshold_3sigma = mean + 3 * std
p99 = np.percentile(all_errors, 99)
p995 = np.percentile(all_errors, 99.5)
p997 = np.percentile(all_errors, 99.7)

print("\n🎯 Thresholdات:")
print("3-Sigma:", threshold_3sigma)
print("99%:", p99)
print("99.5%:", p995)
print("99.7%:", p997)