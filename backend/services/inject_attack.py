import numpy as np
import os
import random

SOURCE_FOLDER = "dataset_reduced"
ATTACK_FOLDER = "dataset_attacked"

os.makedirs(ATTACK_FOLDER, exist_ok=True)

print("🚀 إنشاء بيانات مهاجمة...")

for file in os.listdir(SOURCE_FOLDER):
    if file.endswith(".npy"):
        data = np.load(os.path.join(SOURCE_FOLDER, file))

        attacked = data.copy()

        # نختار 5% من العينات عشوائياً
        num_samples = attacked.shape[0]
        attack_indices = random.sample(range(num_samples), int(0.05 * num_samples))

        for idx in attack_indices:
            # نضيف spike قوي
            attacked[idx] += np.random.normal(5, 1, attacked[idx].shape)

        np.save(os.path.join(ATTACK_FOLDER, file), attacked)

print("✅ تم إنشاء dataset_attacked")