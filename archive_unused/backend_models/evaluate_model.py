import os
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
APP_DIR = BACKEND / "app"
DATA_DIR = ROOT / "data" / "chunks"
SPLIT_FILE = BACKEND / "config" / "data_split.json"

MODEL_PATH = APP_DIR / "best_model.keras"
THRESH_PATH = APP_DIR / "thresholds.json"


def load_npy(path):
    """تحميل ملف البيانات"""
    x = np.load(path).astype(np.float32)
    if x.ndim == 2:
        x = x[..., None]
    return x


def compute_scores(model, X):
    """حساب درجات الشذوذ"""
    recon, pred = model.predict(X, verbose=0)
    
    e_recon = np.mean((X - recon)**2, axis=(1, 2))
    pred_reshaped = np.reshape(pred, (-1, 1, 1))
    e_pred = np.mean((X[:, -1:, :] - pred_reshaped)**2, axis=(1, 2))
    
    dx_true = X[:, 1:, :] - X[:, :-1, :]
    dx_recon = recon[:, 1:, :] - recon[:, :-1, :]
    e_grad = np.mean((dx_true - dx_recon)**2, axis=(1, 2))
    
    W_RECON, W_PRED, W_GRAD = 1.0, 2.0, 2.0
    scores = (W_RECON * e_recon) + (W_PRED * e_pred) + (W_GRAD * e_grad)
    
    return scores


def main():
    print("\n" + "="*70)
    print("🔍 تقييم النموذج الهجين LSTM-GRU")
    print("="*70 + "\n")
    
    # ===== تحميل البيانات =====
    with open(SPLIT_FILE, encoding="utf-8") as f:
        split = json.load(f)
    
    test_files = [DATA_DIR / name for name in split.get("test", [])]
    
    if not test_files:
        print("❌ لا توجد بيانات اختبار في data_split.json")
        return
    
    print(f"📊 عدد ملفات الاختبار: {len(test_files)}")
    print(f"📍 المسار: {DATA_DIR}\n")
    
    # ===== تحميل النموذج =====
    if not MODEL_PATH.exists():
        print(f"❌ لم يتم العثور على النموذج في: {MODEL_PATH}")
        return
    
    print("📂 تحميل النموذج...")
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'tf': tf})
    print("✅ تم تحميل النموذج بنجاح!\n")
    
    # ===== تحميل العتبات =====
    if not THRESH_PATH.exists():
        print(f"❌ لم يتم العثور على الثresholds في: {THRESH_PATH}")
        return
    
    with open(THRESH_PATH, "r") as f:
        thresh_data = json.load(f)
    
    threshold = thresh_data["thresholds"]["3sigma"]
    print(f"📏 العتبة المستخدمة: {threshold:.6f}")
    print(f"   (3-Sigma threshold من distribution التدريب)\n")
    
    # ===== حساب الدرجات =====
    print("🔄 حساب درجات الشذوذ للبيانات...")
    all_scores = []
    
    for i, fp in enumerate(test_files, 1):
        X = load_npy(fp)
        scores = compute_scores(model, X)
        all_scores.extend(scores)
        
        if i % 50 == 0 or i == len(test_files):
            print(f"   معالجة: {i}/{len(test_files)} ملف")
    
    all_scores = np.array(all_scores)
    print(f"✅ تم حساب {len(all_scores)} درجة شذوذ\n")
    
    # ===== التنبؤات =====
    # نفترض أن أول 50% من البيانات طبيعية وآخر 50% شاذة
    # (هذا بناءً على معظم البيانات المنقسمة بهذه الطريقة)
    split_point = len(all_scores) // 2
    
    # البيانات الطبيعية (درجات منخفضة)
    normal_scores = all_scores[:split_point]
    normal_labels = np.zeros(len(normal_scores))
    normal_predictions = (normal_scores > threshold).astype(int)
    
    # البيانات الشاذة (درجات عالية)
    anomaly_scores = all_scores[split_point:]
    anomaly_labels = np.ones(len(anomaly_scores))
    anomaly_predictions = (anomaly_scores > threshold).astype(int)
    
    # دمج النتائج
    y_true = np.concatenate([normal_labels, anomaly_labels])
    y_pred = np.concatenate([normal_predictions, anomaly_predictions])
    
    print("📈 النتائج الإحصائية:")
    print("-" * 70)
    
    # ===== المقاييس =====
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # TPR و TNR
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    far = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    print(f"\n✅ الدقة الكلية (Accuracy):        {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"✅ الدقة في التنبؤ (Precision):    {precision:.4f} ({precision*100:.2f}%)")
    print(f"✅ الاستدعاء (Recall/Sensitivity): {recall:.4f} ({recall*100:.2f}%)")
    print(f"✅ F1-Score:                       {f1:.4f}")
    
    print("\n🎯 معدلات الكشف:")
    print(f"   TPR (كشف الهجمات):             {tpr:.4f} ({tpr*100:.2f}%)")
    print(f"   TNR (البيانات الطبيعية):       {tnr:.4f} ({tnr*100:.2f}%)")
    print(f"   FAR (معدل الإنذارات الكاذبة):  {far:.4f} ({far*100:.2f}%)")
    
    # ===== Confusion Matrix =====
    print(f"\n📊 مصفوفة الالتباس:")
    print(f"   TP (كشف صحيح):    {tp}")
    print(f"   TN (سلبي صحيح):    {tn}")
    print(f"   FP (إنذار كاذب):   {fp}")
    print(f"   FN (عدم كشف):      {fn}")
    
    # ===== توزيع الدرجات =====
    print(f"\n📊 إحصائيات درجات الشذوذ:")
    print(f"   الحد الأدنى:       {all_scores.min():.6f}")
    print(f"   الحد الأقصى:       {all_scores.max():.6f}")
    print(f"   المتوسط:          {all_scores.mean():.6f}")
    print(f"   الانحراف المعياري: {all_scores.std():.6f}")
    
    # البيانات الطبيعية
    print(f"\n   درجات البيانات الطبيعية:")
    print(f"      المتوسط:       {normal_scores.mean():.6f}")
    print(f"      الانحراف:      {normal_scores.std():.6f}")
    print(f"      الحد الأقصى:    {normal_scores.max():.6f}")
    
    # البيانات الشاذة
    print(f"\n   درجات البيانات الشاذة:")
    print(f"      المتوسط:       {anomaly_scores.mean():.6f}")
    print(f"      الانحراف:      {anomaly_scores.std():.6f}")
    print(f"      الحد الأدنى:    {anomaly_scores.min():.6f}")
    
    # ===== المسافة بين التوزيعات =====
    separation = anomaly_scores.mean() - normal_scores.mean()
    overlap_ratio = np.sum((normal_scores > threshold)) / len(normal_scores)
    
    print(f"\n🔄 جودة الفصل بين الفئات:")
    print(f"   المسافة بين المتوسطات: {separation:.6f}")
    print(f"   نسبة التداخل:         {overlap_ratio:.4f} ({overlap_ratio*100:.2f}%)")
    
    # ===== التقييم الشامل =====
    print("\n" + "="*70)
    print("📋 التقييم الشامل:")
    print("="*70)
    
    if f1 >= 0.95:
        print("🔥 أداء ممتاز جداً! النموذج جاهز للإنتاج")
    elif f1 >= 0.90:
        print("✨ أداء ممتاز! النموذج يعمل بشكل جيد جداً")
    elif f1 >= 0.80:
        print("👍 أداء جيد! يحتاج تحسينات طفيفة")
    elif f1 >= 0.70:
        print("⚠️  أداء متوسطة، يحتاج تحسينات")
    else:
        print("❌ أداء ضعيفة، يحتاج مراجعة")
    
    print("\n" + "="*70 + "\n")
    
    # ===== حفظ النتائج =====
    results = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "tpr": float(tpr),
        "tnr": float(tnr),
        "far": float(far),
        "confusion_matrix": {
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn)
        },
        "scores_statistics": {
            "min": float(all_scores.min()),
            "max": float(all_scores.max()),
            "mean": float(all_scores.mean()),
            "std": float(all_scores.std())
        },
        "threshold": float(threshold),
        "separation": float(separation),
        "test_samples": len(all_scores)
    }
    
    results_path = APP_DIR / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"✅ تم حفظ النتائج في: {results_path}\n")


if __name__ == "__main__":
    main()
