CyberSatDetect — حزمة رفع Google Drive / Google Colab
================================================

ما بداخل الـ ZIP:
  - كود التدريب والتقييم وتوليد الهجمات حسب المسارات داخل المجلد.
  - تقسيمة QC: backend/config/data_split_qc_filtered.json

البيانات (chunk_*.npy):
  إذا وُجد ملف data/reduced.zip عند بناء الأرشيف، يُدمَج تلقائياً.
  إذا لم يُدمَج (حجم أو غير موجود): انسخ مجلد data/reduced كامل أو reduced.zip يدوياً إلى Drive بجانب هذا الأرشيف.

التشغيل على كولاب بعد فك الضغط:
  python backend/experiments/colab_qc_package/colab_run_qc_training.py ^
    --repo-root . ^
    --output-root ./qc_outputs

ثبّت: pip install tensorflow numpy
