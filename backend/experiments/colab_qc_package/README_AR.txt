================================================================================
تدريب على Google Colab — تقسيمة QC فقط (data_split_qc_filtered.json)
================================================================================

1) الملفات والمجلدات التي تحتاج ترفعها إلى Colab (أو تخليها في Drive بعد رفع المشروع)

   إلزامي للتدريب بنفس المنهجية:
   - backend/models/train_hybrid_model.py
   - backend/config/data_split_qc_filtered.json
   - مجلد البيانات: data/reduced/  (كل ملفات chunk_*.npy المطلوبة للتقسيمة؛ أو ارفع reduced.zip وفك الضغط)

   ملف مساعد لهذا التشغيل (يعيد توجيه المخرجات إلى مجلد أنت تحدده، مع split الجديد):
   - backend/experiments/colab_qc_package/colab_run_qc_training.py

   اختياري (تقييم لاحقاً بنفس البروتوكول محلياً أو على كولاب):
   - backend/models/evaluate_model_strict_v2.py
   - backend/models/regenerate_attacked_dataset.py

2) في Colab — تثبيت سريع:

   !pip install -q "tensorflow>=2.13" numpy

3) في Colab — بعد رفع المشروع أو ربط Drive، عدّل المتغيرات في نهاية الملف:
   colab_run_qc_training.py

   أو شغّل من الطرفية:

   python backend/experiments/colab_qc_package/colab_run_qc_training.py ^
     --repo-root /content/CyberSatDetectprojct ^
     --output-root /content/qc_colab_outputs

4) المخرجات (كلها تحت --output-root فقط):

   output-root/model/best_model.keras
   output-root/model/final_model.keras
   output-root/results/thresholds_qc_filtered.json
   output-root/logs/training.log

5) ملاحظات:

   - البيانات كبيرة: إما ترفع reduced.zip وتفك ضغط، أو تنسخ مجلد reduced كامل لـ Drive.
   - التدريب ثقيل؛ استخدم GPU runtime في Colab.
   - لا حاجة لتعديل train_hybrid_model.py إذا استخدمت colab_run_qc_training.py (يعيد تعريف المسارات وقت التشغيل فقط).

================================================================================
