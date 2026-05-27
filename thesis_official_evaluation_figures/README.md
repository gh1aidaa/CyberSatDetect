# Thesis official evaluation figures

**Protocol:** QC-filtered model, four-attack subset (`drift`, `freeze`, `noise`, `spike`), strict window-level evaluation.

**Generator:** `scripts/eval_four_attacks_standalone.py`  
**Regenerate all:**

```powershell
.\scripts\run_thesis_official_evaluation_figures.ps1
```

## Figures (LaTeX filenames)

| File | Thesis reference |
|------|------------------|
| `roc_curve_4attacks.png` | Fig. ROC (a) |
| `pr_curve_4attacks.png` | Fig. ROC/PR (b) |
| `score_distribution_4attacks.png` | Fig. score distribution |
| `threshold_vs_f1_4attacks.png` | Fig. threshold analysis (a) |
| `threshold_vs_recall_far_4attacks.png` | Fig. threshold analysis (b) |
| `threshold_vs_balanced_accuracy_4attacks.png` | Fig. threshold analysis (c) |
| `threshold_policy_bar_table52.png` | Table 5.2 policy trade-offs (bars + dual axis) |
| `threshold_policy_small_multiples_table52.png` | **Alt.** three panels: Recall / FAR / F1 |
| `threshold_policy_recall_far_scatter_table52.png` | **Alt.** Recall–FAR trade-off (size = F1) |
| `threshold_policy_heatmap_table52.png` | **Alt.** compact heatmap |
| `confusion_matrix_best_f1_4attacks.png` | Fig. CM (a) |
| `confusion_matrix_p99_4attacks.png` | Fig. CM (b) |
| `confusion_matrix_p99_5_4attacks.png` | (extra) |
| `per_attack_bars_p99_4attacks.png` | Fig. per-attack bars |
| `per_attack_bars_best_f1_4attacks.png` | Fig. per-attack bars |

## Tables / JSON

- `overall_threshold_metrics_4attacks.csv` — Table thresholds
- `per_attack_full_metrics_4attacks.csv` — per-attack table
- `evaluation_summary_4attacks.json` — counts, ROC/PR-AUC
- `best_f1_from_sweep_4attacks.json`

**Not included here:** Validation-phase table (71 files, 321,142 windows) — see `backend/experiments/chapter7_testing_validation_evaluation/results/validation_score_summary.csv`.
