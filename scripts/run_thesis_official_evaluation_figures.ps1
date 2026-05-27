# Official thesis evaluation (4-attack QC-filtered) — all figures to external folder.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$OutDir = Join-Path $Root "thesis_official_evaluation_figures"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

Write-Host "=== Running four-attack evaluation (thesis protocol) ==="
Write-Host "Output: $OutDir"

python scripts/eval_four_attacks_standalone.py `
  --model backend/app/best_model_qc_filtered.keras `
  --thresholds backend/app/thresholds_qc_filtered.json `
  --split backend/config/data_split_qc_filtered.json `
  --normal-dir data/reduced `
  --attacked-dir data/attacked_v2 `
  --out-dir $OutDir `
  --window-size 100 `
  --stride 50 `
  --split-key test `
  --batch-size 256 `
  --sweep-points 500

Write-Host "=== Threshold policy bar (Table 5.2) ==="
python scripts/plot_thesis_threshold_policy_bar.py `
  --metrics-csv "$OutDir\overall_threshold_metrics_4attacks.csv" `
  --summary-json "$OutDir\evaluation_summary_4attacks.json" `
  --output "$OutDir\threshold_policy_bar_table52.png"

Write-Host "Done. Figures in: $OutDir"
