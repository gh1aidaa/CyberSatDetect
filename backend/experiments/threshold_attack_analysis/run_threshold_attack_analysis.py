"""
Offline threshold vs attack-type analysis (isolated experiment).

Does not modify production config, models, or api. Writes only under --output-dir.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Paths: all writes must stay inside output_dir
# ---------------------------------------------------------------------------
def _resolve_under(repo_root: Path, p: str | Path) -> Path:
    path = Path(p)
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    else:
        path = path.resolve()
    return path


def _safe_output_path(output_dir: Path, *parts: str) -> Path:
    out = (output_dir.joinpath(*parts)).resolve()
    output_dir = output_dir.resolve()
    try:
        out.relative_to(output_dir)
    except ValueError as e:
        raise ValueError(f"Refusing to write outside output_dir: {out}") from e
    return out


def _load_eval_strict(repo_root: Path):
    mod_path = repo_root / "backend" / "models" / "evaluate_model_strict_v2.py"
    if not mod_path.is_file():
        raise FileNotFoundError(f"Missing evaluator module: {mod_path}")
    spec = importlib.util.spec_from_file_location("_threshold_attack_eval_strict", mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load spec for {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[str(spec.name)] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


# ---------------------------------------------------------------------------
# Scoring: hybrid recon + pred + grad with weights from thresholds.json
# (mirrors backend/app/api.py compute_scores_hybrid; separation loss not used)
# ---------------------------------------------------------------------------
def compute_scores_hybrid_weighted(
    model: Any,
    X: np.ndarray,
    weights: Dict[str, Any],
    batch_size: int = 256,
) -> np.ndarray:
    recon, pred = model.predict(X, verbose=0, batch_size=int(batch_size))
    recon = np.asarray(recon, dtype=np.float32)
    pred = np.asarray(pred, dtype=np.float32)

    w = weights if isinstance(weights, dict) else {}
    w_recon = float(w.get("W_RECON", 1.0))
    w_pred = float(w.get("W_PRED", 2.0))
    w_grad = float(w.get("W_GRAD", 2.0))

    e_recon = np.mean((X - recon) ** 2, axis=(1, 2))

    dx_true = X[:, 1:, :] - X[:, :-1, :]
    dx_recon = recon[:, 1:, :] - recon[:, :-1, :]
    e_grad = np.mean((dx_true - dx_recon) ** 2, axis=(1, 2))

    t = X.shape[1]
    if pred.ndim == 3 and pred.shape[1] == t - 1:
        y_true = X[:, 1:, :]
        e_pred = np.mean((y_true - pred) ** 2, axis=(1, 2))
    elif pred.ndim == 2:
        pred_exp = pred[:, None, :]
        e_pred = np.mean((X[:, -1:, :] - pred_exp) ** 2, axis=(1, 2))
    elif pred.ndim == 3:
        if pred.shape[1] != 1:
            pred = pred[:, :1, :]
        e_pred = np.mean((X[:, -1:, :] - pred) ** 2, axis=(1, 2))
    else:
        e_pred = np.zeros(X.shape[0], dtype=np.float32)

    return (w_recon * e_recon + w_pred * e_pred + w_grad * e_grad).astype(np.float32)


@dataclass(frozen=True)
class Confusion:
    TP: int
    TN: int
    FP: int
    FN: int


def confusion_at_threshold(y_true: np.ndarray, y_score: np.ndarray, thr: float) -> Confusion:
    y_true = np.asarray(y_true).astype(np.uint8)
    y_pred = (np.asarray(y_score) > float(thr)).astype(np.uint8)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    return Confusion(TP=tp, TN=tn, FP=fp, FN=fn)


def metrics_from_confusion(cm: Confusion) -> Dict[str, float]:
    tp, tn, fp, fn = cm.TP, cm.TN, cm.FP, cm.FN
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    far = fp / (fp + tn) if (fp + tn) else 0.0
    bal_acc = 0.5 * (recall + tnr)
    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tnr": float(tnr),
        "far": float(far),
        "fpr": float(far),
        "balanced_accuracy": float(bal_acc),
    }


def best_f1_threshold_from_scores(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    """Returns (best_f1, threshold_at_best_f1) using score-sorted sweep (no sklearn)."""
    y_true = np.asarray(y_true).astype(np.uint8)
    y_score = np.asarray(y_score).astype(np.float64)
    order = np.argsort(-y_score, kind="mergesort")
    y_true_sorted = y_true[order]
    y_score_sorted = y_score[order]

    tps = np.cumsum(y_true_sorted)
    fps = np.cumsum(1 - y_true_sorted)
    p = float(y_true.sum())

    precision = tps / np.maximum(tps + fps, 1.0)
    recall = tps / max(p, 1.0)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    if len(f1) == 0:
        return 0.0, float("nan")
    j = int(np.argmax(f1))
    return float(f1[j]), float(y_score_sorted[j])


def load_thresholds_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_eval_thresholds(cfg: Dict[str, Any]) -> Tuple[List[Tuple[str, float]], bool]:
    """
    Returns list of (canonical_name, value) for thresholds present in cfg['thresholds'].
    Canonical names: p95, p97, p99, p99.5, p99.7, 3sigma, best_f1 (only if present in JSON).

    p99.5 / p99.7 accept JSON keys p99.5 or p995 / p99.7 or p997.
    """
    raw = cfg.get("thresholds")
    if not isinstance(raw, dict):
        raise ValueError("thresholds.json: missing object 'thresholds'")

    def pick(keys: Sequence[str]) -> Optional[float]:
        for k in keys:
            if k in raw and raw[k] is not None:
                return float(raw[k])
        return None

    rows: List[Tuple[str, float]] = []
    mapping: List[Tuple[str, Sequence[str]]] = [
        ("p95", ("p95",)),
        ("p97", ("p97",)),
        ("p99", ("p99",)),
        ("p99.5", ("p99.5", "p995")),
        ("p99.7", ("p99.7", "p997")),
        ("3sigma", ("3sigma",)),
        ("best_f1", ("best_f1", "best_f1_threshold", "bestF1")),
    ]
    had_best_f1_in_file = False
    for name, keys in mapping:
        v = pick(keys)
        if v is None:
            continue
        rows.append((name, v))
        if name == "best_f1":
            had_best_f1_in_file = True
    if not rows:
        raise ValueError("No known threshold keys found in thresholds.json['thresholds']")
    return rows, had_best_f1_in_file


def per_attack_block(
    scores: np.ndarray,
    y_true: np.ndarray,
    attack_type: np.ndarray,
    thr_name: str,
    thr_val: float,
) -> List[Dict[str, Any]]:
    """One row per attack_type present in attack_type array."""
    rows: List[Dict[str, Any]] = []
    types = sorted(set(str(x) for x in attack_type.tolist()))
    pred = scores > float(thr_val)
    y_true = np.asarray(y_true).astype(np.uint8)
    atype_arr = np.asarray(attack_type, dtype=object)

    for at in types:
        m = atype_arr == at
        if not np.any(m):
            continue
        s_sub = scores[m]
        y_sub = y_true[m]
        p_sub = pred[m]

        tp = int(np.sum(p_sub & (y_sub == 1)))
        fn = int(np.sum((~p_sub) & (y_sub == 1)))
        fp = int(np.sum(p_sub & (y_sub == 0)))
        total_attack_windows = int(np.sum(y_sub == 1))
        false_negatives = fn
        missed_rate = float(fn / total_attack_windows) if total_attack_windows else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
        precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        rows.append(
            {
                "attack_type": at,
                "threshold_name": thr_name,
                "threshold_value": float(thr_val),
                "detection_rate": recall,
                "recall": recall,
                "precision": precision,
                "f1": f1,
                "false_negatives": false_negatives,
                "total_attack_windows": total_attack_windows,
                "missed_rate": missed_rate,
            }
        )
    return rows


def pick_best_name(metric_list: List[Dict[str, Any]], key: str, mode: str) -> str:
    if not metric_list:
        return ""
    if mode == "max":
        best = max(
            metric_list,
            key=lambda r: (float(r.get(key, 0.0)), str(r["threshold_name"])),
        )
    elif mode == "min":
        best = min(
            metric_list,
            key=lambda r: (float(r.get(key, 1e9)), str(r["threshold_name"])),
        )
    else:
        raise ValueError(mode)
    return str(best["threshold_name"])


def pick_low_far_operational_candidates(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    prod_like = {"p95", "p97", "p99", "p99.5", "p99.7", "3sigma"}
    return [r for r in rows if str(r["threshold_name"]) in prod_like]


def choose_operational_threshold(overall_rows: List[Dict[str, Any]]) -> Tuple[str, str]:
    """
    Prefer p99 as operational default when present (production-aligned).
    Returns (name, reason_ar).
    """
    by_name = {str(r["threshold_name"]): r for r in overall_rows}
    if "p99" in by_name:
        return "p99", (
            "تم اختيار p99 كعتبة تشغيلية افتراضية لأنها مُعرّفة رسمياً في thresholds.json "
            "وتوازن بين الحساسية والتحكم في الإنذارات الكاذبة مقارنةً بعتبات أعلى مثل p99.5/p99.7 "
            "أو عتبة best_f1 المشتقة من تحسين أكاديمي على خليط البيانات."
        )
    if "p99.5" in by_name:
        return "p99.5", (
            "p99 غير متوفرة في ملف العتبات؛ تم اختيار p99.5 كبديل تشغيلي أكثر تحفظاً على الإنذارات الكاذبة."
        )
    prod = pick_low_far_operational_candidates(overall_rows)
    if not prod:
        return str(overall_rows[0]["threshold_name"]), "لا توجد عتبات إنتاجية معروفة؛ تم استخدام أول عتبة متاحة."
    prod_sorted = sorted(prod, key=lambda r: (float(r["far"]), -float(r["threshold_value"])))
    name = str(prod_sorted[0]["threshold_name"])
    return name, (
        f"تم اختيار {name} لأنها أقل FAR ضمن مجموعة العتبات الإنتاجية المتاحة بعد غياب p99."
    )


def write_threshold_overall_metrics_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    cols = [
        "threshold_name",
        "threshold_value",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "far",
        "fpr",
        "tnr",
        "tp",
        "tn",
        "fp",
        "fn",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})


def write_per_attack_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    cols = [
        "attack_type",
        "threshold_name",
        "threshold_value",
        "detection_rate",
        "recall",
        "precision",
        "f1",
        "false_negatives",
        "total_attack_windows",
        "missed_rate",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})


def write_best_summary_csv(path: Path, row: Dict[str, str]) -> None:
    cols = [
        "best_threshold_by_f1",
        "best_threshold_by_balanced_accuracy",
        "best_threshold_by_low_far",
        "best_operational_threshold",
        "reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerow({k: row.get(k, "") for k in cols})


def build_markdown_report(
    path: Path,
    overall: List[Dict[str, Any]],
    per_attack: List[Dict[str, Any]],
    summary: Dict[str, str],
    weights_note: str,
) -> None:
    def fmt_row(r: Dict[str, Any]) -> str:
        return (
            f"| {r['threshold_name']} | {r['threshold_value']:.6g} | {r['f1']:.4f} | "
            f"{r['balanced_accuracy']:.4f} | {r['far']:.4f} | {r['recall']:.4f} |"
        )

    lines: List[str] = []
    lines.append("# تحليل العتبات مقابل أنواع الهجمات (تجربة معزولة)\n")
    lines.append("هذا التقرير مُولَّد تلقائياً ولا يؤثر على الإنتاج.\n")
    lines.append(f"**صيغة الاستدلال:** {weights_note}\n")
    lines.append("**تسمية النوافذ:** تُستخدم `y_window` من attacked_v2 (نافذة شاذة إذا ≥10% من الخطوات الزمنية تحتوي هجوماً، كما في مولّد البيانات).\n")
    lines.append("\n## ملخص العتبات على المستوى الكلي\n")
    lines.append("| threshold | value | F1 | balanced_acc | FAR | recall (TPR) |\n")
    lines.append("|---:|---:|---:|---:|---:|---:|\n")
    for r in sorted(overall, key=lambda x: str(x["threshold_name"])):
        lines.append(fmt_row(r) + "\n")

    lines.append("\n## نتائج مختارة\n")
    lines.append(f"- **أعلى F1:** `{summary['best_threshold_by_f1']}`\n")
    lines.append(f"- **أقل FAR (ضمن العتبات الإنتاجية p95…3sigma دون best_f1):** `{summary['best_threshold_by_low_far']}`\n")
    lines.append(f"- **أعلى دقة متوازنة:** `{summary['best_threshold_by_balanced_accuracy']}`\n")
    lines.append(f"- **العتبة التشغيلية المقترحة:** `{summary['best_operational_threshold']}`\n")
    lines.append(f"\n**تبرير العتبة التشغيلية:** {summary['reason']}\n")

    lines.append("\n## مقارنة p99 مع p99.5 و p99.7 و 3sigma و best_f1\n")
    want = {"p99", "p99.5", "p99.7", "3sigma", "best_f1"}
    sub = [r for r in overall if str(r["threshold_name"]) in want]
    if sub:
        lines.append("| threshold | F1 | FAR | recall |\n")
        lines.append("|---:|---:|---:|---:|\n")
        for r in sorted(sub, key=lambda x: str(x["threshold_name"])):
            lines.append(f"| {r['threshold_name']} | {r['f1']:.4f} | {r['far']:.4f} | {r['recall']:.4f} |\n")
    else:
        lines.append("_لا توجد صفوف مطابقة لهذه الأسماء في التقييم الحالي._\n")

    lines.append("\n## أفضل عتبة لكل نوع هجوم (حسب أعلى F1 على نوافذ ذلك النوع)\n")
    by_attack: Dict[str, List[Dict[str, Any]]] = {}
    for r in per_attack:
        by_attack.setdefault(str(r["attack_type"]), []).append(r)
    for at in sorted(by_attack.keys()):
        best = max(by_attack[at], key=lambda x: float(x["f1"]))
        lines.append(
            f"- **{at}:** `{best['threshold_name']}` (F1={best['f1']:.4f}, "
            f"recall={best['recall']:.4f}). تفاصيل إضافية في `per_attack_metrics.csv`.\n"
        )

    lines.append("\n## لماذا قد تبقى p99 عتبة الإنتاج حتى لو لم تكن أعلى F1؟\n")
    lines.append(
        "- **الاستقرار والحوكمة:** عتبات المئينيات على شريحة الاختبار الطبيعي تكون قابلة للتفسير "
        "ومُسبقة التعريف، بينما `best_f1` عتبة تُحسَّب من خليط يضم الهجمات وتعكس أقصى أداء أكاديمي "
        "قد يرفع FAR على البيانات الطبيعية.\n"
    )
    lines.append(
        "- **التكلفة التشغيلية:** في الإنتاج، تكلفة الإنذار الكاذب (FP) غالباً أعلى من تكلفة "
        "تأخير اكتشاف نادرة؛ لذلك لا نعتمد دائماً على أعلى F1 على مجموعة تقييم ثابتة.\n"
    )

    lines.append("\n## الفرق بين التحسين الأكاديمي والنشر الإنتاجي\n")
    lines.append(
        "| الجانب | Academic (مثل best_f1) | Production (مثل p99) |\n"
        "|---|---|---|\n"
        "| الهدف | تعظيم مقياس على مجموعة تقييم محددة | ضبط معدل إنذارات كاذبة مقبول مع تغطية معقولة |\n"
        "| العتبة | تُشتق من المنحنى (قد تتغير بإعادة الجمع) | ثابتة نسبياً ومربوطة بتوزيع السلوك الطبيعي |\n"
        "| المخاطر | FAR أعلى، صعوبة التدقيق | قد يُفوّت بعض الهجمات الخفيفة |\n"
    )

    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Threshold vs attack analysis (isolated).")
    ap.add_argument("--repo-root", type=str, required=True)
    ap.add_argument("--model-path", type=str, required=True)
    ap.add_argument("--thresholds-path", type=str, required=True)
    ap.add_argument("--split-file", type=str, required=True)
    ap.add_argument("--normal-dir", type=str, required=True)
    ap.add_argument("--attacked-dir", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--split-key", type=str, default="test")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    model_path = _resolve_under(repo_root, args.model_path)
    thresholds_path = _resolve_under(repo_root, args.thresholds_path)
    split_file = _resolve_under(repo_root, args.split_file)
    normal_dir = _resolve_under(repo_root, args.normal_dir)
    attacked_dir = _resolve_under(repo_root, args.attacked_dir)
    output_dir = _resolve_under(repo_root, args.output_dir)

    for p, label in (
        (model_path, "model"),
        (thresholds_path, "thresholds"),
        (split_file, "split"),
        (normal_dir, "normal-dir"),
        (attacked_dir, "attacked-dir"),
    ):
        if not p.exists():
            raise FileNotFoundError(f"{label} not found: {p}")

    output_dir.mkdir(parents=True, exist_ok=True)

    ev = _load_eval_strict(repo_root)
    cfg_thr = load_thresholds_json(thresholds_path)
    weights = cfg_thr.get("weights", {})
    if not isinstance(weights, dict):
        weights = {}

    w_note = (
        f"W_RECON·recon_err + W_PRED·pred_err + W_GRAD·grad_err "
        f"(W_RECON={weights.get('W_RECON', 1.0)}, W_PRED={weights.get('W_PRED', 2.0)}, "
        f"W_GRAD={weights.get('W_GRAD', 2.0)}؛ بدون separation loss)"
    )

    file_thresholds, had_best_f1_json = collect_eval_thresholds(cfg_thr)

    t_sample, c_sample = ev.infer_T_C_from_sample(normal_dir)
    model = ev.load_keras_model_robust(model_path, t_sample, c_sample)

    names = ev.load_split_filenames(split_file, args.split_key)
    normal_parts: List[np.ndarray] = []
    for fname in names:
        fp = (normal_dir / fname).resolve()
        if not fp.is_file():
            continue
        x = ev.load_windows_npy(fp)
        if x.shape[1] != t_sample:
            raise ValueError(f"T mismatch {fp}: expected {t_sample}, got {x.shape}")
        s = compute_scores_hybrid_weighted(model, x, weights, batch_size=int(args.batch_size))
        normal_parts.append(s.astype(np.float64))
    if not normal_parts:
        raise RuntimeError("No normal test windows scored; check split-file and normal-dir.")
    scores_normal = np.concatenate(normal_parts)

    attacked_files = sorted(attacked_dir.glob("*.npz"))
    if not attacked_files:
        raise RuntimeError(f"No .npz files under attacked-dir: {attacked_dir}")

    atk_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    type_parts: List[np.ndarray] = []

    for p in attacked_files:
        x_att, y_w, meta = ev.load_attacked_npz(p)
        if x_att.shape[1] != t_sample:
            raise ValueError(f"T mismatch in {p}: {x_att.shape}")
        s = compute_scores_hybrid_weighted(model, x_att, weights, batch_size=int(args.batch_size))
        atk_parts.append(s.astype(np.float64))
        y_parts.append(np.asarray(y_w).astype(np.uint8))
        at = str(meta.get("attack_type", "unknown"))
        type_parts.append(np.array([at] * int(len(s)), dtype=object))

    scores_attacked = np.concatenate(atk_parts)
    y_attacked = np.concatenate(y_parts)
    attack_type_per_win = np.concatenate(type_parts)

    y_true = np.concatenate(
        [np.zeros(len(scores_normal), dtype=np.uint8), y_attacked.astype(np.uint8)]
    )
    y_score = np.concatenate([scores_normal, scores_attacked])

    eval_thresholds: List[Tuple[str, float]] = list(file_thresholds)

    if not had_best_f1_json:
        bf1, bf1_thr = best_f1_threshold_from_scores(y_true, y_score)
        eval_thresholds.append(("best_f1", float(bf1_thr)))
        meta_path = _safe_output_path(output_dir, "computed_best_f1_discovery.json")
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "best_f1": bf1,
                    "best_f1_threshold": bf1_thr,
                    "note": "best_f1 غير موجود في thresholds.json؛ تم اشتقاقه من sweep للتحليل فقط.",
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    overall_rows: List[Dict[str, Any]] = []
    for thr_name, thr_val in eval_thresholds:
        cm = confusion_at_threshold(y_true, y_score, thr_val)
        m = metrics_from_confusion(cm)
        overall_rows.append(
            {
                "threshold_name": thr_name,
                "threshold_value": float(thr_val),
                **m,
                "tp": cm.TP,
                "tn": cm.TN,
                "fp": cm.FP,
                "fn": cm.FN,
            }
        )

    per_rows: List[Dict[str, Any]] = []
    for thr_name, thr_val in eval_thresholds:
        per_rows.extend(
            per_attack_block(scores_attacked, y_attacked, attack_type_per_win, thr_name, thr_val)
        )

    prod_only_far = pick_low_far_operational_candidates(overall_rows)
    best_f1_name = pick_best_name(overall_rows, "f1", "max")
    best_bal_name = pick_best_name(overall_rows, "balanced_accuracy", "max")
    if prod_only_far:
        low_far_row = min(
            prod_only_far,
            key=lambda r: (float(r["far"]), -float(r["threshold_value"]), str(r["threshold_name"])),
        )
        low_far_name = str(low_far_row["threshold_name"])
    else:
        low_far_row = min(
            overall_rows,
            key=lambda r: (float(r["far"]), -float(r["threshold_value"]), str(r["threshold_name"])),
        )
        low_far_name = str(low_far_row["threshold_name"])

    op_name, op_reason = choose_operational_threshold(overall_rows)

    summary = {
        "best_threshold_by_f1": best_f1_name,
        "best_threshold_by_balanced_accuracy": best_bal_name,
        "best_threshold_by_low_far": low_far_name,
        "best_operational_threshold": op_name,
        "reason": op_reason.replace("\n", " ").strip(),
    }

    p_overall = _safe_output_path(output_dir, "threshold_overall_metrics.csv")
    p_per = _safe_output_path(output_dir, "per_attack_metrics.csv")
    p_sum = _safe_output_path(output_dir, "best_threshold_summary.csv")
    p_md = _safe_output_path(output_dir, "threshold_attack_analysis_report.md")

    write_threshold_overall_metrics_csv(p_overall, overall_rows)
    write_per_attack_csv(p_per, per_rows)
    write_best_summary_csv(p_sum, summary)
    build_markdown_report(p_md, overall_rows, per_rows, summary, w_note)

    print("\n=== Threshold / attack analysis complete ===")
    print(f"Best threshold by F1: {summary['best_threshold_by_f1']}")
    print(f"Best threshold by balanced accuracy: {summary['best_threshold_by_balanced_accuracy']}")
    print(f"Lowest FAR threshold (production-like set): {summary['best_threshold_by_low_far']}")
    print(f"Recommended operational threshold: {summary['best_operational_threshold']}")
    print("\nOutput files:")
    print(f"  {p_overall}")
    print(f"  {p_per}")
    print(f"  {p_sum}")
    print(f"  {p_md}")
    if not had_best_f1_json:
        print(f"  {_safe_output_path(output_dir, 'computed_best_f1_discovery.json')}")


if __name__ == "__main__":
    main()
