#!/usr/bin/env python3
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import csv
import re

PROJECT_ROOT = Path.cwd()

MODEL_NAME = "v0"
DATASET_NAME = "training-dv0a"
APPROACH_NAME = "nsga"
# APPROACH_NAME = "random"

MODEL_PATH = PROJECT_ROOT / "initial_dataset/best.pt"

ORIG_IMG_DIR = PROJECT_ROOT / "initial_dataset/all_dv0a_train/images"
GT_LABEL_DIR = PROJECT_ROOT / "initial_dataset/all_dv0a_train/labels"

#NSGA OR RANDOM
RESULTS_ROOT = PROJECT_ROOT / "nsga-results-all-dv0a"
# RESULTS_ROOT = PROJECT_ROOT / "random-results-all-dv0a"
OUT_ROOT = PROJECT_ROOT / "final_compiled_eval_training_v0_nsga"
# OUT_ROOT = PROJECT_ROOT / "final_compiled_eval_training_v0_random"

OUT_ROOT.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_ROOT / "dv0a_all_perturbation_eval.csv"


# YOLO inference settings
CONF_INFER = 0.01
IOU_NMS = 0.5
IMGSZ = None

# Failure thresholds
CONF_MATCH = 0.25
IOU_DET = 0.50
IOU_LOC = 0.70
COUNT_AMBIGUOUS_AS_FAILURE = True

PERT_GLOB = "cand_*_pert.png"
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def load_gt_boxes_yolo(label_file: Path, H: int, W: int):
    if not label_file.exists():
        return np.empty((0, 5), dtype=float)

    txt = label_file.read_text().strip()
    if not txt:
        return np.empty((0, 5), dtype=float)

    out = []
    for ln in txt.splitlines():
        cls, xc, yc, bw, bh = map(float, ln.split())
        x1 = (xc - bw / 2) * W
        y1 = (yc - bh / 2) * H
        x2 = (xc + bw / 2) * W
        y2 = (yc + bh / 2) * H
        out.append([x1, y1, x2, y2, int(cls)])
    return np.array(out, dtype=float)


def compute_iou_xyxy(a, b):
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    areaA = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    areaB = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(areaA + areaB - inter, 1e-9)


def predict_one(model, img_bgr_uint8):
    kwargs = dict(
        source=[img_bgr_uint8],
        conf=CONF_INFER,
        iou=IOU_NMS,
        show=False,
        show_labels=False,
        save=False,
        verbose=False
    )
    if IMGSZ is not None:
        kwargs["imgsz"] = IMGSZ
    res = model.predict(**kwargs)
    return res[0]


def extract_preds(result):
    preds = []
    if result.boxes is None or len(result.boxes) == 0:
        return preds

    for b in result.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        cls = int(b.cls)
        conf = float(b.conf[0])
        preds.append([x1, y1, x2, y2, cls, conf])

    return preds


def avg_conf_iou_sameclass(preds, gt_boxes):
    if len(gt_boxes) == 0:
        return 0.0, 0.0

    confs, ious = [], []

    for g in gt_boxes:
        gx1, gy1, gx2, gy2, gcls = g
        best_iou = 0.0
        best_conf = 0.0

        for p in preds:
            px1, py1, px2, py2, pcl, pconf = p
            if pcl != int(gcls):
                continue

            iou = compute_iou_xyxy([gx1, gy1, gx2, gy2], [px1, py1, px2, py2])
            if iou > best_iou:
                best_iou = iou
                best_conf = pconf

        confs.append(best_conf)
        ious.append(best_iou)

    return float(np.mean(confs)), float(np.mean(ious))


def classify_failures_unique(preds, gt_boxes, conf_thr=CONF_MATCH, iou_det=IOU_DET, iou_loc=IOU_LOC):
    """
    Per GT object, assign exactly one of:
      - MISSED
      - AMBIGUOUS
      - MISLOCALIZED
      - MISCLASSIFIED
      - CORRECT
    """
    if len(gt_boxes) == 0:
        return 0, 0, 0, 0, 0, 0

    n_missed = 0
    n_miscls = 0
    n_misloc = 0
    n_amb = 0
    n_corr = 0

    for g in gt_boxes:
        gx1, gy1, gx2, gy2, gcls = g
        gt_cls = int(gcls)
        gt_box = [gx1, gy1, gx2, gy2]

        acceptable = []
        for p in preds:
            px1, py1, px2, py2, pcl, pconf = p
            pcl = int(pcl)
            iou = compute_iou_xyxy(gt_box, [px1, py1, px2, py2])

            if (pconf >= conf_thr) and (iou >= iou_det):
                acceptable.append((float(iou), pcl, float(pconf)))

        if not acceptable:
            n_missed += 1
            continue

        A = max(acceptable, key=lambda t: t[0])  
        B = max(acceptable, key=lambda t: t[2]) 

        best_iou, best_cls, _ = A
        _, confbest_cls, _ = B

        amb = False
        if best_iou >= iou_loc and best_cls != confbest_cls:
            a_match = (best_cls == gt_cls)
            b_match = (confbest_cls == gt_cls)
            if a_match ^ b_match:
                amb = True

        if amb:
            n_amb += 1
            continue

        if best_iou < iou_loc:
            n_misloc += 1
            continue

        if best_cls != gt_cls:
            n_miscls += 1
        else:
            n_corr += 1

    return len(gt_boxes), n_missed, n_miscls, n_misloc, n_amb, n_corr


def compute_failure_flags(n_miss, n_miscls, n_misloc, n_amb):
    has_miss = int(n_miss > 0)
    has_misloc = int(n_misloc > 0)
    has_miscls = int(n_miscls > 0)
    has_amb = int(n_amb > 0)

    any_failure = int(
        has_miss or
        has_misloc or
        has_miscls or
        (COUNT_AMBIGUOUS_AS_FAILURE and has_amb)
    )

    return has_miss, has_misloc, has_miscls, has_amb, any_failure


def find_gt_label(gt_dir: Path, img_path: Path):
    cand = gt_dir / f"{img_path.stem}.txt"
    return cand if cand.exists() else None


def find_run_dirs(results_root: Path):
    run_dirs = []
    for p in sorted(results_root.glob("run-*")):
        if p.is_dir():
            run_dirs.append(p)
    return run_dirs


def cand_id_from_path(pth: Path) -> str:
    name = pth.name
    if name.endswith("_pert.png"):
        return name[:-len("_pert.png")]
    return pth.stem


def metrics_txt_from_pert_path(pert_path: Path) -> Path:
    cid = cand_id_from_path(pert_path)
    return pert_path.parent / f"{cid}_metrics.txt"


def parse_realism_from_metrics(metrics_path: Path):
    """
    Reads:
      mag_p95=...
      area_frac=...
      realism_budget=...
    from the candidate metrics txt.
    """
    if not metrics_path.exists():
        return np.nan, np.nan, np.nan

    txt = metrics_path.read_text()

    mag = np.nan
    area = np.nan
    budget = np.nan

    mag_m = re.search(r"mag_p95=([-+0-9.eE]+)", txt)
    area_m = re.search(r"area_frac=([-+0-9.eE]+)", txt)
    budget_m = re.search(r"realism_budget=([-+0-9.eE]+)", txt)

    if mag_m:
        mag = float(mag_m.group(1))
    if area_m:
        area = float(area_m.group(1))
    if budget_m:
        budget = float(budget_m.group(1))

    return mag, area, budget


def main():
    model = YOLO(MODEL_PATH)

    orig_images = sorted([
        p for p in ORIG_IMG_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    ])

    if not orig_images:
        raise FileNotFoundError(f"No images found in: {ORIG_IMG_DIR}")

    run_dirs = find_run_dirs(RESULTS_ROOT)
    if not run_dirs:
        raise FileNotFoundError(f"No run-* folders found in: {RESULTS_ROOT}")

    print(f"Model        : {MODEL_NAME}")
    print(f"Dataset      : {DATASET_NAME}")
    print(f"Approach     : {APPROACH_NAME}")
    print(f"Results root : {RESULTS_ROOT}")
    print(f"Runs found   : {[p.name for p in run_dirs]}")
    print(f"Images found : {len(orig_images)}")

    rows = []
    skipped_no_gt = 0
    skipped_no_pert_dir = 0
    skipped_no_pert_files = 0

    for run_dir in run_dirs:
        run_name = run_dir.name
        print(f"\n{'=' * 80}")
        print(f"[RUN] {run_name}")
        print(f"{'=' * 80}")

        for img_path in orig_images:
            gt_path = find_gt_label(GT_LABEL_DIR, img_path)
            if gt_path is None:
                print(f"[skip] No GT label for {img_path.name}")
                skipped_no_gt += 1
                continue

            orig_bgr = cv2.imread(str(img_path))
            if orig_bgr is None:
                print(f"[skip] Could not read original image: {img_path}")
                continue

            H, W = orig_bgr.shape[:2]
            gt_boxes = load_gt_boxes_yolo(gt_path, H, W)
            if len(gt_boxes) == 0:
                print(f"[skip] Empty GT for {img_path.name}")
                continue

            pert_dir = run_dir / img_path.stem / "ALL_SAVED"
            if not pert_dir.exists():
                print(f"[skip] No perturbation folder for {img_path.name} in {run_name}")
                skipped_no_pert_dir += 1
                continue

            pert_paths = sorted(pert_dir.glob(PERT_GLOB))
            if not pert_paths:
                print(f"[skip] No perturbation files for {img_path.name} in {run_name}")
                skipped_no_pert_files += 1
                continue

            res_orig = predict_one(model, orig_bgr)
            preds_orig = extract_preds(res_orig)
            orig_avg_conf, orig_avg_iou = avg_conf_iou_sameclass(preds_orig, gt_boxes)

            # Full failure analysis on original image
            (
                orig_n_gt,
                orig_n_miss,
                orig_n_miscls,
                orig_n_misloc,
                orig_n_amb,
                orig_n_corr,
            ) = classify_failures_unique(preds_orig, gt_boxes)

            (
                orig_has_miss,
                orig_has_misloc,
                orig_has_miscls,
                orig_has_amb,
                orig_any_failure,
            ) = compute_failure_flags(
                orig_n_miss, orig_n_miscls, orig_n_misloc, orig_n_amb
            )

            print(
                f"[image] {img_path.name} | candidates={len(pert_paths)} | "
                f"orig_any_failure={orig_any_failure}"
            )

            for pert_path in pert_paths:
                pert_bgr = cv2.imread(str(pert_path))
                if pert_bgr is None:
                    print(f"[skip] Could not read perturbation: {pert_path}")
                    continue

                cid = cand_id_from_path(pert_path)

                res_pert = predict_one(model, pert_bgr)
                preds_pert = extract_preds(res_pert)
                pert_avg_conf, pert_avg_iou = avg_conf_iou_sameclass(preds_pert, gt_boxes)

                (
                    n_gt,
                    n_miss,
                    n_miscls,
                    n_misloc,
                    n_amb,
                    n_corr,
                ) = classify_failures_unique(preds_pert, gt_boxes)

                (
                    has_miss,
                    has_misloc,
                    has_miscls,
                    has_amb,
                    any_failure,
                ) = compute_failure_flags(n_miss, n_miscls, n_misloc, n_amb)

                metrics_path = metrics_txt_from_pert_path(pert_path)
                mag_p95, area_frac, realism_budget = parse_realism_from_metrics(metrics_path)

                rows.append({
                    "model": MODEL_NAME,
                    "dataset": DATASET_NAME,
                    "approach": APPROACH_NAME,
                    "run": run_name,
                    "image": img_path.name,
                    "candidate": cid,

                    # original quality
                    "orig_avg_conf_same": float(orig_avg_conf),
                    "orig_avg_iou_same": float(orig_avg_iou),

                    # original failure info
                    "orig_n_gt": int(orig_n_gt),
                    "orig_gt_missed_count": int(orig_n_miss),
                    "orig_gt_mislocalized_count": int(orig_n_misloc),
                    "orig_gt_misclassified_count": int(orig_n_miscls),
                    "orig_gt_ambiguous_count": int(orig_n_amb),
                    "orig_gt_correct_count": int(orig_n_corr),
                    "orig_has_missed": int(orig_has_miss),
                    "orig_has_mislocalized": int(orig_has_misloc),
                    "orig_has_misclassified": int(orig_has_miscls),
                    "orig_has_ambiguous": int(orig_has_amb),
                    "orig_any_failure": int(orig_any_failure),

                    # perturbed quality
                    "pert_avg_conf_same": float(pert_avg_conf),
                    "pert_avg_iou_same": float(pert_avg_iou),

                    # perturbation realism info
                    "realism_budget": float(realism_budget) if np.isfinite(realism_budget) else np.nan,
                    "mag_p95": float(mag_p95) if np.isfinite(mag_p95) else np.nan,
                    "area_frac": float(area_frac) if np.isfinite(area_frac) else np.nan,

                    # perturbed failure info
                    "n_gt": int(n_gt),
                    "gt_missed_count": int(n_miss),
                    "gt_mislocalized_count": int(n_misloc),
                    "gt_misclassified_count": int(n_miscls),
                    "gt_ambiguous_count": int(n_amb),
                    "gt_correct_count": int(n_corr),
                    "has_missed": int(has_miss),
                    "has_mislocalized": int(has_misloc),
                    "has_misclassified": int(has_miscls),
                    "has_ambiguous": int(has_amb),
                    "any_failure": int(any_failure),
                })

    if not rows:
        print("\nNo rows collected. Nothing to save.")
        return

    fieldnames = [
        "model",
        "dataset",
        "approach",
        "run",
        "image",
        "candidate",

        # original quality
        "orig_avg_conf_same",
        "orig_avg_iou_same",

        # original failure info
        "orig_n_gt",
        "orig_gt_missed_count",
        "orig_gt_mislocalized_count",
        "orig_gt_misclassified_count",
        "orig_gt_ambiguous_count",
        "orig_gt_correct_count",
        "orig_has_missed",
        "orig_has_mislocalized",
        "orig_has_misclassified",
        "orig_has_ambiguous",
        "orig_any_failure",

        # perturbed quality
        "pert_avg_conf_same",
        "pert_avg_iou_same",

        # perturbation realism info
        "realism_budget",
        "mag_p95",
        "area_frac",

        # perturbed failure info
        "n_gt",
        "gt_missed_count",
        "gt_mislocalized_count",
        "gt_misclassified_count",
        "gt_ambiguous_count",
        "gt_correct_count",
        "has_missed",
        "has_mislocalized",
        "has_misclassified",
        "has_ambiguous",
        "any_failure",
    ]

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Saved rows            : {len(rows)}")
    print(f"Saved CSV             : {OUT_CSV.resolve()}")
    print(f"Skipped (no GT)       : {skipped_no_gt}")
    print(f"Skipped (no pert dir) : {skipped_no_pert_dir}")
    print(f"Skipped (no pert img) : {skipped_no_pert_files}")


if __name__ == "__main__":
    main()