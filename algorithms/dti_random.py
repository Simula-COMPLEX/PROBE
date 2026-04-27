#!/usr/bin/env python3
from pathlib import Path
import json
import random
import numpy as np
import cv2
from ultralytics import YOLO
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import random
PROJECT_ROOT = Path.cwd()

IMAGES_DIR = PROJECT_ROOT / "initial_dataset/all_dv0a_train/images"
LABELS_DIR = PROJECT_ROOT / "initial_dataset/all_dv0a_train/labels"
MODEL_PATH = PROJECT_ROOT / "initial_dataset/best.pt"


# Multi-Run Settings (same as NSGA)
N_RUNS = 10
BASE_SEED = random.randint(0, 1_000_000)
ROOT_OUT = Path("./random-results-all-dv0a")
ROOT_OUT.mkdir(parents=True, exist_ok=True)

CONF_INFER = 0.05
IOU_NMS = 0.5
IMGSZ = None

EPS = 48.0
MARGIN = 5

MIN_RADIUS = 8.0
MAX_RADIUS = 80.0
MIN_SIGMA_RATIO = 0.15
MAX_SIGMA_RATIO = 0.80

# Match NSGA budget
POP_SIZE = 40
N_GEN = 500

# Subset visuals
N_FAILURE_LIKE = 6
AMPLIFY = 8.0

# 3 objectives (all minimized)
OBJECTIVE_NAMES = ["avg_conf_same", "avg_iou_same", "realism_budget"]

DEBUG_PRINT_OBJECTIVES = True  


model = YOLO(MODEL_PATH)
if isinstance(model.names, dict):
    NAMES_MAP = {int(k): v for k, v in model.names.items()}
else:
    NAMES_MAP = {i: n for i, n in enumerate(model.names)}


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
    xA = max(a[0], b[0]); yA = max(a[1], b[1])
    xB = min(a[2], b[2]); yB = min(a[3], b[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    areaA = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    areaB = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(areaA + areaB - inter, 1e-9)


def run_inference_bgr(img_bgr_uint8):
    kwargs = dict(
        source=[img_bgr_uint8],
        conf=CONF_INFER,
        iou=IOU_NMS,
        save=False,
        verbose=False,
        show=False,
        show_labels=False
    )
    if IMGSZ is not None:
        kwargs["imgsz"] = IMGSZ
    res = model.predict(**kwargs)
    return res[0]


def extract_preds(res):
    preds = []
    if res.boxes is None or len(res.boxes) == 0:
        return preds
    for b in res.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        cls = int(b.cls)
        conf = float(b.conf[0])
        preds.append([x1, y1, x2, y2, cls, conf])
    return preds


def evaluate_objectives(res, gt_boxes):
    preds = extract_preds(res)
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


def match_predictions_to_gt(res, gt_boxes):
    preds = extract_preds(res)
    out = []
    for gi, g in enumerate(gt_boxes):
        gx1, gy1, gx2, gy2, gcls = g
        gt_box = [gx1, gy1, gx2, gy2]
        gcls = int(gcls)

        best_same_iou, best_same_conf, best_same_cls, best_same_box = 0.0, 0.0, None, None
        best_any_iou, best_any_conf, best_any_cls, best_any_box = 0.0, 0.0, None, None

        for p in preds:
            px1, py1, px2, py2, pcl, pconf = p
            iou = compute_iou_xyxy(gt_box, [px1, py1, px2, py2])

            if iou > best_any_iou:
                best_any_iou, best_any_conf = iou, pconf
                best_any_cls = int(pcl)
                best_any_box = [px1, py1, px2, py2]

            if int(pcl) == gcls and iou > best_same_iou:
                best_same_iou, best_same_conf = iou, pconf
                best_same_cls = int(pcl)
                best_same_box = [px1, py1, px2, py2]

        if best_same_iou > 0.0:
            status = "OK"
        else:
            if best_any_iou > 0.0 and best_any_cls is not None and best_any_cls != gcls:
                status = "WRONG_CLASS"
            else:
                status = "MISSED"

        out.append({
            "gt_index": gi,
            "gt_class": gcls,
            "gt_box": gt_box,
            "same": {
                "matched": best_same_cls is not None and best_same_iou > 0.0,
                "pred_class": best_same_cls,
                "conf": float(best_same_conf),
                "iou": float(best_same_iou),
                "pred_box": best_same_box,
            },
            "any": {
                "matched": best_any_cls is not None and best_any_iou > 0.0,
                "pred_class": best_any_cls,
                "conf": float(best_any_conf),
                "iou": float(best_any_iou),
                "pred_box": best_any_box,
            },
            "status": status
        })
    return out


def format_box_xyxy(box):
    if box is None:
        return "None"
    x1, y1, x2, y2 = box
    return f"[{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}]"


def write_detailed_match_log(f, match_list, tag):
    f.write(f"\n=== {tag} (per GT object) ===\n")
    if len(match_list) == 0:
        f.write("No GT objects.\n")
        return
    for m in match_list:
        gi = m["gt_index"]
        gcls = m["gt_class"]
        gtname = NAMES_MAP.get(gcls, str(gcls))
        f.write(f"GT[{gi}] class={gcls} ({gtname}) box={format_box_xyxy(m['gt_box'])}\n")

        s = m["same"]
        a = m["any"]

        same_name = NAMES_MAP.get(s["pred_class"], str(s["pred_class"])) if s["pred_class"] is not None else "None"
        any_name = NAMES_MAP.get(a["pred_class"], str(a["pred_class"])) if a["pred_class"] is not None else "None"

        f.write(
            f"  SAME-CLASS match: matched={int(s['matched'])}, pred_class={s['pred_class']} ({same_name}), "
            f"conf={s['conf']:.4f}, iou={s['iou']:.4f}, pred_box={format_box_xyxy(s['pred_box'])}\n"
        )
        f.write(
            f"  ANY-CLASS match : matched={int(a['matched'])}, pred_class={a['pred_class']} ({any_name}), "
            f"conf={a['conf']:.4f}, iou={a['iou']:.4f}, pred_box={format_box_xyxy(a['pred_box'])}\n"
        )
        f.write(f"  STATUS: {m['status']}\n\n")


def make_gaussian_alpha(H, W, cx, cy, radius, sigma):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    alpha = np.exp(-d2 / (2.0 * (sigma ** 2) + 1e-9)).astype(np.float32)
    hard = (d2 <= (2.0 * radius) ** 2).astype(np.float32)
    return alpha * hard


def apply_patch_bgr(orig_bgr, params, W, H):
    cx, cy, radius, sigma_ratio, dB, dG, dR = params
    cx = float(np.clip(cx, 0, W - 1))
    cy = float(np.clip(cy, 0, H - 1))
    radius = float(np.clip(radius, MIN_RADIUS, MAX_RADIUS))
    sigma_ratio = float(np.clip(sigma_ratio, MIN_SIGMA_RATIO, MAX_SIGMA_RATIO))
    sigma = max(1.0, radius * sigma_ratio)

    alpha = make_gaussian_alpha(H, W, cx, cy, radius, sigma)
    delta_color_bgr = np.array([dB, dG, dR], dtype=np.float32).reshape(1, 1, 3)
    delta_map = alpha[:, :, None] * delta_color_bgr

    img_f = orig_bgr.astype(np.float32)
    pert = np.clip(img_f + delta_map, 0.0, 255.0).astype(np.uint8)

    patch_mask = (alpha > 0.01).astype(np.float32)
    return pert, delta_map, patch_mask


def mag_p95(delta_map, patch_mask, q=95):
    vals = np.abs(delta_map[patch_mask > 0]).reshape(-1)
    if vals.size == 0:
        return 0.0
    return float(np.percentile(vals, q))


def patch_area_frac(patch_mask):
    return float(np.mean(patch_mask > 0))


def realism_budget(delta_map, patch_mask):
    mag = mag_p95(delta_map, patch_mask, q=95)
    area = patch_area_frac(patch_mask)
    return float(area * (mag / max(EPS, 1e-9)))


def crowding_distance(F):
    F = np.asarray(F, dtype=float)
    N, M = F.shape
    dist = np.zeros(N, dtype=float)
    if N == 0:
        return dist
    for m in range(M):
        idx = np.argsort(F[:, m])
        dist[idx[0]] = np.inf
        dist[idx[-1]] = np.inf
        f_min = F[idx[0], m]
        f_max = F[idx[-1], m]
        denom = (f_max - f_min)
        if denom < 1e-12:
            continue
        for i in range(1, N - 1):
            dist[idx[i]] += (F[idx[i + 1], m] - F[idx[i - 1], m]) / denom
    return dist


def stable_seed_from_stem(stem: str, base_seed: int) -> int:
    v = 0
    for ch in stem.encode("utf-8"):
        v = (v * 131 + ch) % 1_000_000_007
    return int((v + base_seed) % 1_000_000_007)


def select_rank_fill_from_all(F_all: np.ndarray, X_all: np.ndarray, n_select: int):
    F_all = np.asarray(F_all, dtype=float)
    X_all = np.asarray(X_all, dtype=float)

    nds = NonDominatedSorting()
    fronts, rank = nds.do(F_all, return_rank=True)

    selected = []
    for front in fronts:
        front = list(front)
        if len(selected) >= n_select:
            break

        remaining = n_select - len(selected)
        if len(front) <= remaining:
            selected.extend(front)
        else:
            cd = crowding_distance(F_all[front])
            order = np.argsort(-cd)
            chosen = [front[i] for i in order[:remaining]]
            selected.extend(chosen)

    selected = np.asarray(selected, dtype=int)

    if selected.size == 0:
        selected = np.arange(min(n_select, len(F_all)), dtype=int)

    return X_all[selected], F_all[selected], rank[selected]


def save_patch_visuals(orig_bgr, out_dir: Path, prefix: str, pert_bgr, patch_mask):
    cv2.imwrite(str(out_dir / f"{prefix}_pert.png"), pert_bgr)
    overlay = orig_bgr.copy()
    mask = (patch_mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)
    cv2.imwrite(str(out_dir / f"{prefix}_patch_region.png"), overlay)


def write_metrics_file(out_dir: Path, prefix: str, x, gt_boxes, base_res, pert_res, delta_map, patch_mask):
    base_conf, base_iou = evaluate_objectives(base_res, gt_boxes)
    pert_conf, pert_iou = evaluate_objectives(pert_res, gt_boxes)

    mag = mag_p95(delta_map, patch_mask, q=95)
    area = patch_area_frac(patch_mask)
    budget = realism_budget(delta_map, patch_mask)

    base_match = match_predictions_to_gt(base_res, gt_boxes)
    pert_match = match_predictions_to_gt(pert_res, gt_boxes)

    with open(out_dir / f"{prefix}_metrics.txt", "w") as f:
        f.write(
            "Params (BGR deltas): "
            f"cx={x[0]:.2f}, cy={x[1]:.2f}, radius={x[2]:.2f}, sigma_ratio={x[3]:.3f}, "
            f"dB={x[4]:.2f}, dG={x[5]:.2f}, dR={x[6]:.2f}\n"
        )
        write_detailed_match_log(f, base_match, "ORIGINAL")
        write_detailed_match_log(f, pert_match, "PERTURBED")
        f.write("=== AGGREGATED (objectives) ===\n")
        f.write(f"ORIG: avg_conf_same={base_conf:.4f}, avg_iou_same={base_iou:.4f}\n")
        f.write(
            f"PERT: avg_conf_same={pert_conf:.4f}, avg_iou_same={pert_iou:.4f}, "
            f"mag_p95={mag:.4f}, area_frac={area:.4f}, realism_budget={budget:.6f}\n"
        )


def write_meta_json(out_dir: Path, image_path: Path, H: int, W: int, x_bounds, y_bounds, run_seed: int):
    meta = {
        "method": "RANDOM",
        "image": image_path.name,
        "image_stem": image_path.stem,
        "image_shape_hw": [int(H), int(W)],
        "budget": {"pop_size": int(POP_SIZE), "n_gen": int(N_GEN), "n_eval": int(POP_SIZE * N_GEN)},
        "seed": {
            "run_seed": int(run_seed),
            "per_image_seed": int(stable_seed_from_stem(image_path.stem, run_seed)),
        },
        "objectives": OBJECTIVE_NAMES,
        "minimize": [True] * len(OBJECTIVE_NAMES),
        "inference": {"conf": float(CONF_INFER), "iou_nms": float(IOU_NMS), "imgsz": None if IMGSZ is None else int(IMGSZ)},
        "patch": {
            "eps": float(EPS),
            "margin": int(MARGIN),
            "radius": [float(MIN_RADIUS), float(MAX_RADIUS)],
            "sigma_ratio": [float(MIN_SIGMA_RATIO), float(MAX_SIGMA_RATIO)],
            "center_bounds_xy": {"x": [int(x_bounds[0]), int(x_bounds[1])], "y": [int(y_bounds[0]), int(y_bounds[1])]},
        },
        "notes": (
            "ALL_PARETO contains ONLY rank-0 (non-dominated) solutions from all evaluations. "
            "ALL_SAVED contains EXACTLY POP_SIZE solutions selected by rank-fill (rank-0, rank-1, ...) "
            "with crowding-distance tie-break to reach POP_SIZE. "
            "Invalid (NaN/Inf) objective rows are filtered out before sorting/selection."
        ),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def run_one_image_random(image_path: Path, label_path: Path, out_root: Path, run_seed: int):
    np.random.seed(run_seed)
    random.seed(run_seed)

    orig_bgr = cv2.imread(str(image_path))
    if orig_bgr is None:
        print(f"[skip] Could not read image: {image_path}")
        return

    H, W = orig_bgr.shape[:2]
    gt_boxes = load_gt_boxes_yolo(label_path, H, W)
    if len(gt_boxes) == 0:
        print(f"[skip] No GT boxes (empty label): {label_path.name}")
        return

    subset_dir = out_root / "subset"
    all_pareto_dir = out_root / "ALL_PARETO"
    all_saved_dir = out_root / "ALL_SAVED"
    subset_dir.mkdir(parents=True, exist_ok=True)
    all_pareto_dir.mkdir(parents=True, exist_ok=True)
    all_saved_dir.mkdir(parents=True, exist_ok=True)

    x1 = int(max(0, np.min(gt_boxes[:, 0]) - MARGIN))
    y1 = int(max(0, np.min(gt_boxes[:, 1]) - MARGIN))
    x2 = int(min(W - 1, np.max(gt_boxes[:, 2]) + MARGIN))
    y2 = int(min(H - 1, np.max(gt_boxes[:, 3]) + MARGIN))
    XMIN, XMAX, YMIN, YMAX = x1, x2, y1, y2

    base_res = run_inference_bgr(orig_bgr)
    base_conf, base_iou = evaluate_objectives(base_res, gt_boxes)

    n_eval = POP_SIZE * N_GEN
    print(f"\n=== RANDOM: {image_path.name} ===")
    print(f"GT objects: {len(gt_boxes)} | bounds x[{XMIN},{XMAX}] y[{YMIN},{YMAX}] | baseline conf={base_conf:.3f} iou={base_iou:.3f}")
    print(f"Budget: POP_SIZE={POP_SIZE}, N_GEN={N_GEN} => N_EVAL={n_eval}")

    xl = np.array([XMIN, YMIN, MIN_RADIUS, MIN_SIGMA_RATIO, -EPS, -EPS, -EPS], dtype=float)
    xu = np.array([XMAX, YMAX, MAX_RADIUS, MAX_SIGMA_RATIO,  EPS,  EPS,  EPS], dtype=float)

    img_seed = stable_seed_from_stem(image_path.stem, run_seed)
    rng = np.random.default_rng(img_seed)

    X_all = np.zeros((n_eval, 7), dtype=float)
    F_all = np.zeros((n_eval, len(OBJECTIVE_NAMES)), dtype=float)

    for i in range(n_eval):
        x = rng.uniform(xl, xu)

        try:
            pert_bgr, delta_map, patch_mask = apply_patch_bgr(orig_bgr, x, W, H)
            res = run_inference_bgr(pert_bgr)

            avg_conf_same, avg_iou_same = evaluate_objectives(res, gt_boxes)
            budget = realism_budget(delta_map, patch_mask)

            F_row = [avg_conf_same, avg_iou_same, budget]

        except Exception as e:
            if DEBUG_PRINT_OBJECTIVES:
                print(f"[warn] eval failed at i={i}: {e}")
            F_row = [np.nan, np.nan, np.nan]

        X_all[i] = x
        F_all[i] = F_row

        if (i + 1) % max(1, n_eval // 10) == 0:
            print(f"  evaluated {i+1}/{n_eval}")

    valid = np.all(np.isfinite(F_all), axis=1)
    n_valid = int(np.sum(valid))
    print(f"[info] valid solutions: {n_valid}/{n_eval}")

    if DEBUG_PRINT_OBJECTIVES and n_valid > 0:
        Fv = F_all[valid]
        print(f"[debug] F min: {np.min(Fv, axis=0)}")
        print(f"[debug] F max: {np.max(Fv, axis=0)}")
        print(f"[debug] First 5 valid F rows:\n{Fv[:5]}")

    if n_valid == 0:
        print("[ERROR] All objective values are invalid (NaN/Inf). Nothing to save for this image.")
        write_meta_json(all_pareto_dir, image_path, H, W, (XMIN, XMAX), (YMIN, YMAX), run_seed=run_seed)
        return

    X_all_v = X_all[valid]
    F_all_v = F_all[valid]

    nd_idx = NonDominatedSorting().do(F_all_v, only_non_dominated_front=True)
    X_nd = X_all_v[nd_idx]
    F_nd = F_all_v[nd_idx]
    print(f"Random non-dominated (rank-0): {len(nd_idx)} / {len(F_all_v)} (valid evals)")

    np.save(all_pareto_dir / "pareto_X.npy", X_nd.astype(np.float32))
    np.save(all_pareto_dir / "pareto_F.npy", F_nd.astype(np.float32))
    write_meta_json(all_pareto_dir, image_path, H, W, (XMIN, XMAX), (YMIN, YMAX), run_seed=run_seed)

    print(f"Saving ALL_PARETO rank-0 ({len(X_nd)}) -> {all_pareto_dir}")
    for i in range(len(X_nd)):
        prefix = f"cand_{i+1:04d}"
        x = X_nd[i]
        pert_bgr, delta_map, patch_mask = apply_patch_bgr(orig_bgr, x, W, H)
        save_patch_visuals(orig_bgr, all_pareto_dir, prefix, pert_bgr, patch_mask)
        pert_res = run_inference_bgr(pert_bgr)
        write_metrics_file(all_pareto_dir, prefix, x, gt_boxes, base_res, pert_res, delta_map, patch_mask)


    target = min(int(POP_SIZE), int(len(F_all_v)))
    X_saved, F_saved, rank_saved = select_rank_fill_from_all(F_all_v, X_all_v, n_select=target)

    if len(X_saved) == 0 and len(F_all_v) > 0:
        X_saved = X_all_v[:target]
        F_saved = F_all_v[:target]
        rank_saved = np.zeros(target, dtype=int)

    np.save(all_saved_dir / "saved_X.npy", X_saved.astype(np.float32))
    np.save(all_saved_dir / "saved_F.npy", F_saved.astype(np.float32))

    saved_meta = {
        "image": image_path.name,
        "stem": image_path.stem,
        "method": "RANDOM",
        "seed": int(run_seed),
        "saved_n": int(len(X_saved)),
        "target_n": int(target),
        "note": "Saved set built by rank-fill (rank-0, rank-1, ...) with crowding-distance tie-break to reach target_n.",
        "rank_counts": {str(int(r)): int(np.sum(rank_saved == r)) for r in np.unique(rank_saved)} if len(rank_saved) > 0 else {},
        "n_valid_evals": int(len(F_all_v)),
    }
    with open(all_saved_dir / "saved_meta.json", "w") as f:
        json.dump(saved_meta, f, indent=2)

    print(f"Saving ALL_SAVED fair set ({len(X_saved)}) -> {all_saved_dir}")
    for i in range(len(X_saved)):
        prefix = f"cand_{i+1:04d}"
        x = X_saved[i]
        pert_bgr, delta_map, patch_mask = apply_patch_bgr(orig_bgr, x, W, H)
        save_patch_visuals(orig_bgr, all_saved_dir, prefix, pert_bgr, patch_mask)
        pert_res = run_inference_bgr(pert_bgr)
        write_metrics_file(all_saved_dir, prefix, x, gt_boxes, base_res, pert_res, delta_map, patch_mask)

    if len(X_nd) == 0:
        print("[warn] No rank-0 solutions to create subset.")
        return

    idx_min_conf = int(np.argmin(F_nd[:, 0]))
    idx_min_iou = int(np.argmin(F_nd[:, 1]))
    idx_min_bud = int(np.argmin(F_nd[:, 2]))

    chosen = [idx_min_conf, idx_min_iou, idx_min_bud]

    score = F_nd[:, 0] + F_nd[:, 1]
    for i in np.argsort(score)[:min(N_FAILURE_LIKE, len(score))]:
        chosen.append(int(i))

    seen = set()
    chosen_unique = []
    for i in chosen:
        if i not in seen:
            chosen_unique.append(i)
            seen.add(i)

    print(f"Saving subset ({len(chosen_unique)}) -> {subset_dir}")
    for j, idx in enumerate(chosen_unique, start=1):
        prefix = f"cand_{j:02d}"
        x = X_nd[idx]
        pert_bgr, delta_map, patch_mask = apply_patch_bgr(orig_bgr, x, W, H)
        save_patch_visuals(orig_bgr, subset_dir, prefix, pert_bgr, patch_mask)

        amplified = np.clip(orig_bgr.astype(np.float32) + AMPLIFY * delta_map, 0, 255).astype(np.uint8)
        cv2.imwrite(str(subset_dir / f"{prefix}_amplified_x{AMPLIFY:.1f}.png"), amplified)

        pert_res = run_inference_bgr(pert_bgr)
        write_metrics_file(subset_dir, prefix, x, gt_boxes, base_res, pert_res, delta_map, patch_mask)

        print(f"[saved subset] {prefix}: conf={F_nd[idx,0]:.3f}, iou={F_nd[idx,1]:.3f}, realism_budget={F_nd[idx,2]:.6f}")

    print(f"[done] {image_path.stem} -> {out_root}")


def get_missing_run_indices(root_out: Path, n_runs: int):
    missing = []
    for run_idx in range(1, n_runs + 1):
        run_root = root_out / f"run-{run_idx}"
        if not run_root.exists():
            missing.append(run_idx)
    return missing


def main():
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    images = sorted([p for p in IMAGES_DIR.iterdir() if p.is_file() and p.suffix.lower() in exts])

    if not images:
        print(f"No images found in: {IMAGES_DIR}")
        return

    print(f"Found {len(images)} images in: {IMAGES_DIR}")
    print(f"Labels dir: {LABELS_DIR}")
    print(f"Output root: {ROOT_OUT}")
    print(f"RANDOM budget: pop={POP_SIZE}, gen={N_GEN} => N_EVAL={POP_SIZE*N_GEN} per image")
    print(f"Objectives: {OBJECTIVE_NAMES}")
    print(f"Multi-run: N_RUNS={N_RUNS}, BASE_SEED={BASE_SEED} (run folders: run-1..run-{N_RUNS})")

    missing_runs = get_missing_run_indices(ROOT_OUT, N_RUNS)

    if not missing_runs:
        print("All requested run folders already exist. Nothing to do.")
        return

    print(f"Missing runs detected: {missing_runs}")

    for run_idx in missing_runs:
        run_seed = int(BASE_SEED + (run_idx - 1))
        run_root = ROOT_OUT / f"run-{run_idx}"
        run_root.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 80)
        print(f"[RUN {run_idx}/{N_RUNS}] seed={run_seed} -> {run_root}")
        print("=" * 80)

        for img_path in images:
            label_path = LABELS_DIR / f"{img_path.stem}.txt"
            if not label_path.exists():
                print(f"[skip] No label for {img_path.name} -> expected {label_path.name}")
                continue

            out_root = run_root / img_path.stem
            out_root.mkdir(parents=True, exist_ok=True)

            try:
                run_one_image_random(img_path, label_path, out_root, run_seed=run_seed)
            except Exception as e:
                print(f"[ERROR] run={run_idx} seed={run_seed} image={img_path.name}: {e}")

    print("\nAll missing runs completed.")


if __name__ == "__main__":
    main()