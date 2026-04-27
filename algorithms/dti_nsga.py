#!/usr/bin/env python3
from pathlib import Path
import json
import random
import numpy as np
import cv2
from ultralytics import YOLO

from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.indicators.hv import HV
from pymoo.core.termination import Termination
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pathlib import Path
import random
PROJECT_ROOT = Path.cwd()

IMAGES_DIR = PROJECT_ROOT / "initial_dataset/all_dv0a_train/images"
LABELS_DIR = PROJECT_ROOT / "initial_dataset/all_dv0a_train/labels"
MODEL_PATH = PROJECT_ROOT / "initial_dataset/best.pt"

N_RUNS = 10
BASE_SEED = random.randint(0, 1_000_000)
ROOT_OUT = Path("./nsga-results-all-dv0a")
ROOT_OUT.mkdir(parents=True, exist_ok=True)

# Use the global fmin/fmax produced by your HV comparison script
# The script to run, for computing globalref, is found under ./globalref/extract_hv_globalref.py
HV_GLOBALREF_SUMMARY_JSON = PROJECT_ROOT / "hv_out_globalref_all_experiments/hv_globalref_summary.json"

CONF_INFER = 0.05
IOU_NMS = 0.5
IMGSZ = None

EPS = 48.0
MARGIN = 5

MIN_RADIUS = 8.0
MAX_RADIUS = 80.0
MIN_SIGMA_RATIO = 0.15
MAX_SIGMA_RATIO = 0.80

POP_SIZE = 40
N_GEN = 500

AMPLIFY = 8.0

# ======== 3 objectives (all minimized) ========
# 1) avg_conf_same
# 2) avg_iou_same
# 3) realism_budget = area_frac * (mag_p95 / EPS)   (combined realism objective)
OBJECTIVE_NAMES = ["avg_conf_same", "avg_iou_same", "realism_budget"]

EARLYSTOP_ENABLED = False
EARLYSTOP_PATIENCE = 150
EARLYSTOP_REL_DELTA = 1e-3  # 0.1% improvement

EARLYSTOP_ABS_FLOOR = 1e-12  # avoid division/relative issues when best ~ 0


def load_global_norm_params(summary_json: Path, n_obj: int):
    if not summary_json.exists():
        return None
    try:
        j = json.loads(summary_json.read_text())
    except Exception:
        return None

    if "global_fmin" not in j or "global_fmax" not in j:
        return None

    fmin = np.asarray(j["global_fmin"], dtype=float)
    fmax = np.asarray(j["global_fmax"], dtype=float)
    pad = float(j.get("pad", 0.05))

    if fmin.ndim != 1 or fmax.ndim != 1 or fmin.size != fmax.size:
        return None

    if fmin.size < n_obj:
        return None

    if fmin.size > n_obj:
        fmin = fmin[:n_obj]
        fmax = fmax[:n_obj]

    ref = np.ones_like(fmin) * (1.0 + pad)
    return {"fmin": fmin, "fmax": fmax, "ref": ref, "pad": pad, "src": str(summary_json.resolve())}


def normalize(F, fmin, fmax):
    F = np.asarray(F, dtype=float)
    denom = (fmax - fmin)
    denom = np.where(denom < 1e-12, 1.0, denom)
    return (F - fmin) / denom


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
    kwargs = dict(source=[img_bgr_uint8], conf=CONF_INFER, iou=IOU_NMS, save=False, verbose=False)
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

def calc_crowding_distance(F: np.ndarray) -> np.ndarray:
    F = np.asarray(F, dtype=float)
    n, m = F.shape
    if n == 0:
        return np.array([], dtype=float)
    if n <= 2:
        return np.full(n, np.inf, dtype=float)

    cd = np.zeros(n, dtype=float)

    # For each objective
    for j in range(m):
        idx = np.argsort(F[:, j])
        f_sorted = F[idx, j]

        cd[idx[0]] = np.inf
        cd[idx[-1]] = np.inf

        f_min = f_sorted[0]
        f_max = f_sorted[-1]
        denom = f_max - f_min

        if denom < 1e-12:
            continue

        for k in range(1, n - 1):
            if np.isinf(cd[idx[k]]):
                continue
            cd[idx[k]] += (f_sorted[k + 1] - f_sorted[k - 1]) / denom

    return cd


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


def save_hv_artifacts(all_pareto_dir: Path, F: np.ndarray, X: np.ndarray,
                      image_path: Path, label_path: Path, seed: int):
    np.save(all_pareto_dir / "pareto_F.npy", np.asarray(F, dtype=np.float32))
    np.save(all_pareto_dir / "pareto_X.npy", np.asarray(X, dtype=np.float32))
    meta = {
        "image": image_path.name,
        "stem": image_path.stem,
        "label": label_path.name,
        "method": "NSGA-II",
        "pop_size": int(POP_SIZE),
        "n_gen_requested": int(N_GEN),
        "seed": int(seed),
        "n_obj": int(len(OBJECTIVE_NAMES)),
        "objectives": OBJECTIVE_NAMES,
        "conf_infer": float(CONF_INFER),
        "iou_nms": float(IOU_NMS),
        "imgsz": None if IMGSZ is None else int(IMGSZ),
        "eps": float(EPS),
        "margin": int(MARGIN),
        "min_radius": float(MIN_RADIUS),
        "max_radius": float(MAX_RADIUS),
        "min_sigma_ratio": float(MIN_SIGMA_RATIO),
        "max_sigma_ratio": float(MAX_SIGMA_RATIO),
        "earlystop_enabled": bool(EARLYSTOP_ENABLED),
        "earlystop_patience": int(EARLYSTOP_PATIENCE),
        "earlystop_rel_delta": float(EARLYSTOP_REL_DELTA),
        "n_rank0": int(len(F)),
    }
    with open(all_pareto_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def plot_convergence(out_root: Path, gens, hv, hv_best, min_norm_series, stem: str, p_ref: dict | None, obj_names):
    import matplotlib.pyplot as plt

    gens = np.asarray(gens, dtype=int)
    hv = np.asarray(hv, dtype=float)
    hv_best = np.asarray(hv_best, dtype=float)
    min_norm_series = np.asarray(min_norm_series, dtype=float)  

    plt.figure()
    plt.plot(gens, hv, marker="o", linestyle="-", label="HV")
    plt.plot(gens, hv_best, linestyle="--", label="HV best-so-far")
    plt.xlabel("Generation")
    plt.ylabel("Hypervolume (normalized)")
    title = "NSGA-II convergence (HV)"
    if p_ref is not None:
        title += f"\nref={np.round(p_ref['ref'], 3).tolist()}  (pad={p_ref['pad']})"
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_root / "convergence_hv.png", dpi=200)
    plt.close()

    for k, name in enumerate(obj_names):
        plt.figure()
        plt.plot(gens, min_norm_series[:, k], linestyle="-", marker="o")
        plt.xlabel("Generation")
        plt.ylabel(f"{name} (normalized)")
        plt.title(f"NSGA-II convergence: {name}\n(min over rank-0, normalized)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_root / f"convergence_obj_{name}.png", dpi=200)
        plt.close()


class HVRelativeStagnationTermination(Termination):
    def __init__(self, max_gen: int, hv_indicator: HV | None, norm_params: dict | None,
                 patience: int = 150, rel_delta: float = 1e-3, abs_floor: float = 1e-12):
        super().__init__()
        self.max_gen = int(max_gen)
        self.hv_indicator = hv_indicator
        self.norm_params = norm_params
        self.patience = int(patience)
        self.rel_delta = float(rel_delta)
        self.abs_floor = float(abs_floor)

        self.best = -np.inf
        self.last_improve_gen = 0

    def _update(self, algorithm):
        gen = int(getattr(algorithm, "n_gen", 0))

        if gen >= self.max_gen:
            return 1.0

        if self.hv_indicator is None or self.norm_params is None:
            return 0.0

        F = None
        if getattr(algorithm, "opt", None) is not None:
            try:
                F = algorithm.opt.get("F")
            except Exception:
                F = None
        if F is None:
            try:
                F = algorithm.pop.get("F")
            except Exception:
                F = None
        if F is None or len(F) == 0:
            return 0.0

        F = np.asarray(F, dtype=float)
        Fn = normalize(F, self.norm_params["fmin"], self.norm_params["fmax"])
        hv_val = float(self.hv_indicator(Fn))

        base = max(float(self.best), self.abs_floor)

        if hv_val > base * (1.0 + self.rel_delta):
            self.best = hv_val
            self.last_improve_gen = gen

        if gen - self.last_improve_gen >= self.patience:
            return 1.0

        return 0.0


def select_rank_fill_with_crowding(F_pop: np.ndarray, X_pop: np.ndarray, n_select: int):
    F_pop = np.asarray(F_pop, dtype=float)
    X_pop = np.asarray(X_pop, dtype=float)

    nds = NonDominatedSorting()
    fronts, rank = nds.do(F_pop, return_rank=True)

    selected = []
    for front in fronts:
        front = list(front)
        if len(selected) >= n_select:
            break

        remaining = n_select - len(selected)

        if len(front) <= remaining:
            selected.extend(front)
        else:
            cd = calc_crowding_distance(F_pop[front])
            order = np.argsort(-cd)  
            chosen = [front[i] for i in order[:remaining]]
            selected.extend(chosen)

    selected = np.asarray(selected, dtype=int)
    return X_pop[selected], F_pop[selected], rank[selected]


def run_one_image(image_path: Path, label_path: Path, out_root: Path, norm_params: dict | None, seed: int):
    np.random.seed(seed)
    random.seed(seed)

    orig_bgr = cv2.imread(str(image_path))
    if orig_bgr is None:
        print(f"[skip] Could not read image: {image_path}")
        return

    H, W, _ = orig_bgr.shape
    gt_boxes = load_gt_boxes_yolo(label_path, H, W)
    if len(gt_boxes) == 0:
        print(f"[skip] No GT boxes (empty label): {label_path.name}")
        return

    subset_dir = out_root / "subset"
    all_pareto_dir = out_root / "ALL_PARETO"
    subset_dir.mkdir(parents=True, exist_ok=True)
    all_pareto_dir.mkdir(parents=True, exist_ok=True)

    x1 = int(max(0, np.min(gt_boxes[:, 0]) - MARGIN))
    y1 = int(max(0, np.min(gt_boxes[:, 1]) - MARGIN))
    x2 = int(min(W - 1, np.max(gt_boxes[:, 2]) + MARGIN))
    y2 = int(min(H - 1, np.max(gt_boxes[:, 3]) + MARGIN))
    XMIN, XMAX, YMIN, YMAX = x1, x2, y1, y2

    base_res = run_inference_bgr(orig_bgr)
    base_conf, base_iou = evaluate_objectives(base_res, gt_boxes)

    print(f"\n=== {image_path.name} ===")
    print(f"GT objects: {len(gt_boxes)} | bounds x[{XMIN},{XMAX}] y[{YMIN},{YMAX}] | baseline conf={base_conf:.3f} iou={base_iou:.3f}")

    class PatchProblem(ElementwiseProblem):
        def __init__(self):
            xl = np.array([XMIN, YMIN, MIN_RADIUS, MIN_SIGMA_RATIO, -EPS, -EPS, -EPS], dtype=float)
            xu = np.array([XMAX, YMAX, MAX_RADIUS, MAX_SIGMA_RATIO,  EPS,  EPS,  EPS], dtype=float)
            super().__init__(n_var=7, n_obj=len(OBJECTIVE_NAMES), n_constr=0, xl=xl, xu=xu)

        def _evaluate(self, x, out, *args, **kwargs):
            pert_bgr, delta_map, patch_mask = apply_patch_bgr(orig_bgr, x, W, H)
            res = run_inference_bgr(pert_bgr)

            avg_conf_same, avg_iou_same = evaluate_objectives(res, gt_boxes)
            budget = realism_budget(delta_map, patch_mask)

            out["F"] = [avg_conf_same, avg_iou_same, budget]

    def save_candidate_minimal(idx: int, x):
        prefix = f"cand_{idx:04d}"
        pert_bgr, delta_map, patch_mask = apply_patch_bgr(orig_bgr, x, W, H)
        save_patch_visuals(orig_bgr, all_pareto_dir, prefix, pert_bgr, patch_mask)
        pert_res = run_inference_bgr(pert_bgr)
        write_metrics_file(all_pareto_dir, prefix, x, gt_boxes, base_res, pert_res, delta_map, patch_mask)

    def save_candidate_subset(idx: int, x):
        prefix = f"cand_{idx:02d}"
        pert_bgr, delta_map, patch_mask = apply_patch_bgr(orig_bgr, x, W, H)
        save_patch_visuals(orig_bgr, subset_dir, prefix, pert_bgr, patch_mask)
        amplified = np.clip(orig_bgr.astype(np.float32) + AMPLIFY * delta_map, 0, 255).astype(np.uint8)
        cv2.imwrite(str(subset_dir / f"{prefix}_amplified_x{AMPLIFY:.1f}.png"), amplified)
        pert_res = run_inference_bgr(pert_bgr)
        write_metrics_file(subset_dir, prefix, x, gt_boxes, base_res, pert_res, delta_map, patch_mask)

        pert_conf, pert_iou = evaluate_objectives(pert_res, gt_boxes)
        bud = realism_budget(delta_map, patch_mask)
        print(f"[saved subset] {prefix}: conf={pert_conf:.3f}, iou={pert_iou:.3f}, realism_budget={bud:.6f}")

    hv_indicator = None
    if norm_params is not None:
        hv_indicator = HV(ref_point=np.asarray(norm_params["ref"], dtype=float))

    conv_gens = []
    conv_hv = []
    conv_hv_best = []
    conv_min_raw = []
    conv_min_norm = []

    best_so_far = -np.inf

    def callback(algorithm):
        nonlocal best_so_far
        gen = int(getattr(algorithm, "n_gen", len(conv_gens) + 1))
        F = None
        if getattr(algorithm, "opt", None) is not None:
            try:
                F = algorithm.opt.get("F")
            except Exception:
                F = None
        if F is None:
            try:
                F = algorithm.pop.get("F")
            except Exception:
                F = None
        if F is None or len(F) == 0:
            return

        F = np.asarray(F, dtype=float)
        min_raw = np.min(F, axis=0)

        hv_val = np.nan
        min_norm = np.full((len(OBJECTIVE_NAMES),), np.nan, dtype=float)
        if norm_params is not None and hv_indicator is not None:
            Fn = normalize(F, norm_params["fmin"], norm_params["fmax"])
            min_norm = np.min(Fn, axis=0)
            hv_val = float(hv_indicator(Fn))

        if np.isfinite(hv_val):
            best_so_far = max(best_so_far, hv_val)

        conv_gens.append(gen)
        conv_hv.append(float(hv_val) if np.isfinite(hv_val) else np.nan)
        conv_hv_best.append(float(best_so_far) if np.isfinite(best_so_far) else np.nan)
        conv_min_raw.append(min_raw.tolist())
        conv_min_norm.append(min_norm.tolist())

    problem = PatchProblem()
    algorithm = NSGA2(pop_size=POP_SIZE)

    if EARLYSTOP_ENABLED:
        termination = HVRelativeStagnationTermination(
            max_gen=N_GEN,
            hv_indicator=hv_indicator,
            norm_params=norm_params,
            patience=EARLYSTOP_PATIENCE,
            rel_delta=EARLYSTOP_REL_DELTA,
            abs_floor=EARLYSTOP_ABS_FLOOR,
        )
    else:
        termination = HVRelativeStagnationTermination(
            max_gen=N_GEN,
            hv_indicator=None,
            norm_params=None,
            patience=10**9,
            rel_delta=1.0,
            abs_floor=EARLYSTOP_ABS_FLOOR,
        )

    res = minimize(problem, algorithm, termination, seed=int(seed), verbose=True, callback=callback)

    X_pareto = res.X
    F_pareto = res.F
    if X_pareto is None or len(X_pareto) == 0:
        print("[warn] No rank-0 solutions returned.")
        return

    save_hv_artifacts(all_pareto_dir, F_pareto, X_pareto, image_path, label_path, seed=seed)

    conv = {
        "image": image_path.name,
        "stem": image_path.stem,
        "method": "NSGA-II",
        "pop_size": int(POP_SIZE),
        "n_gen_requested": int(N_GEN),
        "seed": int(seed),
        "earlystop_enabled": bool(EARLYSTOP_ENABLED),
        "earlystop_patience": int(EARLYSTOP_PATIENCE),
        "earlystop_rel_delta": float(EARLYSTOP_REL_DELTA),
        "hv_norm_source": None if norm_params is None else norm_params["src"],
        "hv_ref_normalized": None if norm_params is None else np.asarray(norm_params["ref"], dtype=float).tolist(),
        "global_fmin": None if norm_params is None else np.asarray(norm_params["fmin"], dtype=float).tolist(),
        "global_fmax": None if norm_params is None else np.asarray(norm_params["fmax"], dtype=float).tolist(),
        "gens": conv_gens,
        "hv": conv_hv,
        "hv_best": conv_hv_best,
        "rank0_min_raw": conv_min_raw,
        "rank0_min_norm": conv_min_norm,
        "objectives": OBJECTIVE_NAMES,
    }
    (out_root / "convergence.json").write_text(json.dumps(conv, indent=2))

    csv_path = out_root / "convergence.csv"
    with open(csv_path, "w") as f:
        header = ["gen", "hv", "hv_best"]
        header += [f"min_{n}_raw" for n in OBJECTIVE_NAMES]
        header += [f"min_{n}_norm" for n in OBJECTIVE_NAMES]
        f.write(",".join(header) + "\n")

        for i in range(len(conv_gens)):
            mr = conv_min_raw[i]
            mn = conv_min_norm[i]
            row = [str(conv_gens[i]), str(conv_hv[i]), str(conv_hv_best[i])]
            row += [str(v) for v in mr]
            row += [str(v) for v in mn]
            f.write(",".join(row) + "\n")

    if norm_params is not None and len(conv_gens) > 0:
        plot_convergence(out_root, conv_gens, conv_hv, conv_hv_best, conv_min_norm, image_path.stem, norm_params, OBJECTIVE_NAMES)

    print(f"Saving ALL Pareto solutions ({len(X_pareto)}) -> {all_pareto_dir}")
    for i in range(len(X_pareto)):
        save_candidate_minimal(i + 1, X_pareto[i])


    all_saved_dir = out_root / "ALL_SAVED"
    all_saved_dir.mkdir(parents=True, exist_ok=True)

    try:
        F_pop = res.pop.get("F")
        X_pop = res.pop.get("X")
        if F_pop is None or X_pop is None or len(X_pop) == 0:
            raise RuntimeError("res.pop missing F/X or empty")

        X_saved, F_saved, rank_saved = select_rank_fill_with_crowding(F_pop, X_pop, n_select=POP_SIZE)

        np.save(all_saved_dir / "saved_F.npy", np.asarray(F_saved, dtype=np.float32))
        np.save(all_saved_dir / "saved_X.npy", np.asarray(X_saved, dtype=np.float32))

        saved_meta = {
            "image": image_path.name,
            "stem": image_path.stem,
            "method": "NSGA-II",
            "seed": int(seed),
            "saved_n": int(len(X_saved)),
            "target_n": int(POP_SIZE),
            "note": "Saved set built by rank-fill (rank-0, rank-1, ...) with crowding-distance tie-break to reach POP_SIZE.",
            "rank_counts": {str(int(r)): int(np.sum(rank_saved == r)) for r in np.unique(rank_saved)},
        }
        with open(all_saved_dir / "saved_meta.json", "w") as f:
            json.dump(saved_meta, f, indent=2)

        print(f"Saving FAIR saved set ({len(X_saved)}) -> {all_saved_dir}")
        for i in range(len(X_saved)):
            prefix = f"cand_{i+1:04d}"
            pert_bgr, delta_map, patch_mask = apply_patch_bgr(orig_bgr, X_saved[i], W, H)
            save_patch_visuals(orig_bgr, all_saved_dir, prefix, pert_bgr, patch_mask)
            pert_res = run_inference_bgr(pert_bgr)
            write_metrics_file(all_saved_dir, prefix, X_saved[i], gt_boxes, base_res, pert_res, delta_map, patch_mask)

    except Exception as e:
        print(f"[warn] Could not create fair saved set in {all_saved_dir}: {e}")


    idx_min_conf = int(np.argmin(F_pareto[:, 0]))
    idx_min_iou = int(np.argmin(F_pareto[:, 1]))
    idx_min_budget = int(np.argmin(F_pareto[:, 2]))

    chosen = [idx_min_conf, idx_min_iou, idx_min_budget]

    score = F_pareto[:, 0] + F_pareto[:, 1]
    for i in np.argsort(score)[:6]:
        chosen.append(int(i))

    seen = set()
    chosen_unique = []
    for i in chosen:
        if i not in seen:
            chosen_unique.append(i)
            seen.add(i)

    n_save = min(8, len(chosen_unique))
    print(f"Pareto solutions: {len(X_pareto)}. Saving {n_save} subset candidates -> {subset_dir}")
    for j in range(n_save):
        save_candidate_subset(j + 1, X_pareto[chosen_unique[j]])

    # Helpful console line
    if EARLYSTOP_ENABLED and len(conv_gens) > 0:
        print(f"[earlystop] stopped at gen={conv_gens[-1]} (max={N_GEN}), best HV={conv_hv_best[-1]}")
    print(f"[done] {image_path.stem} -> {out_root}")


def get_missing_run_indices(root_out: Path, n_runs: int):
    """
    Return the run indices that are missing entirely, e.g. [9, 10].
    This assumes incomplete runs were removed as folders (run-9, run-10, etc.).
    """
    missing = []
    for run_idx in range(1, n_runs + 1):
        run_root = root_out / f"run-{run_idx}"
        if not run_root.exists():
            missing.append(run_idx)
    return missing


def main():
    n_obj = len(OBJECTIVE_NAMES)
    norm_params = load_global_norm_params(HV_GLOBALREF_SUMMARY_JSON, n_obj=n_obj)

    if norm_params is None:
        print(f"[warn] Could not load compatible global normalization params: {HV_GLOBALREF_SUMMARY_JSON.resolve()}")
        print("[warn] Convergence HV + normalized per-objective plots will not be generated.")
        print("[warn] Early-stopping on HV will effectively be disabled (falls back to max generations).")
    else:
        print(f"[info] Using global normalization from: {norm_params['src']}")
        print(f"[info] Normalized HV ref point: {np.round(norm_params['ref'], 6).tolist()}")
        print(f"[info] Objectives: {OBJECTIVE_NAMES}")

    exts = (".jpg", ".jpeg", ".png", ".bmp")
    images = sorted([p for p in IMAGES_DIR.iterdir() if p.is_file() and p.suffix.lower() in exts])

    if not images:
        print(f"No images found in: {IMAGES_DIR}")
        return

    print(f"Found {len(images)} images in: {IMAGES_DIR}")
    print(f"Labels dir: {LABELS_DIR}")
    print(f"Output root: {ROOT_OUT}")
    print(f"NSGA-II: pop={POP_SIZE}, gen(max)={N_GEN}")
    print(f"Early-stop: enabled={EARLYSTOP_ENABLED}, patience={EARLYSTOP_PATIENCE}, rel_delta={EARLYSTOP_REL_DELTA}")
    print(f"Multi-run: N_RUNS={N_RUNS}, BASE_SEED={BASE_SEED} (run folders: run-1..run-{N_RUNS})")

    missing_runs = get_missing_run_indices(ROOT_OUT, N_RUNS)

    if not missing_runs:
        print("All requested run folders already exist. Nothing to do.")
        return

    print(f"Missing runs detected: {missing_runs}")

    # Run only the missing folders, e.g. run-9, run-10
    for run_idx in missing_runs:
        seed = int(BASE_SEED + (run_idx - 1))
        run_root = ROOT_OUT / f"run-{run_idx}"
        run_root.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 80)
        print(f"[RUN {run_idx}/{N_RUNS}] seed={seed} -> {run_root}")
        print("=" * 80)

        for img_path in images:
            label_path = LABELS_DIR / f"{img_path.stem}.txt"
            if not label_path.exists():
                print(f"[skip] No label for {img_path.name} -> expected {label_path.name}")
                continue

            out_root = run_root / img_path.stem
            out_root.mkdir(parents=True, exist_ok=True)

            try:
                run_one_image(img_path, label_path, out_root, norm_params, seed=seed)
            except Exception as e:
                print(f"[ERROR] run={run_idx} seed={seed} image={img_path.name}: {e}")

    print("\nAll missing runs completed.")


if __name__ == "__main__":
    main()