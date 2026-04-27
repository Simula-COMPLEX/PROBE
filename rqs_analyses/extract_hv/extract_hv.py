#!/usr/bin/env python3
from pathlib import Path
import json
import csv
import numpy as np
from pymoo.indicators.hv import HV

PROJECT_ROOT = Path.cwd()

NSGA_RESULT_DIRS = [
    PROJECT_ROOT / "nsga-results-all-dv0a",
    PROJECT_ROOT / "nsga-results-all-dv0b",
    #PROJECT_ROOT / "random-results-all-dv0a",
    #PROJECT_ROOT / "random-results-all-dv0b",
]

GLOBALREF_JSON = PROJECT_ROOT / "hv_out_globalref_all_experiments" / "hv_globalref_summary.json"

SUBDIR = "ALL_PARETO"
F_NAME = "pareto_F.npy"

# NSGA or RANDOM
OUT_DIR = PROJECT_ROOT / "hv_nsga_v0_per_image"
# OUT_DIR = PROJECT_ROOT / "hv_random_v0_per_image"

OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "nsga_v0_hv_per_image_per_run.csv"
# OUT_CSV = OUT_DIR / "random_v0_hv_per_image_per_run.csv"

OBJECTIVE_NAMES = ["avg_conf_same", "avg_iou_same", "realism_budget"]


def load_globalref(json_path: Path, n_obj: int):
    if not json_path.exists():
        raise FileNotFoundError(f"Missing global reference JSON: {json_path}")

    with open(json_path, "r") as f:
        j = json.load(f)

    fmin = np.asarray(j["global_fmin_raw"], dtype=float)
    fmax = np.asarray(j["global_fmax_raw"], dtype=float)
    ref = np.asarray(j["ref_normalized"], dtype=float)

    if fmin.size < n_obj or fmax.size < n_obj or ref.size < n_obj:
        raise ValueError(
            f"Global reference JSON does not contain enough objective values for {n_obj} objectives."
        )

    return {
        "fmin": fmin[:n_obj],
        "fmax": fmax[:n_obj],
        "ref": ref[:n_obj],
        "pad": float(j.get("pad", 0.05)),
        "src": str(json_path.resolve()),
    }


def normalize(F: np.ndarray, fmin: np.ndarray, fmax: np.ndarray) -> np.ndarray:
    F = np.asarray(F, dtype=float)
    denom = fmax - fmin
    denom = np.where(np.abs(denom) < 1e-12, 1.0, denom)
    return (F - fmin) / denom


def find_run_dirs(results_root: Path):
    if not results_root.exists():
        return []
    return sorted([p for p in results_root.glob("run-*") if p.is_dir()])


def compute_hv(F_raw: np.ndarray, norm_params: dict) -> float:
    Fn = normalize(F_raw, norm_params["fmin"], norm_params["fmax"])
    hv_indicator = HV(ref_point=np.asarray(norm_params["ref"], dtype=float))
    return float(hv_indicator(Fn))


def main():
    norm_params = load_globalref(GLOBALREF_JSON, n_obj=len(OBJECTIVE_NAMES))

    rows = []
    skipped_missing_roots = 0
    skipped_missing_pareto = 0
    skipped_bad_files = 0

    for results_root in NSGA_RESULT_DIRS:
        if not results_root.exists():
            print(f"[skip] Missing root: {results_root}")
            skipped_missing_roots += 1
            continue

        dataset_name = results_root.name.replace("nsga-results-all-", "")
        run_dirs = find_run_dirs(results_root)

        print(f"\n[dataset] {dataset_name} | runs found: {[r.name for r in run_dirs]}")

        for run_dir in run_dirs:
            run_name = run_dir.name
            image_dirs = sorted([p for p in run_dir.iterdir() if p.is_dir()])

            print(f"  [run] {run_name} | image dirs: {len(image_dirs)}")

            for image_dir in image_dirs:
                pareto_dir = image_dir / SUBDIR
                pareto_f_path = pareto_dir / F_NAME
                meta_path = pareto_dir / "meta.json"

                if not pareto_f_path.exists():
                    skipped_missing_pareto += 1
                    continue

                try:
                    F = np.load(pareto_f_path)
                    F = np.asarray(F, dtype=float)

                    if F.ndim != 2 or F.shape[0] == 0 or F.shape[1] < len(OBJECTIVE_NAMES):
                        skipped_bad_files += 1
                        continue

                    F = F[:, :len(OBJECTIVE_NAMES)]
                    hv_value = compute_hv(F, norm_params)

                    meta = {}
                    if meta_path.exists():
                        try:
                            with open(meta_path, "r") as f:
                                meta = json.load(f)
                        except Exception:
                            meta = {}

                    rows.append({
                        "dataset": dataset_name,
                        "results_root": str(results_root),
                        "run": run_name,
                        "image_stem": image_dir.name,
                        "image_file": meta.get("image", ""),
                        "seed": meta.get("seed", ""),
                        "hv": hv_value,
                        "n_pareto_solutions": int(F.shape[0]),
                        "n_objectives": int(F.shape[1]),
                        "min_avg_conf_same": float(np.min(F[:, 0])),
                        "min_avg_iou_same": float(np.min(F[:, 1])),
                        "min_realism_budget": float(np.min(F[:, 2])),
                        "pareto_f_path": str(pareto_f_path),
                        "globalref_json": norm_params["src"],
                    })

                except Exception as e:
                    skipped_bad_files += 1
                    print(f"[warn] Failed to process {pareto_f_path}: {e}")

    if not rows:
        print("No rows collected. Nothing to save.")
        return

    fieldnames = [
        "dataset",
        "results_root",
        "run",
        "image_stem",
        "image_file",
        "seed",
        "hv",
        "n_pareto_solutions",
        "n_objectives",
        "min_avg_conf_same",
        "min_avg_iou_same",
        "min_realism_budget",
        "pareto_f_path",
        "globalref_json",
    ]

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Saved CSV              : {OUT_CSV.resolve()}")
    print(f"Rows saved             : {len(rows)}")
    print(f"Skipped missing roots  : {skipped_missing_roots}")
    print(f"Skipped missing pareto : {skipped_missing_pareto}")
    print(f"Skipped bad files      : {skipped_bad_files}")


if __name__ == "__main__":
    main()