#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np

PROJECT_ROOT = Path.cwd()

# Examples of the directories (results) to be included
RESULT_ROOTS = [
    PROJECT_ROOT / "nsga-results-all-dv0a",
    PROJECT_ROOT / "random-results-all-dv0a",
]

SUBDIR = "ALL_PARETO"
F_NAME = "pareto_F.npy"

OBJECTIVE_NAMES = ["avg_conf_same", "avg_iou_same", "realism_budget"]

PAD = 0.05
REF_NORMALIZED = [1.0 + PAD] * len(OBJECTIVE_NAMES)

OUT_DIR = PROJECT_ROOT / "hv_out_globalref_all_experiments"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_JSON = OUT_DIR / "hv_globalref_summary.json"


def find_run_dirs(root: Path):
    return sorted([p for p in root.glob("run-*") if p.is_dir()])


def collect_all_F(result_roots):
    all_F = []

    included_roots = []
    missing_roots = []
    processed_files = []
    skipped_bad_files = []

    total_run_dirs = 0
    total_image_dirs = 0

    for root in result_roots:
        if not root.exists():
            missing_roots.append(str(root))
            continue

        included_roots.append(str(root.resolve()))
        run_dirs = find_run_dirs(root)
        total_run_dirs += len(run_dirs)

        for run_dir in run_dirs:
            image_dirs = sorted([p for p in run_dir.iterdir() if p.is_dir()])
            total_image_dirs += len(image_dirs)

            for image_dir in image_dirs:
                f_path = image_dir / SUBDIR / F_NAME
                if not f_path.exists():
                    continue

                try:
                    F = np.load(f_path)
                    F = np.asarray(F, dtype=float)

                    if F.ndim != 2 or F.shape[0] == 0 or F.shape[1] < len(OBJECTIVE_NAMES):
                        skipped_bad_files.append(str(f_path))
                        continue

                    F = F[:, :len(OBJECTIVE_NAMES)]
                    all_F.append(F)
                    processed_files.append(str(f_path))

                except Exception:
                    skipped_bad_files.append(str(f_path))

    return {
        "all_F": all_F,
        "included_roots": included_roots,
        "missing_roots": missing_roots,
        "processed_files": processed_files,
        "skipped_bad_files": skipped_bad_files,
        "total_run_dirs": total_run_dirs,
        "total_image_dirs": total_image_dirs,
    }


def main():
    res = collect_all_F(RESULT_ROOTS)

    all_F_list = res["all_F"]
    if not all_F_list:
        raise FileNotFoundError("No valid pareto_F.npy files were found in the provided roots.")

    all_F = np.vstack(all_F_list)

    global_fmin = np.min(all_F, axis=0)
    global_fmax = np.max(all_F, axis=0)

    summary = {
        "objective_names": OBJECTIVE_NAMES,
        "n_objectives": len(OBJECTIVE_NAMES),

        "pad": PAD,
        "ref_normalized": REF_NORMALIZED,

        "global_fmin_raw": global_fmin.tolist(),
        "global_fmax_raw": global_fmax.tolist(),

        "included_result_roots": res["included_roots"],
        "missing_result_roots": res["missing_roots"],

        "subdir_used": SUBDIR,
        "f_name_used": F_NAME,

        "n_result_roots_requested": len(RESULT_ROOTS),
        "n_result_roots_included": len(res["included_roots"]),
        "n_run_dirs_seen": res["total_run_dirs"],
        "n_image_dirs_seen": res["total_image_dirs"],
        "n_valid_pareto_files": len(res["processed_files"]),
        "n_skipped_bad_files": len(res["skipped_bad_files"]),

        "processed_files_example": res["processed_files"][:20],
        "skipped_bad_files_example": res["skipped_bad_files"][:20],

        "notes": [
            "global_fmin_raw/global_fmax_raw were computed from all available pareto_F.npy files listed above",
            "ref_normalized is fixed as [1+pad, 1+pad, 1+pad]",
            "nsga-results-dv2b was intentionally not included yet",
        ],
    }

    OUT_JSON.write_text(json.dumps(summary, indent=2))

    print("\n=== GLOBAL HV NORMALIZATION SUMMARY SAVED ===")
    print(f"Output JSON           : {OUT_JSON.resolve()}")
    print(f"Valid pareto files    : {len(res['processed_files'])}")
    print(f"Skipped bad files     : {len(res['skipped_bad_files'])}")
    print(f"Missing roots         : {len(res['missing_roots'])}")
    print(f"global_fmin_raw       : {global_fmin.tolist()}")
    print(f"global_fmax_raw       : {global_fmax.tolist()}")
    print(f"ref_normalized        : {REF_NORMALIZED}")


if __name__ == "__main__":
    main()