#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path.cwd()

MODEL_ORDER = ["origSCDM", "noscrewSCDM", "fixtureSCDM"]

CONFIG = {
    "origSCDM": {
        "dir": PROJECT_ROOT / "final_compiled_eval_training_v0_nsga",
        "csv_names": [
            "dv0a_all_perturbation_eval.csv",
            "dv0b_all_perturbation_eval.csv",
            "dv0c_all_perturbation_eval.csv",
            "dv0d_all_perturbation_eval.csv",
        ],
    },
    "noscrewSCDM": {
        "dir": PROJECT_ROOT / "final_compiled_eval_training_v1_nsga",
        "csv_names": [
            "dv1a_all_perturbation_eval.csv",
            "dv1b_all_perturbation_eval.csv",
            "dv1c_all_perturbation_eval.csv",
            "dv1d_all_perturbation_eval.csv",
        ],
    },
    "fixtureSCDM": {
        "dir": PROJECT_ROOT / "compiled_eval_training_v2_nsga",
        "csv_names": [
            "dv2a_all_perturbation_eval.csv",
            "dv2b_all_perturbation_eval.csv",
        ],
    },
}

OUT_DIR = PROJECT_ROOT / "rq3_confidence_stability"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_SUMMARY_CSV = OUT_DIR / "confidence_stability_fixed_threshold_summary_nsga.csv"
OUT_PER_SAMPLE_CSV = OUT_DIR / "confidence_stability_fixed_threshold_per_sample_nsga.csv"

ORIG_CONF_COL = "orig_avg_conf_same"
PERT_CONF_COL = "pert_avg_conf_same"
FAILURE_FLAG_COL = "any_failure"

OPTIONAL_ID_COLS = [
    "model",
    "dataset",
    "approach",
    "run",
    "image",
    "candidate",
]

MINOR_THR = 0.01
SMALL_CHANGE_THR = 0.05
MODERATE_CHANGE_THR = 0.10

VIOLATION_THR = MODERATE_CHANGE_THR

def validate_columns(df: pd.DataFrame, required_cols, context: str):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"[{context}] Missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def load_model_rows(model_name: str, folder: Path, csv_names):
    frames = []

    for csv_name in csv_names:
        csv_path = folder / csv_name
        if not csv_path.exists():
            raise FileNotFoundError(f"[{model_name}] Missing CSV: {csv_path}")

        df = pd.read_csv(csv_path)

        validate_columns(
            df,
            [ORIG_CONF_COL, PERT_CONF_COL, FAILURE_FLAG_COL],
            f"{model_name} | {csv_name}"
        )

        keep_cols = [ORIG_CONF_COL, PERT_CONF_COL, FAILURE_FLAG_COL]
        for col in OPTIONAL_ID_COLS:
            if col in df.columns and col not in keep_cols:
                keep_cols.append(col)

        df = df[keep_cols].copy()
        df["model_config"] = model_name
        df["source_csv"] = csv_name

        df[ORIG_CONF_COL] = pd.to_numeric(df[ORIG_CONF_COL], errors="coerce")
        df[PERT_CONF_COL] = pd.to_numeric(df[PERT_CONF_COL], errors="coerce")
        df[FAILURE_FLAG_COL] = pd.to_numeric(df[FAILURE_FLAG_COL], errors="coerce")

        df = df.dropna(subset=[ORIG_CONF_COL, PERT_CONF_COL, FAILURE_FLAG_COL]).copy()
        frames.append(df)

    if not frames:
        raise ValueError(f"[{model_name}] No CSV data loaded.")

    return pd.concat(frames, ignore_index=True)


def compute_non_failing_deviations(df: pd.DataFrame) -> pd.DataFrame:
    non_fail_df = df[df[FAILURE_FLAG_COL] == 0].copy()

    non_fail_df["conf_deviation_abs"] = (
        non_fail_df[ORIG_CONF_COL] - non_fail_df[PERT_CONF_COL]
    ).abs()

    non_fail_df["conf_deviation_signed"] = (
        non_fail_df[PERT_CONF_COL] - non_fail_df[ORIG_CONF_COL]
    )

    return non_fail_df


def categorize_deviation(d: float) -> str:
    if d <= MINOR_THR:
        return "minor"
    if d <= SMALL_CHANGE_THR:
        return "small_change"
    if d <= MODERATE_CHANGE_THR:
        return "moderate_change"
    return "large_change_violation"


def compute_threshold_summary(df_non_fail: pd.DataFrame):
    deviations = df_non_fail["conf_deviation_abs"].to_numpy(dtype=float)
    n = len(deviations)

    if n == 0:
        raise ValueError("No non-failing samples found.")

    minor_mask = deviations <= MINOR_THR
    small_change_mask = (deviations > MINOR_THR) & (deviations <= SMALL_CHANGE_THR)
    moderate_change_mask = (deviations > SMALL_CHANGE_THR) & (deviations <= MODERATE_CHANGE_THR)
    large_change_mask = deviations > MODERATE_CHANGE_THR

    summary = {
        "n_non_failing": n,
        "mean_abs_deviation": float(np.mean(deviations)),
        "median_abs_deviation": float(np.median(deviations)),
        "std_abs_deviation": float(np.std(deviations, ddof=0)),
        "min_abs_deviation": float(np.min(deviations)),
        "max_abs_deviation": float(np.max(deviations)),
        "minor_count": int(np.sum(minor_mask)),
        "minor_rate": float(np.mean(minor_mask)),
        "small_change_count": int(np.sum(small_change_mask)),
        "small_change_rate": float(np.mean(small_change_mask)),
        "moderate_change_count": int(np.sum(moderate_change_mask)),
        "moderate_change_rate": float(np.mean(moderate_change_mask)),
        "large_change_violation_count": int(np.sum(large_change_mask)),
        "large_change_violation_rate": float(np.mean(large_change_mask)),
    }

    return summary


def print_model_results(model_name: str, df_all: pd.DataFrame, df_non_fail: pd.DataFrame, summary: dict):
    total_samples = len(df_all)
    non_failing = len(df_non_fail)
    failing = total_samples - non_failing

    print(f"\n{'=' * 80}")
    print(f"Model: {model_name}")
    print(f"{'=' * 80}")
    print(f"Total samples                  : {total_samples}")
    print(f"Non-failing samples            : {non_failing}")
    print(f"Failing samples                : {failing}")
    print(f"Mean abs deviation             : {summary['mean_abs_deviation']:.6f}")
    print(f"Median abs deviation           : {summary['median_abs_deviation']:.6f}")
    print(f"Std abs deviation              : {summary['std_abs_deviation']:.6f}")
    print(f"Min abs deviation              : {summary['min_abs_deviation']:.6f}")
    print(f"Max abs deviation              : {summary['max_abs_deviation']:.6f}")

    print("\nFixed-threshold confidence stability categories:")
    print(
        f"  Minor         (<= {MINOR_THR:.2f}) : "
        f"{summary['minor_count']:>6} / {non_failing:<6} "
        f"({summary['minor_rate'] * 100:6.2f}%)"
    )
    print(
        f"  Small change   ({MINOR_THR:.2f}, {SMALL_CHANGE_THR:.2f}] : "
        f"{summary['small_change_count']:>6} / {non_failing:<6} "
        f"({summary['small_change_rate'] * 100:6.2f}%)"
    )
    print(
        f"  Moderate change({SMALL_CHANGE_THR:.2f}, {MODERATE_CHANGE_THR:.2f}] : "
        f"{summary['moderate_change_count']:>6} / {non_failing:<6} "
        f"({summary['moderate_change_rate'] * 100:6.2f}%)"
    )
    print(
        f"  Large change / violation (> {VIOLATION_THR:.2f}) : "
        f"{summary['large_change_violation_count']:>6} / {non_failing:<6} "
        f"({summary['large_change_violation_rate'] * 100:6.2f}%)"
    )


def main():
    all_summary_rows = []
    all_per_sample_frames = []

    for model_name in MODEL_ORDER:
        cfg = CONFIG[model_name]

        df_all = load_model_rows(
            model_name=model_name,
            folder=cfg["dir"],
            csv_names=cfg["csv_names"]
        )

        df_non_fail = compute_non_failing_deviations(df_all)

        summary = compute_threshold_summary(df_non_fail)

        df_export = df_non_fail.copy()
        df_export["stability_category"] = df_export["conf_deviation_abs"].apply(categorize_deviation)
        df_export["is_minor"] = (df_export["conf_deviation_abs"] <= MINOR_THR).astype(int)
        df_export["is_small_change"] = (
            (df_export["conf_deviation_abs"] > MINOR_THR) &
            (df_export["conf_deviation_abs"] <= SMALL_CHANGE_THR)
        ).astype(int)
        df_export["is_moderate_change"] = (
            (df_export["conf_deviation_abs"] > SMALL_CHANGE_THR) &
            (df_export["conf_deviation_abs"] <= MODERATE_CHANGE_THR)
        ).astype(int)
        df_export["is_large_change_violation"] = (
            df_export["conf_deviation_abs"] > VIOLATION_THR
        ).astype(int)

        all_per_sample_frames.append(df_export)

        # Summary
        all_summary_rows.append({
            "model_config": model_name,
            "total_samples": len(df_all),
            "n_non_failing": summary["n_non_failing"],
            "n_failing": len(df_all) - summary["n_non_failing"],
            "mean_abs_deviation": summary["mean_abs_deviation"],
            "median_abs_deviation": summary["median_abs_deviation"],
            "std_abs_deviation": summary["std_abs_deviation"],
            "min_abs_deviation": summary["min_abs_deviation"],
            "max_abs_deviation": summary["max_abs_deviation"],
            "minor_threshold_leq": MINOR_THR,
            "small_change_threshold_upper": SMALL_CHANGE_THR,
            "moderate_change_threshold_upper": MODERATE_CHANGE_THR,
            "violation_threshold_gt": VIOLATION_THR,
            "minor_count": summary["minor_count"],
            "minor_rate": summary["minor_rate"],
            "small_change_count": summary["small_change_count"],
            "small_change_rate": summary["small_change_rate"],
            "moderate_change_count": summary["moderate_change_count"],
            "moderate_change_rate": summary["moderate_change_rate"],
            "large_change_violation_count": summary["large_change_violation_count"],
            "large_change_violation_rate": summary["large_change_violation_rate"],
        })

        print_model_results(
            model_name=model_name,
            df_all=df_all,
            df_non_fail=df_non_fail,
            summary=summary
        )

    summary_df = pd.DataFrame(all_summary_rows)
    summary_df.to_csv(OUT_SUMMARY_CSV, index=False)

    per_sample_df = pd.concat(all_per_sample_frames, ignore_index=True)
    per_sample_df.to_csv(OUT_PER_SAMPLE_CSV, index=False)

    print(f"\nSaved summary CSV   : {OUT_SUMMARY_CSV}")
    print(f"Saved per-sample CSV: {OUT_PER_SAMPLE_CSV}")


if __name__ == "__main__":
    main()