#!/usr/bin/env python3
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

PROJECT_ROOT = Path.cwd()

CONFIG = {
    "PROBE": {
        "origSCDM": PROJECT_ROOT / "compiled_eval_test_v0_nsga",
        "noscrewSCDM": PROJECT_ROOT / "compiled_eval_test_v1_nsga",
        "fixtureSCDM": PROJECT_ROOT / "compiled_eval_test_v2_nsga",
    },
    "Random Search": {
        "origSCDM": PROJECT_ROOT / "compiled_eval_test_v0_random",
        "noscrewSCDM": PROJECT_ROOT / "compiled_eval_test_v1_random",
        "fixtureSCDM": PROJECT_ROOT / "compiled_eval_test_v2_random",
    }
}

SOURCE_DATASETS = {
    "origSCDM": "final_dv0_all_perturbation_eval.csv",
    "noscrewSCDM": "final_dv1_all_perturbation_eval.csv",
    "fixtureSCDM": "final_dv2_all_perturbation_eval.csv",
}

OUT_DIR = PROJECT_ROOT / "transferability_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PLOT = OUT_DIR / "transferability_failure_rate.png"

PROBE_BASE = "#5B7DB1"
RANDOM_BASE = "#E07A5F"
BG_COLOR = "white"
GRID_COLOR = "#F2F2F2"
TEXT_DARK = "#222222"

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.edgecolor": "#444444",
    "figure.facecolor": BG_COLOR,
    "axes.facecolor": BG_COLOR,
})


def make_cmap(light_hex, dark_hex):
    return LinearSegmentedColormap.from_list("custom_map", [light_hex, dark_hex])


PROBE_CMAP = make_cmap("#EEF3FA", "#0F3B7A")
RANDOM_CMAP = make_cmap("#FCEFEA", "#C95F3D")


def load_failure_rate(csv_path: Path) -> float:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")

    vals = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                vals.append(float(r["any_failure"]))
            except Exception:
                continue

    if not vals:
        raise ValueError(f"No valid any_failure values found in: {csv_path}")

    return float(np.mean(vals))


def build_matrix(method_dirs: dict, source_datasets: dict):
    target_names = list(method_dirs.keys())
    source_names = list(source_datasets.keys())

    matrix = np.zeros((len(target_names), len(source_names)), dtype=float)

    for i, target_model in enumerate(target_names):
        folder = method_dirs[target_model]

        for j, source_model in enumerate(source_names):
            csv_name = source_datasets[source_model]
            csv_path = folder / csv_name
            matrix[i, j] = load_failure_rate(csv_path)

    return target_names, source_names, matrix


def print_matrix(title: str, target_names, source_names, matrix):
    print(f"\n=== {title} ===")
    print("Rows = Target Model | Columns = Source Model")
    header = ["Target \\ Source"] + source_names
    print(" | ".join(header))
    for i, target in enumerate(target_names):
        vals = [f"{matrix[i, j]:.4f}" for j in range(len(source_names))]
        print(" | ".join([target] + vals))


def annotate_heatmap(ax, matrix, threshold=None):
    if threshold is None:
        threshold = matrix.max() / 2.0 if matrix.size > 0 else 0.5

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            color = "white" if val > threshold else TEXT_DARK
            ax.text(
                j, i, f"{val:.2f}",
                ha="center", va="center",
                color=color, fontsize=12, fontweight="medium"
            )


def plot_heatmap(ax, matrix, row_labels, col_labels, title, cmap, vmin, vmax):
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=18, ha="right")
    ax.set_yticklabels(row_labels)

    ax.set_xlabel("Source Model (Perturbation Dataset)")
    ax.set_title(title, fontsize=17, pad=12)

    ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color=GRID_COLOR, linestyle="-", linewidth=1.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    annotate_heatmap(ax, matrix)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    return im


def main():
    target_probe, source_probe, matrix_probe = build_matrix(
        CONFIG["PROBE"], SOURCE_DATASETS
    )
    target_rand, source_rand, matrix_rand = build_matrix(
        CONFIG["Random Search"], SOURCE_DATASETS
    )

    print_matrix("PROBE", target_probe, source_probe, matrix_probe)
    print_matrix("Random Search", target_rand, source_rand, matrix_rand)

    vmin = min(matrix_probe.min(), matrix_rand.min())
    vmax = max(matrix_probe.max(), matrix_rand.max())

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
    fig.patch.set_facecolor(BG_COLOR)

    im1 = plot_heatmap(
        axes[0], matrix_probe, target_probe, source_probe,
        title="PROBE", cmap=PROBE_CMAP, vmin=vmin, vmax=vmax
    )
    im2 = plot_heatmap(
        axes[1], matrix_rand, target_rand, source_rand,
        title="Random Search", cmap=RANDOM_CMAP, vmin=vmin, vmax=vmax
    )

    axes[0].set_ylabel("Target Model")

    cbar1 = fig.colorbar(im1, ax=axes[0], fraction=0.045, pad=0.03)
    cbar1.set_label("Failure Rate")

    cbar2 = fig.colorbar(im2, ax=axes[1], fraction=0.045, pad=0.03)
    cbar2.set_label("Failure Rate")

    plt.subplots_adjust(wspace=0.35, left=0.09, right=0.96, top=0.88, bottom=0.18)
    plt.savefig(OUT_PLOT, dpi=300, facecolor=BG_COLOR, bbox_inches="tight")
    plt.close()

    print(f"\nSaved transferability plot: {OUT_PLOT}")


if __name__ == "__main__":
    main()