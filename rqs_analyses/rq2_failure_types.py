#!/usr/bin/env python3
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

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

OUT_DIR = PROJECT_ROOT / "rq2_failure_discovery_overall_with_runs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PLOT = OUT_DIR / "rq2_failure_type_distribution_overall.png"

CANDIDATES_PER_IMAGE = 40
FIXED_YMAX = 41

FAILURE_TYPES = [
    ("Missed", "gt_missed_count"),
    ("Mislocalized", "gt_mislocalized_count"),
    ("Misclassified", "gt_misclassified_count"),
    ("Ambiguous", "gt_ambiguous_count"),
]

COLORS = {
    "Missed": "#7BAE7F",
    "Mislocalized": "#B08CC2",
    "Misclassified": "#5B7DB1",
    "Ambiguous": "#E07A5F",
}

GRID_COLOR = "#DDDDDD"
BG_COLOR = "#FAFAFA"
BAND_COLOR = "#F0F0F0"

plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "axes.edgecolor": "#444444",
})


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def safe_int(x, default=0):
    try:
        return int(float(x))
    except Exception:
        return default


def sort_run_key(run_name):
    s = str(run_name).strip()
    try:
        return (0, int(s))
    except Exception:
        digits = "".join(ch for ch in s if ch.isdigit())
        if digits:
            return (0, int(digits))
        return (1, s)


def run_label(run_name):
    digits = "".join(filter(str.isdigit, str(run_name)))
    return f"R{digits}" if digits else f"R{run_name}"


def load_rows(folder: Path, csv_names):
    rows = []

    for name in csv_names:
        path = folder / name
        if not path.exists():
            raise FileNotFoundError(f"Missing CSV: {path}")

        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)

            for r in reader:
                run = str(r.get("run", "")).strip()
                image = str(r.get("image", "")).strip()

                if not run or not image:
                    continue

                out = {
                    "run": run,
                    "image": image,
                    "any_failure": safe_int(r.get("any_failure", 0), 0),
                }

                for disp, count_col in FAILURE_TYPES:
                    out[disp] = safe_float(r.get(count_col, 0.0), 0.0)

                rows.append(out)

    return rows


def aggregate_per_image_run(rows):
    grouped = {}

    for r in rows:
        key = (r["run"], r["image"])
        grouped.setdefault(key, []).append(r)

    result = {}

    for key, vals in grouped.items():
        result[key] = {}

        for disp, _ in FAILURE_TYPES:
            result[key][disp] = float(np.sum([v[disp] for v in vals]))

        result[key]["num_failing_candidates"] = int(
            np.sum([v["any_failure"] for v in vals])
        )

    return result


def build_run_data(per_image_run_counts):
    out = {}

    for (run, image), counts in per_image_run_counts.items():
        if run not in out:
            out[run] = {
                "distributions": {disp: [] for disp, _ in FAILURE_TYPES},
                "total_failures_per_image": [],
                "num_images": 0,
                "total_failure_occurrences_in_run": 0.0,
                "total_candidates_in_run": 0.0,
                "num_failing_candidates_in_run": 0,
                "failure_rate": 0.0,
            }

        total_for_image = 0.0

        for disp, _ in FAILURE_TYPES:
            value = float(counts[disp])
            out[run]["distributions"][disp].append(value)
            total_for_image += value

        out[run]["total_failures_per_image"].append(total_for_image)
        out[run]["num_failing_candidates_in_run"] += int(
            counts["num_failing_candidates"]
        )

    for run in out:
        totals = np.asarray(out[run]["total_failures_per_image"], dtype=float)
        num_images = len(totals)
        total_failure_occurrences = float(np.sum(totals))
        total_candidates = float(num_images * CANDIDATES_PER_IMAGE)

        num_failing_candidates = out[run]["num_failing_candidates_in_run"]
        failure_rate = (
            num_failing_candidates / total_candidates
            if total_candidates > 0 else 0.0
        )

        out[run]["num_images"] = num_images
        out[run]["total_failure_occurrences_in_run"] = total_failure_occurrences
        out[run]["total_candidates_in_run"] = total_candidates
        out[run]["failure_rate"] = failure_rate

    return out


def build_overall_model_data(run_data):
    distributions = {disp: [] for disp, _ in FAILURE_TYPES}

    total_failure_occurrences = 0.0
    total_failing_candidates = 0
    total_candidates = 0.0
    num_images = 0

    for run in run_data:
        num_images += run_data[run]["num_images"]
        total_failure_occurrences += run_data[run]["total_failure_occurrences_in_run"]
        total_failing_candidates += run_data[run]["num_failing_candidates_in_run"]
        total_candidates += run_data[run]["total_candidates_in_run"]

        for disp, _ in FAILURE_TYPES:
            distributions[disp].extend(run_data[run]["distributions"][disp])

    failure_rate = (
        total_failing_candidates / total_candidates
        if total_candidates > 0 else 0.0
    )

    return {
        "distributions": distributions,
        "num_images": num_images,
        "total_failure_occurrences": total_failure_occurrences,
        "total_failing_candidates": total_failing_candidates,
        "total_candidates": total_candidates,
        "failure_rate": failure_rate,
    }


def print_overall_summary(model_name, model_data):
    print(f"\n=== Overall Summary: {model_name} ===")
    print(f"Images across all runs: {model_data['num_images']}")
    print(f"Total failure occurrences: {model_data['total_failure_occurrences']:.2f}")
    print(f"Total failing candidates: {model_data['total_failing_candidates']}")
    print(f"Total candidates: {model_data['total_candidates']:.2f}")
    print(
        f"Failure rate: {model_data['failure_rate']:.4f} "
        f"({model_data['failure_rate'] * 100:.2f}%)"
    )

    for disp, _ in FAILURE_TYPES:
        vals = np.asarray(model_data["distributions"][disp], dtype=float)

        print(
            f"{disp:<14} | "
            f"mean={vals.mean():8.2f} | "
            f"median={np.median(vals):8.2f} | "
            f"std={vals.std(ddof=0):8.2f} | "
            f"sum={vals.sum():8.2f}"
        )


def get_dominant_failure_type(model_data):
    sums = {}
    for disp, _ in FAILURE_TYPES:
        vals = np.asarray(model_data["distributions"][disp], dtype=float)
        sums[disp] = vals.sum()

    return max(sums, key=sums.get)


def get_bottom_global_ymax(all_run_data):
    max_y = 0.0

    for model_name in MODEL_ORDER:
        run_data = all_run_data[model_name]

        for run in run_data:
            for disp, _ in FAILURE_TYPES:
                vals = np.asarray(run_data[run]["distributions"][disp], dtype=float)
                if len(vals) > 0:
                    max_y = max(max_y, float(vals.mean()))

    return max_y * 1.25 if max_y > 0 else 1.0


def add_soft_background_bands(ax, ymax):
    step = 2.5 if ymax <= 10 else 5.0

    y = 0.0
    band_index = 0

    while y < ymax:
        if band_index % 2 == 0:
            ax.axhspan(
                y,
                min(y + step, ymax),
                facecolor=BAND_COLOR,
                alpha=0.22,
                zorder=0
            )
        y += step
        band_index += 1


def plot_model_subplot(ax, model_name, model_data):
    labels = [disp for disp, _ in FAILURE_TYPES]
    distributions = model_data["distributions"]
    data = [np.asarray(distributions[disp], dtype=float) for disp in labels]

    ax.set_facecolor(BG_COLOR)
    ax.set_axisbelow(True)

    parts = ax.violinplot(
        data,
        showmeans=False,
        showmedians=False,
        showextrema=False
    )

    for body, label in zip(parts["bodies"], labels):
        body.set_facecolor(COLORS[label])
        body.set_alpha(0.25)

    box = ax.boxplot(
        data,
        widths=0.22,
        patch_artist=True,
        showfliers=False
    )

    for patch, label in zip(box["boxes"], labels):
        patch.set_facecolor(COLORS[label])
        patch.set_alpha(0.90)
        patch.set_edgecolor("#333333")
        patch.set_linewidth(1.2)

    for median in box["medians"]:
        median.set_color("black")
        median.set_linewidth(2.4)

    means = [
        np.mean(np.asarray(distributions[label], dtype=float))
        for label in labels
    ]

    ax.scatter(
        range(1, len(labels) + 1),
        means,
        color="black",
        s=30,
        zorder=3
    )

    ax.text(
        0.03,
        0.97,
        f"Failure rate: {model_data['failure_rate'] * 100:.2f}%",
        transform=ax.transAxes,
        fontsize=11,
        fontstyle="italic",
        color="#333333",
        ha="left",
        va="top",
        bbox=dict(
            boxstyle="round,pad=0.24",
            facecolor="white",
            alpha=0.85,
            edgecolor="none"
        )
    )

    ax.set_title(model_name, fontstyle="italic")

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=15, ha="right")

    ax.set_ylim(0, FIXED_YMAX)
    ax.grid(axis="y", linestyle="--", alpha=0.35, color=GRID_COLOR)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def plot_failure_types_per_run_subplot(ax, run_data, model_data, bottom_ymax):
    run_names = sorted(run_data.keys(), key=sort_run_key)
    x = np.arange(1, len(run_names) + 1)

    dominant_type = get_dominant_failure_type(model_data)

    ax.set_facecolor(BG_COLOR)
    ax.set_axisbelow(True)

    add_soft_background_bands(ax, bottom_ymax)

    for disp, _ in FAILURE_TYPES:
        means_per_run = []

        for run in run_names:
            vals = np.asarray(run_data[run]["distributions"][disp], dtype=float)
            mean_val = vals.mean() if len(vals) > 0 else 0.0
            means_per_run.append(mean_val)

        means_per_run = np.asarray(means_per_run, dtype=float)

        line_alpha = 1.0 if disp == dominant_type else 0.72
        line_width = 2.9 if disp == dominant_type else 1.9
        marker_size = 5.5 if disp == dominant_type else 4.4

        ax.plot(
            x,
            means_per_run,
            marker="o",
            linewidth=line_width,
            markersize=marker_size,
            alpha=line_alpha,
            color=COLORS[disp],
            markeredgecolor="white",
            markeredgewidth=0.7,
            label=disp,
            zorder=3
        )

        overall_mean = np.asarray(model_data["distributions"][disp], dtype=float).mean()

        ax.axhline(
            overall_mean,
            linestyle="--",
            linewidth=0.9,
            color=COLORS[disp],
            alpha=0.30,
            zorder=1
        )

    ax.text(
        0.98,
        0.94,
        f"Dominant: {dominant_type}",
        transform=ax.transAxes,
        fontsize=9.5,
        fontstyle="italic",
        color="#333333",
        ha="right",
        va="top",
        bbox=dict(
            boxstyle="round,pad=0.22",
            facecolor="white",
            alpha=0.78,
            edgecolor="none"
        )
    )

    ax.set_title("Mean failures per image across runs", fontsize=11)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [run_label(r) for r in run_names],
        rotation=35,
        ha="right"
    )

    ax.set_ylim(0, bottom_ymax)
    ax.grid(axis="y", linestyle="--", alpha=0.35, color=GRID_COLOR)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def main():
    all_model_data = {}
    all_run_data = {}

    for model_name in MODEL_ORDER:
        cfg = CONFIG[model_name]

        rows = load_rows(cfg["dir"], cfg["csv_names"])
        per_image_run_counts = aggregate_per_image_run(rows)

        run_data = build_run_data(per_image_run_counts)
        model_data = build_overall_model_data(run_data)

        all_run_data[model_name] = run_data
        all_model_data[model_name] = model_data

        print(f"\n{'=' * 60}")
        print(f"Selected model: {model_name}")
        print(f"Candidates per image: {CANDIDATES_PER_IMAGE}")
        print(f"Fixed y-axis upper bound: {FIXED_YMAX}")
        print_overall_summary(model_name, model_data)

    bottom_ymax = get_bottom_global_ymax(all_run_data)

    fig, axes = plt.subplots(
        2,
        len(MODEL_ORDER),
        figsize=(15.5, 8.8),
        sharey="row",
        gridspec_kw={
            "height_ratios": [2.45, 1.05],
            "hspace": 0.48,
            "wspace": 0.16,
        }
    )

    for col, model_name in enumerate(MODEL_ORDER):
        plot_model_subplot(
            axes[0, col],
            model_name,
            all_model_data[model_name]
        )

        plot_failure_types_per_run_subplot(
            axes[1, col],
            all_run_data[model_name],
            all_model_data[model_name],
            bottom_ymax
        )

        if col > 0:
            axes[0, col].tick_params(labelleft=False)
            axes[1, col].tick_params(labelleft=False)

    handles, labels = axes[1, 0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.03)
    )

    plt.subplots_adjust(
        left=0.068,
        right=0.985,
        top=0.94,
        bottom=0.13,
        hspace=0.48,
        wspace=0.16
    )

    fig.canvas.draw()

    top_bbox = axes[0, 0].get_position()
    bottom_bbox = axes[1, 0].get_position()

    top_center = (top_bbox.y0 + top_bbox.y1) / 2.0
    bottom_center = (bottom_bbox.y0 + bottom_bbox.y1) / 2.0

    label_x = top_bbox.x0 - 0.045

    fig.text(
        label_x,
        top_center,
        "Number of Failures per Image",
        fontsize=14,
        rotation=90,
        va="center",
        ha="center"
    )

    fig.text(
        label_x,
        bottom_center,
        "(Mean) Failures per Image",
        fontsize=14,
        rotation=90,
        va="center",
        ha="center"
    )

    plt.savefig(OUT_PLOT, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"\nSaved: {OUT_PLOT}")


if __name__ == "__main__":
    main()