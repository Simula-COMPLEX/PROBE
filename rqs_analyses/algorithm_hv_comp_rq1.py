#!/usr/bin/env python3
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

PROJECT_ROOT = Path.cwd()

CONFIG = {
    "origSCDM": {
        "nsga": PROJECT_ROOT / "hv_nsga_v0_per_image" / "nsga_v0_hv_per_image_per_run.csv",
        "rand": PROJECT_ROOT / "hv_random_v0_per_image" / "random_v0_hv_per_image_per_run.csv"
    },
    "noscrewSCDM": {
        "nsga": PROJECT_ROOT / "hv_nsga_v1_per_image" / "nsga_v1_hv_per_image_per_run.csv",
        "rand": PROJECT_ROOT / "hv_random_v1_per_image" / "random_v1_hv_per_image_per_run.csv"
    },
    "fixtureSCDM": {
        "nsga": PROJECT_ROOT / "hv_nsga_v2_per_image" / "nsga_v2_hv_per_image_per_run.csv",
        "rand": PROJECT_ROOT / "hv_random_v2_per_image" / "random_v2_hv_per_image_per_run.csv"
    }
}

OUT_DIR = PROJECT_ROOT / "hv_plots_combined"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PLOT = OUT_DIR / "hv_distribution_v0_v1_v2.png"


NSGA_COLOR = "#5B7DB1"   
RAND_COLOR = "#E07A5F"   

GRID_COLOR = "#DDDDDD"
BG_COLOR = "#FAFAFA"


plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "axes.edgecolor": "#444444",
})


def load_rows(csv_path: Path):
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")

    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                hv = float(r["hv"])
            except Exception:
                continue

            rows.append({
                "run": str(r["run"]).strip(),
                "image_stem": str(r["image_stem"]).strip(),
                "hv": hv
            })
    return rows


def merge(nsga_rows, rand_rows):
    nsga_map = {(r["run"], r["image_stem"]): r for r in nsga_rows}
    rand_map = {(r["run"], r["image_stem"]): r for r in rand_rows}

    common_keys = sorted(set(nsga_map.keys()).intersection(rand_map.keys()))
    nsga_only = sorted(set(nsga_map.keys()) - set(rand_map.keys()))
    rand_only = sorted(set(rand_map.keys()) - set(nsga_map.keys()))

    merged = [(nsga_map[k]["hv"], rand_map[k]["hv"]) for k in common_keys]
    return merged, nsga_only, rand_only


def effect_size_label(rbc_abs: float) -> str:
    if rbc_abs < 0.10:
        return "negligible"
    if rbc_abs < 0.30:
        return "small"
    if rbc_abs < 0.50:
        return "medium"
    return "large"


def effect_size_text(rbc):
    r = abs(rbc)

    if r < 0.10:
        label = "negligible"
    elif r < 0.30:
        label = "small"
    elif r < 0.50:
        label = "medium"
    else:
        label = "large"

    return rf"$|r_{{rb}}| = {r:.3f}$ ({label})"


def paired_rank_biserial_from_pairs(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    diff = x - y
    wins = int(np.sum(diff > 0))
    losses = int(np.sum(diff < 0))
    ties = int(np.sum(np.isclose(diff, 0.0)))

    denom = wins + losses
    rbc = 0.0 if denom == 0 else (wins - losses) / denom
    return rbc, wins, losses, ties


def print_stats(model_name, hv_nsga, hv_rand, n_nsga_total, n_rand_total, n_nsga_only, n_rand_only):
    hv_nsga = np.asarray(hv_nsga, dtype=float)
    hv_rand = np.asarray(hv_rand, dtype=float)

    print(f"\n=== {model_name} ===")
    print(f"NSGA total rows    : {n_nsga_total}")
    print(f"Random total rows  : {n_rand_total}")
    print(f"Matched samples    : {len(hv_nsga)}")
    print(f"NSGA-only skipped  : {n_nsga_only}")
    print(f"Random-only skipped: {n_rand_only}")
    print(f"Mean HV (PROBE)    : {hv_nsga.mean():.4f}")
    print(f"Mean HV (Random)   : {hv_rand.mean():.4f}")
    print(f"Mean Diff          : {(hv_nsga - hv_rand).mean():.4f}")

    rbc, wins_nsga, wins_rand, ties = paired_rank_biserial_from_pairs(hv_nsga, hv_rand)
    magnitude = effect_size_label(abs(rbc))

    print(f"PROBE wins         : {wins_nsga}")
    print(f"Random wins        : {wins_rand}")
    print(f"Ties               : {ties}")

    try:
        stat, p = wilcoxon(hv_nsga, hv_rand, alternative="greater")
        print(f"Wilcoxon p-value   : {p:.6f}")
    except Exception as e:
        p = np.nan
        print(f"Wilcoxon failed    : {e}")

    print(f"Effect size (r_rb) : {rbc:.4f} ({magnitude})")

    return p, rbc


def plot_subplot(ax, hv_nsga, hv_rand, title, p_value, rbc):
    data = [hv_nsga, hv_rand]

    ax.set_facecolor(BG_COLOR)

    # Violin
    parts = ax.violinplot(data, showextrema=False)
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(NSGA_COLOR if i == 0 else RAND_COLOR)
        body.set_alpha(0.35)

    # Boxplot
    box = ax.boxplot(data, widths=0.22, patch_artist=True, showfliers=False)
    for i, patch in enumerate(box["boxes"]):
        patch.set_facecolor(NSGA_COLOR if i == 0 else RAND_COLOR)
        patch.set_alpha(0.75)
        patch.set_edgecolor("#333333")

    # Median
    for median in box["medians"]:
        median.set_color("black")
        median.set_linewidth(2)

    # Mean
    means = [np.mean(hv_nsga), np.mean(hv_rand)]
    ax.scatter([1, 2], means, color="black", s=26, zorder=3)

    # Stats text
    p_text = "p < 0.001" if np.isfinite(p_value) and p_value < 0.001 else (
        f"p = {p_value:.3f}" if np.isfinite(p_value) else "p = n/a"
    )
    effect_txt = effect_size_text(rbc)

    ax.text(
        0.97, 0.97,
        f"{p_text}\n{effect_txt}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#444444",
        linespacing=1.2,
        bbox=dict(
            boxstyle="round,pad=0.26",
            facecolor="white",
            edgecolor="none",
            alpha=0.85
        )
    )

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["PROBE", "Random Search"], fontsize=12)
    ax.set_title(title, fontstyle="italic")

    ax.grid(axis="y", linestyle="--", alpha=0.35, color=GRID_COLOR)
    ax.tick_params(axis="y", labelsize=10)
    ax.set_box_aspect(1)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def main():
    n_models = len(CONFIG)

    fig, axes = plt.subplots(
        1, n_models,
        figsize=(8.4, 3.2),   
        sharey=True,
        gridspec_kw={"wspace": 0.12}
    )
    fig.patch.set_facecolor("white")

    if n_models == 1:
        axes = [axes]

    all_values = []

    for ax, (model_name, paths) in zip(axes, CONFIG.items()):
        nsga_rows = load_rows(paths["nsga"])
        rand_rows = load_rows(paths["rand"])

        merged, nsga_only, rand_only = merge(nsga_rows, rand_rows)

        if not merged:
            raise ValueError(f"No matched samples for {model_name}")

        hv_nsga = [m[0] for m in merged]
        hv_rand = [m[1] for m in merged]

        p_value, rbc = print_stats(
            model_name,
            hv_nsga,
            hv_rand,
            n_nsga_total=len(nsga_rows),
            n_rand_total=len(rand_rows),
            n_nsga_only=len(nsga_only),
            n_rand_only=len(rand_only),
        )

        if rand_only:
            print(f"Note: {model_name} currently uses only the overlap between NSGA and Random results.")

        all_values.extend(hv_nsga + hv_rand)
        plot_subplot(ax, hv_nsga, hv_rand, title=model_name, p_value=p_value, rbc=rbc)

    axes[0].set_ylabel("Hypervolume", fontsize=13)

    ymin, ymax = min(all_values), max(all_values)
    margin = 0.05 * (ymax - ymin) if ymax > ymin else 0.05

    for ax in axes:
        ax.set_ylim(ymin - margin, ymax + margin)

    plt.subplots_adjust(left=0.08, right=0.99, top=0.92, bottom=0.20, wspace=0.16)
    plt.savefig(OUT_PLOT, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close()

    print(f"\nSaved combined plot: {OUT_PLOT}")


if __name__ == "__main__":
    main()