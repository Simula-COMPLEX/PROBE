#!/usr/bin/env python3
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, wilcoxon

PROJECT_ROOT = Path.cwd()

CONFIG = {
    "origSCDM": {
        "nsga_dir": PROJECT_ROOT / "final_compiled_eval_training_v0_nsga",
        "rand_dir": PROJECT_ROOT / "final_compiled_eval_training_v0_random",
        "csv_names": [
            "dv0a_all_perturbation_eval.csv",
            "dv0b_all_perturbation_eval.csv",
            "dv0c_all_perturbation_eval.csv",
            "dv0d_all_perturbation_eval.csv",
        ],
    },
    "noscrewSCDM": {
        "nsga_dir": PROJECT_ROOT / "final_compiled_eval_training_v1_nsga",
        "rand_dir": PROJECT_ROOT / "final_compiled_eval_training_v1_random",
        "csv_names": [
            "dv1a_all_perturbation_eval.csv",
            "dv1b_all_perturbation_eval.csv",
            "dv1c_all_perturbation_eval.csv",
            "dv1d_all_perturbation_eval.csv",
        ],
    },
    "fixtureSCDM": {
        "nsga_dir": PROJECT_ROOT / "compiled_eval_training_v2_nsga",
        "rand_dir": PROJECT_ROOT / "compiled_eval_training_v2_random",
        "csv_names": [
            "dv2a_all_perturbation_eval.csv",
            "dv2b_all_perturbation_eval.csv",
        ],
    },
}

OUT_DIR = PROJECT_ROOT / "rq1_failure_rate_and_magnitude_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PLOT = OUT_DIR / "rq1_failure_rate_and_magnitude.png"

NSGA_COLOR = "#5B7DB1"
RAND_COLOR = "#E07A5F"
GRID_COLOR = "#DDDDDD"

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.edgecolor": "#444444",
})

def load_rows_from_csv(csv_path: Path):
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")

    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                any_failure = int(float(r["any_failure"]))
                realism_budget = float(r["realism_budget"])
            except Exception:
                continue

            rows.append({
                "run": str(r["run"]).strip(),
                "image": str(r["image"]).strip(),
                "candidate": str(r["candidate"]).strip(),
                "any_failure": any_failure,
                "realism_budget": realism_budget,
            })
    return rows


def load_rows_from_folder(folder: Path, csv_names):
    all_rows = []
    for name in csv_names:
        all_rows.extend(load_rows_from_csv(folder / name))
    return all_rows


def compute_failure_rate_per_image_run(rows):
    grouped = {}
    for r in rows:
        key = (r["run"], r["image"])
        grouped.setdefault(key, []).append(r["any_failure"])

    return {k: float(np.mean(v)) for k, v in grouped.items()}


def pair_failure_rates(rate_dict_nsga, rate_dict_rand):
    common_keys = sorted(set(rate_dict_nsga.keys()).intersection(rate_dict_rand.keys()))
    nsga_only = sorted(set(rate_dict_nsga.keys()) - set(rate_dict_rand.keys()))
    rand_only = sorted(set(rate_dict_rand.keys()) - set(rate_dict_nsga.keys()))

    vals_nsga = [rate_dict_nsga[k] for k in common_keys]
    vals_rand = [rate_dict_rand[k] for k in common_keys]

    return vals_nsga, vals_rand, common_keys, nsga_only, rand_only


def extract_failure_magnitudes(rows):
    return [r["realism_budget"] for r in rows if r["any_failure"] == 1]


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


def rrb_label(rbc_abs):
    if rbc_abs < 0.10:
        return "negligible"
    if rbc_abs < 0.30:
        return "small"
    if rbc_abs < 0.50:
        return "medium"
    return "large"


def rrb_text(rbc):
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


def compute_a12(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    n_x = len(x)
    n_y = len(y)
    if n_x == 0 or n_y == 0:
        return np.nan

    greater = 0
    equal = 0
    for xi in x:
        greater += np.sum(xi > y)
        equal += np.sum(np.isclose(xi, y))

    return (greater + 0.5 * equal) / (n_x * n_y)


def a12_label(a12):
    if not np.isfinite(a12):
        return "n/a"
    if 0.44 < a12 < 0.56:
        return "Negligible"
    elif (0.34 < a12 <= 0.44) or (0.56 <= a12 < 0.64):
        return "Small"
    elif (0.29 < a12 <= 0.34) or (0.64 <= a12 < 0.71):
        return "Medium"
    else:
        return "Large"


def a12_text(a12, direction="greater"):
    if not np.isfinite(a12):
        return r"$\hat{A}_{12}$ = n/a"

    shown = 1.0 - a12 if direction == "less" else a12

    if 0.44 < shown < 0.56:
        label = "Negligible"
    elif (0.34 < shown <= 0.44) or (0.56 <= shown < 0.64):
        label = "Small"
    elif (0.29 < shown <= 0.34) or (0.64 <= shown < 0.71):
        label = "Medium"
    else:
        label = "Large"

    return rf"$\hat{{A}}_{{12}} = {shown:.3f}$ ({label})"

def print_stats(model_name, rates_nsga, rates_rand, mags_nsga, mags_rand,
                n_rate_pairs, n_rate_nsga_only, n_rate_rand_only):
    rates_nsga = np.asarray(rates_nsga, dtype=float)
    rates_rand = np.asarray(rates_rand, dtype=float)
    mags_nsga = np.asarray(mags_nsga, dtype=float)
    mags_rand = np.asarray(mags_rand, dtype=float)

    print(f"\n=== {model_name} ===")

    print("\n[Failure rate | paired per image/run]")
    print(f"Matched image/run pairs         : {n_rate_pairs}")
    print(f"PROBE-only pairs skipped        : {n_rate_nsga_only}")
    print(f"Random-only pairs skipped       : {n_rate_rand_only}")
    print(f"Mean failure rate (PROBE)       : {rates_nsga.mean():.4f}")
    print(f"Mean failure rate (Random)      : {rates_rand.mean():.4f}")
    print(f"Median failure rate (PROBE)     : {np.median(rates_nsga):.4f}")
    print(f"Median failure rate (Random)    : {np.median(rates_rand):.4f}")

    rrb_fr, wins_fr, losses_fr, ties_fr = paired_rank_biserial_from_pairs(rates_nsga, rates_rand)
    print(f"PROBE wins                      : {wins_fr}")
    print(f"Random wins                     : {losses_fr}")
    print(f"Ties                            : {ties_fr}")

    try:
        _, p_fr = wilcoxon(rates_nsga, rates_rand, alternative="greater")
        print(f"Wilcoxon p-value                : {p_fr:.6f}")
    except Exception as e:
        p_fr = np.nan
        print(f"Wilcoxon failed                 : {e}")

    print(f"Effect size (r_rb)              : {rrb_fr:.4f} ({rrb_label(abs(rrb_fr))})")

    print("\n[Perturbation magnitude | failures only | unpaired]")
    print(f"Failure samples (PROBE)         : {len(mags_nsga)}")
    print(f"Failure samples (Random)        : {len(mags_rand)}")
    print(f"Mean magnitude (PROBE)          : {mags_nsga.mean():.6f}")
    print(f"Mean magnitude (Random)         : {mags_rand.mean():.6f}")
    print(f"Median magnitude (PROBE)        : {np.median(mags_nsga):.6f}")
    print(f"Median magnitude (Random)       : {np.median(mags_rand):.6f}")

    try:
        _, p_mag = mannwhitneyu(mags_nsga, mags_rand, alternative="less")
        print(f"Mann–Whitney U p-value          : {p_mag:.6f}")
    except Exception as e:
        p_mag = np.nan
        print(f"Mann–Whitney U failed           : {e}")

    a12_mag = compute_a12(mags_nsga, mags_rand)
    shown_mag = 1.0 - a12_mag if np.isfinite(a12_mag) else np.nan
    print(f"A12 (Perturbation Magnitude)    : {shown_mag:.4f} ({a12_label(shown_mag) if np.isfinite(shown_mag) else 'n/a'})")

    return p_fr, rrb_fr, p_mag, a12_mag


def p_text(p):
    if not np.isfinite(p):
        return "p = n/a"
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def plot_violin_box(ax, vals_nsga, vals_rand, title, ylabel=None,
                    pval=None, effect_text=None):
    data = [vals_nsga, vals_rand]

    parts = ax.violinplot(data, showextrema=False)
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(NSGA_COLOR if i == 0 else RAND_COLOR)
        body.set_alpha(0.35)

    box = ax.boxplot(data, widths=0.2, patch_artist=True, showfliers=False)
    for i, patch in enumerate(box["boxes"]):
        patch.set_facecolor(NSGA_COLOR if i == 0 else RAND_COLOR)
        patch.set_alpha(0.7)

    for median in box["medians"]:
        median.set_color("black")
        median.set_linewidth(2)

    means = [np.mean(vals_nsga), np.mean(vals_rand)]
    ax.scatter([1, 2], means, color="black", s=40, zorder=3)

    ax.text(
        0.5, 0.95,
        f"{p_text(pval)}\n{effect_text}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.5,
        color="#444444",
        linespacing=1.25,
        bbox=dict(
            boxstyle="round,pad=0.28",
            facecolor="white",
            edgecolor="#D6D6D6",
            linewidth=0.8,
            alpha=0.88
        )
    )

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["PROBE", "Random Search"])
    ax.set_title(title, fontstyle="italic")

    if ylabel:
        ax.set_ylabel(ylabel)

    ax.grid(axis="y", linestyle="--", alpha=0.4, color=GRID_COLOR)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def main():
    n_models = len(CONFIG)
    fig, axes = plt.subplots(2, n_models, figsize=(5 * n_models, 7))
    fig.patch.set_facecolor("white")

    all_rates, all_mags = [], []

    for col, (model_name, cfg) in enumerate(CONFIG.items()):
        nsga_rows = load_rows_from_folder(cfg["nsga_dir"], cfg["csv_names"])
        rand_rows = load_rows_from_folder(cfg["rand_dir"], cfg["csv_names"])

        rate_dict_nsga = compute_failure_rate_per_image_run(nsga_rows)
        rate_dict_rand = compute_failure_rate_per_image_run(rand_rows)
        rates_nsga, rates_rand, _, nsga_only_pairs, rand_only_pairs = pair_failure_rates(
            rate_dict_nsga, rate_dict_rand
        )

        mags_nsga = extract_failure_magnitudes(nsga_rows)
        mags_rand = extract_failure_magnitudes(rand_rows)

        if not rates_nsga or not rates_rand:
            raise ValueError(f"Missing paired failure-rate data for {model_name}")
        if not mags_nsga or not mags_rand:
            raise ValueError(f"Missing failure-magnitude data for {model_name}")

        p_fr, rrb_fr, p_mag, a12_mag = print_stats(
            model_name,
            rates_nsga,
            rates_rand,
            mags_nsga,
            mags_rand,
            n_rate_pairs=len(rates_nsga),
            n_rate_nsga_only=len(nsga_only_pairs),
            n_rate_rand_only=len(rand_only_pairs),
        )

        all_rates.extend(rates_nsga + rates_rand)
        all_mags.extend(mags_nsga + mags_rand)

        plot_violin_box(
            axes[0, col],
            rates_nsga,
            rates_rand,
            model_name,
            ylabel="Failure Rate" if col == 0 else None,
            pval=p_fr,
            effect_text=rrb_text(rrb_fr),
        )

        plot_violin_box(
            axes[1, col],
            mags_nsga,
            mags_rand,
            #model_name,
            title="",
            ylabel="Perturbation Magnitude" if col == 0 else None,
            pval=p_mag,
            effect_text=a12_text(a12_mag, direction="less"),
        )

    rate_min, rate_max = min(all_rates), max(all_rates)
    rate_margin = 0.05 * (rate_max - rate_min if rate_max > rate_min else 1.0)

    mag_min, mag_max = min(all_mags), max(all_mags)
    mag_margin = 0.05 * (mag_max - mag_min if mag_max > mag_min else 1.0)

    for ax in axes[0]:
        ax.set_ylim(rate_min - rate_margin, rate_max + rate_margin)
    for ax in axes[1]:
        ax.set_ylim(mag_min - mag_margin, mag_max + mag_margin)

    # for ax in axes[0]:
    #     ax.set_xticklabels([])
    #     ax.set_xlabel("")

    # for ax in axes[0]:
    #     ax.set_xticks([])

    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=300, facecolor="white")
    plt.close()

    print(f"\nSaved plot: {OUT_PLOT}")


if __name__ == "__main__":
    main()