"""Cross-blur-type comparison figures for the paper.

Loads all three completed blur runs (motion, glass, defocus) and produces
figures that compare *across* blur types — the actual core research question
(does robustness/ranking hold across corruption types?) — rather than the
per-blur-type figures scripts/generate_figures.py already makes.

Usage: python -m scripts.generate_overall_figures
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import CIFAR10_LABELS, MODEL_IDS, SEVERITIES

FIGURES_DIR = "results/overall"
BLUR_NAMES = ["motion", "glass", "defocus"]
BLUR_TYPE_TO_NAME = {f"{b}_blur": b for b in BLUR_NAMES}

BLUR_LABELS = {"motion": "Motion", "glass": "Glass", "defocus": "Defocus"}
# From the project's validated 3-series categorical palette (all-pairs safe).
BLUR_COLORS = {"motion": "#2a78d6", "glass": "#eb6834", "defocus": "#1baf7a"}

MODEL_ORDER = list(MODEL_IDS.keys())
MODEL_LABELS = {
    "claude_opus": "Claude Opus 5",
    "claude_sonnet": "Claude Sonnet 5",
    "claude_haiku": "Claude Haiku 4.5",
    "gpt_sol": "GPT-5.6 Sol",
    "gpt_terra": "GPT-5.6 Terra",
    "gpt_luna": "GPT-5.6 Luna",
    "gemini": "Gemini 3.1 Flash Lite",
    "grok": "Grok 4.3",
    "kimi": "Kimi K3",
}
MODEL_COLORS = {m: plt.cm.tab10(i) for i, m in enumerate(MODEL_ORDER)}

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
    }
)


def load_all():
    frames = []
    for blur in BLUR_NAMES:
        df = pd.read_csv(f"results/{blur}/results_{blur}.csv")
        df["cost_usd"] = pd.to_numeric(df["cost_usd"], errors="coerce")
        df["blur_name"] = blur
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def chart_overall_accuracy_by_blur(df, out_dir):
    """The headline number: which blur type hurts these models most overall?"""
    acc = df.groupby("blur_name")["correct"].mean().reindex(BLUR_NAMES) * 100
    fig, ax = plt.subplots(figsize=(7, 5.5))
    bars = ax.bar(
        [BLUR_LABELS[b] for b in BLUR_NAMES], acc.values, color=[BLUR_COLORS[b] for b in BLUR_NAMES]
    )
    ax.set_ylabel("Overall accuracy (%)")
    ax.set_title("Overall Accuracy by Blur Type\n(all 9 models, all severities)")
    ax.set_ylim(0, 100)
    for bar, v in zip(bars, acc.values):
        ax.annotate(f"{v:.1f}%", (bar.get_x() + bar.get_width() / 2, v), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "01_overall_accuracy_by_blur_type.png"), bbox_inches="tight")
    plt.close(fig)


def chart_accuracy_by_blur_and_severity(df, out_dir):
    """Averaged across all models: how does each blur type degrade with severity?"""
    acc = df.groupby(["blur_name", "severity"])["correct"].mean().unstack("blur_name") * 100
    fig, ax = plt.subplots(figsize=(8, 6))
    for blur in BLUR_NAMES:
        ax.plot(
            acc.index, acc[blur], marker="o", linewidth=2.5, markersize=7,
            label=BLUR_LABELS[blur], color=BLUR_COLORS[blur],
        )
    ax.set_xlabel("Severity")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(SEVERITIES)
    ax.set_ylim(0, 100)
    ax.set_title("Average Accuracy vs. Severity, by Blur Type\n(averaged across all 9 models)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "02_accuracy_by_blur_and_severity.png"), bbox_inches="tight")
    plt.close(fig)


def chart_grid_model_by_blur(df, out_dir):
    """One panel per model, 3 lines (blur types) each — the full picture."""
    acc = df.groupby(["model", "blur_name", "severity"])["correct"].mean()
    fig, axes = plt.subplots(3, 3, figsize=(13, 11), sharex=True, sharey=True)
    for ax, model in zip(axes.flat, MODEL_ORDER):
        for blur in BLUR_NAMES:
            row = acc.loc[model, blur]
            ax.plot(
                row.index, row.values * 100, marker="o", markersize=4, linewidth=2,
                label=BLUR_LABELS[blur], color=BLUR_COLORS[blur],
            )
        ax.set_title(MODEL_LABELS[model], fontsize=11)
        ax.set_xticks(SEVERITIES)
        ax.set_ylim(0, 100)
    for ax in axes[-1, :]:
        ax.set_xlabel("Severity")
    for ax in axes[:, 0]:
        ax.set_ylabel("Accuracy (%)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=3, frameon=False)
    fig.suptitle("Accuracy vs. Severity, by Model and Blur Type", fontsize=14, y=1.08)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "03_accuracy_grid_by_model_and_blur.png"), bbox_inches="tight")
    plt.close(fig)


def chart_model_blur_heatmap(df, out_dir):
    """Compact at-a-glance grid: every model x blur-type accuracy."""
    acc = (df.groupby(["model", "blur_name"])["correct"].mean() * 100).unstack("blur_name")
    acc = acc.reindex(index=MODEL_ORDER, columns=BLUR_NAMES)

    fig, ax = plt.subplots(figsize=(6.5, 8))
    im = ax.imshow(acc.values, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(BLUR_NAMES)))
    ax.set_xticklabels([BLUR_LABELS[b] for b in BLUR_NAMES])
    ax.set_yticks(range(len(MODEL_ORDER)))
    ax.set_yticklabels([MODEL_LABELS[m] for m in MODEL_ORDER])
    for i in range(len(MODEL_ORDER)):
        for j in range(len(BLUR_NAMES)):
            v = acc.values[i, j]
            ax.text(
                j, i, f"{v:.1f}%", ha="center", va="center",
                color="white" if v > 55 else "black", fontsize=9,
            )
    ax.set_title("Accuracy Heatmap: Model x Blur Type")
    fig.colorbar(im, ax=ax, label="Accuracy (%)", shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "04_model_blur_heatmap.png"), bbox_inches="tight")
    plt.close(fig)


def chart_dropoff_grouped(df, out_dir):
    """Does the robustness ranking (accuracy lost, sev.1->5) hold across blur types?"""
    acc = df.groupby(["model", "blur_name", "severity"])["correct"].mean()
    dropoff = pd.DataFrame(
        {blur: (acc.xs(blur, level="blur_name").xs(1, level="severity")
                - acc.xs(blur, level="blur_name").xs(5, level="severity")) * 100
         for blur in BLUR_NAMES}
    ).reindex(MODEL_ORDER)
    # Order models by their average drop across blur types for a readable ranking.
    order = dropoff.mean(axis=1).sort_values(ascending=False).index

    x = np.arange(len(order))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, blur in enumerate(BLUR_NAMES):
        ax.bar(x + (i - 1) * width, dropoff.loc[order, blur], width, label=BLUR_LABELS[blur], color=BLUR_COLORS[blur])
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in order], rotation=40, ha="right")
    ax.set_ylabel("Accuracy drop, severity 1 -> 5 (pp)")
    ax.set_title("Robustness Ranking Across Blur Types\n(ordered by average decline)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "05_robustness_dropoff_by_blur.png"), bbox_inches="tight")
    plt.close(fig)


def chart_ranking_bumpchart(df, out_dir):
    """Bump chart: does the #1 model change depending on the blur type?"""
    acc = df.groupby(["blur_name", "model"])["correct"].mean().unstack("model") * 100
    acc = acc.reindex(BLUR_NAMES)
    ranks = acc.rank(axis=1, ascending=False, method="first")

    fig, ax = plt.subplots(figsize=(9, 8))
    x = np.arange(len(BLUR_NAMES))
    for model in MODEL_ORDER:
        y = ranks[model].values
        ax.plot(x, y, marker="o", markersize=10, linewidth=2.5, color=MODEL_COLORS[model], zorder=3)
        ax.annotate(
            MODEL_LABELS[model], (x[0], y[0]), xytext=(-10, 0), textcoords="offset points",
            ha="right", va="center", fontsize=9,
        )
        ax.annotate(
            MODEL_LABELS[model], (x[-1], y[-1]), xytext=(10, 0), textcoords="offset points",
            ha="left", va="center", fontsize=9,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([BLUR_LABELS[b] for b in BLUR_NAMES])
    ax.set_yticks(range(1, len(MODEL_ORDER) + 1))
    ax.set_ylabel("Rank (1 = most accurate)")
    ax.invert_yaxis()
    ax.set_xlim(-0.6, len(BLUR_NAMES) - 0.4)
    ax.set_title("Does the Accuracy Ranking Change by Blur Type?")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "06_model_ranking_bumpchart.png"), bbox_inches="tight")
    plt.close(fig)


def chart_class_heatmap(df, out_dir):
    """Fresh angle: which CIFAR-10 classes are hardest, and does that depend on blur type?"""
    acc = (df.groupby(["true_label", "blur_name"])["correct"].mean() * 100).unstack("blur_name")
    acc = acc.reindex(index=CIFAR10_LABELS, columns=BLUR_NAMES)

    fig, ax = plt.subplots(figsize=(6, 7))
    im = ax.imshow(acc.values, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(BLUR_NAMES)))
    ax.set_xticklabels([BLUR_LABELS[b] for b in BLUR_NAMES])
    ax.set_yticks(range(len(CIFAR10_LABELS)))
    ax.set_yticklabels([label.capitalize() for label in CIFAR10_LABELS])
    for i in range(len(CIFAR10_LABELS)):
        for j in range(len(BLUR_NAMES)):
            v = acc.values[i, j]
            ax.text(j, i, f"{v:.0f}%", ha="center", va="center", color="white" if v > 55 else "black", fontsize=9)
    ax.set_title("Accuracy by CIFAR-10 Class and Blur Type\n(averaged across all models/severities)")
    fig.colorbar(im, ax=ax, label="Accuracy (%)", shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "07_class_accuracy_heatmap.png"), bbox_inches="tight")
    plt.close(fig)


def chart_cost_latency_by_blur(df, out_dir):
    """Sanity + curiosity: did cost/latency actually stay consistent across blur types?"""
    summary = df.groupby(["model", "blur_name"]).agg(
        avg_latency=("latency_seconds", "mean"), total_cost=("cost_usd", "sum")
    )
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    x = np.arange(len(MODEL_ORDER))
    width = 0.25

    for i, blur in enumerate(BLUR_NAMES):
        lat = [summary.loc[(m, blur), "avg_latency"] for m in MODEL_ORDER]
        axes[0].bar(x + (i - 1) * width, lat, width, label=BLUR_LABELS[blur], color=BLUR_COLORS[blur])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], rotation=45, ha="right")
    axes[0].set_ylabel("Avg. latency (seconds)")
    axes[0].set_title("Latency by Model and Blur Type")
    axes[0].legend(frameon=False)

    for i, blur in enumerate(BLUR_NAMES):
        cost = [summary.loc[(m, blur), "total_cost"] for m in MODEL_ORDER]
        axes[1].bar(x + (i - 1) * width, cost, width, label=BLUR_LABELS[blur], color=BLUR_COLORS[blur])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], rotation=45, ha="right")
    axes[1].set_ylabel("Total cost (USD, 7,000 calls)")
    axes[1].set_title("Cost by Model and Blur Type")
    axes[1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "08_cost_latency_consistency.png"), bbox_inches="tight")
    plt.close(fig)


def chart_best_worst_summary(df, out_dir):
    """One clean scorecard: best/worst model per blur type."""
    acc = df.groupby(["blur_name", "model"])["correct"].mean().unstack("model") * 100
    acc = acc.reindex(BLUR_NAMES)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    best = acc.idxmax(axis=1)
    worst = acc.idxmin(axis=1)
    y = np.arange(len(BLUR_NAMES))
    for i, blur in enumerate(BLUR_NAMES):
        b_model, w_model = best[blur], worst[blur]
        ax.plot([acc.loc[blur, w_model], acc.loc[blur, b_model]], [i, i], color="#c3c2b7", zorder=1, linewidth=2)
        ax.scatter(acc.loc[blur, b_model], i, s=200, color=MODEL_COLORS[b_model], zorder=3)
        ax.scatter(acc.loc[blur, w_model], i, s=200, color=MODEL_COLORS[w_model], zorder=3, marker="X")
        ax.annotate(
            f"Best: {MODEL_LABELS[b_model]} ({acc.loc[blur, b_model]:.0f}%)",
            (acc.loc[blur, b_model], i), xytext=(10, 8), textcoords="offset points", fontsize=9,
        )
        ax.annotate(
            f"Worst: {MODEL_LABELS[w_model]} ({acc.loc[blur, w_model]:.0f}%)",
            (acc.loc[blur, w_model], i), xytext=(10, -14), textcoords="offset points", fontsize=9,
        )
    ax.set_yticks(y)
    ax.set_yticklabels([BLUR_LABELS[b] for b in BLUR_NAMES])
    ax.set_xlabel("Accuracy (%)")
    ax.set_xlim(0, 100)
    ax.set_title("Best vs. Worst Model, by Blur Type\n(circle = best, X = worst)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "09_best_worst_by_blur.png"), bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    df = load_all()
    chart_overall_accuracy_by_blur(df, FIGURES_DIR)
    chart_accuracy_by_blur_and_severity(df, FIGURES_DIR)
    chart_grid_model_by_blur(df, FIGURES_DIR)
    chart_model_blur_heatmap(df, FIGURES_DIR)
    chart_dropoff_grouped(df, FIGURES_DIR)
    chart_ranking_bumpchart(df, FIGURES_DIR)
    chart_class_heatmap(df, FIGURES_DIR)
    chart_cost_latency_by_blur(df, FIGURES_DIR)
    chart_best_worst_summary(df, FIGURES_DIR)
    print(f"Saved 9 figures to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
