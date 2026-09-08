"""Generate readable figures from a completed blur-run CSV for the paper.

Reads results/<blur>/results_<blur>.csv and writes a set of PNG charts into
that same folder. Works for any blur type that follows the
results/<blur>/results_<blur>.csv convention (motion, glass, defocus) —
pass the blur name as the one CLI argument.

Usage:
    python -m scripts.generate_figures motion
    python -m scripts.generate_figures glass
    python -m scripts.generate_figures defocus
    python -m scripts.generate_figures          # defaults to motion
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import MODEL_IDS, SEVERITIES

BLUR_NAME = sys.argv[1] if len(sys.argv) > 1 else "motion"
BLUR_LABEL = {"motion": "Motion Blur", "glass": "Glass Blur", "defocus": "Defocus Blur"}.get(
    BLUR_NAME, BLUR_NAME.title()
)
RESULTS_CSV = f"results/{BLUR_NAME}/results_{BLUR_NAME}.csv"
FIGURES_DIR = f"results/{BLUR_NAME}"

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
COLORS = {m: plt.cm.tab10(i) for i, m in enumerate(MODEL_ORDER)}

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


def load_data():
    df = pd.read_csv(RESULTS_CSV)
    df["cost_usd"] = pd.to_numeric(df["cost_usd"], errors="coerce")
    return df


def accuracy_table(df):
    return df.groupby(["model", "severity"])["correct"].mean().unstack("severity")


def bubble_sizes(values, min_size=150, max_size=2000):
    lo, hi = values.min(), values.max()
    return min_size + (values - lo) / (hi - lo) * (max_size - min_size)


def chart_accuracy_grid(df, out_dir):
    """One panel per model: accuracy vs. severity."""
    acc = accuracy_table(df)
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharex=True, sharey=True)
    for ax, model in zip(axes.flat, MODEL_ORDER):
        row = acc.loc[model]
        ax.plot(row.index, row.values * 100, marker="o", color=COLORS[model], linewidth=2)
        ax.set_title(MODEL_LABELS[model], fontsize=11)
        ax.set_xticks(SEVERITIES)
        ax.set_ylim(0, 100)
    for ax in axes[-1, :]:
        ax.set_xlabel("Severity")
    for ax in axes[:, 0]:
        ax.set_ylabel("Accuracy (%)")
    fig.suptitle(f"{BLUR_LABEL}: Accuracy vs. Severity, by Model", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "01_accuracy_by_model_grid.png"), bbox_inches="tight")
    plt.close(fig)


def chart_accuracy_comparison(df, out_dir):
    """All 9 models overlaid on one chart."""
    acc = accuracy_table(df)
    fig, ax = plt.subplots(figsize=(9, 6))
    for model in MODEL_ORDER:
        row = acc.loc[model]
        ax.plot(
            row.index,
            row.values * 100,
            marker="o",
            label=MODEL_LABELS[model],
            color=COLORS[model],
            linewidth=2,
        )
    ax.set_xlabel("Severity")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(SEVERITIES)
    ax.set_ylim(0, 100)
    ax.set_title(f"{BLUR_LABEL}: All Models Compared")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "02_accuracy_all_models_comparison.png"), bbox_inches="tight")
    plt.close(fig)


def chart_latency(df, out_dir):
    lat = df.groupby("model")["latency_seconds"].mean().reindex(MODEL_ORDER)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        [MODEL_LABELS[m] for m in MODEL_ORDER], lat.values, color=[COLORS[m] for m in MODEL_ORDER]
    )
    ax.set_ylabel("Avg. latency (seconds)")
    ax.set_title(f"{BLUR_LABEL}: Average Response Latency by Model")
    plt.xticks(rotation=40, ha="right")
    for bar, v in zip(bars, lat.values):
        ax.annotate(
            f"{v:.2f}s",
            (bar.get_x() + bar.get_width() / 2, v),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "03_latency_by_model.png"), bbox_inches="tight")
    plt.close(fig)


def chart_cost(df, out_dir):
    cost = df.groupby("model")["cost_usd"].sum().reindex(MODEL_ORDER)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        [MODEL_LABELS[m] for m in MODEL_ORDER], cost.values, color=[COLORS[m] for m in MODEL_ORDER]
    )
    ax.set_ylabel("Total cost (USD)")
    ax.set_title(f"{BLUR_LABEL}: Total Cost by Model (7,000 calls each)")
    plt.xticks(rotation=40, ha="right")
    for bar, v in zip(bars, cost.values):
        ax.annotate(
            f"${v:.2f}",
            (bar.get_x() + bar.get_width() / 2, v),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "04_cost_by_model.png"), bbox_inches="tight")
    plt.close(fig)


def chart_cost_vs_accuracy(df, out_dir):
    """Which models give the most accuracy per dollar."""
    summary = (
        df.groupby("model")
        .agg(total_cost=("cost_usd", "sum"), accuracy=("correct", "mean"))
        .reindex(MODEL_ORDER)
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    for model in MODEL_ORDER:
        row = summary.loc[model]
        ax.scatter(row["total_cost"], row["accuracy"] * 100, s=120, color=COLORS[model])
        ax.annotate(
            MODEL_LABELS[model],
            (row["total_cost"], row["accuracy"] * 100),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
        )
    ax.set_xlabel("Total cost (USD, 7,000 calls)")
    ax.set_ylabel("Overall accuracy (%)")
    ax.set_title(f"{BLUR_LABEL}: Cost vs. Accuracy — Which Models Are Efficient?")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "05_cost_vs_accuracy.png"), bbox_inches="tight")
    plt.close(fig)


def chart_dropoff(df, out_dir):
    """Robustness ranking: how much accuracy each model loses, severity 1 -> 5."""
    acc = accuracy_table(df)
    dropoff = ((acc[1] - acc[5]) * 100).reindex(MODEL_ORDER).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [COLORS[m] for m in dropoff.index]
    bars = ax.barh([MODEL_LABELS[m] for m in dropoff.index], dropoff.values, color=colors)
    ax.set_xlabel("Accuracy drop, severity 1 -> 5 (percentage points)")
    ax.set_title(f"{BLUR_LABEL}: Robustness Ranking (Accuracy Decline)")
    ax.invert_yaxis()
    for bar, v in zip(bars, dropoff.values):
        ax.annotate(
            f"{v:.1f}pp",
            (v, bar.get_y() + bar.get_height() / 2),
            va="center",
            ha="left",
            fontsize=8,
            xytext=(4, 0),
            textcoords="offset points",
        )
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "06_accuracy_dropoff_ranking.png"), bbox_inches="tight")
    plt.close(fig)


def chart_latency_vs_accuracy(df, out_dir):
    """Is a slower model actually a more accurate one?"""
    summary = (
        df.groupby("model")
        .agg(avg_latency=("latency_seconds", "mean"), accuracy=("correct", "mean"))
        .reindex(MODEL_ORDER)
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    for model in MODEL_ORDER:
        row = summary.loc[model]
        ax.scatter(row["avg_latency"], row["accuracy"] * 100, s=120, color=COLORS[model])
        ax.annotate(
            MODEL_LABELS[model],
            (row["avg_latency"], row["accuracy"] * 100),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
        )
    ax.set_xlabel("Avg. latency (seconds)")
    ax.set_ylabel("Overall accuracy (%)")
    ax.set_title(f"{BLUR_LABEL}: Latency vs. Accuracy — Does Slower Mean Smarter?")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "07_latency_vs_accuracy.png"), bbox_inches="tight")
    plt.close(fig)


def _bubble_legend(ax, values, fmt, title):
    lo, hi = values.min(), values.max()
    for ref in (lo, (lo + hi) / 2, hi):
        ax.scatter([], [], s=bubble_sizes(pd.Series([lo, hi, ref]))[2], color="gray", alpha=0.5, label=fmt(ref))
    ax.legend(
        title=title,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
        labelspacing=2.2,
        borderpad=1.5,
    )


def chart_latency_cost_accuracy(df, out_dir):
    """All three at once: x=latency, y=accuracy, bubble size=total cost."""
    summary = (
        df.groupby("model")
        .agg(
            avg_latency=("latency_seconds", "mean"),
            accuracy=("correct", "mean"),
            total_cost=("cost_usd", "sum"),
        )
        .reindex(MODEL_ORDER)
    )
    sizes = bubble_sizes(summary["total_cost"])

    fig, ax = plt.subplots(figsize=(9, 7))
    for model in MODEL_ORDER:
        row = summary.loc[model]
        ax.scatter(
            row["avg_latency"],
            row["accuracy"] * 100,
            s=sizes[model],
            color=COLORS[model],
            alpha=0.75,
            edgecolors="white",
            linewidths=1,
        )
        ax.annotate(
            MODEL_LABELS[model],
            (row["avg_latency"], row["accuracy"] * 100),
            textcoords="offset points",
            xytext=(0, 14),
            ha="center",
            fontsize=8,
        )
    ax.set_xlabel("Avg. latency (seconds)")
    ax.set_ylabel("Overall accuracy (%)")
    ax.set_title(f"{BLUR_LABEL}: Latency, Accuracy & Cost Together\n(bubble size = total cost for 7,000 calls)")
    _bubble_legend(ax, summary["total_cost"], lambda v: f"${v:.2f} total", "Bubble size = cost")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "08_latency_cost_accuracy_bubble.png"), bbox_inches="tight")
    plt.close(fig)


def chart_latency_cost_accuracy_size(df, out_dir):
    """Same three variables, different encoding: x=latency, y=cost, bubble size=accuracy."""
    summary = (
        df.groupby("model")
        .agg(
            avg_latency=("latency_seconds", "mean"),
            accuracy=("correct", "mean"),
            total_cost=("cost_usd", "sum"),
        )
        .reindex(MODEL_ORDER)
    )
    sizes = bubble_sizes(summary["accuracy"], min_size=120, max_size=1300)

    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    for i, model in enumerate(MODEL_ORDER):
        row = summary.loc[model]
        ax.scatter(
            row["avg_latency"],
            row["total_cost"],
            s=sizes[model],
            color=COLORS[model],
            alpha=0.75,
            edgecolors="white",
            linewidths=1,
            zorder=3,
        )
        # Alternate label placement above/below to reduce collisions among
        # the tightly clustered cheaper/faster models.
        y_offset = 16 if i % 2 == 0 else -18
        va = "bottom" if i % 2 == 0 else "top"
        ax.annotate(
            MODEL_LABELS[model],
            (row["avg_latency"], row["total_cost"]),
            textcoords="offset points",
            xytext=(0, y_offset),
            ha="center",
            va=va,
            fontsize=8,
        )
    ax.set_xlabel("Avg. latency (seconds)")
    ax.set_ylabel("Total cost (USD, 7,000 calls)")
    ax.margins(x=0.15, y=0.2)
    ax.set_title(
        f"{BLUR_LABEL}: Latency, Cost & Accuracy Together\n(bubble size = overall accuracy)"
    )
    _bubble_legend(ax, summary["accuracy"] * 100, lambda v: f"{v:.0f}% accuracy", "Bubble size = accuracy")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "09_latency_cost_bubble_accuracy_size.png"), bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    df = load_data()
    chart_accuracy_grid(df, FIGURES_DIR)
    chart_accuracy_comparison(df, FIGURES_DIR)
    chart_latency(df, FIGURES_DIR)
    chart_cost(df, FIGURES_DIR)
    chart_cost_vs_accuracy(df, FIGURES_DIR)
    chart_dropoff(df, FIGURES_DIR)
    chart_latency_vs_accuracy(df, FIGURES_DIR)
    chart_latency_cost_accuracy(df, FIGURES_DIR)
    chart_latency_cost_accuracy_size(df, FIGURES_DIR)
    print(f"Saved 9 figures to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
