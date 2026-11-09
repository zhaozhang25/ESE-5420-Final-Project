"""Figures for the paper."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"


def plot_pr_curves(oof, metrics, path):
    """Plot out-of-fold PR curves with and without PageValues, plus the floor."""
    with_pv, without_pv = metrics["with_pagevalues"], metrics["without_pagevalues"]
    floor = metrics["floor"]["mean"]

    fig, ax = plt.subplots(figsize=(7, 4.6), dpi=200, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    series = [
        ("score_with_pagevalues", BLUE,
         f"With PageValues: {with_pv['mean']:.3f} ± {with_pv['se']:.3f}"),
        ("score_without_pagevalues", ORANGE,
         f"Without PageValues: {without_pv['mean']:.3f} ± {without_pv['se']:.3f}"),
    ]
    for column, color, label in series:
        precision, recall, _ = precision_recall_curve(oof["y"], oof[column])
        ax.step(recall, precision, where="post", color=color, lw=1.5, label=label)
    ax.axhline(floor, color=INK_2, lw=1.2, ls=(0, (4, 3)),
               label=f"Always no purchase: {floor:.3f}")

    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Recall", ylabel="Precision")
    fig.suptitle(
        f"Removing PageValues drops CV AUPRC from {with_pv['mean']:.2f} to {without_pv['mean']:.2f}",
        ha="left", fontsize=12, fontweight="bold", color=INK)
    ax.set_title("Dev set, out-of-fold predictions. Legend: CV AUPRC ± SE over 5 folds.",
                 loc="left", fontsize=9, color=INK_2, pad=8)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GRID)
    legend = ax.legend(loc="upper right", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(INK)
    for x, y, text in [(0.42, 0.78, "With PageValues"),
                       (0.42, 0.37, "Without PageValues"),
                       (0.02, 0.105, "Always no purchase")]:
        ax.text(x, y, text, fontsize=9, color=INK)

    fig.tight_layout()
    fig._suptitle.set_x(ax.get_position().x0)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
