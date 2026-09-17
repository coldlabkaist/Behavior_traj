"""Shared palette and PNG/SVG export; panel dimensions remain panel-specific."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"Raw SLEAP": "#a8e6a3", "Seg-Cont SLEAP": "#1b5e20",
          "Raw DLC": "#e4de8e", "Seg-Cont DLC": "#e2bb0c",
          "Raw MovAl": "#f4aaaa", "Seg-Cont MovAl": "#b71c1c"}


def apply_style():
    plt.rcParams.update({"font.family": "Arial", "font.weight": "regular",
                         "axes.unicode_minus": False, "svg.fonttype": "none"})


def significance_label(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def save_figure(fig, folder, stem, transparent=True):
    folder.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for extension in ("png", "svg"):
        fig.savefig(folder / f"{stem}.{extension}", dpi=300,
                    transparent=transparent, bbox_inches="tight")
    plt.close(fig)
