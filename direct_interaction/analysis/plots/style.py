"""Shared colors, typography and PNG/SVG export for the paper figures."""
from pathlib import Path
import matplotlib.pyplot as plt

CONDITIONS = ("cont", "exp")


COLORS = {"cont": "#304F78", "exp": "#C4475B"}


DISPLAY_LABELS = {"cont": "Control", "exp": "VPA"}


TITLE_LABELS = {"Social": "Reciprocal direct social behavior"}


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "Arial",
        "axes.linewidth": 1.7,
        "xtick.major.width": 1.6,
        "ytick.major.width": 1.6,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, dpi: int) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=dpi, facecolor="white")
    with plt.rc_context({"svg.fonttype": "none"}):
        fig.savefig(output_dir / f"{stem}.svg", facecolor="white")
    plt.close(fig)
