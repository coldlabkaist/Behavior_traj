"""Consistent PNG/SVG export for paper plots; callers own figure lifetime."""
from __future__ import annotations

from pathlib import Path

from matplotlib.figure import Figure


def save_figure(
    figure: Figure,
    stem: str | Path,
    *,
    dpi: float = 300,
    svg_dpi: float | None = None,
    bbox_inches: str | None = None,
    facecolor: str | None = None,
) -> tuple[Path, Path]:
    """Save both formats, preserving the panel's canvas and vector rasterization.

    ``stem`` may be extensionless or end in .png/.svg. ``dpi`` controls PNG;
    ``svg_dpi`` optionally controls raster layers embedded in SVG. None retains
    Matplotlib's figure/default DPI, as in the original vector exports.
    No PDF is produced. The figure is not closed or restyled here.
    """
    stem = Path(stem)
    if stem.suffix.lower() == '.pdf':
        raise ValueError('Paper figures are exported as PNG and SVG only')
    if stem.suffix.lower() in {'.png', '.svg'}:
        stem = stem.with_suffix('')
    stem.parent.mkdir(parents=True, exist_ok=True)
    png, svg = Path(str(stem) + '.png'), Path(str(stem) + '.svg')
    options = {}
    if bbox_inches is not None:
        options['bbox_inches'] = bbox_inches
    if facecolor is not None:
        options['facecolor'] = facecolor
    figure.savefig(png, dpi=dpi, **options)
    svg_options = {} if svg_dpi is None else {'dpi': svg_dpi}
    figure.savefig(svg, **options, **svg_options)
    return png, svg
