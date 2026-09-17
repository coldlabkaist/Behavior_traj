"""Shared paper palette and exact panel-specific Matplotlib settings.

Only font.family is universal. Differences in font sizes, axes and SVG text
representation are explicit profiles so consolidation does not redesign figures.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy

import matplotlib as mpl

FONT_FAMILY = 'Arial'
CONTROL_COLOR = '#315780'
VPA_COLOR = '#C74B5D'
CONDITION_COLORS = {'control': CONTROL_COLOR, 'vpa': VPA_COLOR}
MOTIF_COLORS = ('#cfbad1', '#7d1f78', '#cdb18a', '#684720')
# S6 used a distinct palette; retain it explicitly rather than recoloring it.
NETWORK_MOTIF_COLORS = {'M0': '#C8B2CA', 'M1': '#86227F', 'M2': '#D2B184', 'M3': '#825820'}
NETWORK_INK = '#202936'
NETWORK_DECREASE = '#99A2AB'

BASE_STYLE = {'font.family': FONT_FAMILY}

PANEL_STYLES = {
    # Fig4EFG
    'boi': {
        'font.size': 17,
        'axes.titlesize': 22,
        'axes.titleweight': 'bold',
        'axes.labelsize': 19,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 17,
        'axes.linewidth': 1.8,
        'axes.edgecolor': '#202934',
        'xtick.color': '#202934',
        'ytick.color': '#202934',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'pdf.fonttype': 42,
        'svg.fonttype': 'none',
    },
    # FigS3AB
    'dae_validation': {
        'font.size': 15.0,
        'axes.titlesize': 21.0,
        'axes.titleweight': 'bold',
        'axes.labelsize': 17.0,
        'axes.labelcolor': '#202833',
        'xtick.color': '#202833',
        'ytick.color': '#202833',
        'text.color': '#202833',
        'legend.fontsize': 14.0,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
    },
    # Fig4B
    'distribution': {
        'font.sans-serif': [FONT_FAMILY],
        'font.size': 16,
        'axes.titlesize': 24,
        'axes.labelsize': 20,
        'xtick.labelsize': 17,
        'ytick.labelsize': 17,
        'text.color': '#202833',
        'axes.labelcolor': '#202833',
        'xtick.color': '#202833',
        'ytick.color': '#202833',
        'axes.edgecolor': 'black',
    },
    # FigS4AB
    'factor_analysis': {
        'font.size': 18.0,
        'axes.titlesize': 30.0,
        'axes.labelsize': 22.0,
        'xtick.labelsize': 19.0,
        'ytick.labelsize': 19.0,
        'legend.fontsize': 18.0,
        'axes.linewidth': 1.5,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        'xtick.major.size': 7,
        'ytick.major.size': 7,
        'svg.fonttype': 'none',
    },
    # Fig5G/I
    'maternal_association': {
        'axes.linewidth': 1.45,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    },
    # FigS5A
    'motif_resolution': {
        'font.size': 20.5,
        'axes.labelsize': 22.5,
        'axes.titlesize': 28.0,
        'axes.linewidth': 1.6,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        'xtick.major.size': 7.0,
        'ytick.major.size': 7.0,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    },
    # Fig4C
    'motifs': {
        'font.sans-serif': [FONT_FAMILY],
        'font.size': 28,
        'axes.titlesize': 40,
        'axes.labelsize': 42,
        'xtick.labelsize': 30,
        'ytick.labelsize': 30,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
    },
    # FigS6AB
    'network_changes': {
        'font.sans-serif': [FONT_FAMILY],
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
    },
    # FigS7AB
    'sex_endpoints': {
        'font.size': 15,
        'axes.titlesize': 20,
        'axes.titleweight': 'bold',
        'axes.labelsize': 18,
        'axes.linewidth': 1.8,
        'axes.edgecolor': '#202934',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.labelsize': 17,
        'ytick.labelsize': 14,
        'pdf.fonttype': 42,
        'svg.fonttype': 'none',
    },
    # FigS5B
    'temporal_shift': {
        'font.size': 17,
        'axes.titlesize': 22,
        'axes.titleweight': 'bold',
        'axes.labelsize': 20,
        'axes.linewidth': 1.7,
        'axes.edgecolor': '#20262D',
        'axes.labelcolor': '#20262D',
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'xtick.color': '#20262D',
        'ytick.color': '#20262D',
        'text.color': '#20262D',
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'pdf.fonttype': 42,
        'svg.fonttype': 'none',
    },
    # Fig4D
    'transitions': {
        'font.sans-serif': [FONT_FAMILY],
        'font.size': 20,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
        'hatch.linewidth': 3.0,
    },
    # Fig4C representative poses
    'pose': {
        'font.sans-serif': [FONT_FAMILY],
        'font.size': 16,
        'axes.titlesize': 22,
        'axes.labelsize': 20,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
        'hatch.linewidth': 0.8,
    },
    # Latent projection validation
    'projection_validation': {
        'font.size': 12,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'pdf.fonttype': 42,
        'svg.fonttype': 'none',
    },
}

SCALED_FONT_KEYS = {'dae_validation': ['font.size', 'axes.titlesize', 'axes.labelsize', 'legend.fontsize'],
 'factor_analysis': ['font.size',
                     'axes.titlesize',
                     'axes.labelsize',
                     'xtick.labelsize',
                     'ytick.labelsize',
                     'legend.fontsize']}


def style_settings(profile: str, *, text_scale: float = 1.0) -> dict:
    """Return a fresh settings dictionary, scaling only originally scaled fonts."""
    settings = {**BASE_STYLE, **deepcopy(PANEL_STYLES[profile])}
    for key in SCALED_FONT_KEYS.get(profile, ()):
        settings[key] *= text_scale
    return settings


def apply_style(profile: str, *, text_scale: float = 1.0) -> None:
    """Apply a named profile inside the caller's style_context."""
    mpl.rcParams.update(style_settings(profile, text_scale=text_scale))


@contextmanager
def style_context(profile: str | None = None, *, text_scale: float = 1.0):
    """Start from Matplotlib defaults and restore the caller's settings on exit."""
    with mpl.rc_context(rc=mpl.rcParamsDefault):
        if profile is not None:
            apply_style(profile, text_scale=text_scale)
        yield
