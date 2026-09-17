from __future__ import annotations
from pathlib import Path
import sys
from analysis.paths import ROOT
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

PNDS = [10, 15, 20]

COLORS = {"Control": "#304F78", "VPA": "#C4475B"}

OFFSETS = {"Control": -0.16, "VPA": 0.16}

def _style_axis(ax: plt.Axes) -> None:
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)
	ax.tick_params(direction="out", width=1.2)

def _significance_stars(p_value: float) -> str:
	if p_value < 0.001:
		return "***"
	if p_value < 0.01:
		return "**"
	if p_value < 0.05:
		return "*"
	return "ns"

def _save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
	fig.tight_layout()
	fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
	fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
	plt.close(fig)

def plot_locomotion(
	df: pd.DataFrame,
	between: pd.DataFrame,
	output_dir: Path,
) -> None:
	metric = "avg_velocity_mm_s"
	fig, ax = plt.subplots(figsize=(8.8, 5.5))
	rng = np.random.default_rng(20260728)

	for condition in ("Control", "VPA"):
		group = df[df["condition"] == condition].copy()
		offset = OFFSETS[condition]
		for _, subject in group.groupby("subject_id"):
			subject = subject.sort_values("pnd")
			ax.plot(
				subject["pnd"].to_numpy(dtype=float) + offset,
				pd.to_numeric(subject[metric], errors="coerce"),
				color=COLORS[condition],
				alpha=0.16,
				linewidth=0.8,
				zorder=1,
			)

		means: list[float] = []
		sems: list[float] = []
		for pnd in PNDS:
			values = pd.to_numeric(
				group.loc[group["pnd"] == pnd, metric],
				errors="coerce",
			).dropna()
			x = pnd + offset + rng.uniform(-0.045, 0.045, size=len(values))
			ax.scatter(
				x,
				values,
				s=30,
				facecolors="white",
				edgecolors=COLORS[condition],
				linewidths=1.15,
				alpha=0.9,
				zorder=2,
			)
			means.append(float(values.mean()))
			sems.append(float(values.sem()))
		ax.errorbar(
			np.asarray(PNDS, dtype=float) + offset,
			means,
			yerr=sems,
			color=COLORS[condition],
			marker="o",
			markersize=7,
			linewidth=2.2,
			capsize=4,
			label=condition,
			zorder=3,
		)

	values_all = pd.to_numeric(df[metric], errors="coerce").dropna()
	y_max = float(values_all.max())
	ax.set_ylim(0, y_max * 1.18)
	stat_rows = between[between["metric"] == metric].set_index("pnd")
	for pnd in PNDS:
		if pnd not in stat_rows.index:
			continue
		p_value = stat_rows.loc[pnd, "p_holm_within_metric"]
		if pd.notna(p_value):
			significance_label = _significance_stars(float(p_value))
			ax.text(
				pnd,
				y_max * 1.08,
				significance_label,
				ha="center",
				va="bottom",
				fontsize=22 if significance_label == "ns" else 32,
			)

	ax.set_xticks(PNDS)
	ax.set_xlabel("Postnatal day", fontsize=20, labelpad=10)
	ax.set_ylabel("mom mean velocity (mm/s)", fontsize=20, labelpad=12)
	ax.set_title("Maternal movement", fontsize=26, pad=16)
	_style_axis(ax)
	ax.spines["left"].set_linewidth(1.6)
	ax.spines["bottom"].set_linewidth(1.6)
	ax.tick_params(
		axis="both",
		which="major",
		labelsize=17,
		width=1.6,
		length=6,
		pad=6,
	)
	_save(fig, output_dir, "maternal_locomotion_by_pnd")

def plot_pnd10_stable_proximity(
	df: pd.DataFrame,
	stable_stats: pd.DataFrame,
	output_dir: Path,
) -> None:
	metric = "total_stable_proximity_time_sec"
	subset = df[df["pnd"] == 10].copy()
	conditions = ["Control", "VPA"]
	values = [
		pd.to_numeric(
			subset.loc[subset["condition"] == condition, metric],
			errors="coerce",
		).dropna()
		for condition in conditions
	]

	fig, ax = plt.subplots(figsize=(4.8, 4.9))
	box = ax.boxplot(
		values,
		positions=[0, 1],
		widths=0.48,
		patch_artist=True,
		showfliers=False,
		medianprops={"color": "white", "linewidth": 2.0},
		whiskerprops={"linewidth": 1.4},
		capprops={"linewidth": 1.4},
	)
	for patch, condition in zip(box["boxes"], conditions):
		patch.set_facecolor(COLORS[condition])
		patch.set_edgecolor(COLORS[condition])
		patch.set_alpha(0.82)
	for index, condition in enumerate(conditions):
		for artist in (
			box["whiskers"][2 * index],
			box["whiskers"][2 * index + 1],
			box["caps"][2 * index],
			box["caps"][2 * index + 1],
		):
			artist.set_color(COLORS[condition])

	rng = np.random.default_rng(20260728)
	for index, (condition, group_values) in enumerate(zip(conditions, values)):
		x = index + rng.uniform(-0.085, 0.085, size=len(group_values))
		ax.scatter(
			x,
			group_values,
			s=38,
			facecolors="white",
			edgecolors=COLORS[condition],
			linewidths=1.3,
			zorder=3,
		)

	y_max = max(float(max(group_values)) for group_values in values if len(group_values))
	bracket_y = y_max * 1.07
	ax.plot([0, 0, 1, 1], [bracket_y * 0.985, bracket_y, bracket_y, bracket_y * 0.985], color="#222222", linewidth=1.2)
	pnd10 = stable_stats[stable_stats["pnd"] == 10]
	if not pnd10.empty:
		p_value = pnd10.iloc[0]["p_holm_mann_whitney"]
		ax.text(
			0.5,
			bracket_y * 1.015,
			_significance_stars(float(p_value)),
			ha="center",
			va="bottom",
			fontsize=15,
		)

	ax.set_ylim(0, y_max * 1.18)
	ax.set_xticks([0, 1], conditions)
	ax.set_ylabel("Total stable proximity time (s)")
	ax.set_title("Stable proximity at PND10")
	_style_axis(ax)
	_save(fig, output_dir, "pnd10_stable_proximity")
