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

try:
	import pingouin as pg
except ImportError:
	pg = None

from analysis.core.mom_pup.development import PNDS

from analysis.core.mom_pup.development import COLORS

def plot_metric(
	df: pd.DataFrame,
	metric: str,
	ylabel: str,
	between: pd.DataFrame,
	output_path: Path,
) -> None:
	fig, ax = plt.subplots(figsize=(6.4, 5.0))
	for condition in ("Control", "VPA"):
		group = df[df["condition"] == condition]
		for subject_id, subject in group.groupby("subject_id"):
			subject = subject.sort_values("pnd")
			ax.plot(
				subject["pnd"],
				pd.to_numeric(subject[metric], errors="coerce"),
				color=COLORS[condition],
				alpha=0.18,
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
			means.append(values.mean() if len(values) else np.nan)
			sems.append(values.sem() if len(values) > 1 else 0.0)
			ax.scatter(
				np.full(len(values), pnd),
				values,
				s=24,
				facecolors="white",
				edgecolors=COLORS[condition],
				linewidths=1.0,
				alpha=0.75,
				zorder=2,
			)
		ax.errorbar(
			PNDS,
			means,
			yerr=sems,
			color=COLORS[condition],
			marker="o",
			linewidth=2.2,
			capsize=4,
			label=condition,
			zorder=3,
		)

	valid_values = pd.to_numeric(df[metric], errors="coerce").dropna()
	value_range = valid_values.max() - valid_values.min() if len(valid_values) else 1.0
	value_range = value_range if value_range > 0 else 1.0
	for row in between.to_dict(orient="records"):
		q_value = row.get("p_holm_within_metric")
		if pd.notna(q_value):
			ax.text(
				int(row["pnd"]),
				valid_values.max() + 0.07 * value_range,
				f"Holm p={float(q_value):.3g}",
				ha="center",
				va="bottom",
				fontsize=8,
			)
	ax.set_xticks(PNDS)
	ax.set_xlabel("Postnatal day")
	ax.set_ylabel(ylabel)
	ax.legend(frameon=False)
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)
	fig.tight_layout()
	fig.savefig(output_path, dpi=250)
	plt.close(fig)
