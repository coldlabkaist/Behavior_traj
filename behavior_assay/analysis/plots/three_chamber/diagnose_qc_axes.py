from __future__ import annotations
from pathlib import Path
import sys
from analysis.paths import ROOT
import math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analysis.core.three_chamber.diagnose_qc_axes import _condition_sort_key
from analysis.core.three_chamber.diagnose_qc_axes import _sem

from analysis.core.three_chamber.diagnose_qc_axes import FIGURE_DPI

from analysis.core.three_chamber.diagnose_qc_axes import CONDITION_COLORS

from analysis.core.three_chamber.diagnose_qc_axes import SEX_ORDER

from analysis.core.three_chamber.diagnose_qc_axes import PHASE_ORDER

from analysis.core.three_chamber.diagnose_qc_axes import LOCO_PHASE_ORDER

from analysis.core.three_chamber.diagnose_qc_axes import PHASE_LABELS

def _plot_preference_by_axis(pref_df: pd.DataFrame, axis: str, output_path: Path) -> None:
	phases = [phase for phase in PHASE_ORDER if phase in set(pref_df["phase"])]
	fig, axes = plt.subplots(1, len(phases), figsize=(7.2 * max(len(phases), 1), 5.2), dpi=FIGURE_DPI, squeeze=False)
	for ax, phase in zip(axes.ravel(), phases):
		phase_df = pref_df[pref_df["phase"].eq(phase)].copy()
		if axis == "date":
			labels = sorted(phase_df["date"].dropna().astype(str).unique())
			x_positions = np.arange(len(labels), dtype=float)
			for condition in sorted(phase_df["condition"].dropna().astype(str).unique(), key=_condition_sort_key):
				cond_df = phase_df[phase_df["condition"].astype(str).eq(condition)]
				xs = []
				means = []
				sems = []
				for idx, label in enumerate(labels):
					vals = pd.to_numeric(cond_df[cond_df["date"].astype(str).eq(label)]["preference_index"], errors="coerce")
					if vals.dropna().empty:
						continue
					xs.append(x_positions[idx])
					means.append(float(vals.mean()))
					sems.append(_sem(vals))
					points = vals.dropna().to_numpy(dtype=float)
					jitter = np.linspace(-0.06, 0.06, len(points)) if len(points) else []
					ax.scatter(np.full(len(points), x_positions[idx]) + jitter, points, color=CONDITION_COLORS.get(condition, "#777777"), edgecolors="black", linewidths=0.35, s=24, alpha=0.75)
				if xs:
					ax.errorbar(xs, means, yerr=sems, color=CONDITION_COLORS.get(condition, "#777777"), marker="o", linewidth=2.0, capsize=4, label=condition)
			ax.set_xticks(x_positions)
			ax.set_xticklabels(labels, rotation=35, ha="right")
		else:
			labels = SEX_ORDER
			x_positions = np.arange(len(labels), dtype=float)
			width = 0.28
			conditions = sorted(phase_df["condition"].dropna().astype(str).unique(), key=_condition_sort_key)
			for cond_idx, condition in enumerate(conditions):
				cond_df = phase_df[phase_df["condition"].astype(str).eq(condition)]
				offset = (cond_idx - (len(conditions) - 1) / 2.0) * width
				means = []
				sems = []
				for label in labels:
					vals = pd.to_numeric(cond_df[cond_df["sex"].astype(str).str.lower().eq(label)]["preference_index"], errors="coerce")
					means.append(float(vals.mean()) if not vals.dropna().empty else math.nan)
					sems.append(_sem(vals))
				ax.bar(x_positions + offset, means, width=width, color=CONDITION_COLORS.get(condition, "#777777"), alpha=0.85, label=condition)
				ax.errorbar(x_positions + offset, means, yerr=sems, fmt="none", color="black", capsize=3)
				for idx, label in enumerate(labels):
					vals = pd.to_numeric(cond_df[cond_df["sex"].astype(str).str.lower().eq(label)]["preference_index"], errors="coerce").dropna().to_numpy(dtype=float)
					if len(vals):
						jitter = np.linspace(-0.04, 0.04, len(vals))
						ax.scatter(np.full(len(vals), x_positions[idx] + offset) + jitter, vals, color=CONDITION_COLORS.get(condition, "#777777"), edgecolors="black", linewidths=0.35, s=24)
			ax.set_xticks(x_positions)
			ax.set_xticklabels(["M", "F"])
		ax.axhline(0.0, color="#666666", linewidth=1.0)
		ax.set_title(PHASE_LABELS.get(phase, phase), fontsize=16)
		ax.set_ylabel("Preference index")
		ax.grid(axis="y", alpha=0.25)
		ax.legend(frameon=False)
	fig.suptitle(f"Preference index by {axis}", fontsize=21)
	fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.92))
	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output_path, dpi=FIGURE_DPI, facecolor="white")
	plt.close(fig)

def _plot_locomotion(loco_df: pd.DataFrame, output_path: Path) -> None:
	phases = [phase for phase in LOCO_PHASE_ORDER if phase in set(loco_df["phase"])]
	fig, axes = plt.subplots(1, len(phases), figsize=(5.2 * max(len(phases), 1), 5.0), dpi=FIGURE_DPI, squeeze=False)
	conditions = sorted(loco_df["condition"].dropna().astype(str).unique(), key=_condition_sort_key)
	for ax, phase in zip(axes.ravel(), phases):
		phase_df = loco_df[loco_df["phase"].eq(phase)].copy()
		xs = np.arange(len(conditions), dtype=float)
		means = []
		sems = []
		for idx, condition in enumerate(conditions):
			vals = pd.to_numeric(phase_df[phase_df["condition"].astype(str).eq(condition)]["distance_per_min_norm"], errors="coerce")
			means.append(float(vals.mean()) if not vals.dropna().empty else math.nan)
			sems.append(_sem(vals))
			points = vals.dropna().to_numpy(dtype=float)
			if len(points):
				jitter = np.linspace(-0.08, 0.08, len(points))
				ax.scatter(np.full(len(points), xs[idx]) + jitter, points, color=CONDITION_COLORS.get(condition, "#777777"), edgecolors="black", linewidths=0.35, s=24)
		ax.bar(xs, means, width=0.62, color=[CONDITION_COLORS.get(c, "#777777") for c in conditions], alpha=0.85)
		ax.errorbar(xs, means, yerr=sems, fmt="none", color="black", capsize=4)
		ax.set_xticks(xs)
		ax.set_xticklabels(conditions)
		ax.set_title(PHASE_LABELS.get(phase, phase), fontsize=15)
		ax.set_ylabel("Distance per min (norm)")
		ax.grid(axis="y", alpha=0.25)
	fig.suptitle("Locomotion QC", fontsize=21)
	fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.90))
	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output_path, dpi=FIGURE_DPI, facecolor="white")
	plt.close(fig)

def _plot_open_bias(loco_df: pd.DataFrame, output_path: Path) -> None:
	open_df = loco_df[loco_df["phase"].eq("hab_open")].copy()
	fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), dpi=FIGURE_DPI)
	conditions = sorted(open_df["condition"].dropna().astype(str).unique(), key=_condition_sort_key)
	for ax, axis in zip(axes, ["condition", "sex"]):
		if axis == "condition":
			labels = conditions
			xs = np.arange(len(labels), dtype=float)
			for idx, label in enumerate(labels):
				vals = pd.to_numeric(open_df[open_df["condition"].astype(str).eq(label)]["spatial_preference_index"], errors="coerce")
				ax.bar(xs[idx], vals.mean(), width=0.62, color=CONDITION_COLORS.get(label, "#777777"), alpha=0.85)
				ax.errorbar(xs[idx], vals.mean(), yerr=_sem(vals), fmt="none", color="black", capsize=4)
				points = vals.dropna().to_numpy(dtype=float)
				if len(points):
					jitter = np.linspace(-0.08, 0.08, len(points))
					ax.scatter(np.full(len(points), xs[idx]) + jitter, points, color=CONDITION_COLORS.get(label, "#777777"), edgecolors="black", linewidths=0.35, s=24)
			ax.set_xticks(xs)
			ax.set_xticklabels(labels)
		else:
			labels = SEX_ORDER
			xs = np.arange(len(labels), dtype=float)
			width = 0.28
			for cond_idx, condition in enumerate(conditions):
				offset = (cond_idx - (len(conditions) - 1) / 2.0) * width
				for idx, sex in enumerate(labels):
					vals = pd.to_numeric(open_df[open_df["condition"].astype(str).eq(condition) & open_df["sex"].astype(str).str.lower().eq(sex)]["spatial_preference_index"], errors="coerce")
					if vals.dropna().empty:
						continue
					ax.bar(xs[idx] + offset, vals.mean(), width=width, color=CONDITION_COLORS.get(condition, "#777777"), alpha=0.85, label=condition if idx == 0 else None)
					ax.errorbar(xs[idx] + offset, vals.mean(), yerr=_sem(vals), fmt="none", color="black", capsize=3)
					points = vals.dropna().to_numpy(dtype=float)
					jitter = np.linspace(-0.04, 0.04, len(points))
					ax.scatter(np.full(len(points), xs[idx] + offset) + jitter, points, color=CONDITION_COLORS.get(condition, "#777777"), edgecolors="black", linewidths=0.35, s=24)
			ax.set_xticks(xs)
			ax.set_xticklabels(["M", "F"])
			ax.legend(frameon=False)
		ax.axhline(0.0, color="#666666", linewidth=1.0)
		ax.set_title(f"Open habituation bias by {axis}", fontsize=15)
		ax.set_ylabel("Right-side preference index")
		ax.grid(axis="y", alpha=0.25)
	fig.tight_layout()
	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output_path, dpi=FIGURE_DPI, facecolor="white")
	plt.close(fig)
