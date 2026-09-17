from __future__ import annotations
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import pandas as pd
from analysis.core.three_chamber.diagnose_qc_axes import _add_preference_metrics
from analysis.core.three_chamber.diagnose_qc_axes import _aggregate_locomotion
from analysis.core.three_chamber.diagnose_qc_axes import _aggregate_preference
from analysis.core.three_chamber.diagnose_qc_axes import _locomotion_stats
from analysis.plots.three_chamber.diagnose_qc_axes import _plot_locomotion
from analysis.plots.three_chamber.diagnose_qc_axes import _plot_open_bias
from analysis.plots.three_chamber.diagnose_qc_axes import _plot_preference_by_axis
from analysis.core.three_chamber.diagnose_qc_axes import _preference_stats
from analysis.core.three_chamber.diagnose_qc_axes import _summary_by
from analysis.core.three_chamber.diagnose_qc_axes import _to_abs_path
from analysis.core.three_chamber.diagnose_qc_axes import _write_report

import argparse

def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Diagnose batch/date, cage, sex, locomotion, and open-bias axes for 3-chamber data.")
	parser.add_argument("--preference-session", default="output/3chamber/barplots/preference_session_summary__Nose__circle_r1.50.csv")
	parser.add_argument("--locomotion-session", default="output/3chamber/locomotion/locomotion_session_summary__Body_C.csv")
	parser.add_argument("--output-dir", default="output/3chamber/diagnostics")
	return parser.parse_args()

def main() -> None:
	args = parse_args()
	output_dir = _to_abs_path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	pref_session = pd.read_csv(_to_abs_path(args.preference_session))
	pref_session = _add_preference_metrics(pref_session)
	pref_sample = _aggregate_preference(pref_session)
	loco_session = pd.read_csv(_to_abs_path(args.locomotion_session))
	loco_sample = _aggregate_locomotion(loco_session)

	pref_group = _summary_by(pref_sample, ["phase", "condition"], ["preference_index", "target_minus_opposite_time_s"])
	pref_date = _summary_by(pref_sample, ["phase", "condition", "date"], ["preference_index", "target_minus_opposite_time_s"])
	pref_cage = _summary_by(pref_sample, ["phase", "condition", "cage_id"], ["preference_index", "target_minus_opposite_time_s"])
	pref_sex = _summary_by(pref_sample, ["phase", "condition", "sex"], ["preference_index", "target_minus_opposite_time_s"])
	pref_stats = _preference_stats(pref_sample)
	loco_group = _summary_by(loco_sample, ["phase", "condition"], ["distance_per_min_norm", "total_distance_norm", "spatial_preference_index"])
	loco_date = _summary_by(loco_sample, ["phase", "condition", "date"], ["distance_per_min_norm", "total_distance_norm", "spatial_preference_index"])
	loco_sex = _summary_by(loco_sample, ["phase", "condition", "sex"], ["distance_per_min_norm", "total_distance_norm", "spatial_preference_index"])
	loco_stats = _locomotion_stats(loco_sample)
	condition_date = pref_sample.groupby(["condition", "date"], dropna=False)["sample_key"].nunique().reset_index(name="n_samples")

	paths = {
		"preference_sample": output_dir / "diagnostic_preference_sample__Nose__circle_r1.50.csv",
		"preference_group": output_dir / "diagnostic_preference_group_summary__Nose__circle_r1.50.csv",
		"preference_date": output_dir / "diagnostic_preference_by_date__Nose__circle_r1.50.csv",
		"preference_cage": output_dir / "diagnostic_preference_by_cage__Nose__circle_r1.50.csv",
		"preference_sex": output_dir / "diagnostic_preference_by_sex__Nose__circle_r1.50.csv",
		"preference_stats": output_dir / "diagnostic_preference_stats__Nose__circle_r1.50.csv",
		"locomotion_sample": output_dir / "diagnostic_locomotion_sample__Body_C.csv",
		"locomotion_group": output_dir / "diagnostic_locomotion_group_summary__Body_C.csv",
		"locomotion_date": output_dir / "diagnostic_locomotion_by_date__Body_C.csv",
		"locomotion_sex": output_dir / "diagnostic_locomotion_by_sex__Body_C.csv",
		"locomotion_stats": output_dir / "diagnostic_locomotion_stats__Body_C.csv",
		"condition_date": output_dir / "diagnostic_condition_date_balance.csv",
		"report": output_dir / "diagnostic_report.md",
		"pref_date_plot": output_dir / "diagnostic_preference_by_date__Nose__circle_r1.50.png",
		"pref_sex_plot": output_dir / "diagnostic_preference_by_sex__Nose__circle_r1.50.png",
		"loco_plot": output_dir / "diagnostic_locomotion_qc__Body_C.png",
		"open_plot": output_dir / "diagnostic_open_bias__Body_C.png",
	}
	pref_sample.to_csv(paths["preference_sample"], index=False)
	pref_group.to_csv(paths["preference_group"], index=False)
	pref_date.to_csv(paths["preference_date"], index=False)
	pref_cage.to_csv(paths["preference_cage"], index=False)
	pref_sex.to_csv(paths["preference_sex"], index=False)
	pref_stats.to_csv(paths["preference_stats"], index=False)
	loco_sample.to_csv(paths["locomotion_sample"], index=False)
	loco_group.to_csv(paths["locomotion_group"], index=False)
	loco_date.to_csv(paths["locomotion_date"], index=False)
	loco_sex.to_csv(paths["locomotion_sex"], index=False)
	loco_stats.to_csv(paths["locomotion_stats"], index=False)
	condition_date.to_csv(paths["condition_date"], index=False)
	_plot_preference_by_axis(pref_sample, "date", paths["pref_date_plot"])
	_plot_preference_by_axis(pref_sample, "sex", paths["pref_sex_plot"])
	_plot_locomotion(loco_sample, paths["loco_plot"])
	_plot_open_bias(loco_sample, paths["open_plot"])
	_write_report(paths["report"], pref_sample, pref_stats, loco_stats, condition_date)

	print(f"Preference samples: {pref_sample['sample_key'].nunique()}")
	print(f"Locomotion samples: {loco_sample['sample_key'].nunique()}")
	print("Condition/date balance:")
	print(condition_date.to_string(index=False))
	print("\nPreference condition comparisons:")
	show_pref = pref_stats[pref_stats["axis"].eq("condition_comparison") & pref_stats["metric"].eq("preference_index")]
	print(show_pref[["phase", "comparison", "n_a", "n_b", "pvalue", "p_label"]].to_string(index=False))
	print("\nPreference condition comparisons within sex:")
	show_sex = pref_stats[pref_stats["axis"].eq("condition_comparison_within_sex") & pref_stats["metric"].eq("preference_index")]
	print(show_sex[["phase", "sex", "comparison", "n_a", "n_b", "pvalue", "p_label"]].to_string(index=False))
	print("\nLocomotion condition comparisons, distance_per_min_norm:")
	show_loco = loco_stats[loco_stats["axis"].eq("condition_comparison") & loco_stats["metric"].eq("distance_per_min_norm")]
	print(show_loco[["phase", "comparison", "n_a", "n_b", "pvalue", "p_label"]].to_string(index=False))
	for path in paths.values():
		print(f"Wrote {path.relative_to(ROOT)}")

if __name__ == '__main__':
    raise SystemExit(main())
