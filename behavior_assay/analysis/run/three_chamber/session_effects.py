from __future__ import annotations
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import pandas as pd
from analysis.core.three_chamber.session_effects import _condition_sort_key
from analysis.core.three_chamber.session_effects import _default_input_path
from analysis.core.three_chamber.session_effects import _pair_sessions
from analysis.core.three_chamber.session_effects import _parse_phase_list
from analysis.plots.three_chamber.session_effects import _plot_phase
from analysis.core.three_chamber.session_effects import _prepare_session_table
from analysis.plots.three_chamber.session_effects import _summarize_stats
from analysis.core.three_chamber.session_effects import _to_abs_path

import argparse

def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Plot n1 vs n2 QC for 3-chamber cup ROI preference metrics.")
	parser.add_argument("--keypoint", default="Body_C", help="Keypoint name used in preference_session_summary.")
	parser.add_argument("--roi-mode", default="circle", choices=["circle", "polygon"], help="ROI mode used for preference summary.")
	parser.add_argument("--radius-scale", type=float, default=2.5, help="Circle ROI radius scale used for preference summary.")
	parser.add_argument("--input", default=None, help="Optional preference_session_summary CSV.")
	parser.add_argument("--output-dir", default=str(ROOT / "output" / "3chamber" / "session_effects"))
	parser.add_argument("--phases", default="soc,nov", help="Comma-separated phases to plot.")
	return parser.parse_args()

def main() -> None:
	args = parse_args()
	phases = _parse_phase_list(args.phases)
	input_path = _to_abs_path(args.input) if args.input else _default_input_path(args.keypoint, args.roi_mode, args.radius_scale)
	output_dir = _to_abs_path(args.output_dir)
	session_df = pd.read_csv(input_path)
	session_df = _prepare_session_table(session_df)
	session_df = session_df[session_df["phase"].isin(phases)].copy()
	pairs_df, issues_df = _pair_sessions(session_df)
	stats_df = _summarize_stats(pairs_df) if not pairs_df.empty else pd.DataFrame()

	suffix = f"{args.keypoint}__{args.roi_mode}_r{args.radius_scale:.2f}"
	pairs_path = output_dir / f"session_effect_pairs__{suffix}.csv"
	stats_path = output_dir / f"session_effect_stats__{suffix}.csv"
	issues_path = output_dir / f"session_effect_issues__{suffix}.csv"
	output_dir.mkdir(parents=True, exist_ok=True)
	pairs_df.to_csv(pairs_path, index=False)
	stats_df.to_csv(stats_path, index=False)
	issues_df.to_csv(issues_path, index=False)

	conditions = sorted(pairs_df["condition"].dropna().unique(), key=_condition_sort_key) if not pairs_df.empty else []
	for phase in phases:
		if phase not in pairs_df["phase"].unique():
			continue
		fig_path = output_dir / f"session_effect__{phase}__{suffix}.png"
		_plot_phase(pairs_df, stats_df, phase, conditions, args, fig_path)
		print(f"Wrote {fig_path.relative_to(ROOT)}")

	print(f"Input {input_path.relative_to(ROOT) if input_path.is_relative_to(ROOT) else input_path}")
	print(f"Paired samples: {len(pairs_df)}")
	print(f"Pairing issues: {len(issues_df)}")
	print(f"Wrote {pairs_path.relative_to(ROOT)}")
	print(f"Wrote {stats_path.relative_to(ROOT)}")
	print(f"Wrote {issues_path.relative_to(ROOT)}")
	if not stats_df.empty:
		show = stats_df[stats_df["metric"].isin(["preference_index", "target_time_s", "opposite_time_s"])].copy()
		show = show[["phase", "condition", "metric", "n", "mean_n1", "mean_n2", "mean_delta_n2_minus_n1", "pvalue", "p_label"]]
		print(show.to_string(index=False))

if __name__ == '__main__':
    raise SystemExit(main())
