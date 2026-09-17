from __future__ import annotations
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import analysis.run.mom_pup.behavior_metrics as behavior_metrics
import analysis.run.mom_pup.body_scale_proximity as body_scale_proximity
import analysis.run.mom_pup.build_manifest as build_manifest
import analysis.run.mom_pup.cluster_zone_movement as cluster_zone_movement
import analysis.run.mom_pup.development as plot_development
import analysis.run.mom_pup.focused_results as plot_focused_results
import analysis.run.mom_pup.pup_zone_dwell as pup_zone_dwell

def main() -> int:
	stages = (
		("manifest", build_manifest.main),
		("behavior metrics", behavior_metrics.main),
		("body-scale proximity", body_scale_proximity.main),
		("cluster-zone movement", cluster_zone_movement.main),
		("pup-zone dwell", pup_zone_dwell.main),
		("developmental statistics and figures", plot_development.main),
		("focused result figures", plot_focused_results.main),
	)
	for label, stage in stages:
		print(f"\n=== Mom-pup: {label} ===")
		status = int(stage([]))
		if status:
			print(f"Stopped after failed stage: {label}")
			return status
	return 0

if __name__ == '__main__':
    raise SystemExit(main())
