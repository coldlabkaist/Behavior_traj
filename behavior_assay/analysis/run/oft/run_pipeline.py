from __future__ import annotations
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import analysis.run.oft.analyze_sessions as analyze_sessions
import analysis.run.oft.build_manifest as build_manifest
import analysis.run.oft.group_metrics as plot_group_metrics
import analysis.run.oft.heatmaps as plot_heatmaps
import analysis.run.oft.preprocess_sessions as preprocess_sessions

def main() -> int:
	stages = (
		("manifest", build_manifest.main),
		("preprocess", preprocess_sessions.main),
		("metrics and statistics", analyze_sessions.main),
		("metric figures", plot_group_metrics.main),
		("heatmaps", plot_heatmaps.main),
	)
	for label, stage in stages:
		print(f"\n=== OFT: {label} ===")
		status = int(stage([]))
		if status:
			print(f"Stopped after failed stage: {label}")
			return status
	return 0

if __name__ == '__main__':
    raise SystemExit(main())
