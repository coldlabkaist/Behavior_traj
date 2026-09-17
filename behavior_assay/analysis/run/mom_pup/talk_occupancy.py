from __future__ import annotations
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import argparse
from pathlib import Path
import matplotlib
from analysis.plots.mom_pup.talk_occupancy import plot_talk_occupancy

matplotlib.use("Agg")

def main(argv: list[str] | None = None) -> int:
	parser = argparse.ArgumentParser(description="Create the talk-ready 2 x 6 mother-pup occupancy panel.")
	parser.add_argument(
		"--manifest",
		default=str(ROOT / "output" / "mom_pup" / "manifest" / "session_manifest.csv"),
	)
	parser.add_argument(
		"--output-dir",
		default=str(ROOT / "work" / "talkfigure"),
	)
	parser.add_argument("--bins", type=int, default=100)
	parser.add_argument("--vmax-quantile", type=float, default=0.995)
	args = parser.parse_args(argv)

	paths = plot_talk_occupancy(
		Path(args.manifest),
		Path(args.output_dir),
		bins=args.bins,
		vmax_quantile=args.vmax_quantile,
	)
	for path in paths:
		print(path)
	return 0

if __name__ == '__main__':
    raise SystemExit(main())
