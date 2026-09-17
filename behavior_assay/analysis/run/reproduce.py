"""Paper command: load Final inputs, dispatch each assay, verify and export."""
from __future__ import annotations
import argparse
import importlib
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.paths import FINAL, REPRODUCED
from analysis.core.common import read, validate_table

PANELS = ['Fig5B', 'Fig5C', 'Fig5E', 'Fig5F', 'Fig5H', 'FigS8A', 'FigS8B', 'FigS8C', 'FigS8D', 'FigS10A', 'FigS10B']
ASSAYS = {p: ('mom_pup' if p in ['Fig5E', 'Fig5F', 'Fig5H'] else 'oft' if p.startswith('FigS10') else 'three_chamber') for p in PANELS}


def primary_tests(final, panel):
    return importlib.import_module(f'analysis.core.{ASSAYS[panel]}.paper').compute(final, panel)


def plot_panel(final, panel, output, data, tables):
    return importlib.import_module(f'analysis.plots.{ASSAYS[panel]}.paper').render(final, panel, output, data, tables)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panels', nargs='+', choices=PANELS, default=PANELS)
    parser.add_argument('--mode', choices=['all', 'stats', 'figures'], default='all')
    parser.add_argument('--final-dir', type=Path, default=FINAL)
    parser.add_argument('--output-dir', type=Path, default=REPRODUCED)
    args = parser.parse_args(argv)
    final, output = args.final_dir.resolve(), args.output_dir.resolve()
    if output == final or final in output.parents:
        parser.error('Use an output directory outside the adopted Final input folder.')
    matplotlib.rcParams.update({'font.family': 'Arial', 'svg.fonttype': 'none', 'pdf.fonttype': 42, 'ps.fonttype': 42})
    output.mkdir(parents=True, exist_ok=True)
    if args.mode != 'stats' and set(args.panels) & {'Fig5B', 'FigS8A'} and not (final / 'inputs/three_chamber/spatial_maps.npz').exists():
        from analysis.run.prepare_density import prepare
        prepare(final)
    report = []
    figures = []
    for panel in args.panels:
        print(f'{panel}: reading Final inputs', flush=True)
        data, tables = primary_tests(final, panel)
        for name, frame in tables.items():
            validate_table(frame, read(final, panel, 'stat', name), panel + '/' + name, report)
            if args.mode != 'figures':
                dest = output / panel / 'stat' / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                frame.to_csv(dest, index=False)
        if args.mode != 'stats':
            figures.append({'panel': panel, 'status': plot_panel(final, panel, output, data, tables)})
    (output / 'verification.json').write_text(json.dumps({'input_root': str(final), 'statistical_comparisons': report,
        'figures': figures, 'scope': 'Final numerical data to statistics and figures; full raw metric recomputation is not part of this command.'}, indent=2), encoding='utf-8')
    print(f'Finished: {len(report)} statistic tables matched Final. Outputs: {output}', flush=True)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
