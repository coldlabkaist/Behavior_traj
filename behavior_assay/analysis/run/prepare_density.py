"""Build the missing three-chamber density grids from retained raw tracking."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.paths import FINAL
from analysis.run.three_chamber.preprocess_sessions import main as preprocess
from analysis.run.three_chamber.spatial_maps import main as spatial_maps


def prepare(final: Path = FINAL) -> None:
    inputs = final / 'inputs/three_chamber'
    # Temporary preprocessing stays inside the project (legacy exporter writes relative paths).
    work = ROOT / 'output'
    with tempfile.TemporaryDirectory(prefix='.density_', dir=work) as name:
        tmp = Path(name)
        if tmp.resolve().parent != work.resolve() or tmp.is_symlink():
            raise ValueError('Unexpected temporary preprocessing directory')
        manifest = pd.read_csv(inputs / 'session_manifest.csv')
        manifest = manifest.loc[manifest.session_n.eq(1) & manifest.phase.isin(['soc', 'nov'])]
        if len(manifest) != 82:
            raise ValueError(f'Expected 82 adopted sociability/novelty sessions; got {len(manifest)}')
        mp = tmp / 'manifest.csv'
        manifest.to_csv(mp, index=False)
        preprocess([str(mp), '--output-dir', str(tmp / 'preprocessed'), '--keypoints', 'Body_C,Nose',
                    '--method', 'linear', '--max-gap', '0', '--score-threshold', '0'])
        spatial_maps(['--manifest', str(mp), '--preprocess-summary', str(tmp / 'preprocessed/preprocess_summary.csv'),
                      '--roi-vertices', str(inputs / 'roi_vertices.csv'), '--roi-summary', str(inputs / 'roi_summary.csv'),
                      '--output-dir', str(tmp / 'maps'), '--keypoint', 'Nose', '--session-n', '1',
                      '--max-minutes', '5', '--coord-mode', 'normalized', '--cache-only'])
        actual = pd.read_csv(tmp / 'maps/group_summary.csv').sort_values(['phase', 'condition'])
        expected = pd.concat([pd.read_csv(final / panel / 'data/group_summary.csv') for panel in ['Fig5B', 'FigS8A']]).sort_values(['phase', 'condition'])
        for col in ['session_count', 'heatmap_max', 'heatmap_sum']:
            np.testing.assert_allclose(actual[col], expected[col], rtol=1e-10, atol=1e-12, err_msg=col)
        actual_sessions = pd.read_csv(tmp / 'maps/sessions.csv').sort_values('session_key')
        expected_sessions = pd.concat([pd.read_csv(final / panel / 'data/sessions.csv') for panel in ['Fig5B', 'FigS8A']]).sort_values('session_key')
        assert actual_sessions.session_key.tolist() == expected_sessions.session_key.tolist()
        np.testing.assert_array_equal(actual_sessions.valid_points, expected_sessions.valid_points)
        target = inputs / 'spatial_maps.npz'
        target.write_bytes((tmp / 'maps/spatial_maps.npz').read_bytes())
        report = {'source': 'Raw tracking in data/3chamber; Final session manifest and ROI vertices',
                  'sessions': 82, 'bins': 140, 'sigma': 1.8, 'coordinate_mode': 'normalized',
                  'density_mode': 'pooled', 'keypoint': 'Nose', 'minutes': 5, 'fps': 30,
                  'interpolation': 'Body_C,Nose; linear; unlimited interior gaps; no smoothing',
                  'exclude_body_in_cup': True, 'group_summaries_match': True,
                  'per_session_valid_point_counts_match': True,
                  'sha256': hashlib.sha256(target.read_bytes()).hexdigest()}
        (inputs / 'spatial_maps.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Three-chamber density cache verified against adopted group summaries and all 82 session counts.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--final-dir', type=Path, default=FINAL)
    prepare(parser.parse_args().final_dir.resolve())
