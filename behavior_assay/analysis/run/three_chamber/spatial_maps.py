from __future__ import annotations
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from analysis.core.three_chamber.spatial_maps import compute_spatial_maps
import argparse
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from analysis.core.three_chamber.homography import CANONICAL_HEIGHT, CANONICAL_WIDTH, apply_homography, wall_homography
from analysis.core.three_chamber.spatial_maps import _buffer_polygon
from analysis.core.three_chamber.spatial_maps import _condition_sex_label
from analysis.core.three_chamber.spatial_maps import _condition_sort_key
from analysis.core.three_chamber.spatial_maps import _display_path
from analysis.core.three_chamber.spatial_maps import _filter_first_minutes
from analysis.core.three_chamber.spatial_maps import _format_minutes_suffix
from analysis.core.three_chamber.spatial_maps import _hist2d_counts
from analysis.core.three_chamber.spatial_maps import _hist2d_prob
from analysis.core.three_chamber.spatial_maps import _make_group_heat
from analysis.core.three_chamber.spatial_maps import _normalize_session_n
from analysis.core.three_chamber.spatial_maps import _order_polygon
from analysis.core.three_chamber.spatial_maps import _parse_phase_list
from analysis.core.three_chamber.spatial_maps import _parse_session_n_filter
from analysis.core.three_chamber.spatial_maps import _point_in_polygon_mask
from analysis.core.three_chamber.spatial_maps import _resample_polygon
from analysis.core.three_chamber.spatial_maps import _roi_output_tag
from analysis.plots.three_chamber.spatial_maps import _save_combined_figure
from analysis.plots.three_chamber.spatial_maps import _save_phase_figure
from analysis.core.three_chamber.spatial_maps import _sex_sort_key
from analysis.core.three_chamber.spatial_maps import _to_abs_path
from analysis.core.three_chamber.spatial_maps import _transform_polygon
from analysis.core.three_chamber.spatial_maps import SEX_ORDER

def main(argv: list[str] | None=None) -> int:
    parser = argparse.ArgumentParser(description='Plot target-aligned 3-chamber spatial occupancy maps.')
    parser.add_argument('--manifest', default='output/3chamber/manifest/session_manifest.csv', help='Manifest CSV produced by build_manifest.py')
    parser.add_argument('--preprocess-summary', default='output/3chamber/preprocessed/preprocess_summary.csv', help='Preprocess summary CSV produced by preprocess_sessions.py')
    parser.add_argument('--roi-vertices', default='output/3chamber/roi/roi_vertices.csv', help='ROI vertices CSV produced by build_roi_geometry.py')
    parser.add_argument('--roi-summary', default='output/3chamber/roi/roi_summary.csv', help='ROI summary CSV produced by build_roi_geometry.py')
    parser.add_argument('--output-dir', default='output/3chamber/spatial_maps', help='Directory to write spatial map figures and summaries')
    parser.add_argument('--keypoint', default='Body_C', help='Keypoint to plot, e.g. Body_C or Nose')
    parser.add_argument('--phases', default='soc,nov', help='Comma-separated phase list to plot')
    parser.add_argument('--bins', type=int, default=140, help='2D histogram bin count per axis')
    parser.add_argument('--sigma', type=float, default=1.8, help='Gaussian smoothing sigma in bins')
    parser.add_argument('--fps', type=float, default=30.0, help='Frames per second for time-window filtering')
    parser.add_argument('--max-minutes', type=float, default=0.0, help='Plot only the first N minutes of each recording. Default 0 uses the full session.')
    parser.add_argument('--density-mode', default='pooled', choices=['pooled', 'mean_session'], help='pooled sums all frames before normalization; mean_session averages per-session probability maps.')
    parser.add_argument('--roi-mode', default='contact', choices=['contact', 'circle'], help='ROI overlay definition. contact buffers the pinned cup polygon; circle uses the previous radius overlay.')
    parser.add_argument('--roi-radius-scale', type=float, default=1.2, help='ROI scale. contact uses (scale - 1) as the polygon buffer fraction.')
    parser.add_argument('--target-side', default='right', choices=['right', 'left'], help='Align target side to the requested direction before aggregation')
    parser.add_argument('--coord-mode', default='normalized', choices=['normalized', 'wall_bbox', 'homography'], help='Coordinate mode. normalized uses x_norm/y_norm directly; wall_bbox aligns each wall bounding box; homography requires calibrated floor-plane wall corners.')
    parser.add_argument('--session-n', default='', help='Optional session number filter, e.g. 1, n_1, or 1,2')
    parser.add_argument('--display-width-height-ratio', type=float, default=1.5, help='Apparatus width:height ratio used only for map rendering (default 60:40 = 1.5).')
    parser.add_argument('--phase-font-scale', type=float, default=1.0, help='Scale title, group, and ROI label fonts in per-phase figures (default: 1.0).')
    parser.add_argument('--exclude-body-in-cup', action=argparse.BooleanOptionalAction, default=True, help='Exclude trajectory frames when Body_C is inside either pinned cup polygon (default: enabled).')
    parser.add_argument('--cache-only', action='store_true', help='Export numerical maps and overlays for the paper runner without rendering exploratory figures.')
    args = parser.parse_args(argv)
    result = compute_spatial_maps(args)
    file_suffix = result['file_suffix']
    group_analysis_circles = result['group_analysis_circles']
    group_analysis_polygon_df = result['group_analysis_polygon_df']
    group_analysis_polygons = result['group_analysis_polygons']
    group_heatmaps = result['group_heatmaps']
    group_object_circles = result['group_object_circles']
    group_polygon_df = result['group_polygon_df']
    group_polygons = result['group_polygons']
    group_roi_summary_df = result['group_roi_summary_df']
    group_session_histograms = result['group_session_histograms']
    group_summary_rows = result['group_summary_rows']
    group_support_masks = result['group_support_masks']
    issue_rows = result['issue_rows']
    output_dir = result['output_dir']
    phase_list = result['phase_list']
    session_n_filter = result['session_n_filter']
    session_summary_rows = result['session_summary_rows']
    sessions_df = result['sessions_df']
    sex_group_analysis_circles = result['sex_group_analysis_circles']
    sex_group_analysis_polygon_df = result['sex_group_analysis_polygon_df']
    sex_group_analysis_polygons = result['sex_group_analysis_polygons']
    sex_group_heatmaps = result['sex_group_heatmaps']
    sex_group_object_circles = result['sex_group_object_circles']
    sex_group_polygon_df = result['sex_group_polygon_df']
    sex_group_polygons = result['sex_group_polygons']
    sex_group_roi_summary_df = result['sex_group_roi_summary_df']
    sex_group_summary_rows = result['sex_group_summary_rows']
    sex_group_support_masks = result['sex_group_support_masks']
    if args.cache_only:
        if issue_rows or len(session_summary_rows) != len(sessions_df):
            raise RuntimeError(f'Incomplete density inputs: {issue_rows}')
        arrays = {}
        for (prefix, values) in (('heat', group_heatmaps), ('support', group_support_masks), ('object_polygon', group_polygons), ('analysis_polygon', group_analysis_polygons)):
            for (key, value) in values.items():
                arrays[prefix + '__' + '__'.join(key)] = np.asarray(value)
        for (key, values) in group_session_histograms.items():
            arrays['count__' + '__'.join(key)] = np.asarray(len(values))
        np.savez_compressed(output_dir / 'spatial_maps.npz', **arrays)
        pd.DataFrame(group_summary_rows).to_csv(output_dir / 'group_summary.csv', index=False)
        pd.DataFrame(session_summary_rows).to_csv(output_dir / 'sessions.csv', index=False)
        print(f"Exported {len(session_summary_rows)} sessions to {output_dir / 'spatial_maps.npz'}")
        return 0
    condition_order = sorted({str(c) for c in sessions_df['condition'].dropna().unique()}, key=_condition_sort_key)
    sex_values = sorted({str(s).lower() for s in sessions_df['sex'].dropna().unique() if str(s).lower() in SEX_ORDER}, key=_sex_sort_key)
    condition_sex_order = [_condition_sex_label(condition, sex) for condition in condition_order for sex in sex_values if any((key[1] == _condition_sex_label(condition, sex) for key in sex_group_heatmaps))]
    figure_paths: list[Path] = []
    available_phases = [phase for phase in phase_list if any((key[0] == phase for key in group_heatmaps))]
    if available_phases:
        group_counts = {group_key: len(session_arrays) for (group_key, session_arrays) in group_session_histograms.items()}
        for phase in available_phases:
            phase_path = _save_phase_figure(phase=phase, condition_order=condition_order, group_heatmaps=group_heatmaps, group_support_masks=group_support_masks, group_counts=group_counts, group_object_circles=group_object_circles, group_analysis_circles=group_analysis_circles, group_object_polygons=group_polygons if args.roi_mode == 'contact' else None, group_analysis_polygons=group_analysis_polygons if args.roi_mode == 'contact' else None, output_dir=output_dir, keypoint=args.keypoint, target_side=args.target_side, roi_radius_scale=args.roi_radius_scale, file_suffix=file_suffix, x_max=CANONICAL_WIDTH if args.coord_mode == 'homography' else 1.0, display_width_height_ratio=float(args.display_width_height_ratio), font_scale=float(args.phase_font_scale))
            if phase_path is not None:
                figure_paths.append(phase_path)
        figure_paths.append(_save_combined_figure(phase_list=available_phases, condition_order=condition_order, group_heatmaps=group_heatmaps, group_support_masks=group_support_masks, group_object_circles=group_object_circles, group_analysis_circles=group_analysis_circles, group_object_polygons=group_polygons if args.roi_mode == 'contact' else None, group_analysis_polygons=group_analysis_polygons if args.roi_mode == 'contact' else None, output_dir=output_dir, keypoint=args.keypoint, target_side=args.target_side, roi_radius_scale=args.roi_radius_scale, density_mode=args.density_mode, file_suffix=file_suffix, x_max=CANONICAL_WIDTH if args.coord_mode == 'homography' else 1.0, display_width_height_ratio=float(args.display_width_height_ratio)))
    sex_available_phases = [phase for phase in phase_list if any((key[0] == phase for key in sex_group_heatmaps))]
    if sex_available_phases and condition_sex_order:
        sex_file_suffix = f'__by_sex{file_suffix}'
        figure_paths.append(_save_combined_figure(phase_list=sex_available_phases, condition_order=condition_sex_order, group_heatmaps=sex_group_heatmaps, group_support_masks=sex_group_support_masks, group_object_circles=sex_group_object_circles, group_analysis_circles=sex_group_analysis_circles, group_object_polygons=sex_group_polygons if args.roi_mode == 'contact' else None, group_analysis_polygons=sex_group_analysis_polygons if args.roi_mode == 'contact' else None, output_dir=output_dir, keypoint=args.keypoint, target_side=args.target_side, roi_radius_scale=args.roi_radius_scale, density_mode=args.density_mode, file_suffix=sex_file_suffix, x_max=CANONICAL_WIDTH if args.coord_mode == 'homography' else 1.0, display_width_height_ratio=float(args.display_width_height_ratio)))
    session_summary_df = pd.DataFrame(session_summary_rows)
    group_summary_df = pd.DataFrame(group_summary_rows)
    sex_group_summary_df = pd.DataFrame(sex_group_summary_rows)
    issue_df = pd.DataFrame(issue_rows)
    csv_suffix = f'__{_roi_output_tag(args.roi_mode, float(args.roi_radius_scale))}{file_suffix}'
    session_summary_path = output_dir / f'spatial_map_session_summary__{args.keypoint}{csv_suffix}.csv'
    group_summary_path = output_dir / f'spatial_map_group_summary__{args.keypoint}{csv_suffix}.csv'
    sex_group_summary_path = output_dir / f'spatial_map_sex_group_summary__{args.keypoint}{csv_suffix}.csv'
    group_polygon_path = output_dir / f'spatial_map_group_polygons__{args.keypoint}{csv_suffix}.csv'
    group_analysis_polygon_path = output_dir / f'spatial_map_group_analysis_polygons__{args.keypoint}{csv_suffix}.csv'
    sex_group_polygon_path = output_dir / f'spatial_map_sex_group_polygons__{args.keypoint}{csv_suffix}.csv'
    sex_group_analysis_polygon_path = output_dir / f'spatial_map_sex_group_analysis_polygons__{args.keypoint}{csv_suffix}.csv'
    group_roi_path = output_dir / f'spatial_map_group_roi_summary__{args.keypoint}{csv_suffix}.csv'
    sex_group_roi_path = output_dir / f'spatial_map_sex_group_roi_summary__{args.keypoint}{csv_suffix}.csv'
    issue_path = output_dir / f'spatial_map_issues__{args.keypoint}{csv_suffix}.csv'
    session_summary_df.to_csv(session_summary_path, index=False)
    group_summary_df.to_csv(group_summary_path, index=False)
    sex_group_summary_df.to_csv(sex_group_summary_path, index=False)
    group_polygon_df.to_csv(group_polygon_path, index=False)
    group_analysis_polygon_df.to_csv(group_analysis_polygon_path, index=False)
    sex_group_polygon_df.to_csv(sex_group_polygon_path, index=False)
    sex_group_analysis_polygon_df.to_csv(sex_group_analysis_polygon_path, index=False)
    group_roi_summary_df.to_csv(group_roi_path, index=False)
    sex_group_roi_summary_df.to_csv(sex_group_roi_path, index=False)
    issue_df.to_csv(issue_path, index=False)
    print(f'Keypoint: {args.keypoint}')
    print(f"Phases plotted: {', '.join(phase_list)}")
    print(f'Coordinate mode: {args.coord_mode}')
    print(f'ROI mode: {args.roi_mode}')
    print(f'ROI radius scale: {args.roi_radius_scale:.2f}')
    print(f'Density mode: {args.density_mode}')
    print(f'Display width:height ratio: {float(args.display_width_height_ratio):g}:1')
    print(f'Exclude {args.keypoint} inside pinned cup: no')
    print(f"Exclude Body_C inside pinned cup: {('yes' if bool(args.exclude_body_in_cup) else 'no')}")
    print(f'FPS: {float(args.fps):g}')
    if float(args.max_minutes) > 0:
        print(f'Analysis window: first {float(args.max_minutes):g} min')
    if session_n_filter:
        print(f"Session n filter: {', '.join(session_n_filter)}")
    print(f'Sessions used: {len(session_summary_df)}')
    print(f'Sessions with issues: {len(issue_df)}')
    for fig_path in figure_paths:
        print(f'Wrote {_display_path(fig_path)}')
    print(f'Wrote {_display_path(session_summary_path)}')
    print(f'Wrote {_display_path(group_summary_path)}')
    print(f'Wrote {_display_path(sex_group_summary_path)}')
    print(f'Wrote {_display_path(group_polygon_path)}')
    print(f'Wrote {_display_path(group_analysis_polygon_path)}')
    print(f'Wrote {_display_path(sex_group_polygon_path)}')
    print(f'Wrote {_display_path(sex_group_analysis_polygon_path)}')
    print(f'Wrote {_display_path(group_roi_path)}')
    print(f'Wrote {_display_path(sex_group_roi_path)}')
    print(f'Wrote {_display_path(issue_path)}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
