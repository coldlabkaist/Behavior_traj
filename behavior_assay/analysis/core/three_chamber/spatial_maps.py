from __future__ import annotations
from pathlib import Path
import sys
from analysis.paths import ROOT, resolve_input_path
from pathlib import Path
import numpy as np
import pandas as pd
from analysis.core.three_chamber.contact_geometry import buffer_polygon

FIGURE_DPI = 1200

PHASE_LABELS = {
	"soc": ("E", "S"),
	"nov": ("F", "N"),
}

PHASE_TITLES = {
	"soc": "SOC occupancy",
	"nov": "NOV occupancy",
}

PHASE_FIGURE_TITLES = {
	"soc": "Spatial Map of Sociability Test",
	"nov": "Spatial Map of Social Recognition Test",
}

PREFERRED_CONDITION_ORDER = ["Control", "control", "VPA", "vpa"]

SEX_ORDER = ["m", "f"]

def _to_abs_path(path_str: str) -> Path:
    return resolve_input_path(path_str)

def _display_path(path: Path) -> str:
	try:
		return path.relative_to(ROOT).as_posix()
	except ValueError:
		return str(path)

def _parse_phase_list(arg: str) -> list[str]:
	phases = [token.strip().lower() for token in str(arg).split(",") if token.strip()]
	if not phases:
		raise ValueError("At least one phase must be provided")
	return phases

def _normalize_session_n(value: object) -> str | None:
	if value is None or pd.isna(value):
		return None
	text = str(value).strip().lower()
	if not text:
		return None
	if text.startswith("n_"):
		text = text[2:]
	elif text.startswith("n"):
		text = text[1:]
	try:
		num = float(text)
	except ValueError:
		return None
	if not np.isfinite(num):
		return None
	if abs(num - round(num)) < 1e-9:
		return str(int(round(num)))
	return str(num)

def _parse_session_n_filter(arg: str) -> list[str]:
	values = []
	for token in str(arg).split(","):
		norm = _normalize_session_n(token)
		if norm is not None and norm not in values:
			values.append(norm)
	return values

def _format_minutes_suffix(max_minutes: float) -> str:
	if not np.isfinite(max_minutes) or max_minutes <= 0:
		return ""
	if abs(max_minutes - round(max_minutes)) < 1e-9:
		return f"__first{int(round(max_minutes))}min"
	return f"__first{max_minutes:g}min".replace(".", "p")

def _filter_first_minutes(df: pd.DataFrame, *, max_minutes: float, fps: float) -> pd.DataFrame:
	if not np.isfinite(max_minutes) or max_minutes <= 0:
		return df
	if "frame_idx" in df.columns:
		frame_series = pd.to_numeric(df["frame_idx"], errors="coerce")
	elif "frame" in df.columns:
		frame_series = pd.to_numeric(df["frame"], errors="coerce")
	else:
		return df
	max_frames = int(round(max_minutes * 60.0 * max(float(fps), 1e-9)))
	if max_frames <= 0:
		return df.iloc[0:0].copy()
	return df[frame_series < max_frames].copy()

def _condition_sort_key(condition: str) -> tuple[int, str]:
	if condition in PREFERRED_CONDITION_ORDER:
		return (PREFERRED_CONDITION_ORDER.index(condition), condition)
	return (len(PREFERRED_CONDITION_ORDER), condition)

def _sex_sort_key(sex: str) -> tuple[int, str]:
	sex = str(sex).lower()
	if sex in SEX_ORDER:
		return (SEX_ORDER.index(sex), sex)
	return (len(SEX_ORDER), sex)

def _sex_label(sex: str) -> str:
	sex = str(sex).lower()
	if sex == "m":
		return "M"
	if sex == "f":
		return "F"
	return sex.upper() if sex else "?"

def _condition_sex_label(condition: str, sex: str) -> str:
	return f"{condition} {_sex_label(sex)}"

def _gaussian_kernel1d(sigma: float) -> np.ndarray:
	if sigma <= 0:
		return np.array([1.0], dtype=float)
	radius = int(max(1, round(3.0 * sigma)))
	x = np.arange(-radius, radius + 1, dtype=float)
	k = np.exp(-(x * x) / (2.0 * sigma * sigma))
	k /= float(k.sum())
	return k

def _smooth2d(h: np.ndarray, *, sigma: float) -> np.ndarray:
	if sigma <= 0:
		return h
	k = _gaussian_kernel1d(float(sigma))
	radius = (len(k) - 1) // 2

	def _conv_axis(a: np.ndarray, axis: int) -> np.ndarray:
		pad_width = [(0, 0)] * a.ndim
		pad_width[axis] = (radius, radius)
		ap = np.pad(a, pad_width, mode="reflect")
		return np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), axis, ap)

	out = _conv_axis(h, axis=0)
	out = _conv_axis(out, axis=1)
	return out

def _hist2d_counts(x: np.ndarray, y: np.ndarray, *, bins: int, x_max: float = 1.0) -> np.ndarray:
	x_bins = max(int(round(float(bins) * float(x_max))), 1)
	h, _, _ = np.histogram2d(x, y, bins=[x_bins, bins], range=[[0.0, x_max], [0.0, 1.0]])
	return h

def _hist2d_prob(x: np.ndarray, y: np.ndarray, *, bins: int, x_max: float = 1.0) -> np.ndarray:
	h = _hist2d_counts(x, y, bins=bins, x_max=x_max)
	total = float(h.sum())
	return (h / total) if total > 0 else h

def _point_in_polygon_mask(x: np.ndarray, y: np.ndarray, polygon_xy: np.ndarray) -> np.ndarray:
	if polygon_xy.shape[0] < 3:
		return np.zeros(len(x), dtype=bool)
	poly = _order_polygon(polygon_xy.astype(float))
	px = poly[:, 0]
	py = poly[:, 1]
	px_next = np.roll(px, -1)
	py_next = np.roll(py, -1)
	inside = np.zeros(len(x), dtype=bool)
	for x1, y1, x2, y2 in zip(px, py, px_next, py_next):
		crosses = (y1 > y) != (y2 > y)
		x_intersect = (x2 - x1) * (y - y1) / ((y2 - y1) + 1e-12) + x1
		inside ^= crosses & (x < x_intersect)
	return inside

def _order_polygon(vertices_xy: np.ndarray) -> np.ndarray:
	if vertices_xy.size == 0:
		return vertices_xy
	centroid = vertices_xy.mean(axis=0)
	angles = np.arctan2(vertices_xy[:, 1] - centroid[1], vertices_xy[:, 0] - centroid[0])
	order = np.argsort(angles)
	return vertices_xy[order]

def _buffer_polygon(vertices_xy: np.ndarray, distance: float) -> np.ndarray:
	return buffer_polygon(vertices_xy, distance, resolution=16)

def _resample_polygon(vertices_xy: np.ndarray, count: int = 180) -> np.ndarray:
	poly = _order_polygon(np.asarray(vertices_xy, dtype=float))
	poly = poly[np.isfinite(poly).all(axis=1)]
	if poly.shape[0] < 3:
		return np.empty((0, 2), dtype=float)
	closed = np.vstack([poly, poly[0]])
	segments = np.sqrt(np.sum(np.diff(closed, axis=0) ** 2, axis=1))
	total = float(np.sum(segments))
	if total <= 0:
		return np.empty((0, 2), dtype=float)
	cumulative = np.concatenate([[0.0], np.cumsum(segments)])
	targets = np.linspace(0.0, total, count, endpoint=False)
	out = np.empty((count, 2), dtype=float)
	for idx, target in enumerate(targets):
		seg_idx = min(int(np.searchsorted(cumulative, target, side="right") - 1), len(segments) - 1)
		fraction = (target - cumulative[seg_idx]) / max(segments[seg_idx], 1e-12)
		out[idx] = closed[seg_idx] + fraction * (closed[seg_idx + 1] - closed[seg_idx])
	return out

def _roi_output_tag(roi_mode: str, radius_scale: float) -> str:
	if roi_mode == "contact":
		return f"contact{int(round(max(radius_scale - 1.0, 0.0) * 100.0))}"
	return f"r{radius_scale:.2f}"

def _transform_polygon(
	vertices_xy: np.ndarray,
	*,
	wall_bounds: tuple[float, float, float, float],
	mirror_x: bool,
) -> np.ndarray:
	if vertices_xy.size == 0:
		return vertices_xy
	min_x, max_x, min_y, max_y = wall_bounds
	width = max(max_x - min_x, 1e-9)
	height = max(max_y - min_y, 1e-9)
	out = vertices_xy.astype(float).copy()
	out[:, 0] = (out[:, 0] - min_x) / width
	out[:, 1] = (out[:, 1] - min_y) / height
	if mirror_x:
		out[:, 0] = 1.0 - out[:, 0]
	return _order_polygon(out)

def _make_group_heat(
	group_arrays: list[np.ndarray],
	*,
	sigma: float,
	density_mode: str,
) -> np.ndarray:
	if not group_arrays:
		return np.zeros((1, 1), dtype=float)
	stack = np.stack(group_arrays, axis=0)
	if density_mode == "pooled":
		h = stack.sum(axis=0)
		total = float(h.sum())
		h = (h / total) if total > 0 else h
	else:
		h = stack.mean(axis=0)
	return _smooth2d(h, sigma=sigma) if sigma > 0 else h

def _roi_label_position(
	*,
	phase: str,
	condition: str,
	roi_id: str,
	group_object_circles: dict[tuple[str, str, str], tuple[float, float, float, float]],
	group_object_polygons: dict[tuple[str, str, str], np.ndarray] | None,
) -> tuple[float, float] | None:
	poly = (group_object_polygons or {}).get((phase, condition, roi_id))
	if poly is not None and poly.shape[0] >= 3:
		return float(np.mean(poly[:, 0])), float(np.mean(poly[:, 1]))
	circle = group_object_circles.get((phase, condition, roi_id))
	if circle is not None:
		return float(circle[0]), float(circle[1])
	return None


from pathlib import Path
import sys
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from analysis.core.three_chamber.homography import CANONICAL_HEIGHT, CANONICAL_WIDTH, apply_homography, wall_homography

def compute_spatial_maps(args):
    if args.roi_mode == 'contact' and float(args.roi_radius_scale) < 1.0:
        raise ValueError('contact ROI requires --roi-radius-scale >= 1.0')
    if not np.isfinite(args.display_width_height_ratio) or float(args.display_width_height_ratio) <= 0:
        raise ValueError('--display-width-height-ratio must be > 0')
    if not np.isfinite(args.phase_font_scale) or float(args.phase_font_scale) <= 0:
        raise ValueError('--phase-font-scale must be > 0')
    manifest_path = _to_abs_path(args.manifest)
    preprocess_path = _to_abs_path(args.preprocess_summary)
    roi_vertices_path = _to_abs_path(args.roi_vertices)
    roi_summary_path = _to_abs_path(args.roi_summary)
    output_dir = _to_abs_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    phase_list = _parse_phase_list(args.phases)
    session_n_filter = _parse_session_n_filter(args.session_n)
    file_suffix = '' if args.roi_mode == 'contact' or args.coord_mode == 'normalized' else '__raw' if args.coord_mode == 'wall_bbox' else '__homography'
    file_suffix += f"__n{'_'.join(session_n_filter)}" if session_n_filter else ''
    file_suffix += _format_minutes_suffix(float(args.max_minutes))
    for path in (manifest_path, preprocess_path, roi_vertices_path, roi_summary_path):
        if not path.exists():
            raise FileNotFoundError(f'Required input not found: {path}')
    manifest_df = pd.read_csv(manifest_path)
    preprocess_df = pd.read_csv(preprocess_path)
    roi_vertices_df = pd.read_csv(roi_vertices_path)
    roi_summary_df = pd.read_csv(roi_summary_path)
    sessions_df = preprocess_df.merge(manifest_df[['session_key', 'condition', 'phase', 'heatmap_target_side', 'mirror_lr_for_target_right', 'layout_raw', 'subject_id', 'trial_id', 'session_n']].drop_duplicates('session_key'), on=['session_key', 'condition', 'phase'], how='left', suffixes=('', '_manifest'))
    sessions_df = sessions_df[sessions_df['phase'].isin(phase_list)].copy()
    if 'sex' not in sessions_df.columns:
        sessions_df['sex'] = ''
    if session_n_filter:
        sessions_df['session_n_norm'] = sessions_df['session_n'].map(_normalize_session_n)
        sessions_df = sessions_df[sessions_df['session_n_norm'].isin(session_n_filter)].copy()
    if sessions_df.empty:
        print('No sessions matched the requested phases.')
        return 0
    xcol = f'{args.keypoint}.x'
    ycol = f'{args.keypoint}.y'
    group_session_histograms: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    group_session_supports: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    sex_group_session_histograms: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    sex_group_session_supports: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    group_polygon_rows: list[dict[str, object]] = []
    group_analysis_polygon_rows: list[dict[str, object]] = []
    sex_group_polygon_rows: list[dict[str, object]] = []
    sex_group_analysis_polygon_rows: list[dict[str, object]] = []
    group_roi_rows: list[dict[str, object]] = []
    sex_group_roi_rows: list[dict[str, object]] = []
    session_summary_rows: list[dict[str, object]] = []
    issue_rows: list[dict[str, object]] = []
    for row in sessions_df.to_dict(orient='records'):
        session_key = str(row['session_key'])
        condition = str(row.get('condition', ''))
        sex = str(row.get('sex', '')).lower().strip()
        phase = str(row.get('phase', '')).lower()
        output_path = _to_abs_path(str(row.get('output_path', '')))
        if not output_path.exists():
            issue_rows.append({'session_key': session_key, 'issue': 'preprocessed_file_not_found', 'detail': str(output_path)})
            continue
        session_roi = roi_vertices_df[roi_vertices_df['session_key'] == session_key].copy()
        if session_roi.empty:
            issue_rows.append({'session_key': session_key, 'issue': 'missing_roi_vertices', 'detail': ''})
            continue
        roi_polys: dict[str, np.ndarray] = {}
        roi_polys_raw: dict[str, np.ndarray] = {}
        for (roi_id, group) in session_roi.groupby('roi_id', sort=False):
            g = group.sort_values('vertex_idx')
            if not {'x_norm', 'y_norm'}.issubset(g.columns):
                continue
            roi_polys[str(roi_id)] = g[['x_norm', 'y_norm']].astype(float).to_numpy()
            if {'x', 'y'}.issubset(g.columns):
                roi_polys_raw[str(roi_id)] = g[['x', 'y']].astype(float).to_numpy()
        wall_poly = roi_polys.get('wall')
        wall_poly_raw = roi_polys_raw.get('wall')
        if wall_poly is None or wall_poly.shape[0] < 3:
            issue_rows.append({'session_key': session_key, 'issue': 'missing_wall_polygon', 'detail': ''})
            continue
        homography = None
        if args.coord_mode == 'homography':
            try:
                homography = wall_homography(wall_poly)
            except ValueError as exc:
                issue_rows.append({'session_key': session_key, 'issue': 'invalid_wall_homography', 'detail': str(exc)})
                continue
        df = pd.read_csv(output_path)
        df = _filter_first_minutes(df, max_minutes=float(args.max_minutes), fps=float(args.fps))
        if xcol not in df.columns or ycol not in df.columns:
            issue_rows.append({'session_key': session_key, 'issue': 'missing_keypoint_columns', 'detail': args.keypoint})
            continue
        if bool(args.exclude_body_in_cup) and ('Body_C.x' not in df.columns or 'Body_C.y' not in df.columns):
            issue_rows.append({'session_key': session_key, 'issue': 'missing_body_center_columns', 'detail': 'Body_C'})
            continue
        x = pd.to_numeric(df[xcol], errors='coerce').to_numpy(dtype=float)
        y = pd.to_numeric(df[ycol], errors='coerce').to_numpy(dtype=float)
        body_on_cup = np.zeros(len(df), dtype=bool)
        if bool(args.exclude_body_in_cup):
            body_x = pd.to_numeric(df['Body_C.x'], errors='coerce').to_numpy(dtype=float)
            body_y = pd.to_numeric(df['Body_C.y'], errors='coerce').to_numpy(dtype=float)
            body_finite = np.isfinite(body_x) & np.isfinite(body_y)
            for roi_id in ('chamber_l', 'chamber_r'):
                cup_poly = roi_polys.get(roi_id)
                if cup_poly is not None and cup_poly.shape[0] >= 3:
                    body_on_cup |= body_finite & _point_in_polygon_mask(body_x, body_y, cup_poly)
        finite_mask = np.isfinite(x) & np.isfinite(y)
        x = x[finite_mask]
        y = y[finite_mask]
        body_on_cup = body_on_cup[finite_mask]
        if x.size == 0:
            issue_rows.append({'session_key': session_key, 'issue': 'no_finite_points', 'detail': args.keypoint})
            continue
        if args.coord_mode == 'homography' and homography is not None:
            points = apply_homography(np.column_stack([x, y]), homography)
            x = points[:, 0]
            y = points[:, 1]
            wall_poly = apply_homography(wall_poly, homography)
            finite_after = np.isfinite(x) & np.isfinite(y)
            x = x[finite_after]
            y = y[finite_after]
            body_on_cup = body_on_cup[finite_after]
            if x.size == 0:
                issue_rows.append({'session_key': session_key, 'issue': 'no_finite_homography_points', 'detail': args.keypoint})
                continue
        in_wall = _point_in_polygon_mask(x, y, wall_poly)
        keypoint_in_cup_frames_removed = 0
        body_in_cup_frames_removed = int(np.sum(in_wall & body_on_cup))
        cup_exclusion_frames_removed = body_in_cup_frames_removed
        keep_points = in_wall & ~body_on_cup
        x = x[keep_points]
        y = y[keep_points]
        if x.size == 0:
            issue_rows.append({'session_key': session_key, 'issue': 'no_points_inside_wall', 'detail': args.keypoint})
            continue
        wall_bounds = (0.0, 1.0, 0.0, 1.0)
        wall_bounds_raw = (0.0, 1.0, 0.0, 1.0)
        if args.coord_mode in {'normalized', 'wall_bbox'}:
            wall_min_x = float(np.nanmin(wall_poly[:, 0]))
            wall_max_x = float(np.nanmax(wall_poly[:, 0]))
            wall_min_y = float(np.nanmin(wall_poly[:, 1]))
            wall_max_y = float(np.nanmax(wall_poly[:, 1]))
            wall_bounds = (wall_min_x, wall_max_x, wall_min_y, wall_max_y)
            if args.coord_mode == 'wall_bbox':
                if wall_poly_raw is None or wall_poly_raw.shape[0] < 3:
                    issue_rows.append({'session_key': session_key, 'issue': 'missing_raw_wall_polygon', 'detail': ''})
                    continue
                wall_bounds_raw = (float(np.nanmin(wall_poly_raw[:, 0])), float(np.nanmax(wall_poly_raw[:, 0])), float(np.nanmin(wall_poly_raw[:, 1])), float(np.nanmax(wall_poly_raw[:, 1])))
            x = (x - wall_min_x) / max(wall_max_x - wall_min_x, 1e-09)
            y = (y - wall_min_y) / max(wall_max_y - wall_min_y, 1e-09)
        plot_x_max = CANONICAL_WIDTH if args.coord_mode == 'homography' else 1.0
        target_side = str(row.get('heatmap_target_side', '')).lower()
        if args.target_side == 'right':
            mirror_x = bool(str(row.get('mirror_lr_for_target_right', '')).lower() == 'true' or row.get('mirror_lr_for_target_right') is True)
        else:
            mirror_x = bool(target_side == 'r')
        if mirror_x:
            x = plot_x_max - x
        x = np.clip(x, 0.0, plot_x_max)
        y = np.clip(y, 0.0, 1.0)
        session_counts = _hist2d_counts(x, y, bins=args.bins, x_max=plot_x_max)
        session_prob = _hist2d_prob(x, y, bins=args.bins, x_max=plot_x_max)
        session_heat = session_counts if args.density_mode == 'pooled' else session_prob
        group_session_histograms[phase, condition].append(session_heat)
        group_session_supports[phase, condition].append(session_counts > 0.0)
        if sex in SEX_ORDER:
            sex_group_session_histograms[phase, _condition_sex_label(condition, sex)].append(session_heat)
            sex_group_session_supports[phase, _condition_sex_label(condition, sex)].append(session_counts > 0.0)
        for roi_id in ('chamber_l', 'chamber_r'):
            if args.roi_mode == 'contact':
                raw_poly = roi_polys_raw.get(roi_id)
                if raw_poly is None or raw_poly.shape[0] < 3 or wall_poly_raw is None or (wall_poly_raw.shape[0] < 3):
                    continue
                roi_row = roi_summary_df[(roi_summary_df['session_key'] == session_key) & (roi_summary_df['roi_id'] == roi_id)]
                if roi_row.empty:
                    continue
                buffer_px = float(roi_row.iloc[0]['mean_radius_px']) * (float(args.roi_radius_scale) - 1.0)
                buffered_raw = _buffer_polygon(raw_poly, buffer_px)
                if buffered_raw.shape[0] < 3:
                    continue
                raw_bounds = (float(np.nanmin(wall_poly_raw[:, 0])), float(np.nanmax(wall_poly_raw[:, 0])), float(np.nanmin(wall_poly_raw[:, 1])), float(np.nanmax(wall_poly_raw[:, 1])))
                object_display = _resample_polygon(_transform_polygon(raw_poly, wall_bounds=raw_bounds, mirror_x=mirror_x))
                analysis_display = _resample_polygon(_transform_polygon(buffered_raw, wall_bounds=raw_bounds, mirror_x=mirror_x))
                if object_display.shape[0] < 3 or analysis_display.shape[0] < 3:
                    continue
                aligned_roi_id = roi_id
                if mirror_x:
                    aligned_roi_id = 'chamber_r' if roi_id == 'chamber_l' else 'chamber_l'
                for (vertex_rank, (vx, vy)) in enumerate(object_display):
                    row_data = {'phase': phase, 'condition': condition, 'roi_id': aligned_roi_id, 'vertex_rank': vertex_rank, 'x': vx, 'y': vy}
                    group_polygon_rows.append(row_data)
                    if sex in SEX_ORDER:
                        sex_group_polygon_rows.append({**row_data, 'condition': _condition_sex_label(condition, sex)})
                for (vertex_rank, (vx, vy)) in enumerate(analysis_display):
                    row_data = {'phase': phase, 'condition': condition, 'roi_id': aligned_roi_id, 'vertex_rank': vertex_rank, 'x': vx, 'y': vy}
                    group_analysis_polygon_rows.append(row_data)
                    if sex in SEX_ORDER:
                        sex_group_analysis_polygon_rows.append({**row_data, 'condition': _condition_sex_label(condition, sex)})
                continue
            poly = roi_polys.get(roi_id)
            if poly is None or poly.shape[0] < 3:
                continue
            if args.coord_mode == 'homography' and homography is not None:
                physical_poly = apply_homography(poly, homography)
                physical_poly = physical_poly[np.isfinite(physical_poly).all(axis=1)]
                if physical_poly.shape[0] < 3:
                    continue
                physical_cx = float(np.mean(physical_poly[:, 0]))
                physical_cy = float(np.mean(physical_poly[:, 1]))
                physical_radius = float(np.mean(np.sqrt((physical_poly[:, 0] - physical_cx) ** 2 + (physical_poly[:, 1] - physical_cy) ** 2)))
                aligned_poly = physical_poly.copy()
                if mirror_x:
                    aligned_poly[:, 0] = plot_x_max - aligned_poly[:, 0]
                aligned_poly = _order_polygon(aligned_poly)
            else:
                aligned_poly = _transform_polygon(poly, wall_bounds=wall_bounds, mirror_x=mirror_x)
            if aligned_poly.shape[0] < 3:
                continue
            aligned_roi_id = roi_id
            if mirror_x:
                aligned_roi_id = 'chamber_r' if roi_id == 'chamber_l' else 'chamber_l'
            cx = float(np.mean(aligned_poly[:, 0]))
            cy = float(np.mean(aligned_poly[:, 1]))
            if args.coord_mode == 'homography':
                object_radius_x = physical_radius
                object_radius_y = physical_radius
            elif args.coord_mode == 'normalized':
                roi_row = roi_summary_df[(roi_summary_df['session_key'] == session_key) & (roi_summary_df['roi_id'] == roi_id)]
                if roi_row.empty:
                    continue
                r0 = roi_row.iloc[0]
                (wall_min_x, wall_max_x, wall_min_y, wall_max_y) = wall_bounds
                wall_width = max(wall_max_x - wall_min_x, 1e-09)
                wall_height = max(wall_max_y - wall_min_y, 1e-09)
                cx = (float(r0['centroid_x_norm']) - wall_min_x) / wall_width
                cy = (float(r0['centroid_y_norm']) - wall_min_y) / wall_height
                if mirror_x:
                    cx = 1.0 - cx
                object_radius_x = float(r0['mean_radius_norm']) / wall_width
                object_radius_y = float(r0['mean_radius_norm']) / wall_height
            else:
                roi_row = roi_summary_df[(roi_summary_df['session_key'] == session_key) & (roi_summary_df['roi_id'] == roi_id)]
                if roi_row.empty:
                    continue
                r0 = roi_row.iloc[0]
                (raw_min_x, raw_max_x, raw_min_y, raw_max_y) = wall_bounds_raw
                raw_width = max(raw_max_x - raw_min_x, 1e-09)
                raw_height = max(raw_max_y - raw_min_y, 1e-09)
                cx = (float(r0['centroid_x']) - raw_min_x) / raw_width
                cy = (float(r0['centroid_y']) - raw_min_y) / raw_height
                if mirror_x:
                    cx = 1.0 - cx
                raw_radius = float(r0['mean_radius_px'])
                object_radius_x = raw_radius / raw_width
                object_radius_y = raw_radius / raw_height
            analysis_radius_x = object_radius_x * float(args.roi_radius_scale)
            analysis_radius_y = object_radius_y * float(args.roi_radius_scale)
            group_roi_rows.append({'phase': phase, 'condition': condition, 'roi_id': aligned_roi_id, 'center_x': cx, 'center_y': cy, 'object_radius_x': object_radius_x, 'object_radius_y': object_radius_y, 'analysis_radius_x': analysis_radius_x, 'analysis_radius_y': analysis_radius_y})
            if sex in SEX_ORDER:
                sex_group_roi_rows.append({'phase': phase, 'condition': _condition_sex_label(condition, sex), 'sex': sex, 'roi_id': aligned_roi_id, 'center_x': cx, 'center_y': cy, 'object_radius_x': object_radius_x, 'object_radius_y': object_radius_y, 'analysis_radius_x': analysis_radius_x, 'analysis_radius_y': analysis_radius_y})
            for (vertex_rank, (vx, vy)) in enumerate(aligned_poly):
                group_polygon_rows.append({'phase': phase, 'condition': condition, 'roi_id': aligned_roi_id, 'vertex_rank': vertex_rank, 'x': vx, 'y': vy})
        session_summary_rows.append({'session_key': session_key, 'condition': condition, 'phase': phase, 'subject_id': row.get('subject_id', ''), 'sex': sex, 'trial_id': row.get('trial_id', pd.NA), 'session_n': row.get('session_n', pd.NA), 'layout_raw': row.get('layout_raw', ''), 'keypoint': args.keypoint, 'coord_mode': args.coord_mode, 'roi_mode': args.roi_mode, 'roi_radius_scale': float(args.roi_radius_scale), 'contact_buffer_fraction': float(args.roi_radius_scale) - 1.0 if args.roi_mode == 'contact' else math.nan, 'fps': float(args.fps), 'max_minutes': float(args.max_minutes), 'exclude_keypoint_in_cup': False, 'keypoint_in_cup_frames_removed': keypoint_in_cup_frames_removed, 'exclude_body_in_cup': bool(args.exclude_body_in_cup), 'body_in_cup_frames_removed': body_in_cup_frames_removed, 'cup_exclusion_frames_removed': cup_exclusion_frames_removed, 'density_mode': args.density_mode, 'target_side_original': target_side, 'target_side_aligned': args.target_side, 'mirrored_x': mirror_x, 'valid_points': int(len(x))})
    group_heatmaps: dict[tuple[str, str], np.ndarray] = {}
    group_support_masks: dict[tuple[str, str], np.ndarray] = {}
    group_summary_rows: list[dict[str, object]] = []
    for (group_key, session_arrays) in group_session_histograms.items():
        (phase, condition) = group_key
        group_heat = _make_group_heat(session_arrays, sigma=args.sigma, density_mode=args.density_mode)
        group_heatmaps[group_key] = group_heat
        support_arrays = group_session_supports.get(group_key, [])
        if support_arrays:
            group_support_masks[group_key] = np.stack(support_arrays, axis=0).any(axis=0)
        group_summary_rows.append({'phase': phase, 'condition': condition, 'session_count': len(session_arrays), 'density_mode': args.density_mode, 'max_minutes': float(args.max_minutes), 'fps': float(args.fps), 'heatmap_max': float(np.max(group_heat)) if group_heat.size else math.nan, 'heatmap_sum': float(np.sum(group_heat)) if group_heat.size else math.nan})
    sex_group_heatmaps: dict[tuple[str, str], np.ndarray] = {}
    sex_group_support_masks: dict[tuple[str, str], np.ndarray] = {}
    sex_group_summary_rows: list[dict[str, object]] = []
    for (group_key, session_arrays) in sex_group_session_histograms.items():
        (phase, condition_sex) = group_key
        group_heat = _make_group_heat(session_arrays, sigma=args.sigma, density_mode=args.density_mode)
        sex_group_heatmaps[group_key] = group_heat
        support_arrays = sex_group_session_supports.get(group_key, [])
        if support_arrays:
            sex_group_support_masks[group_key] = np.stack(support_arrays, axis=0).any(axis=0)
        condition = condition_sex.rsplit(' ', 1)[0]
        sex_label = condition_sex.rsplit(' ', 1)[-1].lower()
        sex = 'm' if sex_label == 'm' else 'f' if sex_label == 'f' else sex_label
        sex_group_summary_rows.append({'phase': phase, 'condition': condition, 'sex': sex, 'condition_sex': condition_sex, 'session_count': len(session_arrays), 'density_mode': args.density_mode, 'max_minutes': float(args.max_minutes), 'fps': float(args.fps), 'heatmap_max': float(np.max(group_heat)) if group_heat.size else math.nan, 'heatmap_sum': float(np.sum(group_heat)) if group_heat.size else math.nan})
    group_polygons: dict[tuple[str, str, str], np.ndarray] = {}
    if group_polygon_rows:
        group_polygon_df = pd.DataFrame(group_polygon_rows)
        grouped = group_polygon_df.groupby(['phase', 'condition', 'roi_id', 'vertex_rank'], as_index=False)[['x', 'y']].mean().sort_values(['phase', 'condition', 'roi_id', 'vertex_rank'])
        for ((phase, condition, roi_id), group) in grouped.groupby(['phase', 'condition', 'roi_id'], sort=False):
            group_polygons[str(phase), str(condition), str(roi_id)] = group[['x', 'y']].to_numpy(dtype=float)
    else:
        group_polygon_df = pd.DataFrame(columns=['phase', 'condition', 'roi_id', 'vertex_rank', 'x', 'y'])

    def _aggregate_polygon_rows(rows: list[dict[str, object]]) -> tuple[pd.DataFrame, dict[tuple[str, str, str], np.ndarray]]:
        if not rows:
            return (pd.DataFrame(columns=['phase', 'condition', 'roi_id', 'vertex_rank', 'x', 'y']), {})
        df = pd.DataFrame(rows)
        df = df.groupby(['phase', 'condition', 'roi_id', 'vertex_rank'], as_index=False)[['x', 'y']].mean().sort_values(['phase', 'condition', 'roi_id', 'vertex_rank'])
        polygons: dict[tuple[str, str, str], np.ndarray] = {}
        for ((phase, condition, roi_id), group) in df.groupby(['phase', 'condition', 'roi_id'], sort=False):
            polygons[str(phase), str(condition), str(roi_id)] = group[['x', 'y']].to_numpy(dtype=float)
        return (df, polygons)
    (group_analysis_polygon_df, group_analysis_polygons) = _aggregate_polygon_rows(group_analysis_polygon_rows)
    (sex_group_polygon_df, sex_group_polygons) = _aggregate_polygon_rows(sex_group_polygon_rows)
    (sex_group_analysis_polygon_df, sex_group_analysis_polygons) = _aggregate_polygon_rows(sex_group_analysis_polygon_rows)
    group_roi_summary_df = pd.DataFrame(group_roi_rows)
    group_object_circles: dict[tuple[str, str, str], tuple[float, float, float, float]] = {}
    group_analysis_circles: dict[tuple[str, str, str], tuple[float, float, float, float]] = {}
    if not group_roi_summary_df.empty:
        group_roi_summary_df = group_roi_summary_df.groupby(['phase', 'condition', 'roi_id'], as_index=False)[['center_x', 'center_y', 'object_radius_x', 'object_radius_y', 'analysis_radius_x', 'analysis_radius_y']].mean().sort_values(['phase', 'condition', 'roi_id']).reset_index(drop=True)
        for row in group_roi_summary_df.to_dict(orient='records'):
            group_object_circles[str(row['phase']), str(row['condition']), str(row['roi_id'])] = (float(row['center_x']), float(row['center_y']), float(row['object_radius_x']), float(row['object_radius_y']))
            group_analysis_circles[str(row['phase']), str(row['condition']), str(row['roi_id'])] = (float(row['center_x']), float(row['center_y']), float(row['analysis_radius_x']), float(row['analysis_radius_y']))
    else:
        group_roi_summary_df = pd.DataFrame(columns=['phase', 'condition', 'roi_id', 'center_x', 'center_y', 'object_radius_x', 'object_radius_y', 'analysis_radius_x', 'analysis_radius_y'])
    sex_group_roi_summary_df = pd.DataFrame(sex_group_roi_rows)
    sex_group_object_circles: dict[tuple[str, str, str], tuple[float, float, float, float]] = {}
    sex_group_analysis_circles: dict[tuple[str, str, str], tuple[float, float, float, float]] = {}
    if not sex_group_roi_summary_df.empty:
        sex_group_roi_summary_df = sex_group_roi_summary_df.groupby(['phase', 'condition', 'sex', 'roi_id'], as_index=False)[['center_x', 'center_y', 'object_radius_x', 'object_radius_y', 'analysis_radius_x', 'analysis_radius_y']].mean().sort_values(['phase', 'condition', 'sex', 'roi_id']).reset_index(drop=True)
        for row in sex_group_roi_summary_df.to_dict(orient='records'):
            key = (str(row['phase']), str(row['condition']), str(row['roi_id']))
            sex_group_object_circles[key] = (float(row['center_x']), float(row['center_y']), float(row['object_radius_x']), float(row['object_radius_y']))
            sex_group_analysis_circles[key] = (float(row['center_x']), float(row['center_y']), float(row['analysis_radius_x']), float(row['analysis_radius_y']))
    else:
        sex_group_roi_summary_df = pd.DataFrame(columns=['phase', 'condition', 'sex', 'roi_id', 'center_x', 'center_y', 'object_radius_x', 'object_radius_y', 'analysis_radius_x', 'analysis_radius_y'])
    return {'file_suffix': file_suffix, 'group_analysis_circles': group_analysis_circles, 'group_analysis_polygon_df': group_analysis_polygon_df, 'group_analysis_polygons': group_analysis_polygons, 'group_heatmaps': group_heatmaps, 'group_object_circles': group_object_circles, 'group_polygon_df': group_polygon_df, 'group_polygons': group_polygons, 'group_roi_summary_df': group_roi_summary_df, 'group_session_histograms': group_session_histograms, 'group_summary_rows': group_summary_rows, 'group_support_masks': group_support_masks, 'issue_rows': issue_rows, 'output_dir': output_dir, 'phase_list': phase_list, 'session_n_filter': session_n_filter, 'session_summary_rows': session_summary_rows, 'sessions_df': sessions_df, 'sex_group_analysis_circles': sex_group_analysis_circles, 'sex_group_analysis_polygon_df': sex_group_analysis_polygon_df, 'sex_group_analysis_polygons': sex_group_analysis_polygons, 'sex_group_heatmaps': sex_group_heatmaps, 'sex_group_object_circles': sex_group_object_circles, 'sex_group_polygon_df': sex_group_polygon_df, 'sex_group_polygons': sex_group_polygons, 'sex_group_roi_summary_df': sex_group_roi_summary_df, 'sex_group_summary_rows': sex_group_summary_rows, 'sex_group_support_masks': sex_group_support_masks}
