"""Tracking coordinate preparation and CSV/video matching for SiMBA import."""
from pathlib import Path
import re
import numpy as np
import pandas as pd
from .video import video_info, rewrite_video, validate_seek_samples

DEFAULT_BAD_PREFIXES = ("Euclidean_distance_", "Movement_", "Mean_", "Sum_", "All_bp_", "Low_prob_")
VIDEO_EXTENSIONS = (".avi", ".mp4", ".mov", ".mkv")

def normalize_stem(stem):
    """Normalize MovAl/Cutie derived file names for video/CSV matching."""
    normalized = stem.strip()

    # Common prediction output prefixes:
    #   predict__VIDEO_260505_152935
    #   predict_VIDEO
    #   prediction__VIDEO
    normalized = re.sub(r"^(predict|prediction|pred)[_-]+", "", normalized, flags=re.IGNORECASE)

    # Common timestamp suffixes from prediction/export folders.
    timestamp_patterns = (
        r"[_-]\d{6}[_-]\d{6}$",       # _260505_152935
        r"[_-]\d{8}[_-]\d{6}$",       # _20260505_152935
        r"[_-]\d{6}T\d{6}$",          # _260505T152935
        r"[_-]\d{8}T\d{6}$",          # _20260505T152935
    )
    changed = True
    while changed:
        changed = False
        for pattern in timestamp_patterns:
            new_value = re.sub(pattern, "", normalized)
            if new_value != normalized:
                normalized = new_value
                changed = True

    # Cutie/ffmpeg cut videos often carry _cut while CSVs may not.
    normalized = re.sub(r"[_-]cut$", "", normalized, flags=re.IGNORECASE)
    return normalized



def parse_expected_tracks(value):
    try:
        tracks = tuple(int(x.strip()) for x in value.split(",") if x.strip() != "")
    except ValueError:
        raise argparse.ArgumentTypeError("--expected-tracks must be comma-separated integers")
    if not tracks:
        raise argparse.ArgumentTypeError("--expected-tracks cannot be empty")
    return tracks



def parse_drop_bodyparts(value):
    return tuple(x.strip() for x in value.split(",") if x.strip() != "")



def parse_track_to_numeric(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if float(value).is_integer():
            return int(value)
        raise ValueError("Non-integer numeric track value: {}".format(value))

    text = str(value).strip()
    if text == "":
        return np.nan
    if re.fullmatch(r"\d+(\.0+)?", text):
        return int(float(text))
    match = re.search(r"(\d+)$", text)
    if match:
        return int(match.group(1))
    raise ValueError("Cannot parse track value: {}".format(value))



def get_coord_columns(df):
    x_cols = []
    y_cols = []
    for col in df.columns:
        if col.startswith(DEFAULT_BAD_PREFIXES):
            continue
        if col.endswith(".x") or col.endswith("_x"):
            x_cols.append(col)
        elif col.endswith(".y") or col.endswith("_y"):
            y_cols.append(col)
    return x_cols, y_cols



def get_score_columns(df):
    score_cols = []
    for col in df.columns:
        if col == "instance.score" or col.endswith(".score") or col.endswith("_score") or col.endswith("_p"):
            score_cols.append(col)
    return score_cols



def drop_bodypart_columns(df, drop_bodyparts):
    drop_cols = []
    for bodypart in drop_bodyparts:
        for col in df.columns:
            if col in (
                "{}.x".format(bodypart),
                "{}.y".format(bodypart),
                "{}.score".format(bodypart),
                "{}_x".format(bodypart),
                "{}_y".format(bodypart),
                "{}_p".format(bodypart),
                "{}_score".format(bodypart),
            ):
                drop_cols.append(col)
            if re.fullmatch(r"{}_\d+_(x|y|p|score)".format(re.escape(bodypart)), col):
                drop_cols.append(col)
    drop_cols = sorted(set(drop_cols))
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df, drop_cols



def scale_coords_to_video(df, x_cols, y_cols, video_width, video_height, source_width, source_height):
    df = df.copy()
    max_x = pd.to_numeric(df[x_cols].stack(), errors="coerce").max()
    max_y = pd.to_numeric(df[y_cols].stack(), errors="coerce").max()
    min_x = pd.to_numeric(df[x_cols].stack(), errors="coerce").min()
    min_y = pd.to_numeric(df[y_cols].stack(), errors="coerce").min()

    if pd.isna(max_x) or pd.isna(max_y):
        raise ValueError("Coordinate columns are all NaN or invalid")

    if max_x <= 1.5 and max_y <= 1.5:
        coord_type = "normalized_0_1"
        x_scale = float(video_width)
        y_scale = float(video_height)
    elif max_x <= video_width * 1.05 and max_y <= video_height * 1.05:
        coord_type = "already_video_resolution"
        x_scale = 1.0
        y_scale = 1.0
    elif source_width and source_height:
        coord_type = "larger_than_video_assume_source_resolution"
        x_scale = float(video_width) / float(source_width)
        y_scale = float(video_height) / float(source_height)
    else:
        raise ValueError(
            "Coordinates exceed video bounds, but no source resolution was provided"
        )

    for col in x_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce") * x_scale
    for col in y_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce") * y_scale

    return df, {
        "coord_type": coord_type,
        "x_scale": x_scale,
        "y_scale": y_scale,
        "min_x_before": float(min_x),
        "max_x_before": float(max_x),
        "min_y_before": float(min_y),
        "max_y_before": float(max_y),
    }



def fill_missing_track_rows(df, expected_tracks):
    df = df.copy()
    original_cols = list(df.columns)

    if "frame_idx" not in df.columns:
        raise ValueError("CSV has no frame_idx column")
    if "track" not in df.columns:
        raise ValueError("CSV has no track column")

    df["frame_idx"] = pd.to_numeric(df["frame_idx"], errors="coerce")
    df["track"] = pd.to_numeric(df["track"], errors="coerce")

    if df["frame_idx"].isna().any():
        raise ValueError("frame_idx contains NaN after numeric conversion")
    if df["track"].isna().any():
        raise ValueError("track contains NaN after numeric conversion")

    df["frame_idx"] = df["frame_idx"].astype(int)
    df["track"] = df["track"].astype(int)

    unexpected_tracks = sorted(set(df["track"].dropna().unique().tolist()) - set(expected_tracks))
    if unexpected_tracks:
        raise ValueError("Unexpected tracks found: {}".format(unexpected_tracks))

    duplicate_count = int(df.duplicated(["frame_idx", "track"]).sum())
    if duplicate_count:
        if "instance.score" in df.columns:
            df = df.sort_values(
                ["frame_idx", "track", "instance.score"],
                ascending=[True, True, False],
            )
        else:
            df = df.sort_values(["frame_idx", "track"])
        df = df.drop_duplicates(["frame_idx", "track"], keep="first")

    frame_min = int(df["frame_idx"].min())
    frame_max = int(df["frame_idx"].max())
    full_index = pd.MultiIndex.from_product(
        [range(frame_min, frame_max + 1), expected_tracks],
        names=["frame_idx", "track"],
    )

    original_pairs = set(zip(df["frame_idx"], df["track"]))
    df_filled = df.set_index(["frame_idx", "track"]).sort_index().reindex(full_index)
    added_mask = pd.Series(
        [(frame, track) not in original_pairs for frame, track in df_filled.index],
        index=df_filled.index,
    )
    added_rows = int(added_mask.sum())

    for col in df_filled.columns:
        df_filled[col] = pd.to_numeric(df_filled[col], errors="coerce")

    temp = df_filled.reset_index()
    x_cols, y_cols = get_coord_columns(temp)
    coord_cols = x_cols + y_cols
    score_cols = get_score_columns(df_filled)

    if coord_cols:
        df_filled[coord_cols] = df_filled.groupby(level="track")[coord_cols].transform(
            lambda s: s.interpolate(method="linear", limit_direction="both")
        )

    if score_cols:
        df_filled[score_cols] = df_filled.groupby(level="track")[score_cols].transform(
            lambda s: s.interpolate(method="linear", limit_direction="both")
        )
        df_filled.loc[added_mask, score_cols] = 0.0

    df_filled = df_filled.groupby(level="track").ffill().bfill()
    return df_filled.reset_index()[original_cols], added_rows, duplicate_count, frame_min, frame_max



def find_video_for_csv(csv_path, input_dir, recursive, keep_original_names):
    for extension in VIDEO_EXTENSIONS:
        candidate = input_dir / (csv_path.stem + extension)
        if candidate.exists():
            return candidate

    if keep_original_names:
        return None

    csv_norm = normalize_stem(csv_path.stem)
    search_root = input_dir if not recursive else input_dir
    pattern = "**/*" if recursive else "*"
    matches = []
    for path in search_root.glob(pattern):
        if not path.is_file():
            continue
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        if "simba_ready" in [part.lower() for part in path.parts]:
            continue
        if normalize_stem(path.stem) == csv_norm:
            matches.append(path)

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        same_folder = [path for path in matches if path.parent == csv_path.parent]
        if len(same_folder) == 1:
            return same_folder[0]
        raise ValueError(
            "Multiple videos match normalized stem '{}': {}".format(
                csv_norm, [str(path) for path in matches]
            )
        )
    return None



def list_csv_files(input_dir, recursive):
    pattern = "**/*.csv" if recursive else "*.csv"
    csv_files = []
    for path in input_dir.glob(pattern):
        if "simba_ready" in [part.lower() for part in path.parts]:
            continue
        if path.name.lower().endswith("_report.csv"):
            continue
        csv_files.append(path)
    return sorted(csv_files)






def process_one_pair(csv_path, video_path, output_dir, args, output_stem):
    source_info = video_info(video_path)
    df = pd.read_csv(str(csv_path))
    original_rows = len(df)
    original_cols = len(df.columns)

    df, dropped_cols = drop_bodypart_columns(df, args.drop_bodyparts)

    if "track" not in df.columns or "frame_idx" not in df.columns:
        raise ValueError("CSV must contain track and frame_idx columns")

    df["track"] = df["track"].apply(parse_track_to_numeric)

    x_cols, y_cols = get_coord_columns(df)
    if not x_cols or not y_cols:
        raise ValueError("No coordinate columns found")

    df, scale_report = scale_coords_to_video(
        df=df,
        x_cols=x_cols,
        y_cols=y_cols,
        video_width=source_info["width"],
        video_height=source_info["height"],
        source_width=args.source_width,
        source_height=args.source_height,
    )

    df, added_rows, duplicate_count, frame_min, frame_max = fill_missing_track_rows(
        df, args.expected_tracks
    )

    frame_count = int(df["frame_idx"].nunique())
    output_csv = output_dir / "csv" / (output_stem + ".csv")
    output_video = output_dir / "video" / (output_stem + ".avi")

    if not args.dry_run:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        output_video.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(str(output_csv), index=False)
        written_frames = rewrite_video(
            src=video_path,
            dst=output_video,
            expected_count=frame_count,
            codec=args.codec,
        )
        output_info = video_info(output_video)
        seek_mismatches = validate_seek_samples(
            output_video, frame_count, args.seek_validation
        )
    else:
        written_frames = 0
        output_info = {"meta_frames": None, "fps": source_info["fps"], "width": source_info["width"], "height": source_info["height"]}
        seek_mismatches = ""

    x_final, y_final = get_coord_columns(df)
    coord_cols = x_final + y_final
    track_counts = df.groupby("frame_idx")["track"].nunique()
    bad_track_frames = int((track_counts != len(args.expected_tracks)).sum())
    duplicate_final = int(df.duplicated(["frame_idx", "track"]).sum())
    nan_coord_count = int(df[coord_cols].isna().sum().sum())
    neck_cols = [col for col in df.columns for bp in args.drop_bodyparts if bp in col]

    min_x = pd.to_numeric(df[x_final].stack(), errors="coerce").min()
    max_x = pd.to_numeric(df[x_final].stack(), errors="coerce").max()
    min_y = pd.to_numeric(df[y_final].stack(), errors="coerce").min()
    max_y = pd.to_numeric(df[y_final].stack(), errors="coerce").max()

    output_frame_ok = True if args.dry_run else (output_info["meta_frames"] == frame_count)
    coord_ok = (
        pd.notna(min_x)
        and pd.notna(max_x)
        and pd.notna(min_y)
        and pd.notna(max_y)
        and float(min_x) >= 0
        and float(max_x) <= source_info["width"]
        and float(min_y) >= 0
        and float(max_y) <= source_info["height"]
    )
    status = "OK" if (
        output_frame_ok
        and len(df) == frame_count * len(args.expected_tracks)
        and frame_min == 0
        and bad_track_frames == 0
        and duplicate_final == 0
        and nan_coord_count == 0
        and not neck_cols
        and coord_ok
        and seek_mismatches == ""
    ) else "CHECK"

    report = {
        "file": csv_path.stem,
        "normalized_file": output_stem,
        "status": "DRY_RUN_" + status if args.dry_run else status,
        "input_csv": str(csv_path),
        "input_video": str(video_path),
        "output_csv": str(output_csv),
        "output_video": str(output_video),
        "input_video_meta_frames": source_info["meta_frames"],
        "output_video_meta_frames": output_info["meta_frames"],
        "written_video_frames": written_frames,
        "video_w": source_info["width"],
        "video_h": source_info["height"],
        "fps": source_info["fps"],
        "original_rows": original_rows,
        "final_rows": len(df),
        "original_col_count": original_cols,
        "final_col_count": len(df.columns),
        "final_csv_frames": frame_count,
        "frame_min": frame_min,
        "frame_max": frame_max,
        "added_missing_track_rows": added_rows,
        "duplicate_rows_removed": duplicate_count,
        "bad_track_frames_final": bad_track_frames,
        "duplicate_frame_track_final": duplicate_final,
        "nan_coord_count": nan_coord_count,
        "final_tracks": str(sorted(df["track"].dropna().unique().tolist())),
        "dropped_cols": str(dropped_cols),
        "dropped_col_count": len(dropped_cols),
        "remaining_drop_bodypart_cols": str(neck_cols),
        "min_x_final": float(min_x) if pd.notna(min_x) else None,
        "max_x_final": float(max_x) if pd.notna(max_x) else None,
        "min_y_final": float(min_y) if pd.notna(min_y) else None,
        "max_y_final": float(max_y) if pd.notna(max_y) else None,
        "seek_sample_mismatches": seek_mismatches,
    }
    report.update(scale_report)
    return report

