"""Sequential decoding, seek diagnostics and video re-encoding."""
from pathlib import Path
import csv
import subprocess
import cv2
import numpy as np
import pandas as pd

PROBE_LIMIT = 300
MATCH_THRESHOLD = 0.05
DOWNSAMPLE_SIZE = (160, 90)


def get_csv_frame_info(csv_path):
    """Support long MovAl tracking CSVs and three-header SiMBA pose CSVs."""
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle), [])
    if "frame_idx" not in header:
        count = 0
        with Path(csv_path).open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.reader(handle):
                if not row:
                    continue
                try:
                    value = float(row[0])
                except ValueError:
                    continue
                if value.is_integer():
                    count += 1
        if not count:
            raise ValueError(f"No frame rows: {csv_path}")
        return {"csv_rows": count, "csv_frame_count": count,
                "csv_frame_min": 0, "csv_frame_max": count-1,
                "csv_unique_tracks": "", "bad_track_frame_count": 0}
    frame = pd.read_csv(csv_path, usecols=["frame_idx", "track"])
    frame["frame_idx"] = pd.to_numeric(frame.frame_idx, errors="raise").astype(int)
    frames = frame.frame_idx.unique()
    if len(frames) == 0:
        raise ValueError(f"No frame rows: {csv_path}")
    if min(frames) != 0 or max(frames) + 1 != len(frames):
        raise ValueError("Tracking frame indices must start at zero and be contiguous")
    return {"csv_rows": len(frame), "csv_frame_count": len(frames),
            "csv_frame_min": int(min(frames)), "csv_frame_max": int(max(frames)),
            "csv_unique_tracks": ",".join(sorted(frame.track.astype(str).unique())),
            "bad_track_frame_count": int((frame.groupby("frame_idx").track.nunique() != 2).sum())}

def prep(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, DOWNSAMPLE_SIZE, interpolation=cv2.INTER_AREA)



def frame_diff(a, b):
    return float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16))))



def video_info(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open video: {video_path}")
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    meta = {
        "meta_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fourcc": "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4)),
    }
    cap.release()
    return meta



def sequential_count_and_needed(video_path, needed_indices):
    needed = set(i for i in needed_indices if i is not None and i >= 0)
    frames = {}
    cap = cv2.VideoCapture(str(video_path))
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        if count in needed:
            frames[count] = prep(frame)
        count += 1
    cap.release()
    return count, frames



def read_seek_frame(video_path, frame_idx):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return None
    return prep(frame)



def best_offset_for_seek(video_path, seek_idx, seq_frames, max_extra, previous_offset):
    img = read_seek_frame(video_path, seek_idx)
    if img is None:
        return None, None, None

    candidates = []
    for dropped in range(0, max_extra + 1):
        seq_idx = seek_idx - dropped
        if seq_idx in seq_frames:
            candidates.append((seq_idx - seek_idx, seq_idx, frame_diff(img, seq_frames[seq_idx])))

    if not candidates:
        return None, None, None

    candidates.sort(key=lambda x: (x[2], abs(x[0] - previous_offset), abs(x[0])))
    best_offset, best_seq_idx, best_score = candidates[0]
    if best_score > MATCH_THRESHOLD:
        return None, best_seq_idx, best_score
    return best_offset, best_seq_idx, best_score



def probe_offsets(video_path, seq_frames, max_extra):
    offsets = {}
    details = []
    previous_offset = 0
    for seek_idx in range(0, PROBE_LIMIT + 1):
        offset, seq_idx, score = best_offset_for_seek(
            video_path, seek_idx, seq_frames, max_extra, previous_offset
        )
        if offset is None:
            continue
        offsets[seek_idx] = offset
        previous_offset = offset
        details.append(f"seek {seek_idx}->seq {seq_idx}, offset {offset}, score {score:.3f}")
    return offsets, details



def derive_delete_ranges(offsets):
    if not offsets:
        return []
    ranges = []
    seek_indices = sorted(offsets)
    last_offset = offsets[seek_indices[0]]
    for seek_idx in seek_indices[1:]:
        cur_offset = offsets[seek_idx]
        if cur_offset < last_offset:
            drop_count = last_offset - cur_offset
            ranges.append((seek_idx, seek_idx + drop_count - 1))
        last_offset = cur_offset
    return ranges



def compact_ranges(ranges):
    parts = []
    for start, end in ranges:
        parts.append(str(start) if start == end else f"{start}-{end}")
    return ";".join(parts)



def count_deleted(ranges):
    return sum(end - start + 1 for start, end in ranges)



def new_idx_to_old_seek(new_idx, ranges):
    old_seek = new_idx
    for start, end in ranges:
        if old_seek >= start:
            old_seek += end - start + 1
    return old_seek



def validate_delete_mapping(video_path, seq_frames, expected_count, ranges):
    samples = set([
        0, 1, 28, 29, 30, 31, 34, 35, 36, 39, 40,
        100, 1000, 3000, 6000, expected_count - 1,
    ])
    for start, end in ranges:
        samples.update([max(0, start - 2), max(0, start - 1), start, start + 1, end, end + 1])
    samples = sorted(i for i in samples if 0 <= i < expected_count)

    checked = 0
    mismatches = []
    for new_idx in samples:
        if new_idx not in seq_frames:
            continue
        old_seek = new_idx_to_old_seek(new_idx, ranges)
        seek_img = read_seek_frame(video_path, old_seek)
        if seek_img is None:
            mismatches.append(f"new {new_idx}: old_seek {old_seek} unreadable")
            continue
        score = frame_diff(seek_img, seq_frames[new_idx])
        checked += 1
        if score > MATCH_THRESHOLD:
            mismatches.append(f"new {new_idx}: old_seek {old_seek}, score {score:.3f}")
    return checked, "; ".join(mismatches)



def diagnose_one(csv_path, video_path):
    stem = csv_path.stem
    row = {
        "stem": stem,
        "status": "",
        "csv_rows": "",
        "csv_frame_count": "",
        "csv_frame_min": "",
        "csv_frame_max": "",
        "csv_unique_tracks": "",
        "bad_track_frame_count": "",
        "video_meta_count": "",
        "sequential_count": "",
        "meta_minus_csv_frames": "",
        "seq_minus_csv_frames": "",
        "delete_seek_ranges": "",
        "delete_seek_count": "",
        "validation_checked": "",
        "validation_mismatches": "",
        "probe_tail": "",
        "video_path": str(video_path) if video_path else "",
        "csv_path": str(csv_path),
    }
    info = get_csv_frame_info(csv_path)
    row.update(info)

    if video_path is None:
        row["status"] = "VIDEO_NOT_FOUND"
        return row

    expected_count = info["csv_frame_count"]
    meta = video_info(video_path)
    row["video_meta_count"] = meta["meta_frames"]
    row["meta_minus_csv_frames"] = meta["meta_frames"] - expected_count
    max_extra = max(0, meta["meta_frames"] - expected_count)

    needed = set(range(0, PROBE_LIMIT + 1))
    needed.update([0, 1, 28, 29, 30, 31, 34, 35, 36, 39, 40, 100, 1000, 3000, 6000, expected_count - 1])
    seq_count, seq_frames = sequential_count_and_needed(video_path, needed)
    row["sequential_count"] = seq_count
    row["seq_minus_csv_frames"] = seq_count - expected_count

    if seq_count != expected_count:
        row["status"] = "SEQ_CSV_COUNT_MISMATCH"
        return row

    if max_extra == 0:
        mismatches = validate_seek_samples(video_path, expected_count, "deep")
        row["validation_mismatches"] = mismatches
        row["status"] = "SEEK_MISMATCH" if mismatches else "OK_NO_DELETE"
        return row

    offsets, details = probe_offsets(video_path, seq_frames, max_extra)
    ranges = derive_delete_ranges(offsets)
    row["probe_tail"] = " | ".join(details[-20:])
    row["delete_seek_ranges"] = compact_ranges(ranges)
    row["delete_seek_count"] = count_deleted(ranges)

    extra_needed = set(needed)
    for start, end in ranges:
        extra_needed.update([max(0, start - 2), max(0, start - 1), start, start + 1, end, end + 1])
    _, seq_frames = sequential_count_and_needed(video_path, extra_needed)

    checked, mismatches = validate_delete_mapping(video_path, seq_frames, expected_count, ranges)
    row["validation_checked"] = checked
    row["validation_mismatches"] = mismatches

    if count_deleted(ranges) == max_extra and not mismatches:
        row["status"] = "OK_DELETE_THESE_SEEK_FRAMES"
    elif not ranges:
        row["status"] = "DELETE_START_NOT_FOUND_IN_PROBE"
    else:
        row["status"] = "CHECK_COMPLEX_MAPPING"

    return row



def validate_output(video_path, expected_count):
    samples = set(i for i in [0, 1, 28, 29, 30, 31, 34, 35, 36, 39, 40, 100, 1000, 3000, 6000, expected_count - 1] if 0 <= i < expected_count)
    meta = video_info(video_path)
    seq_count, seq_frames = sequential_count_and_needed(video_path, samples)

    mismatches = []
    for idx in sorted(samples):
        seq_img = seq_frames.get(idx)
        seek_img = read_seek_frame(video_path, idx)
        if seq_img is None or seek_img is None:
            mismatches.append(f"{idx}: unreadable")
            continue
        score = frame_diff(seq_img, seek_img)
        if score > MATCH_THRESHOLD:
            mismatches.append(f"{idx}: score {score:.3f}")

    ok = meta["meta_frames"] == expected_count and seq_count == expected_count and not mismatches
    return {
        "output_meta_count": meta["meta_frames"],
        "output_seq_count": seq_count,
        "output_fps": meta["fps"],
        "output_width": meta["width"],
        "output_height": meta["height"],
        "output_fourcc": meta["fourcc"],
        "seek_mismatches": "; ".join(mismatches),
        "validation_ok": ok,
    }


def rewrite_video(src, dst, expected_count, codec="XVID", ffmpeg=None):
    src, dst = Path(src), Path(dst)
    if src.resolve() == dst.resolve():
        raise ValueError("Video output must differ from its source")
    if dst.exists():
        raise FileExistsError(dst)
    src_meta = video_info(src)
    fps = src_meta["fps"] if src_meta["fps"] and src_meta["fps"] > 0 else 30.0
    width = src_meta["width"]
    height = src_meta["height"]

    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid source resolution: {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".writing.avi")
    if tmp.exists():
        tmp.unlink()

    if ffmpeg is not None:
        cmd = [
            str(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(src),
            "-map",
            "0:v:0",
            "-an",
            "-vf",
            f"setpts=N/({fps}*TB)",
            "-frames:v",
            str(expected_count),
            "-r",
            str(fps),
            "-c:v",
            "mpeg4",
            "-q:v",
            "3",
            "-vtag",
            codec,
            str(tmp),
        ]
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        if proc.returncode != 0:
            if tmp.exists():
                tmp.unlink()
            raise RuntimeError(
                "ffmpeg failed: "
                + proc.stderr.strip().replace("\r", " ").replace("\n", " ")
            )

        if dst.exists():
            dst.unlink()
        tmp.replace(dst)
        return expected_count

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open source video: {src}")

    writer = cv2.VideoWriter(
        str(tmp),
        cv2.VideoWriter_fourcc(*codec),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        writer.release()
        raise RuntimeError(f"Cannot open VideoWriter with codec {codec}: {tmp}")

    written = 0
    while written < expected_count:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        writer.write(frame)
        written += 1

    cap.release()
    writer.release()

    if written != expected_count:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(
            f"Sequential source ended at {written}, expected {expected_count}: {src}"
        )

    if dst.exists():
        dst.unlink()
    tmp.replace(dst)

    return written


def checksum_frame(frame):
    return int(np.sum(frame.astype(np.uint64)))



def validate_seek_samples(video_path, frame_count, sample_mode):
    if sample_mode == "none":
        return ""

    samples = [0, 1, 10, 28, 35, 100]
    if sample_mode == "deep":
        samples.extend([frame_count // 2, frame_count - 2, frame_count - 1])
    samples = sorted(set([idx for idx in samples if 0 <= idx < frame_count]))
    if not samples:
        return ""

    max_sample = max(samples)
    cap = cv2.VideoCapture(str(video_path))
    seq_checks = {}
    idx = 0
    while idx <= max_sample:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        if idx in samples:
            seq_checks[idx] = checksum_frame(frame)
        idx += 1
    cap.release()

    mismatches = []
    cap = cv2.VideoCapture(str(video_path))
    for sample_idx in samples:
        cap.set(cv2.CAP_PROP_POS_FRAMES, sample_idx)
        ret, frame = cap.read()
        if (
            not ret
            or frame is None
            or sample_idx not in seq_checks
            or checksum_frame(frame) != seq_checks[sample_idx]
        ):
            mismatches.append(sample_idx)
    cap.release()

    return ",".join(str(x) for x in mismatches)

