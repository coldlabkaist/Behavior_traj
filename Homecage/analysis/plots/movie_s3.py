"""Movie S3 compilation: synchronized source frames and pose overlays."""
from functools import lru_cache
from pathlib import Path
from datasets.paths import resolve_data_path
import json
import subprocess
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from matplotlib.font_manager import FontProperties, findfont
from analysis.core.archive import ROOT
from analysis.core.definitions import MOTIF_NAMES

W, H, ASPECT = 960, 540, 1920 / 1080
ANIMAL = ['#4c78a8', '#f58518', '#54a24b']
POINT = ['#e07a2d', '#e53935', '#f2c94c', '#9b59b6', '#4c78a8', '#6b7280']
EDGES = [(0,4),(4,1),(1,5),(4,2),(4,3),(0,2),(0,3),(2,3)]
COLORS = ['#cfbad1','#7d1f78','#cdb18a','#684720']
KEYS = ['Nose','Body_C','Ear_L','Ear_R','Neck','Tail']
SELECTION = ROOT / 'analysis/resources/movie_s3.json'

@lru_cache(maxsize=4)
def movie_font(size, bold=False):
    path=findfont(FontProperties(family='Arial', weight='bold' if bold else 'normal'))
    return ImageFont.truetype(path, size)

def decode(source, start):
    info = json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
        '-show_entries','stream=r_frame_rate,width,height','-of','json',str(source)]))['streams'][0]
    a,b = map(float, info['r_frame_rate'].split('/'))
    if abs(a/b-30)>0.001:
        raise ValueError(f'Unexpected source FPS: {info}')
    raw = subprocess.check_output(['ffmpeg','-v','error','-ss',str(start/30),'-i',str(source),
        '-frames:v','30','-vf',f'scale={W}:{H}','-f','rawvideo','-pix_fmt','rgb24','pipe:1'])
    frames = np.frombuffer(raw,np.uint8).reshape(-1,H,W,3)
    if len(frames)!=30: raise ValueError('Incomplete source clip')
    return frames

def pose_layer(draw, xy, index, offset, overlay=False, labels=True):
    for animal in range(3):
        trail=xy[:index+1,animal,1]
        for a,b in zip(trail[:-1],trail[1:]):
            if np.isfinite([a,b]).all():
                draw.line([tuple(a+[0,offset]),tuple(b+[0,offset])],fill=ANIMAL[animal],width=3)
        pts=xy[index,animal].copy(); pts[:,1]+=offset
        for a,b in EDGES:
            if np.isfinite(pts[[a,b]]).all():
                draw.line([tuple(pts[a]),tuple(pts[b])],fill=ANIMAL[animal] if overlay else '#6b7280',width=3)
        for k,p in enumerate(pts):
            if not np.isfinite(p).all(): continue
            x,y=p
            if k==1:
                draw.line((x-5,y-5,x+5,y+5),fill=POINT[k],width=3)
                draw.line((x-5,y+5,x+5,y-5),fill=POINT[k],width=3)
            else: draw.ellipse((x-3,y-3,x+3,y+3),fill=POINT[k])
        if overlay and np.isfinite(pts).all():
            x,y=pts.min(axis=0)-8; xx,yy=pts.max(axis=0)+8
            draw.rectangle((x,y,xx,yy),outline=ANIMAL[animal],width=2)
            if labels:
                draw.text((x,y-24),f'Mouse {animal+1}',font=movie_font(22),fill=ANIMAL[animal],stroke_width=1,stroke_fill='white')


def load_selection(video_roots):
    """Check retained scenes against Final; resolve videos in supplied folders."""
    records = json.loads(SELECTION.read_text())['clips']
    assignments = pd.read_csv(
        ROOT/'analysis/output/Final/Fig4C/data/Fig4C_window_assignments.csv.gz'
    ).set_index('window_id')
    if {(r['set'], r['motif']) for r in records} != {
        (s, m) for s in range(1, 11) for m in range(4)
    } or len(records) != 40:
        raise ValueError('Expected one selected window per set and motif')
    for r in records:
        row = assignments.loc[r['window_id']]
        if (row.condition != 'control' or int(row.hard_token) != r['motif']
                or int(row.clip_start_frame_id) != r['start_frame']
                or str(row.file_path).replace('\\', '/') != r['pose_csv']):
            raise ValueError(f"Selection differs from Final: {r['window_id']}")
        # Search in command-line order, retaining the original primary-folder priority.
        matches = [(Path(folder)/r['source_video']).resolve() for folder in video_roots
                   if (Path(folder)/r['source_video']).is_file()]
        if not matches:
            raise FileNotFoundError(f"Source video not found: {r['source_video']}")
        r['video_path'] = matches[0]
        paths = [resolve_data_path(r['pose_csv'])]
        if r['pose_npz']:
            paths.append(ROOT/r['pose_npz'])
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
    return sorted(records, key=lambda r: (r['set'], r['motif']))


def coordinates(record, tables):
    """Retain the original frozen-pose/raw-CSV distinction for set 1."""
    if record['pose_npz']:
        with np.load(ROOT/record['pose_npz'], allow_pickle=False) as data:
            return data['coordinates'].copy() * [W/ASPECT, H]
    path = record['pose_csv']
    if path not in tables:
        tables[path] = pd.read_csv(resolve_data_path(path))
    table = tables[path]
    start = record['start_frame']
    index = pd.MultiIndex.from_product(
        [range(start, start+30), sorted(table.track.unique())],
        names=['frame_idx', 'track'],
    )
    sub = table.set_index(['frame_idx', 'track']).reindex(index)
    xy = np.stack([sub[[k+'.x', k+'.y']].to_numpy(float) for k in KEYS], axis=1)
    xy = xy.reshape(30, 3, 6, 2)
    if not np.isfinite(xy).all() or not (sub[[k+'.score' for k in KEYS]].to_numpy() >= .5).all():
        raise ValueError(f"Unresolved keypoints: {record['window_id']}")
    return xy * [W, H]


def compose_frame(panels, frame_index):
    header, height = 120, 120+2*H
    combined = Image.new('RGB', (4*W, height), 'white')
    for motif, (frames, xy) in enumerate(panels):
        panel = Image.new('RGB', (W, height), 'white')
        panel.paste(Image.fromarray(frames[frame_index]), (0, header))
        draw = ImageDraw.Draw(panel)
        draw.text((W/2, 30), f'M{motif}', anchor='mm', font=movie_font(52, True), fill='#202934')
        draw.text((W/2, 87), MOTIF_NAMES[motif], anchor='mm', font=movie_font(44, True), fill='#202934')
        pose_layer(draw, xy, frame_index, header, True, labels=False)
        pose_layer(draw, xy, frame_index, header+H, labels=False)
        draw.rectangle((1, header+H, W-2, height-1), outline=COLORS[motif], width=4)
        combined.paste(panel, (motif*W, 0))
    return combined


def render_movie(records, destination):
    """Render ten three-second sets at one-third of source playback speed."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = ['ffmpeg', '-v', 'error', '-n', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
               '-s', f'{4*W}x{120+2*H}', '-r', '10', '-i', 'pipe:0', '-an',
               '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p',
               '-movflags', '+faststart', str(destination)]
    pipe = subprocess.Popen(command, stdin=subprocess.PIPE)
    tables = {}
    try:
        for number in range(1, 11):
            chosen = [r for r in records if r['set'] == number]
            panels = [(decode(r['video_path'], r['start_frame']), coordinates(r, tables)) for r in chosen]
            for t in range(30):
                pipe.stdin.write(compose_frame(panels, t).tobytes())
            print(f'Set {number}/10 rendered', flush=True)
    except BaseException:
        pipe.terminate()
        try:
            pipe.stdin.close()
        except BrokenPipeError:
            pass
        pipe.wait()
        raise
    pipe.stdin.close()
    if pipe.wait() != 0:
        raise RuntimeError('Movie encoding failed')
    return destination
