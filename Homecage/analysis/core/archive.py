"""Read and verify the frozen publication archive; shared by core and plots."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / 'analysis/output/Final'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def read(panel, name, **kwargs):
    return pd.read_csv(FINAL/panel/name, **kwargs)

def coordinates():
    meta=read('Fig4C','data/Fig4C_window_assignments.csv.gz')
    with np.load(FINAL/'Fig4B/data/Fig4B_coordinates.npz',allow_pickle=False) as a:
        meta=meta.set_index('window_id').loc[a['window_id']].reset_index()
        values=a['factor_coordinates'].copy()
    return values,meta

def materialize_coordinates(path):
    """Rebuild only the legacy adapter format in disposable working storage."""
    values,meta=coordinates()
    payload={k:meta[k].to_numpy(dtype=str if k in ['condition','cage_id','sex','file_path'] else None)
             for k in ['window_id','condition','cage_id','sex','week','file_path','clip_start_frame_id']}
    payload.update(factor_coordinates=values,coordinate_model_id=np.asarray(meta.coordinate_model_id.iloc[0]))
    np.savez_compressed(path,**payload)
    return path

def verify_integrity():
    result={}
    for p in FINAL.glob('*/manifest.json'):
        m=json.loads(p.read_text(encoding='utf-8'))
        for r in m['files']:
            f=p.parent/r['file']
            assert sha(f)==r['sha256'],str(f)
            result[str(f.relative_to(ROOT))]=r['sha256']
    return result
