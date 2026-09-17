"""Render selected publication panels from Final into working storage."""
from argparse import ArgumentParser
from pathlib import Path
import json
import os
import sys
import matplotlib
matplotlib.use('Agg')
from threadpoolctl import threadpool_limits
from analysis.core.archive import ROOT, FINAL, verify_integrity, sha
from analysis.plots.panels import PANELS, render
from analysis.plots.style import style_context

def main():
    panels=list(PANELS)
    p=ArgumentParser(description=__doc__)
    p.add_argument('--panels',nargs='+',choices=panels,default=panels)
    p.add_argument('--output',type=Path,default=ROOT/'analysis/work/paper_figures')
    args=p.parse_args();out=args.output.resolve()
    for protected in [ROOT/'analysis/output',ROOT.parent/'MovAl_benchmark',ROOT.parent/'direct_interaction',ROOT.parent/'behavior_assay',ROOT.parent/'data',ROOT/'data',ROOT/'checkpoints']:
        if out==protected or protected in out.parents:raise ValueError('Output overlaps preserved inputs')
    os.chdir(ROOT);sys.path.insert(0,str(ROOT/'analysis'))
    old_output=(ROOT/'analysis/output').resolve()
    def guard(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)):
            path=Path(os.fsdecode(args[0])).resolve()
            if old_output in path.parents and not FINAL in path.parents:
                raise RuntimeError('Uncurated output access: '+str(path))
    sys.addaudithook(guard)
    before=verify_integrity()
    with threadpool_limits(limits=1):
        for panel in args.panels:
            print('Rendering '+panel,flush=True)
            with style_context():
                render(panel,out/panel)
    assert verify_integrity()==before
    files=[p for p in out.rglob('*') if p.suffix in ['.png','.svg']]
    (out/'render_manifest.json').write_text(json.dumps({'panels':args.panels,'archive_unchanged':True,
        'note':'Re-rendered review copies; pixel identity and final Illustrator assembly are not assumed.',
        'files':[{'file':str(p.relative_to(out)),'sha256':sha(p)} for p in files]},indent=2),encoding='utf-8')
    print('Rendered',len(files),'PNG/SVG files; Final unchanged.',flush=True)

if __name__ == '__main__':
    main()
