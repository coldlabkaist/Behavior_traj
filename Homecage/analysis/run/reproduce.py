"""Verify published numbers. Optional refits: --refit-projection, --refit-s5."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import tempfile
from threadpoolctl import threadpool_limits
from analysis.core.archive import ROOT, FINAL, verify_integrity
from analysis.core.verification import CHECKS, verify_boi, verify_motifs, verify_s3_s4, verify_s5, verify_fig5

def main():
    CHECKS.clear()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'analysis/work/paper')
    parser.add_argument('--refit-projection',action='store_true')
    parser.add_argument('--refit-s5',action='store_true')
    parser.add_argument('--rscript',default='C:/Program Files/R/R-4.4.1/bin/Rscript.exe')
    args=parser.parse_args()
    out=args.output.resolve()
    # A caller must not direct regenerated results into curated inputs or data.
    for protected in [ROOT/'analysis/output',ROOT.parent/'MovAl_benchmark',ROOT.parent/'direct_interaction',ROOT.parent/'behavior_assay',ROOT.parent/'data',ROOT/'data',ROOT/'checkpoints']:
        if out==protected or protected in out.parents:raise ValueError('Output overlaps preserved inputs')
    out.mkdir(parents=True,exist_ok=True)
    os.chdir(ROOT)
    legacy_output=(ROOT/'analysis/output').resolve()
    def reject_old_output_reads(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes)):return
        p=Path(os.fsdecode(args[0])).resolve()
        if legacy_output in p.parents and not (p==FINAL or FINAL in p.parents):
            raise RuntimeError('Paper code attempted to access an uncurated output: '+str(p))
    sys.addaudithook(reject_old_output_reads)
    before=verify_integrity()
    CHECKS['verified_archive_files']=len(before)
    with tempfile.TemporaryDirectory(prefix='paper_',dir=out) as tmp,threadpool_limits(limits=1):
        work=Path(tmp)
        for label,fn in [('BOI',lambda:verify_boi(work,args.refit_projection)),
                         ('motifs, age effect, S6',lambda:verify_motifs(work)),
                         ('S3 training and S4 factors',verify_s3_s4),
                         ('S5',lambda:verify_s5(work,args.refit_s5)),
                         ('Fig5G/I',lambda:verify_fig5(work))]:
            print('Recalculating '+label,flush=True);fn()
        subprocess.run([args.rscript,str(ROOT/'analysis/run/inference.R'),str(work)],check=True,cwd=ROOT)
        for p in work.glob('*.csv'):
            if p.name!='primary_soft_joint_cage_week_distributions.csv':
                (out/p.name).write_bytes(p.read_bytes())
        for p in work.glob('*.json'):(out/p.name).write_bytes(p.read_bytes())
    assert before==verify_integrity()
    report={'status':'passed','inputs':'analysis/output/Final','checks':CHECKS,
            'original_archive_unchanged':True,'projection_refitted':args.refit_projection,
            'S5_GMM_refits_executed':args.refit_s5,
            'limits':['Default S5 verification reaggregates saved refits; it does not refit every GMM.',
                      'S3 nonlinear probe refitting is a separate command: python -m analysis.run.probe.',
                      'No claim of pixel-identical figure rendering or Illustrator reassembly.']}
    (out/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__ == '__main__':
    main()
