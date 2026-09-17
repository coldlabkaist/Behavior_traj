"""Recalculate and compare published measurements from frozen Final inputs.

CLI parsing, subprocesses and execution order live in analysis.run.
"""
from pathlib import Path
import io
import json
import numpy as np
import pandas as pd
import torch
from analysis.core.archive import ROOT, FINAL, read, coordinates, materialize_coordinates, sha
from analysis.core.motif_stability import summarize_resolution
from analysis.core.temporal_shift import summarize_temporal_shift, _mean_ci

CHECKS = {}

def compare(name, actual, expected, keys, columns, tolerance=1e-10):
    columns=list(columns)
    a,b=actual.copy(),expected.copy()
    if 'shift_magnitude_seconds' in keys:
        for d in [a,b]:
            frames=np.rint(d.shift_magnitude_seconds*30).astype(int)
            np.testing.assert_allclose(d.shift_magnitude_seconds,frames/30,atol=1e-15,rtol=0)
            d['shift_magnitude_seconds']=frames
    a=a.set_index(keys).sort_index();b=b.set_index(keys).sort_index()
    pd.testing.assert_index_equal(a.index,b.index,exact=False)
    av=a[columns].to_numpy(float);bv=b[columns].to_numpy(float)
    np.testing.assert_allclose(av,bv,atol=tolerance,rtol=0,equal_nan=True,err_msg=name)
    CHECKS[name]=float(np.nanmax(np.abs(av-bv)))

def verify_boi(work, refit_projection):
    import analysis.core.projection as pipeline
    import analysis.core.boi as boi
    values,meta=coordinates()
    frame=meta[['window_id','coordinate_model_id','condition','cage_id','sex','week','file_path','clip_start_frame_id']].copy()
    frame[['spacing','orientation','dynamics']]=values
    if refit_projection:
        print('Refitting original 12 Control-cage MLP folds',flush=True)
        fitdir=work/'projection_refit';fitdir.mkdir(exist_ok=True)
        frame,_,_=pipeline._fit_projection(FINAL/'Fig4B/data/latent_windows.pt',
            FINAL/'FigS4AB/data/primary_factor_model.npz',fitdir,seed=42,cap_per_cage_week=300)
        actual=frame[['spacing','orientation','dynamics']].to_numpy()
        np.testing.assert_allclose(actual,values,atol=1e-10,rtol=0)
        CHECKS['MLP_projection_max_error']=float(np.max(np.abs(actual-values)))
        expected=read('Fig4B','data/projection_folds.csv')
        got=pd.read_csv(fitdir/'01_latent_projection/control_loco_projection_folds.csv')
        compare('projection_folds',got,expected,['fold','held_out_control_cage'],
                list(expected.select_dtypes('number').columns.drop('fold')),1e-9)
    # Original BOI consumes coordinates AFTER %.10g CSV serialization.
    frame=pd.read_csv(io.StringIO(frame.to_csv(index=False,float_format='%.10g')))
    summaries=boi._summarize(frame)
    scores,_,_=boi._crossfit(summaries)
    expected=read('Fig4EFG','data/Fig4_EFG_model_observations.csv')
    compare('BOI_scores',scores,expected,['condition','cage_id','week'],['developmental_score'],1e-10)
    scores.to_csv(work/'BOI_recalculated.csv',index=False)

def verify_motifs(work):
    import analysis.core.transitions as age
    frame=read('Fig4C','data/Fig4C_window_assignments.csv.gz')
    from sklearn.mixture import GaussianMixture
    from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky
    params=dict(np.load(FINAL/'Fig4C/data/Fig4C_codebook_parameters.npz',allow_pickle=False))
    model=GaussianMixture(n_components=4,covariance_type='full')
    model.weights_=params['weights'];model.means_=params['means'];model.covariances_=params['covariances']
    model.precisions_cholesky_=_compute_precision_cholesky(model.covariances_,'full')
    model.n_features_in_=3
    posterior=model.predict_proba(frame[['spacing','orientation','dynamics']].to_numpy())
    stored=frame[[f'p_M{i}' for i in range(4)]].to_numpy()
    np.testing.assert_allclose(posterior,stored,atol=1e-8,rtol=0)
    np.testing.assert_array_equal(posterior.argmax(axis=1),frame.hard_token)
    CHECKS['GMM_posterior']=float(np.max(np.abs(posterior-stored)))
    transitions,occupancy,weekly,weekocc=age._transition_tables(frame)
    for name,actual,file,keys in [
        ('weekly_transition',weekly,'Fig4D_weekly_transition_summary.csv',['condition','week','source','target']),
        ('weekly_occupancy',weekocc,'Fig4D_weekly_occupancy_summary.csv',['condition','week','token'])]:
        expected=read('Fig4D','stat/'+file)
        compare(name,actual,expected,keys,list(expected.select_dtypes('number').columns.difference(keys)))
    rows=[]
    for key,g in frame.groupby(['condition','cage_id','week']):
        g=g.sort_values(['file_path','clip_start_frame_id'])
        p=g[[f'p_M{i}' for i in range(4)]].to_numpy()
        f=g.file_path.to_numpy();starts=g.clip_start_frame_id.to_numpy()
        valid=(f[1:]==f[:-1])&(np.diff(starts)==30)
        joint=(p[:-1,:,None]*p[1:,None,:]).reshape(-1,16)[valid].mean(axis=0)
        joint/=joint.sum()
        rows.append(dict(zip(['condition','cage_id','week'],key))|{'n_adjacent_pairs':int(valid.sum())}|dict(zip(age.JOINT_COLUMNS,joint)))
    joint=pd.DataFrame(rows)
    compare('soft_joint',joint,read('Fig4D','data/Fig4D_soft_joint_cage_week.csv'),
            ['condition','cage_id','week'],age.JOINT_COLUMNS)
    # The original transform accepts a directory; use temporary output, never a date folder.
    joint.to_csv(work/'primary_soft_joint_cage_week_distributions.csv',index=False)
    transformed=age.load_joint_table(work)
    rng=np.random.default_rng(20260906);results=[]
    for condition in age.CONDITIONS:
        subset=transformed.loc[transformed.condition.eq(condition)]
        results.append({'condition':condition,**age.global_age_permutation(subset,n_permutations=100000,rng=rng)})
    result=pd.DataFrame(results)
    result['p_holm_across_conditions']=age.holm(result.p_value.to_numpy())
    expected=read('Fig4D','stat/Fig4D_global_age_effect.csv')
    compare('global_age',result,expected,['condition'],['p_value','p_holm_across_conditions'])
    result.to_csv(work/'Fig4D_age_recalculated.csv',index=False)
    changes=age.build_change_summary(None,transition_mode='hard',assignments=frame)
    expected=read('FigS6AB','stat/FigS6AB_network_changes.csv',keep_default_na=False)
    compare('S6_changes',changes,expected,['condition','start_week','end_week','feature_type','feature','source','target'],
            ['n_cages','mean_change_pp','sd_change_pp'])
    changes.to_csv(work/'FigS6_changes_recalculated.csv',index=False)

def verify_s3_s4():
    import analysis.core.factor_analysis as fa
    checkpoint=torch.load(ROOT/'checkpoints/best_clean.pth',map_location='cpu',weights_only=False)
    history=read('FigS3AB','data/FigS3A_training_history.csv')
    np.testing.assert_allclose(history.train_loss,checkpoint['train_losses'],rtol=0,atol=1e-14)
    val=np.asarray([np.nan if v is None else v for v in checkpoint['val_losses']])
    np.testing.assert_allclose(history.validation_loss,val,rtol=0,atol=1e-14,equal_nan=True)
    CHECKS['S3_training_history_epochs']=len(history)
    cache=torch.load(FINAL/'Fig4B/data/latent_windows.pt',map_location='cpu',weights_only=False)
    model=dict(np.load(FINAL/'FigS4AB/data/primary_factor_model.npz',allow_pickle=False))
    metadata=fa._metadata(cache);ix=fa._balanced_control_indices(metadata,300,42)
    np.testing.assert_array_equal(ix,model['reference_indices'])
    spec=fa.DESIGNS['legacy_fa8']
    raw=cache['relation_mean'].numpy().astype(float)[ix][:,spec['indices']]
    values,_=fa._standardize(raw)
    observed,null=fa._block_parallel_analysis(values,metadata.loc[ix].reset_index(drop=True),300,52)
    pattern,phi,converged=fa._fit_three_factor(values,spec['groups'],spec['anchors'])
    assert converged
    for name,actual,expected in [('FA_loadings',pattern,model['pattern_loadings']),
        ('FA_correlations',phi,model['factor_correlation']),('parallel_observed',observed,model['observed_eigenvalues']),
        ('parallel_null',null,model['block_null_95_eigenvalues'])]:
        np.testing.assert_allclose(actual,expected,atol=1e-10,rtol=0)
        CHECKS[name]=float(np.max(np.abs(actual-expected)))

def verify_s5(work, refit):
    import analysis.core.motif_stability as stability
    import analysis.core.temporal_shift as shift
    base=FINAL/'FigS5AB/data'
    a=read('FigS5AB','data/FigS5A_cage_metrics.csv')
    n=stability._cage_novelty(FINAL/'Fig4B/data/latent_windows.pt',base/'codebooks')
    compare('S5A_novelty',n,a,['cage','k'],['novelty'],1e-8)
    raw=pd.read_csv(base/'stability_refits.csv')
    hr=pd.read_csv(base/'temporal_shift_hard_refits.csv')
    sr=pd.read_csv(base/'temporal_shift_soft_refits.csv')
    if refit:
        coordinate_path=materialize_coordinates(work/'coordinates_adapter.npz')
        raw=stability._stability_by_evaluation_cage(coordinate_path,base/'codebooks',work/'stability_refits.csv',
              iterations=50,per_cage_week=220,seed=42)
        inputs=shift.load_inputs(coordinate_path,base/'shifted_coordinates.npz')
        hr=shift.run_hard_validation(inputs,iterations=8,per_cage_week=300,seed=20260826,cache_path=work/'hard_refits.csv')
        sr=shift.run_soft_validation(inputs,iterations=8,per_cage_week=300,seed=20260826,cache_path=work/'soft_refits.csv')
    assert raw.groupby(['evaluation_cage','k']).size().eq(50).all()
    st=raw.groupby(['evaluation_cage','k'],as_index=False).soft_agreement.mean().rename(
        columns={'evaluation_cage':'cage','soft_agreement':'stability'})
    compare('S5A_stability',st,a,['cage','k'],['stability'],1e-8 if refit else 1e-12)
    compare('S5A_summary',summarize_resolution(a),read('FigS5AB','stat/FigS5A_metric_summary.csv'),
            ['metric','k'],['mean','ci95_low','ci95_high'])
    b=read('FigS5AB','data/FigS5B_cage_metrics.csv')
    for label,got,cols in [('hard',shift.hard_cage_level(hr),['hard_agreement']),
                            ('soft',shift.soft_cage_level(sr),['js_similarity'])]:
        compare('S5B_'+label,got,b,['held_out_cage','shift_magnitude_seconds'],cols,1e-8 if refit else 1e-12)
    expected=read('FigS5AB','stat/FigS5B_shift_summary.csv')
    compare('S5B_summary',summarize_temporal_shift(b),expected,['metric','shift_magnitude_seconds'],
            ['mean','ci95_low','ci95_high','decrease_from_no_shift'])
    for _,row in expected.iterrows():
        pivot=b.pivot(index='held_out_cage',columns='shift_magnitude_seconds',values=row.metric)
        shift=pivot.columns[np.argmin(abs(pivot.columns-row.shift_magnitude_seconds))]
        _,lo,hi=_mean_ci((pivot[0]-pivot[shift]).to_numpy())
        np.testing.assert_allclose([lo,hi],[row.decrease_ci95_low,row.decrease_ci95_high],atol=1e-12,rtol=0)
    CHECKS['S5B_paired_CI_verified']=True

def verify_fig5(work):
    import analysis.core.rank_tests as source
    g=read('Fig5G','data/Fig5G_litter_data.csv')
    i=read('Fig5I','data/Fig5I_litter_data.csv').merge(g[['condition','maternal_cage_id','score_3w']],
        on=['condition','maternal_cage_id'],validate='one_to_one',sort=False)
    score=read('Fig4EFG','data/Fig4_EFG_model_observations.csv');score=score.loc[score.week.eq(3)].copy()
    score['maternal_cage_id']=[l.removeprefix(c+'.') for l,c in zip(score.litter,score.condition)]
    score=score.groupby(['condition','maternal_cage_id'],as_index=False).developmental_score.mean().rename(columns={'developmental_score':'score_3w'})
    compare('Fig5_litter_BOI_link',g,score,['condition','maternal_cage_id'],['score_3w'])
    for panel,frame,config in [('Fig5G',g,{'predictor':'filtered_total_pct_1s'}),('Fig5I',i,{'predictor':'avg_velocity_mm_s'})]:
        actual=source.calculate_statistics(frame,config['predictor'])
        compare(panel+'_Spearman',actual,read(panel,'stat/'+panel+'_spearman_tests.csv'),
                ['analysis'],['spearman_rho','exact_permutation_p'],1e-14)
        actual.to_csv(work/(panel+'_statistics.csv'),index=False)

def refit_probe(out):
    from analysis.core.nonlinear_probe import run_nonlinear_probe
    CHECKS.clear()
    targets,_,_=run_nonlinear_probe(FINAL/'Fig4B/data/latent_windows.pt',out,
                                  cap_per_cage_week=300,seed=42)
    expected=read('FigS3AB','stat/FigS3B_latent_feature_r2.csv')
    compare('nonlinear_probe',targets[targets.target.isin(expected.target)],expected,
            ['target'],['nonlinear_r2'],1e-9)
    return {'status':'passed','checks':dict(CHECKS)}


def extract_and_verify_latent(destination, device):
    from analysis.common import load_yaml, load_frozen_model
    from analysis.core.latent_cache import extract_latent_cache, load_latent_cache
    config=load_yaml(ROOT/'cfg/train_config.yaml')
    model,_=load_frozen_model(config,ROOT/'checkpoints/best_clean.pth',device)
    reference=load_latent_cache(FINAL/'Fig4B/data/latent_windows.pt')
    actual=extract_latent_cache(model=model,config=config,device=device,
        cohorts={'control':'data/cont','vpa':'data/experiments'},
        output_path=destination,batch_size=256)
    errors={}
    for key in ['condition','cage_id','sex','week','file_path','clip_start_frame_id']:
        np.testing.assert_array_equal(np.asarray(actual[key]),np.asarray(reference[key]),err_msg=key)
    for key in ['z','relation_mean']:
        a,b=actual[key].cpu().numpy(),reference[key].cpu().numpy()
        errors[key]=float(np.max(np.abs(a-b)))
    report={'batch_size':256,'torch_threads':4,'device':str(device),
            'torch_version':str(torch.__version__),'metadata_identical':True,
            'max_absolute_errors':errors,'tensor_values_identical':all(x==0 for x in errors.values()),
            'candidate_sha256':sha(destination),
            'reference_sha256':sha(FINAL/'Fig4B/data/latent_windows.pt')}

    return report
