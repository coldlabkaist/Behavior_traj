"""Connect published panel names to their plot functions and Final inputs."""
from argparse import Namespace
import numpy as np
import pandas as pd
from analysis.core.archive import FINAL, coordinates
from analysis.core.motif_stability import summarize_resolution
from analysis.core.temporal_shift import summarize_temporal_shift

PANELS = ('Fig4B', 'Fig4C', 'Fig4D', 'Fig4EFG', 'Fig5G', 'Fig5I',
          'FigS3AB', 'FigS4AB', 'FigS5AB', 'FigS6AB', 'FigS7AB')

def render(panel,out):
    out.mkdir(parents=True,exist_ok=True)
    if panel=='Fig4B':
        import analysis.plots.distribution as s
        values,meta=coordinates();ranges=s._shared_ranges(values)
        panels,audit,norm=s._prepare_panels(values,meta,ranges,bins=30,points_per_cage_week=400,points_per_cage_overall=600)
        s._plot_by_week_talk(panels,ranges,norm,out,poster=False)
        s._plot_colorbar(norm,out)
    elif panel=='Fig4C':
        import analysis.plots.motifs as s
        s.render_motif_space(FINAL/panel/'data/Fig4C_window_assignments.csv.gz',
                         FINAL/panel/'data/Fig4C_codebook_parameters.npz',out/'Fig4C_motifs_3d')
        pose={}
        for i in range(4):
            with np.load(FINAL/panel/f'data/Fig4C_M{i}_pose.npz',allow_pickle=False) as a:
                pose[f'M{i}']=a['coordinates']
        s.render_representative_poses(pose,out/'Fig4C_representative_poses')
    elif panel=='Fig4D':
        import analysis.plots.transitions as s
        s.render_weekly_transitions(FINAL/panel/'stat/Fig4D_weekly_transition_summary.csv',
                 FINAL/panel/'stat/Fig4D_weekly_occupancy_summary.csv',out/'Fig4D_transition_network')
    elif panel=='Fig4EFG':
        from analysis.plots.boi import render_trajectory as plot
        plot(out)
    elif panel=='FigS7AB':
        from analysis.plots.boi import render_sex_endpoints as plot
        plot(out)
    elif panel in ['Fig5G','Fig5I']:
        import analysis.plots.associations as s
        g=pd.read_csv(FINAL/'Fig5G/data/Fig5G_litter_data.csv')
        if panel=='Fig5G':data=g;config=s.PANELS[0]
        else:
            data=pd.read_csv(FINAL/'Fig5I/data/Fig5I_litter_data.csv').merge(
                g[['condition','maternal_cage_id','score_3w']],on=['condition','maternal_cage_id'],validate='one_to_one',sort=False)
            config=s.PANELS[1]
        stats=pd.read_csv(FINAL/panel/f'stat/{panel}_spearman_tests.csv')
        s.render(data,stats,config,out)
    elif panel=='FigS3AB':
        from analysis.plots.validation import render_dae_validation
        render_dae_validation(out,talk=True)
    elif panel=='FigS4AB':
        from analysis.plots.validation import render_factor_validation
        render_factor_validation(out,talk=True)
    elif panel=='FigS5AB':
        import analysis.plots.stability as s
        metrics=pd.read_csv(FINAL/panel/'data/FigS5A_cage_metrics.csv')
        s.render_resolution(metrics,summarize_resolution(metrics),pd.DataFrame(),out)
        data=pd.read_csv(FINAL/panel/'data/FigS5B_cage_metrics.csv')
        s.render_temporal_shift(data,summarize_temporal_shift(data),out)
    elif panel=='FigS6AB':
        import analysis.plots.transitions as s
        s.render_network_changes(Namespace(motif_dir=FINAL/'Fig4C/data',transition_mode='hard',output_dir=out),
              assignments=pd.read_csv(FINAL/'Fig4C/data/Fig4C_window_assignments.csv.gz'))
    else:raise ValueError(panel)
