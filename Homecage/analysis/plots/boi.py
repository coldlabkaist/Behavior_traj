"""BOI trajectory and sex endpoint figures."""
from __future__ import annotations
from analysis.plots.style import apply_style, CONTROL_COLOR, VPA_COLOR
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import t, false_discovery_control
from PIL import Image
from analysis.plots.export import save_figure
from matplotlib.lines import Line2D


# BOI developmental trajectory

def render_trajectory(output: Path):
    ROOT = Path(__file__).resolve().parents[2]
    BASE = ROOT / 'analysis/output/Final/Fig4EFG'
    OUT = Path(output)
    OUT.mkdir(parents=True, exist_ok=True)
    STEM = 'BOI6_dam_CR1_Satterthwaite_EFG'
    inputs = [BASE/f'stat/Fig4_{p}_statistics.csv' for p in ['E','F','G']] + [BASE/'stat/Fig4_E_weekly_means_CR1.csv', BASE/'stat/Fig4_F_auc_means_CR1.csv', BASE/'data/Fig4_F_cage_auc.csv']
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    stats = pd.concat([pd.read_csv(p) for p in inputs[:3]],ignore_index=True).set_index('comparison')
    weekly = pd.read_csv(inputs[3])
    auc_model = pd.read_csv(inputs[4]).set_index('condition')
    auc_raw = pd.read_csv(inputs[5])
    keys = [f'{co}_W{w}_W{w+1}' for co in ['control','vpa'] for w in [3,4,5]]
    g = stats.loc[keys]
    np.testing.assert_allclose(false_discovery_control(g.p.to_numpy()), g.q, atol=1e-12)
    np.testing.assert_allclose(false_discovery_control(stats.loc[['E_W4','E_W5'],'p']), stats.loc[['E_W4','E_W5'],'q'], atol=1e-12)
    pp = np.where(stats.one_sided, t.sf(stats.t,stats.df), 2*t.sf(abs(stats.t),stats.df))
    np.testing.assert_allclose(pp, stats.p, atol=1e-12)
    apply_style('boi')
    fig, axs = plt.subplots(1,3,figsize=(23,6.7),gridspec_kw={'width_ratios':[1.65,.85,1.65]})
    ax,mid,right = axs
    marks = []
    def symbol(p):
        return '***' if p<.001 else '**' if p<.01 else '*' if p<.05 else '†' if p<.1 else ''
    def mark(axis,x,y,p,key):
        s=symbol(p)
        marks.append({'comparison':key,'symbol':s,'p_or_q':float(p)})
        if s:
            axis.text(x,y,s,ha='center',va='bottom',fontsize=20,fontweight='bold',color='#202934')
    for j,(co,col,m,ls,label) in enumerate([('control',CONTROL_COLOR,'o','-','Control'),('vpa',VPA_COLOR,'s','--','VPA')]):
        h=weekly[weekly.condition==co].sort_values('week_f')
        x=h.week_f.to_numpy();y=h.model_mean.to_numpy();se=h.CR1_SE.to_numpy()
        ax.plot(x,y,color=col,marker=m,ls=ls,lw=3,ms=10,mec='white',mew=1.1,label=label)
        ax.fill_between(x,y-se,y+se,color=col,alpha=.18,lw=0)
        values=auc_raw[auc_raw.condition==co].normalized_auc.to_numpy()
        mid.scatter(j+np.random.default_rng(91+j).uniform(-.08,.08,len(values)),values,
                    color=col,marker=m,s=90,alpha=.85,edgecolor='white',linewidth=.8)
        am=auc_model.loc[co]
        mid.errorbar(j,am.model_mean,yerr=am.CR1_SE,fmt=m,mfc='white',mec=col,color=col,
                     ms=14,mew=2.4,capsize=7,capthick=2,lw=2.4)
        for w in [3,4,5]:
            key=f'{co}_W{w}_W{w+1}';row=g.loc[key];xx=w-3+(j-.5)*.24
            right.errorbar(xx,row.estimate,yerr=row.SE,fmt=m,color=col,ms=12,mec='white',mew=1.2,
                           capsize=6,capthick=2,lw=2.2,label=label if w==3 else None)
            mark(right,xx,row.estimate+row.SE+.045,row.q,key)
    for y in [0,1]:
        ax.axhline(y,color='#A7B4C3',ls='--',alpha=.85,lw=2,zorder=0)
    top=json.loads((BASE/'manifest.json').read_text(encoding='utf-8'))['plot_layout']['e_annotation_top']
    for w in [4,5]:
        mark(ax,w,top,stats.loc[f'E_W{w}','q'],f'E_W{w}')
    ax.plot([4,5],[top+.16,top+.16],color='#202934',lw=1.8)
    mark(ax,4.5,top+.18,stats.loc['E_W4_W5','p'],'E_W4_W5')
    ax.set(xlabel='Postnatal age',ylabel='Behavioral organization index (BOI)',xticks=range(3,9),
           xticklabels=[f'{i}W' for i in range(3,9)],ylim=(-.4,top+.38))
    atop=auc_raw.normalized_auc.max()+.13
    mid.plot([0,0,1,1],[atop-.03,atop,atop,atop-.03],color='#202934',lw=1.8)
    mark(mid,.5,atop+.02,stats.loc['F_AUC_W3_W6','p'],'F_AUC_W3_W6')
    mid.set(xticks=[0,1],xticklabels=['Control','VPA'],ylabel='3–6W BOI AUC',xlim=(-.4,1.4),
            ylim=(min(-.7,auc_raw.normalized_auc.min()-.08),atop+.17))
    right.axhline(0,color='#9DA6AF',lw=1.4)
    right.set(xlabel='Adjacent-week interval',ylabel='BOI gain',xticks=range(3),
              xticklabels=['3–4W','4–5W','5–6W'],xlim=(-.5,2.5),ylim=(-.25,1.12))
    ax.legend(frameon=False,ncol=2,loc='lower left',bbox_to_anchor=(0,1.005),borderaxespad=0)
    right.legend(frameon=False,ncol=2,loc='lower right',bbox_to_anchor=(1,1.005),borderaxespad=0)
    for axis,title in zip(axs,['BOI trajectory','Early-period BOI','Condition-specific adjacent-week BOI gain']):
        axis.set_title(title,pad=42)
        axis.tick_params(width=1.6,length=7,colors='#202934')
    fig.subplots_adjust(left=.055,right=.99,bottom=.16,top=.85,wspace=.35)
    save_figure(fig, OUT/STEM, svg_dpi=300)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    for axis in axs:
        for artist in [axis.title,axis.xaxis.label,axis.yaxis.label,*axis.texts]:
            box=artist.get_window_extent(renderer)
            assert box.x0>=0 and box.y0>=0 and box.x1<=fig.bbox.width and box.y1<=fig.bbox.height
    plt.close(fig)
    with Image.open(OUT/f'{STEM}.png') as im:
        assert im.size==(6900,2010)
        assert all(abs(v-300)<.1 for v in im.info['dpi'])
    expected={'E_W4':'†','E_W5':'*','E_W4_W5':'*','F_AUC_W3_W6':'*',
              'control_W3_W4':'','control_W4_W5':'†','control_W5_W6':'†',
              'vpa_W3_W4':'*','vpa_W4_W5':'†','vpa_W5_W6':''}
    assert {x['comparison']:x['symbol'] for x in marks}==expected
    assert hashes=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    audit={'model':'dam/litter/cage random intercepts; condition x categorical week; heterogeneous residuals',
     'inference':'dam-clustered CR1 + Satterthwaite','dam_counts':{'control':5,'vpa':8},
     'E_BH':2,'G_BH':6,'gain_alternative':'greater','SE_display':'CR1 standard error, not raw SEM',
     'between_gain_not_drawn':True,
     'marks':marks,'dpi':300,'dimensions':[6900,2010],'source_hashes':hashes,'text_inside_canvas':True}
    (OUT/'figure_verification.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(audit,indent=2,ensure_ascii=False))


# Sex endpoints

def render_sex_endpoints(output: Path):
    ROOT = Path(__file__).resolve().parents[2]
    BASE = ROOT / 'analysis/output/Final/FigS7AB'
    OUT = Path(output)
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(BASE / 'data/FigS7AB_cage_endpoints.csv')
    tests = pd.read_csv(BASE / 'stat/FigS7AB_factorial_tests.csv')
    scores = pd.read_csv(ROOT / 'analysis/output/Final/Fig4EFG/data/Fig4_EFG_model_observations.csv')
    sex = pd.read_csv(ROOT / 'analysis/output/Final/Fig4C/data/Fig4C_window_assignments.csv.gz', usecols=['condition','cage_id','sex']).drop_duplicates()
    scores = scores.merge(sex, on=['condition','cage_id'], validate='many_to_one')
    keys = ['condition', 'sex', 'cage_id']
    wide = scores.pivot(index=keys, columns='week', values='developmental_score').sort_index()
    obs = raw.set_index(keys).sort_index()
    np.testing.assert_allclose(obs.W4_W5_mean, (wide[4]+wide[5])/2, atol=1e-12)
    np.testing.assert_allclose(obs.W3_W6_AUC, (wide[3]+2*wide[4]+2*wide[5]+wide[6])/6, atol=1e-12)
    assert not raw.cage_id.duplicated().any() and len(raw) == 22
    apply_style('sex_endpoints')
    palette = [('control', CONTROL_COLOR, 'o', 'Control'), ('vpa', VPA_COLOR, 's', 'VPA')]
    endpoints = [('W4_W5_mean', 'W4–W5 mean BOI', 'Mean BOI'),
                 ('W3_W6_AUC', 'W3–W6 BOI AUC', 'Normalized BOI AUC')]
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.8))
    fig.subplots_adjust(left=.085, right=.985, bottom=.12, top=.86, wspace=.28)
    records = []
    n_points = 0
    for ax, (ep, title, ylabel) in zip(axes, endpoints):
        for i, sex in enumerate(['m', 'f']):
            for j, (condition, color, marker, label) in enumerate(palette):
                x = i + (j-.5)*.36
                values = raw.loc[(raw.sex==sex)&(raw.condition==condition), ep].to_numpy()
                assert len(values) == (4 if sex=='f' and condition=='vpa' else 6)
                ax.boxplot([values], positions=[x], widths=.27, whis=1.5,
                    patch_artist=True, showfliers=False, manage_ticks=False,
                    boxprops=dict(facecolor=matplotlib.colors.to_rgba(color,.18), edgecolor=color, linewidth=1.8),
                    medianprops=dict(color=color, linewidth=2.4),
                    whiskerprops=dict(color=color, linewidth=1.6),
                    capprops=dict(color=color, linewidth=1.6))
                jitter = np.random.default_rng(920+i*2+j).uniform(-.08,.08,len(values))
                ax.scatter(x+jitter, values, color=color, marker=marker, s=55,
                    edgecolor='white', linewidth=.7, alpha=.9, zorder=4)
                n_points += len(values)
                q1, med, q3 = np.quantile(values, [.25,.5,.75])
                inside = values[(values>=q1-1.5*(q3-q1)) & (values<=q3+1.5*(q3-q1))]
                records.append(dict(endpoint=ep,sex=sex,condition=condition,n=len(values),
                    q1=q1,median=med,q3=q3,whisker_low=inside.min(),whisker_high=inside.max(),
                    minimum=values.min(),maximum=values.max()))
        ax.axhline(0, color='#B9C2CD', linewidth=1, zorder=0)
        ax.set(xlim=(-.55,1.55), xticks=[0,1], xticklabels=['Male','Female'], ylabel=ylabel)
        low, high = raw[ep].min(), raw[ep].max()
        pad = max(.1,(high-low)*.09)
        ax.set_ylim(low-pad, high+max(pad,(high-low)*.25))
        ax.tick_params(width=1.6, length=6)
        ax.set_title(title, pad=18)
        legend = [Line2D([0],[0],marker=marker,color=color,linestyle='none',markersize=8,label=label)
                  for _,color,marker,label in palette]
        ax.legend(handles=legend,loc='upper right',ncol=2,frameon=False,fontsize=14,
                  handletextpad=.4,columnspacing=1.1)
    assert n_points==44
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    for ax in axes:
        for artist in [ax.title,ax.yaxis.label,*ax.texts]:
            b=artist.get_window_extent(renderer)
            assert b.x0>=0 and b.y0>=0 and b.x1<=fig.bbox.width and b.y1<=fig.bbox.height
    save_figure(fig, OUT/'BOI6_sex_factorial_boxplots', svg_dpi=300)
    plt.close(fig)
    pd.DataFrame(records).to_csv(OUT/'boxplot_summary.csv',index=False)
    caption = ('Cage-level mean BOI across W4–W5 (left) and normalized W3–W6 BOI AUC (right), '
        'calculated from the unchanged six-feature Ridge BOI. Boxes show the median and interquartile range; '
        'whiskers extend to the most extreme observations within 1.5 times the interquartile range. '
        'All cage values, including values beyond the whiskers, are shown as points (Control, circles; VPA, squares). '
        'Sample sizes are Control: 6 male and 6 female cages; VPA: 6 male and 4 female cages. '
        'No sample-size labels or statistical annotations are displayed in the figure. The accompanying report contains '
        'endpoint-specific condition, sex, and condition-by-sex contrasts in the existing '
        'longitudinal mixed model with approximate Satterthwaite degrees of freedom, accounting for dam/litter/cage '
        'clustering and condition- and week-dependent residual variances. BH correction was applied across the two '
        'endpoints separately for each effect type. Tests concern model-estimated means, not box medians. '
        'No within-sex pairwise comparison brackets are shown. See REPORT.md for model limitations.')
    (OUT/'BOXPLOT_CAPTION.txt').write_text(caption,encoding='utf-8')
    (OUT/'boxplot_verification.json').write_text(json.dumps(dict(cages=22,points_across_panels=n_points,
        endpoints_recomputed=True,boxes=8,all_observations_shown=True,model_refitted=False),indent=2),encoding='utf-8')
    print(OUT/'BOI6_sex_factorial_boxplots.png')
    print('Verified: 22 cages, 44 points, 8 boxes; original endpoints reproduced.')
