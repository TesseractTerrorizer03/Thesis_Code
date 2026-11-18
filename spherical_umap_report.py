"""Plotting/report helpers for the spherical UMAP notebook; no model selection."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from IPython.display import Markdown, display
from spherical_umap import to_xyz

COLORS={'Spherical UMAP':'#2563eb','Planar UMAP':'#d97706','PCA2':'#059669','PCA3':'#7c3aed'}


def save(fig,bundle,name):
    folder=Path(bundle['run_dir'])/'figures';folder.mkdir(exist_ok=True)
    fig.savefig(folder/f'{name}.png',dpi=155,bbox_inches='tight')
    fig.savefig(folder/f'{name}.pdf',bbox_inches='tight')
    plt.show()


def summary(bundle):
    cards=bundle['scorecards']
    fig,axes=plt.subplots(2,2,figsize=(12,7),sharey=True)
    for ax,(track,g) in zip(axes.flat,cards.groupby('track',sort=False)):
        for x,(_,r) in enumerate(g.iterrows()):
            ax.errorbar(x,r.test_gain*100,yerr=np.array([[r.test_gain-r.lower],[r.upper-r.test_gain]])*100,
                        fmt='o',color='#2563eb',capsize=5,ms=7)
        ax.axhline(0,color='#334155',lw=1)
        ax.axhline(-2,color='#b91c1c',ls='--',lw=1,label='Noninferiority margin')
        ax.set_xticks(range(len(g)),g.metric)
        ax.set_title(track);ax.set_ylabel('Spherical − planar recall (percentage points)')
        ax.grid(axis='y',alpha=.2)
    fig.suptitle('Retrospective validation: does the sphere preserve more neighbors?',fontsize=14)
    fig.tight_layout();save(fig,bundle,'01_primary_comparison')


def sphere(ax,angles,color,label):
    xyz=to_xyz(angles)
    theta=np.linspace(0,2*np.pi,36);phi=np.linspace(0,np.pi,18)
    ax.plot_wireframe(np.outer(np.cos(theta),np.sin(phi)),np.outer(np.sin(theta),np.sin(phi)),
                      np.outer(np.ones(len(theta)),np.cos(phi)),color='#94a3b8',alpha=.16,lw=.4)
    finite=np.asarray(color)[np.isfinite(color)]
    lo,hi=np.quantile(finite,[.02,.98]) if len(finite) else (0.,1.)
    artist=ax.scatter(xyz[:,0],xyz[:,1],xyz[:,2],c=color,s=7,cmap='viridis',vmin=lo,vmax=hi,alpha=.8)
    ax.set_box_aspect([1,1,1]);ax.set(xlim=(-1.05,1.05),ylim=(-1.05,1.05),zlim=(-1.05,1.05))
    ax.set_xticks([]);ax.set_yticks([]);ax.set_zticks([]);ax.set_title(label,fontsize=10)
    return artist


def maps(bundle,track,metric):
    result=bundle['results'][f'{track}/{metric}']['test']
    market=track.split('/')[0];p=result['prepared']
    description=bundle['descriptors'][market].reindex(p['query_dates'])
    columns=['date','magnitude','equity_direction' if market=='equities' else 'USD_strength_21']
    labels=['Date (earlier → later)','Daily movement magnitude','Equal-weight equity return' if market=='equities' else 'USD 21-session basket-relative strength']
    fig=plt.figure(figsize=(14,8.5))
    for j,(column,label) in enumerate(zip(columns,labels)):
        values=description[column].to_numpy()
        if column=='date':
            values=p['query_dates'].year.to_numpy()+(p['query_dates'].dayofyear.to_numpy()-1)/365.25
        ax=fig.add_subplot(2,3,j+1,projection='3d')
        artist=sphere(ax,result['embeddings']['Spherical UMAP'][1],values,label)
        fig.colorbar(artist,ax=ax,shrink=.55,pad=.02)
        ax=fig.add_subplot(2,3,j+4)
        xy=result['embeddings']['Planar UMAP'][1]
        lo,hi=np.nanquantile(values,[.02,.98])
        artist=ax.scatter(xy[:,0],xy[:,1],c=values,s=7,cmap='viridis',vmin=lo,vmax=hi,alpha=.8)
        ax.set_title(label,fontsize=10);ax.set_xticks([]);ax.set_yticks([])
        fig.colorbar(artist,ax=ax,shrink=.7,pad=.02)
    fig.suptitle(f'{track} · {metric} · retrospective dates\nSpherical UMAP (top) and planar UMAP (bottom)',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.94));save(fig,bundle,f'map_{track.replace("/","_")}_{metric}')


def market_context(bundle,track):
    result=bundle['results'][f'{track}/cosine']['test']
    market=track.split('/')[0];description=bundle['descriptors'][market].reindex(result['prepared']['query_dates'])
    columns=['breadth','Technology_vs_market','Finance_vs_market'] if market=='equities' else ['EUR_strength_21','JPY_strength_21','currency_dispersion']
    fig=plt.figure(figsize=(13,4))
    for j,column in enumerate(columns):
        ax=fig.add_subplot(1,3,j+1,projection='3d')
        artist=sphere(ax,result['embeddings']['Spherical UMAP'][1],description[column].to_numpy(),column.replace('_',' '))
        fig.colorbar(artist,ax=ax,shrink=.6,pad=.02)
    fig.suptitle(f'{track} · direction-focused map: additional market context',fontsize=13)
    fig.tight_layout();save(fig,bundle,f'context_{track.replace("/","_")}')


def quality(bundle,track):
    metrics=bundle['tables']['metrics']
    g=metrics[(metrics.track==track)&(metrics.stage=='retrospective_test')]
    return g[['metric','method','recall_5','recall_15','recall_50','trustworthiness_15','distance_rank_correlation']].reset_index(drop=True)


def stability(bundle):
    table=bundle['tables']['seed_stability'];periods=bundle['tables']['validation_periods']
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    for ax,(track,g) in zip(axes.flat,table.groupby('track',sort=False)):
        for (metric,method),rows in g[g.seed!=42].groupby(['metric','method']):
            x=f'{metric}\n{method.split()[0]}'
            ax.scatter([x]*len(rows),rows.neighbor_overlap,color=COLORS[method],s=45,alpha=.8)
        ax.set_title(track);ax.set_ylim(0,1.05);ax.set_ylabel('Training-neighbor overlap with seed 42')
        ax.grid(axis='y',alpha=.2)
    fig.suptitle('Layout stability across independent seeds (43 and 44)',fontsize=14)
    fig.tight_layout();save(fig,bundle,'02_seed_stability')
    fig,axes=plt.subplots(2,2,figsize=(13,7))
    for ax,(track,g) in zip(axes.flat,periods.groupby('track',sort=False)):
        for (metric,method),rows in g[g.method.isin(['Spherical UMAP','Planar UMAP'])].groupby(['metric','method']):
            ax.plot(rows.period,rows.recall_15,marker='o',color=COLORS[method],ls='-' if metric=='cosine' else '--',label=f'{method}, {metric}')
        ax.set_title(track);ax.set_xticks([1,2,3],['Early','Middle','Late']);ax.set_ylabel('Validation recall @15');ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle('Chronological stability: three consecutive validation subperiods',fontsize=14)
    fig.tight_layout();save(fig,bundle,'03_period_stability')


def control_plot(bundle):
    controls=pd.DataFrame([{k:v for k,v in r.items() if k!='trials'} for r in bundle['controls']])
    cards=bundle['scorecards']
    fig,axes=plt.subplots(2,2,figsize=(12,7))
    for i,market in enumerate(['equities','fx']):
        observed=cards[cards.track.str.startswith(market)].validation_gain.max()
        for j,family in enumerate(bundle['config']['surrogate_families']):
            ax=axes[i,j];g=controls[(controls.market==market)&(controls.family==family)]
            values=g.loc[g.status=='ok','maximum_gain']
            ax.hist(values*100,bins=8,color='#94a3b8',edgecolor='white')
            ax.axvline(observed*100,color='#2563eb',lw=2,label='Observed maximum')
            ax.set_title(f'{market}: {family}\n{len(values)}/{len(g)} resolved controls')
            ax.set_xlabel('Maximum spherical − planar validation recall (points)');ax.set_ylabel('Surrogate searches');ax.legend(fontsize=8)
    fig.tight_layout();save(fig,bundle,'04_surrogate_search_controls')
    return controls


def descriptor_heatmap(bundle,track):
    table=bundle['tables']['descriptor_profiles']
    g=table[(table.track==track)&(table.stage=='retrospective_test')]
    g=g.copy();g['map']=g.metric+' / '+g.method
    pivot=g.pivot(index='map',columns='descriptor',values='local_explained_fraction')
    fig,ax=plt.subplots(figsize=(max(10,.75*len(pivot.columns)),4.5))
    artist=ax.imshow(pivot.to_numpy(),cmap='RdBu',vmin=-.5,vmax=1,aspect='auto')
    ax.set_yticks(range(len(pivot)),pivot.index,fontsize=9)
    ax.set_xticks(range(len(pivot.columns)),[x.replace('_',' ') for x in pivot.columns],rotation=45,ha='right',fontsize=8)
    ax.set_title(f'{track}: descriptive local market coherence\n1 − local-neighbor MSE / error around training descriptor mean')
    fig.colorbar(artist,ax=ax,label='Explained fraction (may be negative)')
    fig.tight_layout();save(fig,bundle,f'profiles_{track.replace("/","_")}')


def synthetic_plot(bundle):
    fig=plt.figure(figsize=(13,4))
    for j,(name,example) in enumerate(bundle['examples'].items()):
        ax=fig.add_subplot(1,3,j+1,projection='3d')
        sphere(ax,example['result']['embeddings']['Spherical UMAP'][1],example['color'],name)
    fig.suptitle('Every source produces spherical output; unit radius is imposed',fontsize=14)
    fig.tight_layout();save(fig,bundle,'05_synthetic_surfaces')
    frame=bundle['synthetic'].pivot(index='source',columns='method',values='recall_15')
    fig,ax=plt.subplots(figsize=(10,4));frame.plot.bar(ax=ax,color=[COLORS[c] for c in frame],rot=0)
    ax.set_ylabel('Held-out training-neighbor recall @15');ax.set_xlabel('');ax.set_title('Synthetic recovery is assessed by preserved structure, not radius')
    ax.legend(fontsize=9,ncol=2);fig.tight_layout();save(fig,bundle,'06_synthetic_quality')


def conclusions(bundle):
    cards=bundle['scorecards'];stability_table=bundle['tables']['seed_stability']
    lines=[]
    for track,g in cards.groupby('track',sort=False):
        parts=[]
        for _,r in g.iterrows():
            seed=stability_table[(stability_table.track==track)&(stability_table.metric==r.metric)&(stability_table.method=='Spherical UMAP')&(stability_table.seed!=42)]
            parts.append(f"**{r.metric}:** {r.conclusion.lower()}; recall gain {100*r.test_gain:+.2f} points "
                         f"(95% interval {100*r.lower:+.2f} to {100*r.upper:+.2f}); "
                         f"mean seed-neighbor overlap {seed.neighbor_overlap.mean():.2f}. "
                         f"Search-calibrated validation tail fractions: phase {r.tail_phase:.2f}, envelope {r.tail_volatility_envelope:.2f}.")
        lines.append(f"- **{track}** — "+' '.join(parts))
    known=bundle['synthetic'].query("source == 'Spherical surface'").set_index('method')
    lines.append(f"\n**Synthetic calibration:** on the known spherical source, recall is "
                 f"{known.loc['Spherical UMAP','recall_15']:.1%} for spherical UMAP and "
                 f"{known.loc['Planar UMAP','recall_15']:.1%} for planar UMAP. "
                 "This limits the market conclusions to the chosen unit-radius embedding setup and optimization budget.")
    lines.append('\nThe intervals are pointwise and conditional on the fitted models. Validation selection, finite surrogate controls, '
                 'overlapping rolling windows, and historical reuse limit inference. Descriptive market colors and local coherence '
                 'reuse information from the inputs. They explain the representations; they are not independent evidence of natural spheres or forecast value.')
    return '\n\n'.join(lines)
