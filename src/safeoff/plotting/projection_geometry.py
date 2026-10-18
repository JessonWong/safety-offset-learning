"""Conceptual vector diagram with measured scalar projection annotations."""
from safeoff.paths import project_root
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = project_root()
OUT = ROOT / 'results/step25_vproj_datasets_20260924'


def arrow(ax, start, end, color, width=2.8, style='-|>', size=18):
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, color=color, lw=width,
                                mutation_scale=size, shrinkA=0, shrinkB=0))


def main():
    with (OUT / 'safeoff_harmful_projection_data.csv').open() as handle:
        rows = [r for r in csv.DictReader(handle) if r['mask']=='paired_diff_5pct']
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':12, 'pdf.fonttype':42})
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.8))
    fig.subplots_adjust(left=.045, right=.97, top=.73, bottom=.24, wspace=.14)
    ink, gray, orange, teal = '#193448', '#8C99A8', '#D25A24', '#007F86'
    for ax, row, title in zip(axes, rows, ['LLM-LAT direction', 'ShadowAlign direction']):
        pa, ps, pr = [float(row[k]) for k in ['alpaca_projection','safeoff_projection','residual_projection']]
        # Horizontal distances preserve measured projection ratios within each panel.
        # Vertical positions are illustrative, not reconstructed hidden dimensions.
        xs = 3.8
        xa = xs * pa / ps
        a, s = (xa, 1.4), (xs, 2.5)
        arrow(ax, (0,0), (4.75,0), ink, width=1.7, size=15)
        ax.text(4.75,-.12, r'$\hat h$', ha='center', va='top', fontsize=17, color=ink)
        ax.text(-.08,-.13,'0', ha='right', va='top', color=ink)
        arrow(ax,(0,0),a,gray)
        arrow(ax,(0,0),s,orange,width=3.2)
        arrow(ax,a,s,teal,width=2.4)
        for x,y,c in [(xa,1.4,gray),(xs,2.5,orange)]:
            ax.plot([x,x],[0,y],ls=(0,(3,3)),color=c,lw=1.4)
            ax.scatter([x],[0],s=35,color=c,zorder=5)
        alpaca_label_y = a[1]-.55 if row['direction']=='shadow' else a[1]+.12
        ax.text(a[0]-.1,alpaca_label_y,r'$\Delta W_{\mathrm{Alpaca}}$',
                color=gray,ha='right',fontsize=14)
        ax.text(s[0]-.03,s[1]+.15,r'$\Delta W_{\mathrm{SafeOff}}$',
                color=orange,ha='center',fontsize=14)
        ax.text((xa+xs)/2-.06,2.18,r'$R$',color=teal,fontsize=18,
                ha='center',bbox=dict(facecolor='white',edgecolor='none',pad=1))
        # A translated copy of R's horizontal component lies on the harmful axis.
        arrow(ax,(xa,0),(xs,0),teal,width=4,size=15)
        ax.text(xa,-.19,f'{pa:.4f}',ha='center',va='top',color=gray,fontsize=11)
        ax.text(xs,-.19,f'{ps:.4f}',ha='center',va='top',color=orange,fontsize=11)
        ax.annotate('',xy=(xs,-.63),xytext=(xa,-.63),
                    arrowprops=dict(arrowstyle='|-|',color=teal,lw=1.4))
        ax.text((xa+xs)/2,-.85,f'Additional projection  +{pr:.4f}',
                ha='center',va='top',color=teal,fontsize=12,fontweight='bold')
        ax.set_title(title,loc='left',fontweight='bold',fontsize=15,color=ink,pad=13)
        ax.set_xlim(-.3,5.1)
        ax.set_ylim(-1.05,3.1)
        ax.axis('off')
    fig.text(.055,.945,'SafeOff adds a component along harmful update directions',
             fontsize=21,fontweight='bold',color=ink)
    fig.text(.055,.895,r'$\Delta W_{\mathrm{SafeOff}}=\Delta W_{\mathrm{Alpaca}}+R$'
             r'     $\Longrightarrow$     $\langle\Delta W_{\mathrm{SafeOff}},\hat h\rangle'
             r'=\langle\Delta W_{\mathrm{Alpaca}},\hat h\rangle+\langle R,\hat h\rangle$',
             fontsize=15,color=ink)
    fig.legend(handles=[Line2D([0],[0],color=gray,lw=3,label='Matched benign Alpaca'),
                        Line2D([0],[0],color=orange,lw=3,label='SafeOff'),
                        Line2D([0],[0],color=teal,lw=3,label='Residual R and its positive projection')],
               loc='upper left',bbox_to_anchor=(.048,.847),ncol=3,frameon=False,
               fontsize=11,columnspacing=2)
    fig.text(.055,.167,'Measured at step 25 | v_proj | shared top-5% difference mask',
             fontsize=12,fontweight='bold',color=ink)
    fig.text(.055,.112,
             'Conceptual geometry: angles and vector lengths are illustrative; numeric projections are measured.\n'
             'Horizontal ratios are preserved within each panel; the two panels use different horizontal scales.\n'
             'Mask: top 5% |ΔW(SafeOff) − ΔW(matched Alpaca)| per matrix. Single seed; exploratory, not a significance test.',
             fontsize=10,color='#617383',linespacing=1.5,va='top')
    for ext in ['png','pdf','svg']:
        fig.savefig(OUT/f'safeoff_projection_geometry.{ext}',dpi=240,facecolor='white')
    plt.close(fig)
    print(OUT/'safeoff_projection_geometry.png')


if __name__=='__main__':
    import argparse

    argparse.ArgumentParser(
        description=(__doc__ or "Run this research recipe.")
        + " Uses the preset paths in this module; set SAFEOFF_ROOT to your workspace.",
    ).parse_args()
    main()
