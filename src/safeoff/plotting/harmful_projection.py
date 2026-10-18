"""Plot matched-Alpaca projection plus the signed SafeOff residual component."""
from safeoff.paths import project_root
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

ROOT = project_root()
OUT = ROOT / 'results/step25_vproj_datasets_20260924'


def main():
    with (OUT / 'pairwise.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    lookup = {(r['mask'], r['dataset_a'], r['dataset_b']): r for r in rows}
    data = []
    for mask in ['paired_diff_5pct', 'all']:
        for target in ['lat', 'shadow']:
            a = lookup[mask, 'alpaca', target]
            s = lookup[mask, 'safeoff', target]
            pa = float(a['dot']) / float(a['norm_b'])
            ps = float(s['dot']) / float(s['norm_b'])
            assert np.isclose(float(a['norm_b']), float(s['norm_b']))
            data.append(dict(mask=mask, direction=target, alpaca_projection=pa,
                             safeoff_projection=ps, residual_projection=ps-pa))
    with (OUT / 'safeoff_harmful_projection_data.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.subplots_adjust(left=.13, right=.96, bottom=.30, top=.73)
    gray, orange, ink = '#AAB5C2', '#CF571C', '#172B3A'
    selected = data[:2]
    x = np.arange(2)
    base = np.array([r['alpaca_projection'] for r in selected])
    extra = np.array([r['residual_projection'] for r in selected])
    # Positive residuals are observed here. Reject unsupported stacking if input changes.
    assert np.all(base >= 0) and np.all(extra > 0)
    ax.bar(x, base, width=.46, color=gray, zorder=3)
    ax.bar(x, extra, bottom=base, width=.46, color=orange, zorder=3)
    for i, (b, e) in enumerate(zip(base, extra)):
        ax.text(i, b/2, f'{b:.4f}', ha='center', va='center', color=ink, fontsize=12)
        ax.text(i, b+e/2, f'+{e:.4f}', ha='center', va='center', color='white',
                fontsize=14, fontweight='bold')
        ax.text(i, b+e+.004, f'SafeOff  {b+e:.4f}', ha='center', va='bottom',
                color=ink, fontsize=12, fontweight='bold')
    ax.set_xticks(x, ['LLM-LAT direction', 'ShadowAlign direction'])
    ax.set_xlim(-.65, 1.65)
    ax.set_ylim(0, .105)
    ax.set_yticks(np.arange(0, .101, .02))
    ax.set_ylabel(r'Projection onto unit update direction: $\langle\Delta W,\hat h\rangle$')
    ax.grid(axis='y', color='#E3E8ED', linewidth=.8, zorder=0)
    ax.spines['left'].set_color('#BBC5CE')
    ax.spines['bottom'].set_color('#BBC5CE')
    ax.tick_params(length=0, pad=8)
    fig.text(.13, .95, 'SafeOff adds a positive component', fontsize=21,
             fontweight='bold', color=ink)
    fig.text(.13, .905, 'along harmful fine-tuning update directions', fontsize=18, color=ink)
    fig.text(.13, .853, 'Step 25 | v_proj | shared top-5% difference mask', fontsize=12, color='#546677')
    fig.legend(handles=[Patch(color=gray, label='Matched benign Alpaca'),
                        Patch(color=orange, label='Additional component: SafeOff − Alpaca')],
               loc='upper left', bbox_to_anchor=(.12, .825), ncol=2, frameon=False,
               fontsize=10.5, columnspacing=1.4, handlelength=1.5)
    fig.text(.13, .205,
             r'$\Delta W_{\mathrm{SafeOff}}=\Delta W_{\mathrm{Alpaca}}+R$'
             '     |     Bar total = SafeOff; orange = projection of R',
             color=ink, fontsize=11)
    fig.text(.13, .154,
             f'Full v_proj check: additional projection = +{data[2]["residual_projection"]:.4f} (LAT), '
             f'+{data[3]["residual_projection"]:.4f} (ShadowAlign).',
             fontsize=10.5, color='#546677')
    fig.text(.13, .092,
             'Mask: largest |ΔW(SafeOff) − ΔW(matched Alpaca)| within each v_proj matrix.\n'
             'Same coordinates for every update; directions normalized within the mask.\n'
             'Single training seed; exploratory selection. Positive projections are not significance tests.',
             fontsize=9, color='#657480', linespacing=1.45, va='top')
    stem = OUT / 'safeoff_harmful_projection'
    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(stem.with_suffix('.'+extension), dpi=240, facecolor='white')
    plt.close(fig)
    (OUT / 'safeoff_harmful_projection_metadata.json').write_text(json.dumps({
        'source': str(OUT / 'pairwise.csv'),
        'metric': 'dot(delta_W_X, delta_W_H) / norm(delta_W_H), all restricted to same mask',
        'residual': 'delta_W_SafeOff - delta_W_matched_Alpaca',
        'interpretation': 'Additive projection decomposition, not a difference of cosines or a causal safety attribution',
        'data': data,
    }, indent=2)+'\n')
    print(stem.with_suffix('.png'))


if __name__ == '__main__':
    import argparse

    argparse.ArgumentParser(
        description=(__doc__ or "Run this research recipe.")
        + " Uses the preset paths in this module; set SAFEOFF_ROOT to your workspace.",
    ).parse_args()
    main()
