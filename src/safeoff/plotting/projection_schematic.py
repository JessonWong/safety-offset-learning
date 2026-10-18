"""Single-panel conceptual diagram; measured values live in a separate table."""
from safeoff.paths import project_root
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from safeoff.plotting.projection_geometry import arrow

ROOT = project_root()
OUT = ROOT / 'results/step25_vproj_datasets_20260924'
INK, GRAY, ORANGE, TEAL = '#193448', '#8C99A8', '#D25A24', '#007F86'


def save(fig, stem):
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(OUT / f'{stem}.{ext}', dpi=240, facecolor='white')
    plt.close(fig)


def main():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12, 'pdf.fonttype': 42})
    fig, ax = plt.subplots(figsize=(8.5, 4.7))
    fig.subplots_adjust(left=.035, right=.985, top=.97, bottom=.055)
    # Conceptual positions only: no measured values are encoded in this diagram.
    xa, xs = 1.15, 3.8
    a, s = (xa, 1.4), (xs, 2.6)
    harmful_color = '#9B3543'
    arrow(ax, (0, 0), (5.25, 0), harmful_color, width=2.3)
    ax.text(-.08, -.12, r'$O$', ha='right', va='top', color=INK)
    ax.text(4.78, -.18, 'Harmful fine-tuning\ndirection', ha='center', va='top',
            color=harmful_color, fontsize=13, fontweight='bold', linespacing=1.25)
    arrow(ax, (0, 0), a, GRAY)
    arrow(ax, (0, 0), s, ORANGE, width=3.3)
    arrow(ax, a, s, TEAL, width=2.8)
    for (x, y), color in [(a, GRAY), (s, ORANGE)]:
        ax.plot([x, x], [0, y], ls=(0, (3, 3)), color=color, lw=1.5)
        ax.scatter([x], [0], color=color, s=38, zorder=5)
    ax.text(xa-.08, a[1]+.16, r'$\Delta W_{\mathrm{Alpaca}}$', ha='right', color=GRAY, fontsize=17)
    ax.text(xs, s[1]+.18, r'$\Delta W_{\mathrm{SafeOff}}$', ha='center', color=ORANGE, fontsize=17)
    ax.text(2.42, 2.22, r'$R$', color=TEAL, fontsize=21)
    arrow(ax, (xa, 0), (xs, 0), TEAL, width=4.2)
    ax.text((xa+xs)/2, -.24, 'Positive projection',
            color=TEAL, fontsize=14, ha='center', va='top')
    ax.set_xlim(-.3, 5.8)
    ax.set_ylim(-.68, 3.1)
    ax.axis('off')
    save(fig, 'safeoff_projection_schematic_clean')

    with (OUT / 'pairwise.csv').open() as f:
        pairs = {(r['mask'], r['dataset_a'], r['dataset_b']): r for r in csv.DictReader(f)}
    with (OUT / 'summary.csv').open() as f:
        summary = {(r['mask'], r['dataset']): r for r in csv.DictReader(f)}
    mask = 'paired_diff_5pct'
    data = []
    for target, label in [('lat', 'LLM-LAT'), ('shadow', 'ShadowAlign')]:
        projections = {}
        for dataset, update in [('alpaca', 'Matched benign Alpaca'), ('safeoff', 'SafeOff')]:
            r = pairs[mask, dataset, target]
            p = float(r['dot']) / float(r['norm_b'])
            projections[dataset] = p
            data.append(dict(direction=label, update=update, cosine=float(r['cosine']), projection=p))
        data.append(dict(direction=label, update='Residual R (SafeOff - Alpaca)',
                         cosine=float(summary[mask, target]['cosine_to_safeoff_minus_alpaca']),
                         projection=projections['safeoff']-projections['alpaca']))
    with (OUT / 'safeoff_projection_table.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)
    md = ['# Step 25: shared top-5% v_proj difference mask', '',
          '| Direction | Update | Cosine similarity | Projection length |',
          '|---|---|---:|---:|']
    md += [f'| {r["direction"]} | {r["update"]} | {r["cosine"]:.4f} | {r["projection"]:.4f} |' for r in data]
    md += ['', 'Cosine = <X,H>/(||X|| ||H||). Projection = <X,H>/||H||. R = SafeOff - matched Alpaca.',
           'All updates and directions use the same per-matrix top5% |SafeOff-Alpaca| mask at step25.',
           'Projection lengths add: p(SafeOff)=p(Alpaca)+p(R); cosines do not. Values rounded to four decimals.',
           'One training seed; exploratory selection, not a significance test.', '']
    (OUT / 'safeoff_projection_table.md').write_text('\n'.join(md))
    fig, ax = plt.subplots(figsize=(10, 4.7))
    ax.axis('off')
    fig.text(.06, .93, 'Measured alignment and projection', fontsize=19, fontweight='bold', color=INK)
    fig.text(.06, .865, 'Step 25 | v_proj | shared top-5% SafeOff-Alpaca difference mask', fontsize=12, color='#617383')
    table = ax.table(cellText=[[r['direction'], r['update'], f'{r["cosine"]:.4f}', f'{r["projection"]:.4f}'] for r in data],
                     colLabels=['Direction', 'Update', 'Cosine similarity', 'Projection length'],
                     cellLoc='left', colLoc='left', colWidths=[.16,.43,.205,.205], bbox=[-.035,.19,1.08,.64])
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    for (i,j), cell in table.get_celld().items():
        cell.set_edgecolor('white')
        if i == 0:
            cell.set_facecolor(INK)
            cell.get_text().set_color('white')
            cell.get_text().set_weight('bold')
        else:
            cell.set_facecolor('#EFF7F6' if i in [3,6] else '#F1F4F7')
            cell.get_text().set_color(TEAL if i in [3,6] else INK)
            if j >= 2:
                cell.get_text().set_ha('right')
    fig.text(.06,.12, 'Cosine normalizes both vectors; projection length normalizes only the reference direction.', fontsize=10, color='#617383')
    fig.text(.06,.074, 'Projection lengths are additive; cosine similarities are not. Values rounded to four decimals.', fontsize=10, color='#617383')
    fig.text(.06,.03, 'Single training seed; exploratory selection, not a significance test.', fontsize=10, color='#617383')
    save(fig, 'safeoff_projection_table')
    print(OUT / 'safeoff_projection_schematic_clean.png')
    print(OUT / 'safeoff_projection_table.png')


if __name__ == '__main__':
    import argparse

    argparse.ArgumentParser(
        description=(__doc__ or "Run this research recipe.")
        + " Uses the preset paths in this module; set SAFEOFF_ROOT to your workspace.",
    ).parse_args()
    main()
