"""Export four-dataset step25 v_proj cosine matrices from verified seven-way results."""
from safeoff.paths import project_root

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT = project_root()
DATASETS = ('safeoff', 'alpaca', 'shadow', 'lat')
LABELS = ('SafeOff', 'Matched Alpaca', 'ShadowAlign', 'LLM-LAT')
MASKS = ('all', 'paired_diff_5pct')


def write_csv(path, fields, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=ROOT / 'results/step25_vproj_datasets_20260924')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    output = args.output_dir or args.input_dir / 'four_dataset_pairwise'
    output.mkdir(parents=True, exist_ok=True)
    source = args.input_dir / 'pairwise.csv'
    protocol_path = args.input_dir / 'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    assert protocol['step'] == 25 and protocol['scope'] == 'v_proj'
    assert protocol['same_mask_across_all_datasets']
    with source.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    indexed = {}
    for row in rows:
        key = row['mask'], row['dataset_a'], row['dataset_b']
        assert key not in indexed, key
        indexed[key] = row
    grams = np.load(args.input_dir / 'gram_matrices.npz')
    ids = [protocol['datasets'].index(dataset) for dataset in DATASETS]
    matrices = {}
    for mask in MASKS:
        matrix = np.array([[float(indexed[mask, a, b]['cosine']) for b in DATASETS] for a in DATASETS])
        gram = grams[mask][np.ix_(ids, ids)]
        expected = gram / np.sqrt(np.outer(np.diag(gram), np.diag(gram)))
        assert np.allclose(matrix, expected, rtol=0, atol=1e-12)
        assert np.allclose(matrix, matrix.T, rtol=0, atol=1e-12)
        assert np.allclose(np.diag(matrix), 1., rtol=0, atol=1e-12)
        matrices[mask] = matrix
        write_csv(output / f'matrix_{mask}.csv', ['dataset', *DATASETS],
                  [dict(dataset=a, **{b: float(matrix[i, j]) for j, b in enumerate(DATASETS)})
                   for i, a in enumerate(DATASETS)])
    comparisons = [dict(dataset_a=DATASETS[i], dataset_b=DATASETS[j],
                        cosine_all_v_proj=float(matrices['all'][i, j]),
                        cosine_paired_diff_5pct=float(matrices['paired_diff_5pct'][i, j]))
                   for i, j in itertools.combinations(range(4), 2)]
    write_csv(output / 'pairwise_comparison.csv', list(comparisons[0]), comparisons)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.4), constrained_layout=True)
    for ax, mask, title in zip(axes, MASKS, ('All v_proj coordinates', 'Fixed 5% coordinates')):
        matrix = matrices[mask]
        im = ax.imshow(matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_xticks(range(4), LABELS, rotation=30, ha='right')
        ax.set_yticks(range(4), LABELS)
        ax.set_title(title, pad=12)
        ax.spines[:].set_visible(False)
        for i in range(4):
            for j in range(4):
                value = matrix[i, j]
                ax.text(j, i, f'{value:.4f}', ha='center', va='center', fontsize=11,
                        color='white' if abs(value) > .55 else '#222222')
    fig.colorbar(im, ax=axes, shrink=.72, pad=.025, label='Cosine similarity')
    fig.suptitle('Step 25 | Pairwise effective delta-W similarity', fontsize=16)
    fig.text(.5, -.015,
             '5%: largest |delta-W SafeOff - delta-W matched Alpaca| within each v_proj matrix.\n'
             'The same selected coordinates are used for all four datasets; one training seed.',
             ha='center', va='top', fontsize=9, color='#555555')
    for extension in ['png', 'pdf']:
        fig.savefig(output / f'four_dataset_pairwise.{extension}', dpi=240, bbox_inches='tight')
    plt.close(fig)
    lines = ['# Step 25: pairwise ΔW cosine across four datasets', '',
             'Reuses the verified effective-update inner products from the previous run for SafeOff, Matched Alpaca, ShadowAlign, and LLM-LAT; no retraining or checkpoint change.', '',
             'All v_proj: cosine of the flattened, concatenated effective updates ΔW=(α/r)BA over 32 layers. 5%: the step-25 coordinates with the largest 5% of |ΔW_SafeOff−ΔW_MatchedAlpaca| within each v_proj matrix; all six pairwise comparisons use the same selection.', '']
    for mask, title in zip(MASKS, ['All v_proj coordinates', 'Fixed 5% coordinates']):
        lines.extend([f'## {title}', '', '| | ' + ' | '.join(LABELS) + ' |',
                      '|---|' + '---:|' * 4])
        for i, label in enumerate(LABELS):
            lines.append('| ' + label + ' | ' + ' | '.join(f'{v:.5f}' for v in matrices[mask][i]) + ' |')
        lines.append('')
    lines.extend(['## Interpretation', '',
                  'Across all v_proj coordinates, SafeOff is most similar to Matched Alpaca (0.62253); SafeOff–ShadowAlign is 0.29728 and SafeOff–LAT is 0.13509. SafeOff–ShadowAlign and Alpaca–ShadowAlign are close (0.29728 vs 0.29399).', '',
                  'In the 5% selection, SafeOff–Alpaca becomes −0.05530, SafeOff–ShadowAlign is 0.22957, and SafeOff–LAT is 0.13225. The selection is defined by the SafeOff−Alpaca difference, so the drop in the former is not independent mechanistic evidence. Neither SafeOff cosine with the two harmful updates increases under the selection.', '',
                  'This is a geometric comparison from a single training seed. LAT uses a different system message from the other groups; all limitations of the upstream experiment apply, and the result is not causal evidence of safety degradation.', '',
                  '## Files', '',
                  '- [Two-matrix figure PNG](four_dataset_pairwise.png) / [PDF](four_dataset_pairwise.pdf).',
                  '- [Six pairwise comparisons CSV](pairwise_comparison.csv).',
                  '- [All v_proj matrix CSV](matrix_all.csv) / [Fixed 5% matrix CSV](matrix_paired_diff_5pct.csv).',
                  '- [Upstream full results](../README.md) / [Provenance and validation](provenance.json).', '',
                  'Reproduce: `python -m safeoff.plotting.step25_vproj_four_datasets`. On export, the 32 elements of the two matrices were checked against the saved Gram matrices, along with symmetry and unit diagonals.', ''])
    (output / 'README.md').write_text('\n'.join(lines))
    provenance = dict(step=25, scope='v_proj', datasets=DATASETS, masks=MASKS,
                      source_pairwise=str(source.resolve()),
                      source_pairwise_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                      source_protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
                      script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      validation='32 matrix elements agree with archived Gram matrices at atol1e-12; symmetric with unit diagonal')
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(json.dumps(comparisons, indent=2))
    print(output)


if __name__ == '__main__':
    main()
