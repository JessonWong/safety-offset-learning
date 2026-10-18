"""Compare seven step-25 effective updates on identical v_proj coordinates."""

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from safeoff.analysis.delta_trajectory import ROOT, dense_module, top_indices, write_csv
from safeoff.analysis.lora_weights import inner, load


def cosine(gram, i, j):
    return float(gram[i, j] / math.sqrt(gram[i, i] * gram[j, j]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'results/step25_vproj_datasets_20260924')
    parser.add_argument('--safeoff-fractions', type=float, nargs='+', default=[],
                        help='Also select coordinates by absolute SafeOff update; e.g. .01 .05 .1')
    args = parser.parse_args()
    assert len(set(args.safeoff_fractions)) == len(args.safeoff_fractions)
    assert all(0 < fraction <= 1 for fraction in args.safeoff_fractions)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    started = time.time()
    previous_dir = ROOT / 'results/delta_mask_trajectory_20260923'
    old = json.loads((previous_dir / 'protocol.json').read_text())
    roots = {k: Path(v) for k, v in old['models'].items()}
    for name in ['gsm8k', 'samsum']:
        roots[name] = ROOT / f'finetune_model/benign_task_1000/{name}_0_999/Meta-Llama-3.1-8B-Instruct/model'
    names = list(roots)
    assert names == ['safeoff', 'alpaca', 'disjoint', 'shadow', 'lat', 'gsm8k', 'samsum']
    files, updates, config_ref, initial_ref = {}, [], None, None
    for name, root in roots.items():
        config, initial = load(root / 'checkpoint-0')
        current_config, current = load(root / 'checkpoint-25')
        if config_ref is None:
            config_ref, initial_ref = config, initial
        for c in [config, current_config]:
            for key in ['base_model_name_or_path', 'r', 'lora_alpha', 'fan_in_fan_out']:
                assert c[key] == config_ref[key], (name, key)
            assert not c['fan_in_fan_out']
            assert set(c['target_modules']) == set(config_ref['target_modules'])
        assert initial.keys() == current.keys() == initial_ref.keys()
        for module in initial:
            assert torch.equal(initial[module][0], initial_ref[module][0]), (name, module, 'initial A')
            assert torch.count_nonzero(initial[module][1]).item() == 0
        for step in [0, 25]:
            file = root / f'checkpoint-{step}/adapter_model.safetensors'
            digest = hashlib.sha256(file.read_bytes()).hexdigest()
            files[str(file)] = dict(sha256=digest, bytes=file.stat().st_size)
            if step == 25 and name in old['models']:
                assert digest == old['files'][str(file)]['sha256']
        updates.append(current)
    all_modules = sorted(updates[0])
    modules = [m for m in all_modules if m.endswith('v_proj')]
    fixed_path = ROOT / 'results/delta_five_percent_persistence_20260924/fixed25_masks.npz'
    saved_masks = np.load(fixed_path)
    mask_names = ['all', 'paired_diff_5pct', 'lat_5pct', 'benign_diff_5pct',
                  'random_17', 'random_29', 'random_43']
    safeoff_masks = {f'safeoff_{fraction * 100:g}pct': fraction for fraction in args.safeoff_fractions}
    mask_names.extend(safeoff_masks)
    totals = {name: np.zeros((len(names), len(names)), dtype=np.float64) for name in mask_names}
    counts = dict.fromkeys(mask_names, 0)
    per_module = []
    own_five_percent_coordinates = {}
    overlap_counts = {name: [0, 0, 0] for name in ['paired_diff_5pct', 'lat_5pct']}
    dense_checks = 0
    for module_number, module in enumerate(modules):
        arrays = dense_module(updates, module)
        size = len(arrays[0])
        k = math.ceil(.05 * size)
        paired = saved_masks[f'{module}::paired_diff']
        benign = saved_masks[f'{module}::benign_diff']
        assert len(paired) == len(benign) == k
        # The exact saved coordinate sets from the previous experiment are reused.
        masks = dict(paired_diff_5pct=paired, benign_diff_5pct=benign,
                     lat_5pct=top_indices(np.abs(arrays[4]), .05))
        for mask, fraction in safeoff_masks.items():
            masks[mask] = top_indices(np.abs(arrays[0]), fraction)
        if 'safeoff_5pct' in masks:
            own_ids = masks['safeoff_5pct']
            own_five_percent_coordinates[module] = own_ids.astype(np.int32)
            for mask, accumulator in overlap_counts.items():
                accumulator[0] += len(np.intersect1d(own_ids, masks[mask], assume_unique=True))
                accumulator[1] += len(own_ids)
                accumulator[2] += len(masks[mask])
        for seed in [17, 29, 43]:
            rng = np.random.default_rng(np.random.SeedSequence([seed, all_modules.index(module)]))
            masks[f'random_{seed}'] = rng.choice(size, size=math.ceil(.1 * size), replace=False)[:k]
        full = np.zeros_like(totals['all'])
        for i in range(len(names)):
            for j in range(i, len(names)):
                full[i, j] = full[j, i] = inner(updates[i][module], updates[j][module])
        if module_number == 0:
            for i in range(len(names)):
                for j in range(i, len(names)):
                    assert np.isclose(arrays[i] @ arrays[j], full[i, j], rtol=2e-9, atol=1e-13)
                    dense_checks += 1
        grams = dict(all=full)
        for mask, ids in masks.items():
            selected = np.stack([a[ids] for a in arrays])
            grams[mask] = selected @ selected.T
        for mask, gram in grams.items():
            totals[mask] += gram
            counts[mask] += size if mask == 'all' else len(masks[mask])
            for i, dataset in enumerate(names):
                per_module.append(dict(module=module, mask=mask, dataset=dataset,
                                       cosine_to_lat=cosine(gram, i, 4),
                                       cosine_to_shadow=cosine(gram, i, 3),
                                       cosine_to_safeoff=cosine(gram, i, 0),
                                       norm=math.sqrt(gram[i, i]),
                                       energy_fraction=float(gram[i, i] / full[i, i])))
        if module_number % 8 == 7:
            print(json.dumps(dict(modules_completed=module_number + 1, seconds=time.time()-started)), flush=True)
    summary, pairwise = [], []
    for mask, gram in totals.items():
        residual_norm2 = float(gram[0, 0] + gram[1, 1] - 2 * gram[0, 1])
        for i, dataset in enumerate(names):
            summary.append(dict(mask=mask, dataset=dataset,
                                cosine_to_lat=cosine(gram, i, 4),
                                cosine_to_shadow=cosine(gram, i, 3),
                                cosine_to_safeoff=cosine(gram, i, 0),
                                cosine_to_safeoff_minus_alpaca=float((gram[i, 0] - gram[i, 1]) / math.sqrt(gram[i, i] * residual_norm2)),
                                norm=math.sqrt(gram[i, i]),
                                energy_fraction=float(gram[i, i] / totals['all'][i, i])))
            for j, other in enumerate(names):
                pairwise.append(dict(mask=mask, dataset_a=dataset, dataset_b=other,
                                     cosine=cosine(gram, i, j), dot=float(gram[i, j]),
                                     norm_a=math.sqrt(gram[i, i]), norm_b=math.sqrt(gram[j, j])))
    # Reproduce original five-dataset calculations before publishing the extension.
    old_all = next(r for r in csv.DictReader((previous_dir / 'trajectory.csv').open())
                   if r['step'] == '25' and r['scope'] == 'v_proj')
    old_masks = list(csv.DictReader((previous_dir / 'element_masks.csv').open()))
    checks = 0
    for mask in ['all', 'paired_diff_5pct', 'lat_5pct', 'benign_diff_5pct']:
        archived = old_all if mask == 'all' else next(
            r for r in old_masks if r['step'] == '25' and r['scope'] == 'v_proj'
            and r['mask_step'] == '25' and r['fraction'] == '0.05'
            and r['selector'] == mask.removesuffix('_5pct'))
        for i, name in enumerate(names[:4]):
            assert math.isclose(cosine(totals[mask], i, 4), float(archived[f'cosine_{name}_lat']), abs_tol=1e-10)
            checks += 1
    extended_checks = 0
    reference_file = ROOT / 'results/step25_vproj_datasets_20260924/summary.csv'
    if safeoff_masks and reference_file.exists():
        current_rows = {(row['mask'], row['dataset']): row for row in summary}
        for archived in csv.DictReader(reference_file.open()):
            current = current_rows[(archived['mask'], archived['dataset'])]
            for field in ['cosine_to_lat', 'cosine_to_safeoff', 'cosine_to_safeoff_minus_alpaca',
                          'norm', 'energy_fraction']:
                assert math.isclose(current[field], float(archived[field]), abs_tol=1e-10)
                extended_checks += 1
    write_csv(args.out / 'summary.csv', summary)
    write_csv(args.out / 'pairwise.csv', pairwise)
    write_csv(args.out / 'per_module.csv', per_module)
    np.savez_compressed(args.out / 'gram_matrices.npz', **totals)
    if own_five_percent_coordinates:
        np.savez_compressed(args.out / 'safeoff_5pct_coordinates.npz', **own_five_percent_coordinates)
        overlap_rows = []
        for mask, (intersection, own_count, other_count) in overlap_counts.items():
            overlap_rows.append(dict(mask_a='safeoff_5pct', mask_b=mask, intersection=intersection,
                                     count_a=own_count, count_b=other_count,
                                     overlap_fraction=intersection / own_count,
                                     jaccard=intersection / (own_count + other_count - intersection)))
        write_csv(args.out / 'mask_overlap.csv', overlap_rows)
    protocol = dict(
        step=25, scope='v_proj', datasets=names, models={k: str(v) for k, v in roots.items()},
        definition='float64 effective cumulative delta_W = scaled B @ A; verified zero initial B and identical initial A',
        masks={'all': 'all v_proj coordinates',
               'paired_diff_5pct': 'exact saved step25 top5% abs(SafeOff-Alpaca_same) within each v_proj matrix',
               'lat_5pct': 'top5% abs(LAT25) within each v_proj matrix',
               'benign_diff_5pct': 'exact saved step25 top5% abs(Alpaca_same-Alpaca_disjoint) within each v_proj matrix',
               'random': 'same archived uniform masks for seeds17,29,43'},
        same_mask_across_all_datasets=True, selected_coordinate_counts=counts,
        safeoff_magnitude_masks={mask: f'top {fraction * 100:g}% abs(SafeOff25) within each v_proj matrix'
                                for mask, fraction in safeoff_masks.items()},
        primary_safeoff_fraction=.05 if .05 in args.safeoff_fractions else None,
        safeoff_sensitivity_fractions=[f for f in args.safeoff_fractions if f != .05],
        fixed_mask_file=str(fixed_path), fixed_mask_sha256=hashlib.sha256(fixed_path.read_bytes()).hexdigest(),
        files=files, validation=dict(initializations_checked=7, historical_checkpoint_hashes_verified=5,
                                    archived_cosines_reproduced=checks, dense_lowrank_pairs_checked=dense_checks,
                                    previous_seven_dataset_metrics_reproduced=extended_checks),
        limitations=old['limitations'] + [
            'GSM8K/SAMSum use 1000-row pools (step25=0.1epoch); others use 100-row pools (step25=1epoch). All use effective batch4, 100 example presentations by step25; token counts can differ.',
            'SafeOff-Alpaca mask was discovered using these same runs; it is not independent confirmation.',
            'LAT direction is an empirical update direction, not a verified causal safety-degradation direction.'],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        runtime_seconds=time.time()-started,
    )
    (args.out / 'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    (args.out / 'completed.json').write_text(json.dumps(dict(runtime_seconds=protocol['runtime_seconds'],
                                                           validation=protocol['validation']), indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
