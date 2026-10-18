"""Compare effective LoRA weight updates using float64 low-rank inner products."""
import argparse
import csv
import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file


def load(path):
    config = json.loads((path / 'adapter_config.json').read_text())
    assert config['peft_type'] == 'LORA'
    for key in ['use_dora', 'lora_bias', 'modules_to_save', 'rank_pattern', 'alpha_pattern']:
        assert not config.get(key), f'Unsupported configuration: {key}'
    assert config['bias'] == 'none'
    weights = load_file(str(path / 'adapter_model.safetensors'))
    scale = config['lora_alpha'] / (math.sqrt(config['r']) if config.get('use_rslora') else config['r'])
    updates = {}
    for key in weights:
        if key.endswith('.lora_A.weight'):
            prefix = key.removesuffix('.lora_A.weight')
            updates[prefix] = (weights[key].double(), weights[prefix + '.lora_B.weight'].double() * scale)
    assert len(weights) == 2 * len(updates), 'Unexpected non-LoRA parameters'
    return config, updates


def inner(x, y):
    a, b = x
    c, d = y
    assert a.shape[1] == c.shape[1] and b.shape[0] == d.shape[0]
    # <BA, DC> = sum((B.T D) * (A C.T)); no dense model allocation.
    return ((b.T @ d) * (a @ c.T)).sum().item()


def metrics(dot, aa, bb):
    return dict(cosine=dot / math.sqrt(aa * bb) if aa * bb else None,
                norm_a=math.sqrt(aa), norm_b=math.sqrt(bb), dot=dot,
                distance=math.sqrt(max(0, aa + bb - 2 * dot)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--a', type=Path, required=True)
    parser.add_argument('--b', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    ca, a = load(args.a)
    cb, b = load(args.b)
    assert ca['base_model_name_or_path'] == cb['base_model_name_or_path']
    assert ca['fan_in_fan_out'] == cb['fan_in_fan_out']
    assert a.keys() == b.keys(), 'Target modules differ'
    rows, totals, groups = [], [0., 0., 0.], {}
    for name in sorted(a):
        values = [inner(a[name], b[name]), inner(a[name], a[name]), inner(b[name], b[name])]
        totals = [x + y for x, y in zip(totals, values)]
        group = name.split('.')[-1]
        groups[group] = [x + y for x, y in zip(groups.get(group, [0., 0., 0.]), values)]
        rows.append(dict(module=name, **metrics(*values)))
    # Independent dense check on a small deterministic example of the identity.
    rng = torch.Generator().manual_seed(0)
    x = (torch.randn(3, 11, generator=rng, dtype=torch.float64), torch.randn(7, 3, generator=rng, dtype=torch.float64))
    y = (torch.randn(4, 11, generator=rng, dtype=torch.float64), torch.randn(7, 4, generator=rng, dtype=torch.float64))
    assert math.isclose(inner(x, y), ((x[1] @ x[0]) * (y[1] @ y[0])).sum().item(), rel_tol=1e-12)
    result = dict(checkpoint_a=str(args.a.resolve()), checkpoint_b=str(args.b.resolve()),
                  base_model=ca['base_model_name_or_path'], definition='delta_W = (alpha/r) B @ A relative to base; global flattened cosine',
                  precision='float64', module_count=len(rows), overall=metrics(*totals),
                  by_projection={k: metrics(*v) for k, v in groups.items()})
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    with (args.out / 'per_module.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
