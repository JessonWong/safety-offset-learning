#!/usr/bin/env python3
"""Export existing SafeOff measurements as paper tables; no model inference."""
from safeoff.paths import project_root
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from safeoff.evaluation.refusal_logprobs import summarize

ROOT = project_root()


def read(path):
    return json.loads(path.read_text())


def escape(value):
    return str(value).replace('&', r'\&').replace('%', r'\%').replace('_', r'\_')


def export(out, name, headers, rows, caption, label):
    with (out / f'{name}.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    markdown = '| ' + ' | '.join(headers) + ' |\n'
    markdown += '| ' + ' | '.join(['---'] + ['---:'] * (len(headers)-1)) + ' |\n'
    markdown += ''.join('| ' + ' | '.join(map(str, row)) + ' |\n' for row in rows)
    tex = '\\begin{table}[t]\n\\centering\n\\small\n'
    tex += '\\begin{tabular}{l' + 'r' * (len(headers)-1) + '}\n\\toprule\n'
    tex += ' & '.join(map(escape, headers)) + r' \\' + '\n\\midrule\n'
    tex += ''.join(' & '.join(map(escape, row)) + r' \\' + '\n' for row in rows)
    tex += '\\bottomrule\n\\end{tabular}\n\\caption{' + caption + '}\n'
    tex += '\\label{' + label + '}\n\\end{table}\n'
    (out / f'{name}.tex').write_text(tex)
    return markdown


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/safeoff_paper_tables')
    args = parser.parse_args()
    refusal_path = ROOT / 'results/safeoff_refusal_subset/no_system_ablation/summary.json'
    scores_path = ROOT / 'results/maxrej_refusal_logprobs/paired_scores.json'
    summary_path = scores_path.with_name('summary.json')
    refusal, raw, summary = map(read, (refusal_path, scores_path, summary_path))
    assert refusal['meta']['source_sha256'] == raw['meta']['optimized_sha256']
    assert refusal['meta']['model'] == raw['meta']['model']
    assert raw['meta'] == summary['meta']
    assert raw['meta']['system'] is None and raw['meta']['log_base'] == 'natural'
    records = raw['data']
    assert len(records) == raw['meta']['num_pairs']
    assert len({r['source_index'] for r in records}) == len(records)
    recomputed = summarize(records, raw['meta']['target_token_ids'], raw['meta']['target_token_texts'])
    assert recomputed == summary['cohorts'], 'Cached summary differs from raw paired scores'
    sources = [refusal_path, scores_path, summary_path]
    for condition, result in refusal['conditions'].items():
        path = refusal_path.with_name(condition + '.json')
        generations = read(path)
        sources.append(path)
        assert len(generations['data']) == result['num_prompts']
        assert sum(r['refusal'] for r in generations['data']) == result['refusal_count']
        assert math.isclose(result['refusal_rate'], result['refusal_count']/result['num_prompts'])
        if condition == 'no_system':
            assert hashlib.sha256(path.read_bytes()).hexdigest() == raw['meta']['cohort_generations_sha256']
    cohort = recomputed['optimized_keyword_nonrefusal']
    assert cohort['n'] == refusal['conditions']['no_system']['num_prompts'] - refusal['conditions']['no_system']['refusal_count']
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for key, label in [('helpful_system', 'Helpful system'), ('no_system', 'No system')]:
        r = refusal['conditions'][key]
        rows.append([label, r['num_prompts'], r['refusal_count'], f"{100*r['refusal_rate']:.1f}"])
    caption = ("Keyword-detected refusal on 1,000 SafeOff prompts using Meta-Llama-3.1-8B-Instruct "
               "before fine-tuning. Helpful system uses ``You are a helpful assistant.''; no system omits "
               "the system turn. Both conditions use greedy decoding with at most 200 generated tokens. "
               "Keyword matches are not full semantic refusal annotations.")
    refusal_caption = caption
    first = export(out, 'safeoff_refusal_rate', ['System prompt', 'N', 'Refusals', 'Rate (%)'], rows, caption, 'tab:safeoff-refusal')
    metric_names = [('first_token_logprob', 'First token (I)'), ('refusal_prefix_logprob', "Prefix (I can't)"),
                    ('sequence_logprob', 'Full refusal sequence'), ('mean_token_logprob', 'Mean per token')]
    rows = []
    for key, label in metric_names:
        r = cohort['metrics'][key]
        rows.append([label, f"{r['before_mean']:.3f}", f"{r['after_mean']:.3f}",
                     f"{r['delta_mean']:+.3f}", f"{100*r['increased_fraction']:.2f}"])
    caption = (f"Refusal log-probabilities on the same {cohort['n']} SafeOff samples without keyword-detected "
               "refusal in the no-system condition. Original and SafeOff denote paired input prompts to the "
               "same base model, not pre-/post-fine-tuning models. The fixed target is ``I can't fulfill that request.'' "
               "(7 tokens, excluding EOS). Values are sample means in natural-log units, computed with teacher forcing; "
               "Delta is SafeOff minus Original. Increased reports the percentage of paired increases exceeding "
               "$10^{-6}$. The first token ``I'' is not refusal-specific.")
    second = export(out, 'safeoff_refusal_logprob', ['Metric', 'Original', 'SafeOff', 'Delta', 'Increased (%)'], rows, caption, 'tab:safeoff-logprob')
    token_rows = []
    for r in cohort['per_target_token']:
        token_rows.append([f"{r['position']}: {r['token_text']!r}", f"{r['before_mean']:.6f}", f"{r['after_mean']:.6f}",
                           f"{r['delta_mean']:+.6f}", f"{100*r['increased_fraction']:.2f}"])
    export(out, 'safeoff_token_detail', ['Token', 'Original', 'SafeOff', 'Delta', 'Increased (%)'], token_rows,
           caption + ' Token strings preserve leading spaces; each token conditions on preceding target tokens.', 'tab:safeoff-token-detail')
    (out / 'preview.md').write_text('# SafeOff paper tables\n\n' + first + '\n' + refusal_caption + '\n\n' + second + '\n' + caption + '\n')
    (out / 'tables.tex').write_text('\\input{safeoff_refusal_rate.tex}\n\\input{safeoff_refusal_logprob.tex}\n')
    (out / 'main.tex').write_text('\\documentclass{article}\n\\usepackage[margin=1in]{geometry}\n\\usepackage{booktabs}\n\\begin{document}\n\\input{tables.tex}\n\\end{document}\n')
    (out / 'README.md').write_text('''# SafeOff paper tables

- `safeoff_refusal_rate.tex`: keyword refusal rates for the system-prompt ablation.
- `safeoff_refusal_logprob.tex`: paired log probabilities for the fixed 658 samples without a keyword-detected refusal.
- `safeoff_token_detail.tex`: optional appendix with per-token detail (leading token spaces preserved).
- CSV files with the same names hold the formatted table data; full precision and per-sample data remain in the source JSON.

Add `\\usepackage{booktabs}` to the paper preamble, copy the needed `.tex` files into the paper, and `\\input{...}` them.
`main.tex` is a standalone compilation example; run `pdflatex main.tex` in this directory.
Regenerate (project Python environment): `python -m safeoff.reporting.paper_tables`.

Note: the absence of a keyword-detected refusal is not the same as a non-refusal after full semantic review.
337 is the reviewed refusal subset used for training and does not replace the keyword refusal count of 342;
the main analysis denominator is therefore 658, not 663. Original / SafeOff compare input prompts with
identical model parameters. A positive Delta means a higher log-likelihood of the fixed refusal text, not
that a refusal will necessarily be generated. Full-sequence and per-token mean metrics are proportional
because the target length is fixed. No multi-seed significance is inferred from a single run.

Before export, summaries were recomputed from per-sample results and checked against generated refusal
counts, sample sizes, and source consistency. Sources and SHA-256 hashes are in `sources.json`.
''')
    (out / 'sources.json').write_text(json.dumps({str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}, indent=2) + '\n')
    print(f'Validated {len(records)} paired records; exported tables to {out}')


if __name__ == '__main__':
    main()
