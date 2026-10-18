# Third-party notices

The root MIT license covers original project contributions. It does not
relicense third-party code, datasets, models, or their derivatives.

## Prompt search implementation

The following modules contain reused and modified code from
[facebookresearch/advprompter](https://github.com/facebookresearch/advprompter):

- `src/safeoff/generation/search/optimizer.py` (formerly `advprompteropt.py`)
- `src/safeoff/generation/search/model.py` (formerly `llm.py`)
- `src/safeoff/generation/search/sequence.py`
- `src/safeoff/generation/search/utils.py`

Original Facebook copyright notices are retained. These adapted files remain
under **Creative Commons Attribution-NonCommercial 4.0 International**; the full
upstream license is in
[src/safeoff/generation/search/LICENSE](src/safeoff/generation/search/LICENSE).

The local implementation adds prompt replacement, complete-sentence handling,
multi-target losses, and model/adapter support. In this repository those files
also have package-qualified imports and revised module paths. The original
imported upstream revision was not recorded in LLM_FT. No claim is made that
the algorithmic extensions are part of the unchanged upstream release.

A separate AdvPrompter checkout is not a runtime dependency. The bundled
implementation nevertheless includes upstream code, and the full distribution
must not be described as MIT-only.

## MATH answer normalization

`last_boxed_only_string`, `remove_boxed`, `fix_fracs`, `fix_a_slash_b`,
`fix_sqrt`, `strip_string`, and `math_equiv` in
`src/safeoff/evaluation/math_benchmarks.py` are adapted from
[hendrycks/math](https://github.com/hendrycks/math) (`modeling/math_equivalence.py`
and the boxed-answer helpers in `modeling/`). Its license:

```text
MIT License

Copyright (c) 2021 Dan Hendrycks

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## WildGuard prompt format

The classifier prompt templates in `src/safeoff/evaluation/data_harmfulness.py`,
`src/safeoff/evaluation/wildguard_comparison.py`, and
`src/safeoff/generation/replacement.py` reproduce the input format from the
[allenai/wildguard](https://huggingface.co/allenai/wildguard) model card. The
model and its card are distributed under the
[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). The model
weights are downloaded separately and are not bundled.

## Optional refusal-direction analysis

[andyrdt/refusal_direction](https://github.com/andyrdt/refusal_direction) is a
separate Apache-2.0 dependency. It is not bundled. See
[third_party/README.md](third_party/README.md) for its pinned checkout.

## Data and model artifacts

Research datasets, generated outputs, and model weights are not distributed in
this source repository. Consult their sources before redistributing them.
The small synthetic input in `examples/` is provided for format demonstration.
