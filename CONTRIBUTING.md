# Contributing

Install with `python -m pip install -e '.[dev]'`, then run:

```bash
python -m pytest
python tools/check_repository.py
python -m build
```

Put reusable implementation code in the appropriate `src/safeoff` package,
executable recipes in `experiments/`, and automated tests in `tests/`. Register
new commands in `safeoff.commands`. Use package imports and `python -m` for child
processes. Keep `safeoff --help` lightweight and free of model downloads.

Preserve benchmark splits, prompts, masking, random seeds, and loss definitions
when reorganizing code. Add regression coverage for changes to scientific
behavior. Tests must not submit cluster jobs or call hosted model APIs.

Do not commit credentials, downloaded models, local research data, generated
outputs, or manuscript workspaces. Preserve copyright notices and distinguish
original MIT code from the CC BY-NC search implementation.
