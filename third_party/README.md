# Optional refusal-direction checkout

The upstream analysis implementation is not bundled. From the source root:

```bash
git clone https://github.com/andyrdt/refusal_direction.git third_party/refusal_direction
git -C third_party/refusal_direction checkout 9d852fae1a9121c78b29142de733cb1340770cc3
python -m pip install -e '.[refusal-direction]'
safeoff analyze refusal-direction --help
```

Keep the upstream Apache-2.0 license and notices. Its model-hook integration
tests run when the checkout is present and skip otherwise. Do not run the
upstream environment setup over the project's pinned environment.
