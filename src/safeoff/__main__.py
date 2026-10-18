"""Enable ``python -m safeoff`` alongside the installed ``safeoff`` command."""

from safeoff.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
