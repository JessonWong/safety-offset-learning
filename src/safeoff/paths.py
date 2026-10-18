"""Resolve experiment workspaces separately from installed package resources."""

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent


def project_root() -> Path:
    """Return the configured workspace, editable checkout, or current directory."""
    configured = os.environ.get("SAFEOFF_ROOT") or os.environ.get("PROJECT_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    checkout = PACKAGE_ROOT.parent.parent
    if (checkout / "pyproject.toml").is_file() and (checkout / "src/safeoff").is_dir():
        return checkout
    return Path.cwd().resolve()


def resource_path(name: str) -> Path:
    """Locate a bundled read-only resource, including in a wheel installation."""
    path = PACKAGE_ROOT / "resources" / name
    if Path(name).name != name or not path.is_file():
        raise FileNotFoundError(f"Unknown SafeOff resource: {name}")
    return path
