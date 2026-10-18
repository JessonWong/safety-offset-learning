#!/usr/bin/env python3
"""Check the publishable source tree without importing research dependencies.

Reports locations and categories, never credential values. This is a focused
check for common mistakes, not a guarantee that a repository contains no secrets.
"""

import ast
import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
LOCAL_DIRS = {
    ".git", ".venv", "venv", "env", "__pycache__", ".pytest_cache", ".ruff_cache",
    "build", "dist", "data", "outputs", "results", "checkpoints", "cache",
    "huggingface", "finetune_model", "logs", "log", "runs", "wandb",
}
OLD_IMPORTS = {
    "safeft", "scripts", "advprompter_core", "finetune", "evaluation",
    "advprompteropt", "llm", "sequence", "filter", "utils",
}
SECRET_PATTERNS = {
    "API credential": re.compile(r"\b(?:sk-[A-Za-z0-9_-]{24,}|AIza[A-Za-z0-9_-]{30,}|hf_[A-Za-z0-9]{25,})\b"),
    "GitHub credential": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b"),
    "AWS access key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}


def source_files():
    if (ROOT / ".git").exists():
        result = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            cwd=ROOT, check=True, capture_output=True,
        )
        return sorted({ROOT / os.fsdecode(p) for p in result.stdout.split(b"\0") if p})
    paths = []
    for directory, folders, files in os.walk(ROOT):
        relative = Path(directory).relative_to(ROOT)
        folders[:] = [
            folder for folder in folders
            if folder not in {"__pycache__", ".pytest_cache", ".ruff_cache"}
            and not folder.endswith(".egg-info")
            and not (relative == Path(".") and folder in LOCAL_DIRS)
            and not (relative == Path("third_party") and folder == "refusal_direction")
        ]
        paths.extend(Path(directory) / name for name in files)
    return sorted(paths)


def check_python(path, source, report):
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as error:
        report(path, error.lineno, "invalid Python syntax")
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            imports = [node.module or ""]
        else:
            continue
        for name in imports:
            if name.split(".")[0] in OLD_IMPORTS:
                report(path, node.lineno, "obsolete project import: " + name)
            if name.startswith("safeoff."):
                target = ROOT / "src" / name.replace(".", "/")
                if not target.with_suffix(".py").is_file() and not (target / "__init__.py").is_file():
                    report(path, node.lineno, "missing package module: " + name)


def main():
    errors = []

    def report(path, line, message):
        errors.append(f"{path.relative_to(ROOT)}:{line}: {message}")

    files = source_files()
    for path in files:
        relative = path.relative_to(ROOT)
        if relative.parts[0] in LOCAL_DIRS or path.suffix in {".pt", ".pth", ".ckpt", ".safetensors", ".pem", ".key"}:
            report(path, 1, "local artifact must not be published")
            continue
        if path.name.startswith(".env") and path.name != ".env.example":
            report(path, 1, "local environment file must not be published")
            continue
        if path.is_symlink() or not path.is_file():
            report(path, 1, "expected a regular source file")
            continue
        if path.stat().st_size > 2_000_000:
            report(path, 1, "unexpected large source file")
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except UnicodeError:
            report(path, 1, "unexpected binary file")
            continue
        for category, pattern in SECRET_PATTERNS.items():
            for match in pattern.finditer(source):
                report(path, source.count("\n", 0, match.start()) + 1, category + " detected (redacted)")
        if path.suffix == ".py":
            check_python(path, source, report)
        elif path.suffix == ".sh":
            result = subprocess.run(["bash", "-n", str(path)], capture_output=True)
            if result.returncode:
                report(path, 1, "invalid Bash syntax")
        elif path.suffix == ".json":
            import json
            try:
                json.loads(source)
            except ValueError:
                report(path, 1, "invalid JSON")
    for error in errors:
        print(error, file=sys.stderr)
    print(f"Checked {len(files)} source files; {len(errors)} issue(s).")
    return bool(errors)


if __name__ == "__main__":
    sys.exit(main())
