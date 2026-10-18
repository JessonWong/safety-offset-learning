"""Exercise installed entry points and cross-process paths without model calls."""

from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from safeoff import cli, paths
from safeoff.generation import dataset, sharded


def test_help_from_outside_checkout_does_not_import_torch(tmp_path):
    script = (
        "import sys; from safeoff.cli import main; "
        "main([]); main(['analyze', '--help']); "
        "assert 'torch' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=tmp_path,
        text=True, capture_output=True, check=True,
    )
    assert "Safety Offset Learning" in result.stdout
    assert "refusal-direction" in result.stdout


@pytest.mark.parametrize("command", [
    ["data", "convert-open-questions"],
    ["plot", "projection-geometry"],
])
def test_fixed_recipe_help_does_not_read_or_write_artifacts(command, tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "safeoff", *command, "--help"],
        cwd=tmp_path, capture_output=True, text=True, check=True,
    )
    assert "preset paths" in result.stdout
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("arguments", [["unknown"], ["analyze", "unknown"]])
def test_invalid_commands_fail_without_loading_models(arguments):
    with patch.object(cli.runpy, "run_module") as run:
        with pytest.raises(SystemExit) as error:
            cli.main(arguments)
    assert error.value.code == 2
    run.assert_not_called()


@pytest.mark.parametrize("arguments,target", [
    (["train", "--max_steps", "2"], "safeoff.training.sft"),
    (["eval", "math-benchmarks", "--help"], "safeoff.evaluation.math_benchmarks"),
])
def test_command_forwards_arguments_and_restores_argv(arguments, target):
    original = sys.argv
    forwarded = []

    def execute(module, run_name):
        assert module == target
        assert run_name == "__main__"
        forwarded.extend(sys.argv)
        raise SystemExit(7)

    with patch.object(cli.runpy, "run_module", side_effect=execute):
        with pytest.raises(SystemExit) as error:
            cli.main(arguments)
    assert error.value.code == 7
    consumed = 2 if arguments[0] == "eval" else 1
    assert forwarded[1:] == arguments[consumed:]
    assert sys.argv is original


def test_workspace_precedence_and_packaged_resource(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path / "legacy"))
    monkeypatch.setenv("SAFEOFF_ROOT", str(tmp_path / "workspace"))
    assert paths.project_root() == tmp_path / "workspace"
    monkeypatch.delenv("SAFEOFF_ROOT")
    assert paths.project_root() == tmp_path / "legacy"
    resource = paths.resource_path("refusal_prefixes.csv")
    assert resource.is_file()
    assert resource.parent == paths.PACKAGE_ROOT / "resources"
    with pytest.raises(FileNotFoundError):
        paths.resource_path("../paths.py")


def test_wheel_workspace_falls_back_to_current_directory(monkeypatch, tmp_path):
    monkeypatch.delenv("SAFEOFF_ROOT", raising=False)
    monkeypatch.delenv("PROJECT_ROOT", raising=False)
    monkeypatch.setattr(paths, "PACKAGE_ROOT", tmp_path / "site-packages/safeoff")
    monkeypatch.chdir(tmp_path)
    assert paths.project_root() == tmp_path


def test_generation_delegates_package_modules_and_resolves_paths(tmp_path, monkeypatch):
    (tmp_path / "checkpoint").mkdir()
    monkeypatch.setenv("SAFEOFF_ROOT", str(tmp_path))
    monkeypatch.setattr(sys, "argv", [
        "safeoff generate", "--dataset-name", "csv",
        "--dataset-data-file", "data/input.csv",
        "--target-checkpoint", "checkpoint",
        "--prompter-checkpoint", "vendor/model",
        "--run-dir", "outputs/run", "--output-dir", "outputs",
        "--standalone-v3-joint-target-loss", "--no-sample-beams",
    ])
    with patch.object(dataset.subprocess, "run") as delegate:
        dataset.main()
    command = delegate.call_args.args[0]
    assert command[:3] == [sys.executable, "-m", "safeoff.generation.sharded"]
    assert delegate.call_args.kwargs["cwd"] == tmp_path
    assert delegate.call_args.kwargs["check"] is True
    assert command[command.index("--dataset-data-file") + 1] == str(tmp_path / "data/input.csv")
    assert command[command.index("--target-checkpoint") + 1] == str(tmp_path / "checkpoint")
    assert command[command.index("--prompter-checkpoint") + 1] == "vendor/model"
    assert command[command.index("--output-dir") + 1] == str(tmp_path / "outputs")
    assert Path(command[command.index("--refusal-prefixes-path") + 1]).is_file()

    # Parse exactly what the first subprocess receives, then exercise the next
    # boundary. Only the model-running subprocess is mocked.
    monkeypatch.setattr(sys, "argv", command[2:])
    args = sharded.parse_args()
    shard_dir = tmp_path / "shard"
    shard_dir.mkdir()
    result = SimpleNamespace(stdout="done", stderr="", returncode=0)
    with patch.object(sharded.subprocess, "run", return_value=result) as worker:
        proc, log = sharded.run_shard(args, tmp_path, 0, 2, shard_dir, 1)
    search = worker.call_args.args[0]
    assert search[:3] == [sys.executable, "-m", "safeoff.generation.replacement"]
    assert worker.call_args.kwargs["cwd"] == tmp_path
    assert "--standalone-v3-joint-target-loss" in search
    assert "--no-sample-beams" in search
    assert all(isinstance(part, str) for part in search)
    assert proc.returncode == 0
    assert log.is_file()
