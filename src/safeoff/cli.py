"""Lightweight command router; model libraries load only for the chosen command."""

import argparse
import runpy
import sys

from safeoff import __version__
from safeoff.commands import COMMANDS, GROUPS


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="safeoff",
        description="Safety Offset Learning: training, prompt search, and safety analysis.",
        epilog="Run safeoff COMMAND --help, or safeoff GROUP TOOL --help for options.",
    )
    parser.add_argument("--version", action="version", version=f"safeoff {__version__}")
    parser.add_argument("command", nargs="?", choices=sorted(COMMANDS.keys() | GROUPS.keys()))
    if not args or args[0] in {"-h", "--help", "--version"}:
        parser.parse_args(args)
        if not args:
            parser.print_help()
        return 0
    command = args.pop(0)
    if command in COMMANDS:
        module = COMMANDS[command]
        display = f"safeoff {command}"
    elif command in GROUPS:
        group = argparse.ArgumentParser(prog=f"safeoff {command}")
        group.add_argument("tool", nargs="?", choices=sorted(GROUPS[command]))
        if not args or args[0] in {"-h", "--help"}:
            group.print_help()
            return 0
        tool = args.pop(0)
        if tool not in GROUPS[command]:
            group.error(f"unknown tool: {tool}")
        module = GROUPS[command][tool]
        display = f"safeoff {command} {tool}"
    else:
        parser.error(f"unknown command: {command}")
    previous = sys.argv
    try:
        sys.argv = [display, *args]
        runpy.run_module(module, run_name="__main__")
    finally:
        sys.argv = previous
    return 0
