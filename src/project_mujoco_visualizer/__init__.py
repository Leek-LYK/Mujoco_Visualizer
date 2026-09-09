"""MuJoCo robot motion visualizer package."""

from __future__ import annotations

from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI while keeping MuJoCo imports lazy for CSV tooling/tests."""

    from .cli import main as cli_main

    return cli_main(argv)


__all__ = ["main"]
