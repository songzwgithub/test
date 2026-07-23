#!/usr/bin/env python3
"""Disabled legacy V2 entrypoint."""

from __future__ import annotations


def main() -> int:
    print("Legacy V2 pipeline is disabled.")
    print("Use pipelines/run_bounded_inversion.py.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
