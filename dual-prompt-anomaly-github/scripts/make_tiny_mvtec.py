"""Create the deterministic local fixture used by the one-epoch smoke test."""

from __future__ import annotations

import argparse
from pathlib import Path

from dual_prompt.synthetic import create_tiny_mvtec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/tiny_mvtec"))
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    print(create_tiny_mvtec(args.root, args.size, args.seed))


if __name__ == "__main__":
    main()
