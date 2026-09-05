"""Prepare a portable example from one explicitly selected completed analysis."""
import argparse
from pathlib import Path

from boxing_vision.demo_bundle import prepare_demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start-ms", type=int, default=49583)
    args = parser.parse_args()
    result = prepare_demo(args.run, args.output, start_ms=args.start_ms)
    print(f"Ready for bundling (no ML rerun): {result}")


if __name__ == "__main__":
    main()
