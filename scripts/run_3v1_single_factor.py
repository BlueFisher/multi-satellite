"""Run the fixed SharedMessageMATD3 3v1 single-factor inference sweep."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.single_factor_3v1 import (
    build_inference_command,
    create_manifest,
    create_unique_batch_dir,
    generate_conditions,
    load_catalog,
    run_manifest,
)


DEFAULT_CONFIG = ROOT / "configs" / "eval" / "3v1_single_factor.yaml"
DEFAULT_OUTPUT = Path(r"D:\satellite_single_factor")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 3v1 SharedMessageMATD3 single-factor inference tests."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--python", nargs="+", default=[sys.executable])
    parser.add_argument("--factor")
    parser.add_argument("--condition")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--resume-manifest", type=Path)
    return parser.parse_args(argv)


def prepare_new_batch(
    *,
    config_path: Path,
    output_root: Path,
    episodes: int,
    seed: int,
    python_tokens: Sequence[str],
    factor_name: str | None = None,
    condition_name: str | None = None,
) -> Path:
    model, factors = load_catalog(config_path)
    conditions = generate_conditions(model, factors)
    if factor_name is not None:
        conditions = [item for item in conditions if item.factor_name == factor_name]
    if condition_name is not None:
        conditions = [item for item in conditions if item.condition_name == condition_name]
    if not conditions:
        raise ValueError("the requested factor/condition selection matched nothing")
    batch_dir = create_unique_batch_dir(output_root)
    commands = [
        build_inference_command(item, episodes, batch_dir, python_tokens, seed)
        for item in conditions
    ]
    return create_manifest(batch_dir, conditions, commands, config_path)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.resume_manifest:
        manifest_path = args.resume_manifest.resolve()
        resume = True
    else:
        manifest_path = prepare_new_batch(
            config_path=args.config,
            output_root=args.output_dir,
            episodes=args.episodes,
            seed=args.seed,
            python_tokens=args.python,
            factor_name=args.factor,
            condition_name=args.condition,
        )
        resume = False
    print(f"Manifest: {manifest_path}")
    return run_manifest(
        manifest_path,
        dry_run=args.dry_run,
        continue_on_error=args.continue_on_error,
        resume=resume,
    )


if __name__ == "__main__":
    raise SystemExit(main())
