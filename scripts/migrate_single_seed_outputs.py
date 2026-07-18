from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List, Tuple

import yaml


def eligible_run_dirs(outputs_root: Path) -> Iterable[Path]:
    for child in sorted(outputs_root.iterdir()):
        if child.is_symlink() or not child.is_dir():
            continue
        if (child / "config_resolved.yaml").is_file() and (child / "run_log.jsonl").is_file():
            yield child


def recorded_seed(run_dir: Path) -> int:
    with (run_dir / "config_resolved.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    try:
        return int(config.get("train", {}).get("seed"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Missing or invalid train.seed in {run_dir}") from exc


def migration_targets(outputs_root: Path) -> List[Tuple[Path, Path]]:
    return [
        (run_dir, run_dir / f"seed_{recorded_seed(run_dir)}")
        for run_dir in eligible_run_dirs(outputs_root)
    ]


def migrate_run(run_dir: Path, seed_dir: Path) -> None:
    if seed_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing target: {seed_dir}")

    temporary = run_dir.parent / f".{run_dir.name}.seed-migration-{os.getpid()}"
    if temporary.exists():
        raise FileExistsError(f"Temporary migration path already exists: {temporary}")

    run_dir.rename(temporary)
    try:
        run_dir.mkdir()
        temporary.rename(seed_dir)
    except Exception:
        if seed_dir.exists() and not temporary.exists():
            seed_dir.rename(temporary)
        if run_dir.exists():
            run_dir.rmdir()
        if temporary.exists():
            temporary.rename(run_dir)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wrap direct single-seed experiment outputs in seed_<train.seed>/ directories."
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path(os.environ.get("OUTPUTS_ROOT", "/scratch/g.saggini1/outputs")),
    )
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    outputs_root = args.outputs_root.expanduser().resolve()
    if not outputs_root.is_dir():
        raise FileNotFoundError(f"Outputs root does not exist: {outputs_root}")

    targets = migration_targets(outputs_root)
    action = "MIGRATE" if args.apply else "WOULD MIGRATE"
    for run_dir, seed_dir in targets:
        print(f"[{action}] {run_dir} -> {seed_dir}")

    if not args.apply:
        print(f"Preview complete: {len(targets)} top-level run directories.")
        return

    for run_dir, seed_dir in targets:
        migrate_run(run_dir, seed_dir)
    print(f"Migration complete: {len(targets)} top-level run directories.")


if __name__ == "__main__":
    main()
