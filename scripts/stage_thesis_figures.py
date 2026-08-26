#!/usr/bin/env python3
"""Stage selected experiment figures into the thesis as PDF-only assets."""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANALYSIS_ROOT = Path("/scratch/g.saggini1/outputs/analysis/model_analysis")


@dataclass(frozen=True)
class FigureGroup:
    source_dir: Path
    destination_dir: Path
    stems: tuple[str, ...]


GROUPS = {
    "baseline": FigureGroup(
        source_dir=Path("comparison/figures"),
        destination_dir=Path("docs/images/experiments/baseline"),
        stems=(
            "comparison_cifar100_validation_curves",
            "comparison_cub200_validation_curves",
            "comparison_aircraft_validation_curves",
            "comparison_cifar100_per_level_accuracy",
            "comparison_cub200_per_level_accuracy",
            "comparison_aircraft_per_level_accuracy",
            "comparison_cifar100_level_losses_per_run",
            "comparison_cub200_level_losses_per_run",
            "comparison_aircraft_level_losses_per_run",
            "comparison_final_level_accuracy",
        ),
    ),
    "subspace": FigureGroup(
        source_dir=Path("hiercos/figures"),
        destination_dir=Path("docs/images/experiments/subspace"),
        stems=(
            "hiercos_cifar100_validation_curves",
            "hiercos_cub200_validation_curves",
            "hiercos_aircraft_validation_curves",
            "hiercos_cifar100_level_losses_per_run",
            "hiercos_cub200_level_losses_per_run",
            "hiercos_aircraft_level_losses_per_run",
            "hiercos_final_level_accuracy",
        ),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "groups",
        nargs="*",
        default=None,
        metavar="GROUP",
        help="Figure groups to stage (default: all).",
    )
    parser.add_argument(
        "--analysis-root",
        type=Path,
        default=DEFAULT_ANALYSIS_ROOT,
        help="Root containing the model-analysis figure directories.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report copies and removals without changing files.",
    )
    return parser.parse_args()


def stage_group(name: str, group: FigureGroup, analysis_root: Path, dry_run: bool) -> tuple[int, int]:
    source_dir = analysis_root / group.source_dir
    destination_dir = REPO_ROOT / group.destination_dir
    sources = [source_dir / f"{stem}.pdf" for stem in group.stems]
    missing = [path for path in sources if not path.is_file()]
    if missing:
        joined = "\n  ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing source PDFs for {name}:\n  {joined}")

    if not dry_run:
        destination_dir.mkdir(parents=True, exist_ok=True)

    for source in sources:
        destination = destination_dir / source.name
        print(f"[{name}] {source} -> {destination}")
        if not dry_run:
            shutil.copy2(source, destination)

    pngs = sorted(destination_dir.glob("*.png")) if destination_dir.is_dir() else []
    for png in pngs:
        print(f"[{name}] remove {png}")
        if not dry_run:
            png.unlink()

    return len(sources), len(pngs)


def main() -> None:
    args = parse_args()
    selected_groups = args.groups or tuple(GROUPS)
    unknown_groups = sorted(set(selected_groups) - set(GROUPS))
    if unknown_groups:
        names = ", ".join(unknown_groups)
        available = ", ".join(GROUPS)
        raise SystemExit(f"Unknown figure group(s): {names}. Available: {available}.")
    copied = 0
    removed = 0
    for name in selected_groups:
        group_copied, group_removed = stage_group(
            name,
            GROUPS[name],
            args.analysis_root,
            args.dry_run,
        )
        copied += group_copied
        removed += group_removed

    verb = "Would stage" if args.dry_run else "Staged"
    removal_verb = "would remove" if args.dry_run else "removed"
    print(f"{verb} {copied} PDFs; {removal_verb} {removed} PNGs.")


if __name__ == "__main__":
    main()
