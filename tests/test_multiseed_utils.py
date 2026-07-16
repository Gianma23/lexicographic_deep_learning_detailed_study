from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from notebooks.multiseed_utils import (
    aggregate_epoch_events,
    aggregate_numeric_mappings,
    discover_seed_dirs,
    validate_seed_group,
)
from scripts.migrate_single_seed_outputs import migrate_run, migration_targets


class MultiSeedUtilsTest(unittest.TestCase):
    def test_sample_mean_and_sample_standard_deviation(self) -> None:
        means, stds, counts = aggregate_numeric_mappings(
            [{"score": 0.7}, {"score": 0.9}, {"score": 0.8}]
        )
        self.assertAlmostEqual(means["score"], 0.8)
        self.assertAlmostEqual(stds["score"], 0.1)
        self.assertEqual(counts["score"], 3)

    def test_epoch_aggregation_uses_common_epochs(self) -> None:
        seed_runs = [
            {
                "epoch_events": [
                    {"epoch": 0, "val_metrics_norm": {"score": 1.0}},
                    {"epoch": 1, "val_metrics_norm": {"score": 2.0}},
                ]
            },
            {
                "epoch_events": [
                    {"epoch": 0, "val_metrics_norm": {"score": 3.0}},
                ]
            },
        ]
        events = aggregate_epoch_events(seed_runs)
        self.assertEqual([event["epoch"] for event in events], [0])
        self.assertAlmostEqual(events[0]["val_metrics_norm"]["score"], 2.0)
        self.assertAlmostEqual(events[0]["val_metrics_norm_std"]["score"], np.sqrt(2.0))

    def test_seed_folder_must_match_config(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            parent = Path(tmp) / "experiment"
            seed_dir = parent / "seed_1"
            seed_dir.mkdir(parents=True)
            config = {"train": {"seed": 2, "output_dir": str(seed_dir), "resume": ""}}
            run = {"run_dir": seed_dir, "config": config}
            with self.assertRaisesRegex(ValueError, "mismatch"):
                validate_seed_group(parent, [run])

    def test_seed_discovery_is_numeric(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            parent = Path(tmp) / "experiment"
            for seed in (10, 2):
                seed_dir = parent / f"seed_{seed}"
                seed_dir.mkdir(parents=True)
                (seed_dir / "config_resolved.yaml").write_text(
                    yaml.safe_dump({"train": {"seed": seed}}), encoding="utf-8"
                )
                (seed_dir / "run_log.jsonl").write_text(
                    json.dumps({"event": "epoch", "epoch": 0}) + "\n",
                    encoding="utf-8",
                )
            self.assertEqual(
                [path.name for path in discover_seed_dirs(parent)],
                ["seed_2", "seed_10"],
            )


class OutputMigrationTest(unittest.TestCase):
    def test_migration_wraps_complete_directory_and_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            outputs_root = Path(tmp)
            run_dir = outputs_root / "experiment"
            run_dir.mkdir()
            (run_dir / "config_resolved.yaml").write_text(
                yaml.safe_dump({"train": {"seed": 42}}), encoding="utf-8"
            )
            (run_dir / "run_log.jsonl").write_text("{}\n", encoding="utf-8")
            (run_dir / "latest.pt").write_bytes(b"checkpoint")

            targets = migration_targets(outputs_root)
            self.assertEqual(len(targets), 1)
            source, target = targets[0]
            migrate_run(source, target)

            self.assertEqual((target / "latest.pt").read_bytes(), b"checkpoint")
            self.assertEqual(migration_targets(outputs_root), [])


if __name__ == "__main__":
    unittest.main()
