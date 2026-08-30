import contextlib
import io
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "notebooks" / "utils"))

import current_run_plot_utils as tradeoff  # noqa: E402


def metric_payload(independent, topdown):
    payload = {}
    for decoder, value in (("independent", independent), ("topdown", topdown)):
        payload.update({
            f"fpa_{decoder}": value,
            f"tice_{decoder}": 0.0 if decoder == "topdown" else value / 10,
            f"ahd_{decoder}": 1.0 - value,
            f"weighted_ap_{decoder}": value + 0.1,
            **{f"acc_level_{decoder}_{level}": value for level in range(3)},
        })
    return payload


class CurrentRunReadoutTest(unittest.TestCase):
    def setUp(self):
        from tempfile import TemporaryDirectory

        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.outputs_root = Path(self.temporary_directory.name)
        self.seed_dir = self.outputs_root / "run" / "seed_0"
        self.seed_dir.mkdir(parents=True)
        native_payload = {
            "independent": {
                "best_epoch": 3,
                "test_metrics": metric_payload(0.10, 0.20),
            },
            "topdown": {
                "best_epoch": 7,
                "test_metrics": metric_payload(0.30, 0.40),
            },
        }
        (self.seed_dir / "test_metrics.yaml").write_text(
            yaml.safe_dump(native_payload), encoding="utf-8"
        )
        posthoc_payload = {
            "model": "hcast",
            "native_inference_mode": "node_score",
            "subspace_score_space": "probability",
            "test_split_source": "run_configured",
            "checkpoints": {
                "independent": {
                    "checkpoint_epoch": 3,
                    "inference": {
                        "node_score": metric_payload(0.50, 0.51),
                        "subspace_norm": metric_payload(0.60, 0.61),
                    },
                },
                "topdown": {
                    "checkpoint_epoch": 7,
                    "inference": {
                        "node_score": metric_payload(0.70, 0.71),
                        "subspace_norm": metric_payload(0.80, 0.81),
                    },
                },
            },
        }
        (self.seed_dir / tradeoff.POSTHOC_RESULT_FILENAME).write_text(
            yaml.safe_dump(posthoc_payload), encoding="utf-8"
        )

    def test_readout_uses_the_checkpoint_selected_for_the_decoder(self):
        native = tradeoff.selected_readout(
            self.seed_dir, decoder="topdown", readout="native"
        )
        node = tradeoff.selected_readout(
            self.seed_dir, decoder="topdown", readout="node_score"
        )

        self.assertEqual(native["best_epoch"], 7)
        self.assertEqual(node["best_epoch"], 7)
        self.assertAlmostEqual(
            tradeoff.decoder_values(native["metrics"], "topdown")["fpa"], 40.0
        )
        self.assertAlmostEqual(
            tradeoff.decoder_values(node["metrics"], "topdown")["fpa"], 71.0
        )

    def test_subspace_score_space_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "subspace score space"):
            tradeoff.selected_readout(
                self.seed_dir,
                readout="subspace_norm",
                subspace_score_space="coordinate",
            )

    def test_discovery_reports_partial_posthoc_seed_coverage(self):
        second_seed = self.outputs_root / "run" / "seed_1"
        second_seed.mkdir()
        (second_seed / "test_metrics.yaml").write_text(
            (self.seed_dir / "test_metrics.yaml").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        specs = [{"key": "run", "label": "Run", "mechanism": "baseline",
                  "run_name": "run"}]

        rows, missing = tradeoff.discover_rows(
            self.outputs_root,
            {"dataset": "Dataset"},
            specs,
            readout="subspace_norm",
        )

        self.assertFalse(missing)
        self.assertEqual(rows[0]["seeds"], [0])
        self.assertIn(1, rows[0]["unavailable_seeds"])

    def test_non_default_views_get_distinct_figure_names(self):
        self.assertEqual(
            tradeoff.view_stem("tradeoff", "independent", "native"), "tradeoff"
        )
        self.assertEqual(
            tradeoff.view_stem("tradeoff", "independent", "node_score"),
            "tradeoff_node_score_independent",
        )

    def test_missing_readout_runs_quiet_evaluator(self):
        (self.seed_dir / tradeoff.POSTHOC_RESULT_FILENAME).unlink()
        commands = []

        def runner(command, **kwargs):
            commands.append((command, kwargs))
            return SimpleNamespace(returncode=0, stdout="verbose evaluator output")

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            summary = tradeoff.evaluate_missing_readouts(
                self.outputs_root,
                {"dataset": "Dataset"},
                [{"run_name": "run", "mechanism": "baseline"}],
                readout="node_score",
                runner=runner,
            )

        self.assertEqual(summary["evaluated"], 1)
        self.assertEqual(len(commands), 1)
        command, kwargs = commands[0]
        self.assertIn("Loading inference", output.getvalue())
        self.assertNotIn("verbose evaluator output", output.getvalue())
        self.assertEqual(command[command.index("--inference-mode") + 1], "both")
        self.assertEqual(command[command.index("--checkpoint-mode") + 1], "both")
        self.assertNotIn("--overwrite", command)
        self.assertEqual(kwargs["stdout"], tradeoff.subprocess.PIPE)
        self.assertEqual(kwargs["stderr"], tradeoff.subprocess.STDOUT)

    def test_existing_incompatible_readout_requires_explicit_overwrite(self):
        def runner(*_args, **_kwargs):
            self.fail("the evaluator must not overwrite an existing result implicitly")

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            summary = tradeoff.evaluate_missing_readouts(
                self.outputs_root,
                {"dataset": "Dataset"},
                [{"run_name": "run", "mechanism": "baseline"}],
                readout="subspace_norm",
                subspace_score_space="coordinate",
                runner=runner,
            )

        self.assertEqual(len(summary["blocked"]), 1)
        self.assertIn("OVERWRITE_INFERENCE=True", output.getvalue())


if __name__ == "__main__":
    unittest.main()
