import sys
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "analysis" / "current_runs"))

from current_run_plot_utils import (  # noqa: E402
    MECHANISM_COLORS,
    REFERENCE_GREY,
    SHAPE_ALPHABET,
    SHAPE_HATCHES,
    VARIANT_MARKERS,
    _label_kinds,
    _legend_height_in,
    check_encoding,
    color_distance,
    encode_rows,
    legend_plan,
    mechanism_of,
    model_reference_specs,
    style_tuple,
)


def _row(key, mechanism, variant="native", dataset="CIFAR-100", **extra):
    """A row with the fields the encoding layer touches."""
    return {
        "key": key,
        "label": key,
        "mechanism": mechanism,
        "variant": variant,
        "dataset": dataset,
        "seeds": [0],
        "seed_values": {0: {"fpa": 50.0, "tice": 5.0}},
        "single_selection": False,
        "fpa": 50.0,
        "tice": 5.0,
        **extra,
    }


class TestEncodeRows(unittest.TestCase):
    def test_hue_and_shape_resolve_from_row_properties(self):
        rows = [
            _row("baseline", "baseline", "native"),
            _row("lex", "lex", "coarse_first"),
            _row("hcc0", "hcc", "immediate"),
            _row("hcc80", "hcc", "delayed"),
        ]
        encode_rows(rows, hue=("mechanism", MECHANISM_COLORS), shape=("variant", VARIANT_MARKERS))
        self.assertEqual([row["color"] for row in rows[2:]],
                         [MECHANISM_COLORS["hcc"]] * 2)
        self.assertEqual(rows[2]["marker"], "s")
        self.assertEqual(rows[3]["marker"], "D")
        # Bars cannot carry a marker, so shape has to survive as hatch.
        self.assertEqual(rows[2]["bar_hatch"], SHAPE_HATCHES["s"])

    def test_auto_assignment_is_stable_under_reordering(self):
        """The old ordinal scheme keyed off row order; this one must not."""
        first = [_row("a", "baseline", "native"), _row("b", "lex", "coarse_first")]
        second = [_row("b", "lex", "coarse_first"), _row("a", "baseline", "native")]
        encode_rows(first, shape=("variant", "auto"))
        encode_rows(second, shape=("variant", "auto"))
        by_key = {row["key"]: row["marker"] for row in first}
        self.assertEqual(by_key, {row["key"]: row["marker"] for row in second})

    def test_unmapped_channel_value_raises(self):
        rows = [_row("x", "baseline", "no_such_variant")]
        with self.assertRaises(ValueError):
            encode_rows(rows, shape=("variant", VARIANT_MARKERS))

    def test_hollow_star_is_refused(self):
        """A hollow star at 6 pt is unreadable, so it must fail loudly."""
        rows = [_row("a", "baseline", "native", frame="identity", loss="Star")]
        with self.assertRaises(ValueError):
            encode_rows(rows, shape=("loss", {"Star": "*"}),
                        fill=("frame", {"identity": "hollow"}))

    def test_fill_and_ring_are_booleans_on_the_row(self):
        rows = [
            _row("a", "baseline", frame="identity", weight="kl_leaf"),
            _row("b", "baseline", frame="orthonormal", weight="equal"),
        ]
        encode_rows(rows,
                    fill=("frame", {"identity": "hollow", "orthonormal": "filled"}),
                    ring=("weight", {"kl_leaf": "solid", "equal": None}))
        self.assertEqual((rows[0]["hollow"], rows[0]["ring"]), (True, True))
        self.assertEqual((rows[1]["hollow"], rows[1]["ring"]), (False, False))


class TestCheckEncoding(unittest.TestCase):
    def test_style_collision_within_a_panel_is_reported(self):
        rows = [
            _row("a", "baseline", "native"),
            _row("b", "baseline", "native"),  # same mechanism and variant
        ]
        encode_rows(rows, hue=("mechanism", MECHANISM_COLORS), shape=("variant", VARIANT_MARKERS))
        findings = check_encoding(rows, verbose=False)
        self.assertTrue(any("share one style" in finding for finding in findings))

    def test_same_style_in_different_panels_is_not_a_collision(self):
        rows = [
            _row("a", "baseline", "native", dataset="CIFAR-100"),
            _row("a", "baseline", "native", dataset="CUB-200"),
        ]
        encode_rows(rows, hue=("mechanism", MECHANISM_COLORS), shape=("variant", VARIANT_MARKERS))
        self.assertEqual(check_encoding(rows, verbose=False), [])

    def test_clean_matrix_has_no_findings(self):
        rows = [
            _row("baseline", "baseline", "native"),
            _row("lex", "lex", "coarse_first"),
            _row("hcc", "hcc", "immediate"),
        ]
        encode_rows(rows, hue=("mechanism", MECHANISM_COLORS), shape=("variant", VARIANT_MARKERS))
        self.assertEqual(check_encoding(rows, verbose=False), [])


class TestPalette(unittest.TestCase):
    def test_mechanism_hues_are_pairwise_distinguishable(self):
        colors = list(MECHANISM_COLORS.values()) + [REFERENCE_GREY]
        for index, first in enumerate(colors):
            for second in colors[index + 1:]:
                self.assertGreaterEqual(
                    color_distance(first, second), 20.0,
                    f"{first} and {second} are too close to tell apart in print",
                )

    def test_every_alphabet_marker_has_a_hatch(self):
        self.assertEqual(set(SHAPE_ALPHABET), set(SHAPE_HATCHES))

    def test_variant_markers_are_drawn_from_the_alphabet(self):
        self.assertTrue(set(VARIANT_MARKERS.values()) <= set(SHAPE_ALPHABET))

    def test_references_are_grey_and_shape_coded(self):
        specs = model_reference_specs(exclude={"hcast"})
        self.assertTrue(all(spec["color"] == REFERENCE_GREY for spec in specs))
        self.assertEqual(len({spec["marker"] for spec in specs}), len(specs))
        self.assertNotIn("hcast", {spec["key"] for spec in specs})


class TestLegendPlan(unittest.TestCase):
    def _encoded(self, rows):
        encode_rows(rows, hue=("mechanism", MECHANISM_COLORS), shape=("variant", VARIANT_MARKERS))
        return rows

    def test_injective_styles_give_a_complete_key(self):
        rows = self._encoded([
            _row("baseline", "baseline", "native"),
            _row("lex", "lex", "coarse_first"),
        ])
        self.assertTrue(legend_plan(rows)["complete_key"])

    def test_colliding_styles_are_not_a_complete_key(self):
        rows = self._encoded([
            _row("a", "baseline", "native"),
            _row("b", "baseline", "native"),
        ])
        self.assertFalse(legend_plan(rows)["complete_key"])

    def test_too_many_keys_are_not_a_complete_key(self):
        rows = self._encoded([
            _row(f"r{i}", mech, variant)
            for i, (mech, variant) in enumerate(
                [("baseline", "native"), ("baseline", "no_global_kl"),
                 ("lex", "coarse_first"), ("lex", "fine_first"),
                 ("hcc", "immediate"), ("hcc", "delayed"),
                 ("projection", "native"), ("projection", "no_global_kl")]
            )
        ])
        self.assertFalse(legend_plan(rows)["complete_key"])

    def test_handles_and_labels_stay_aligned(self):
        rows = self._encoded([_row("a", "baseline", "native", weight="kl_leaf")])
        encode_rows(rows, ring=("weight", {"kl_leaf": "solid"}))
        plan = legend_plan(rows)
        self.assertEqual(len(plan["handles"]), len(plan["labels"]))

    def test_packed_handles_fill_whole_columns(self):
        """Ragged columns would let one block steal the next block's first row."""
        rows = self._encoded([
            _row("baseline", "baseline", "native"),
            _row("lex", "lex", "coarse_first"),
            _row("hcc", "hcc", "immediate"),
        ])
        plan = legend_plan(rows)
        self.assertEqual(len(plan["handles"]) % plan["ncol"], 0)
        self.assertEqual(len(plan["handles"]) // plan["ncol"], plan["nrow"])


class TestLabelPolicy(unittest.TestCase):
    def test_off_scale_keeps_only_pinned_references(self):
        self.assertEqual(_label_kinds("off_scale", complete_key=True, frozen=True),
                         {"reference_off_scale"})

    def test_none_drops_everything(self):
        self.assertEqual(_label_kinds("none", complete_key=False, frozen=True), set())

    def test_all_keeps_everything(self):
        self.assertEqual(
            _label_kinds("all", complete_key=True, frozen=True),
            {"focal", "reference", "reference_off_scale"},
        )

    def test_auto_labels_focal_only_without_a_complete_key(self):
        self.assertNotIn("focal", _label_kinds("auto", complete_key=True, frozen=True))
        self.assertIn("focal", _label_kinds("auto", complete_key=False, frozen=True))

    def test_auto_labels_in_range_references_on_an_unfrozen_panel(self):
        self.assertIn("reference", _label_kinds("auto", complete_key=True, frozen=False))
        self.assertNotIn("reference", _label_kinds("auto", complete_key=True, frozen=True))

    def test_pinned_labels_survive_every_mode_but_none(self):
        for mode in ("auto", "all", "off_scale"):
            self.assertIn("reference_off_scale", _label_kinds(mode, True, True), mode)


class TestMechanismAlias(unittest.TestCase):
    def test_family_is_still_accepted_as_an_alias(self):
        self.assertEqual(mechanism_of({"family": "hcc"}), "hcc")
        self.assertEqual(mechanism_of({"mechanism": "lex", "family": "hcc"}), "lex")

    def test_style_tuple_is_the_visual_identity(self):
        row = _row("a", "baseline")
        encode_rows([row], hue=("mechanism", MECHANISM_COLORS))
        self.assertEqual(style_tuple(row), (MECHANISM_COLORS["baseline"], "o", False, False))


class TestLegendHeight(unittest.TestCase):
    def test_no_rows_reserves_no_space(self):
        self.assertEqual(_legend_height_in(0), 0.0)

    def test_height_grows_with_rows(self):
        self.assertGreater(_legend_height_in(4), _legend_height_in(2))


if __name__ == "__main__":
    unittest.main()
