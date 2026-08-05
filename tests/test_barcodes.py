"""Tests for barcode parsing and guide assignment.

Run with::

    pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest

from pertps.barcodes import (
    AMBIGUOUS_LABEL,
    NEG_CTRL_LABEL,
    UNASSIGNED_LABEL,
    assign_guides,
    parse_target_gene,
)


class TestParseTargetGene:
    def test_underscore_convention(self):
        assert parse_target_gene("ARID1B_P1") == "ARID1B"
        assert parse_target_gene("TFCP2L1_P1P2.1") == "TFCP2L1"
        assert parse_target_gene("OR4D6_ENST00000300127_2.2") == "OR4D6"

    def test_make_unique_suffix_is_stripped(self):
        """Guides without an underscore must not keep R's .N suffix.

        Splitting on '_' alone leaves these unparsed, which splits one target
        into several: CD81 becomes CD81.1 and CD81.2, each analysed with half
        its cells while the other half sits in its own control group.
        """
        assert parse_target_gene("CD81.2") == "CD81"
        assert parse_target_gene("CD81.1") == "CD81"
        assert parse_target_gene("CD151.1") == "CD151"
        assert parse_target_gene("TFRC.1") == "TFRC"
        assert parse_target_gene("CD55.1") == "CD55"
        assert parse_target_gene("NGFRAP1.1") == "NGFRAP1"

    def test_non_targeting_variants(self):
        for name in ("non_targeting.10", "non-targeting.7", "non_targeting_3", "NTC_1"):
            assert parse_target_gene(name) == NEG_CTRL_LABEL

    def test_two_guides_of_one_target_collapse(self):
        assert parse_target_gene("CD81.1") == parse_target_gene("CD81.2")


class TestAssignGuides:
    def _frame(self, rows):
        return pd.DataFrame(rows, columns=["cell", "gene", "umi_count"])

    def test_dominant_guide_wins(self):
        out = assign_guides(
            self._frame([("A", "G1", 50), ("A", "G2", 5)]), min_umi=3, dominance_ratio=2.0
        )
        assert out["A"] == "G1"

    def test_codominant_guides_are_ambiguous(self):
        out = assign_guides(
            self._frame([("B", "G1", 20), ("B", "G2", 18)]), min_umi=3, dominance_ratio=2.0
        )
        assert out["B"] == AMBIGUOUS_LABEL

    def test_below_threshold_is_unassigned(self):
        out = assign_guides(self._frame([("C", "G1", 2)]), min_umi=3)
        assert out["C"] == UNASSIGNED_LABEL

    def test_result_does_not_depend_on_row_order(self):
        """The bug this replaces: to_dict() kept whichever row came last.

        With a mean of 2.3 guides per cell in the shipped table, that assigned
        a guide at random for every multiplet.
        """
        rows = [("A", "G1", 50), ("A", "G2", 5)]
        forward = assign_guides(self._frame(rows), min_umi=3)
        backward = assign_guides(self._frame(rows[::-1]), min_umi=3)
        assert forward["A"] == backward["A"] == "G1"

        naive_forward = self._frame(rows).set_index("cell")["gene"].to_dict()
        naive_backward = self._frame(rows[::-1]).set_index("cell")["gene"].to_dict()
        assert naive_forward["A"] != naive_backward["A"], "the old behaviour flips"

    def test_ambiguous_fraction_is_not_zero_on_multiplets(self):
        """Guards a subtle failure: groupby.nth(1) returns the original row
        index, so reindexing it by cell id yields all-NaN, the runner-up count
        reads as 0 and every cell looks dominant."""
        rng = np.random.default_rng(0)
        rows = []
        for i in range(200):
            cell = f"C{i}"
            a, b = rng.integers(10, 60), rng.integers(10, 60)
            rows.append((cell, "G1", int(a)))
            rows.append((cell, "G2", int(b)))
        out = assign_guides(self._frame(rows), min_umi=3, dominance_ratio=2.0)
        frac_ambiguous = (out == AMBIGUOUS_LABEL).mean()
        assert frac_ambiguous > 0.2, f"expected many ambiguous cells, got {frac_ambiguous}"

    def test_requires_a_count_column(self):
        frame = pd.DataFrame({"cell": ["A"], "gene": ["G1"]})
        with pytest.raises(ValueError, match="umi_count"):
            assign_guides(frame)


class TestExpressionCut:
    def test_median_is_degenerate_on_zero_inflated_data(self):
        """Why the quadrant cut moved from the control median to the mean.

        With dropout the control median is exactly 0 for most genes, so
        "low expression" collapses into "expression is exactly zero".
        """
        control = pd.Series([0.0] * 60 + [1.0, 2.0, 3.0] * 10)
        assert np.median(control) == 0.0
        assert control.mean() > 0.0
