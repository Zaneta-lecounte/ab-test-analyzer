import math
from pathlib import Path

import pandas as pd
import pytest

import stats_engine as se

SAMPLE_CSV = Path(__file__).resolve().parent.parent / "sample_data.csv"


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
class TestValidation:
    def test_valid_inputs_pass(self):
        se.validate_inputs(1000, 100, 1000, 120)

    @pytest.mark.parametrize(
        "args, message",
        [
            ((0, 0, 1000, 10), "Control visitors must be greater than zero"),
            ((1000, 10, 0, 0), "Variant visitors must be greater than zero"),
            ((100, 101, 100, 10), "Control conversions cannot exceed"),
            ((100, 10, 100, 101), "Variant conversions cannot exceed"),
            ((-1, 0, 100, 10), "cannot be negative"),
            ((100, -5, 100, 10), "cannot be negative"),
            ((100.5, 10, 100, 10), "whole number"),
            ((100, float("nan"), 100, 10), "whole number"),
            (("100", 10, 100, 10), "must be a number"),
            ((True, 0, 100, 10), "must be a number"),
        ],
    )
    def test_invalid_inputs_raise(self, args, message):
        with pytest.raises(ValueError, match=message):
            se.validate_inputs(*args)

    def test_whole_number_floats_are_accepted(self):
        se.validate_inputs(100.0, 10.0, 100.0, 12.0)

    def test_analyze_test_validates(self):
        with pytest.raises(ValueError):
            se.analyze_test(100, 200, 100, 10)

    @pytest.mark.parametrize("alpha", [0, 1, -0.1, 1.5])
    def test_invalid_alpha(self, alpha):
        with pytest.raises(ValueError, match="Alpha"):
            se.analyze_test(100, 10, 100, 12, alpha=alpha)


# --------------------------------------------------------------------------- #
# Basic metrics
# --------------------------------------------------------------------------- #
class TestMetrics:
    def test_conversion_rate(self):
        assert se.conversion_rate(50, 1000) == pytest.approx(0.05)

    def test_conversion_rate_rejects_zero_visitors(self):
        with pytest.raises(ValueError):
            se.conversion_rate(0, 0)

    def test_absolute_lift(self):
        assert se.absolute_lift(0.10, 0.12) == pytest.approx(0.02)

    def test_relative_lift(self):
        assert se.relative_lift(0.10, 0.12) == pytest.approx(0.20)

    def test_relative_lift_with_zero_control_is_none(self):
        assert se.relative_lift(0.0, 0.05) is None


# --------------------------------------------------------------------------- #
# z-test and confidence interval
# --------------------------------------------------------------------------- #
class TestZTest:
    def test_known_values(self):
        # Hand-computed: p_c = 0.10, p_v = 0.12, pooled = 0.11
        # SE = sqrt(0.11 * 0.89 * (2 / 1000)) = 0.0139929
        # z = 0.02 / 0.0139929 = 1.42929, two-sided p = 0.15291
        z, p = se.two_proportion_z_test(1000, 100, 1000, 120)
        assert z == pytest.approx(1.42929, abs=1e-4)
        assert p == pytest.approx(0.15291, abs=1e-4)

    def test_identical_rates_give_p_of_one(self):
        z, p = se.two_proportion_z_test(1000, 100, 1000, 100)
        assert z == pytest.approx(0.0)
        assert p == pytest.approx(1.0)

    def test_zero_variance_edge_cases(self):
        assert se.two_proportion_z_test(100, 0, 100, 0) == (0.0, 1.0)
        assert se.two_proportion_z_test(100, 100, 100, 100) == (0.0, 1.0)

    def test_symmetry(self):
        z1, p1 = se.two_proportion_z_test(5000, 250, 5000, 300)
        z2, p2 = se.two_proportion_z_test(5000, 300, 5000, 250)
        assert z1 == pytest.approx(-z2)
        assert p1 == pytest.approx(p2)

    def test_matches_scipy(self):
        stats = pytest.importorskip("scipy.stats")
        c_n, c_x, v_n, v_x = 12000, 480, 11800, 560
        z, p = se.two_proportion_z_test(c_n, c_x, v_n, v_x)
        pooled = (c_x + v_x) / (c_n + v_n)
        se_ = math.sqrt(pooled * (1 - pooled) * (1 / c_n + 1 / v_n))
        z_ref = (v_x / v_n - c_x / c_n) / se_
        assert z == pytest.approx(z_ref)
        assert p == pytest.approx(2 * stats.norm.sf(abs(z_ref)))


class TestConfidenceInterval:
    def test_known_values(self):
        # SE_unpooled = sqrt(0.1*0.9/1000 + 0.12*0.88/1000) = 0.0139857
        # 1.959964 * SE = 0.0274115
        low, high = se.difference_confidence_interval(1000, 100, 1000, 120)
        assert low == pytest.approx(0.02 - 0.0274115, abs=1e-5)
        assert high == pytest.approx(0.02 + 0.0274115, abs=1e-5)

    def test_interval_contains_point_estimate(self):
        low, high = se.difference_confidence_interval(4000, 200, 4100, 260)
        diff = 260 / 4100 - 200 / 4000
        assert low < diff < high

    def test_wider_confidence_gives_wider_interval(self):
        l95, h95 = se.difference_confidence_interval(1000, 100, 1000, 120, 0.95)
        l99, h99 = se.difference_confidence_interval(1000, 100, 1000, 120, 0.99)
        assert (h99 - l99) > (h95 - l95)

    def test_more_data_gives_narrower_interval(self):
        small = se.difference_confidence_interval(1000, 100, 1000, 120)
        large = se.difference_confidence_interval(100000, 10000, 100000, 12000)
        assert (large[1] - large[0]) < (small[1] - small[0])

    def test_invalid_confidence(self):
        with pytest.raises(ValueError):
            se.difference_confidence_interval(1000, 100, 1000, 120, confidence=1.0)


# --------------------------------------------------------------------------- #
# Recommendation and full analysis
# --------------------------------------------------------------------------- #
class TestRecommendation:
    def test_win(self):
        assert se.recommend(0.01, 0.02) == se.WIN

    def test_loss(self):
        assert se.recommend(0.01, -0.02) == se.LOSS

    def test_inconclusive_when_not_significant(self):
        assert se.recommend(0.20, 0.02) == se.INCONCLUSIVE

    def test_boundary_p_equal_alpha_is_inconclusive(self):
        assert se.recommend(0.05, 0.02, alpha=0.05) == se.INCONCLUSIVE

    def test_zero_lift_is_inconclusive(self):
        assert se.recommend(0.0, 0.0) == se.INCONCLUSIVE


class TestAnalyzeTest:
    def test_clear_win(self):
        res = se.analyze_test(10000, 500, 10000, 650)
        assert res.control_cvr == pytest.approx(0.05)
        assert res.variant_cvr == pytest.approx(0.065)
        assert res.absolute_lift == pytest.approx(0.015)
        assert res.relative_lift == pytest.approx(0.30)
        assert res.p_value < 0.05
        assert res.ci_low > 0
        assert res.recommendation == se.WIN
        assert res.is_significant

    def test_clear_loss(self):
        res = se.analyze_test(10000, 650, 10000, 500)
        assert res.recommendation == se.LOSS
        assert res.ci_high < 0

    def test_inconclusive(self):
        res = se.analyze_test(1000, 100, 1000, 120)
        assert res.recommendation == se.INCONCLUSIVE
        assert res.ci_low < 0 < res.ci_high

    def test_ci_agrees_with_decision_for_clear_results(self):
        for args in [(20000, 1000, 20000, 1200), (20000, 1200, 20000, 1000)]:
            res = se.analyze_test(*args)
            excludes_zero = res.ci_low > 0 or res.ci_high < 0
            assert excludes_zero == res.is_significant

    def test_accepts_numpy_integers(self):
        np = pytest.importorskip("numpy")
        res = se.analyze_test(np.int64(1000), np.int64(100), np.int64(1000), np.int64(120))
        assert isinstance(res.control_visitors, int)

    def test_to_dict(self):
        d = se.analyze_test(1000, 100, 1000, 120).to_dict()
        assert d["recommendation"] == se.INCONCLUSIVE
        assert set(d) >= {"control_cvr", "variant_cvr", "p_value", "ci_low", "ci_high"}


# --------------------------------------------------------------------------- #
# Segment analysis (Phase 2)
# --------------------------------------------------------------------------- #
@pytest.fixture
def sample_df():
    return se.prepare_dataframe(pd.read_csv(SAMPLE_CSV))


class TestPrepareDataframe:
    def test_sample_csv_loads(self, sample_df):
        assert set(sample_df["group"]) == {"control", "variant"}
        assert list(sample_df.columns) == ["device", "traffic_source", "group", "visitors", "conversions"]

    def test_user_level_rows_are_supported(self):
        raw = pd.DataFrame(
            {
                "device": ["mobile"] * 4,
                "traffic_source": ["paid"] * 4,
                "group": ["control", "control", "variant", "variant"],
                "converted": [0, 1, 1, 1],
            }
        )
        out = se.prepare_dataframe(raw)
        assert out["visitors"].sum() == 4
        assert out["conversions"].sum() == 3

    def test_group_labels_and_column_names_are_normalised(self):
        raw = pd.DataFrame(
            {
                " Device ": ["Mobile", "Mobile"],
                "Traffic_Source": ["Paid", "Paid"],
                "Group": ["A", "B"],
                "Visitors": [100, 100],
                "Conversions": [5, 8],
            }
        )
        out = se.prepare_dataframe(raw)
        assert list(out["group"]) == ["control", "variant"]
        assert list(out["device"]) == ["mobile", "mobile"]

    @pytest.mark.parametrize(
        "frame, message",
        [
            ({"device": ["m"], "traffic_source": ["p"], "visitors": [1], "conversions": [0]}, "group"),
            ({"traffic_source": ["p"], "group": ["control"], "visitors": [1], "conversions": [0]}, "device"),
            ({"device": ["m"], "traffic_source": ["p"], "group": ["control"]}, "visitors"),
            (
                {"device": ["m", "m"], "traffic_source": ["p", "p"], "group": ["control", "variant"],
                 "visitors": [10, 10], "conversions": [11, 1]},
                "exceed",
            ),
            (
                {"device": ["m", "m"], "traffic_source": ["p", "p"], "group": ["control", "variant"],
                 "visitors": [10, -1], "conversions": [1, 0]},
                "negative",
            ),
            (
                {"device": ["m", "m"], "traffic_source": ["p", "p"], "group": ["control", "variant"],
                 "visitors": [10, "abc"], "conversions": [1, 0]},
                "non-numeric",
            ),
            (
                {"device": ["m", "m"], "traffic_source": ["p", "p"], "group": ["control", "oops"],
                 "visitors": [10, 10], "conversions": [1, 0]},
                "Unrecognised group",
            ),
            (
                {"device": ["m"], "traffic_source": ["p"], "group": ["control"],
                 "visitors": [10], "conversions": [1]},
                "both control and variant",
            ),
        ],
    )
    def test_invalid_frames_raise(self, frame, message):
        with pytest.raises(ValueError, match=message):
            se.prepare_dataframe(pd.DataFrame(frame))


class TestAnalyzeSegments:
    def test_overall_totals(self, sample_df):
        overall = se.analyze_overall(sample_df)
        assert overall.control_visitors == sample_df.query("group == 'control'")["visitors"].sum()
        assert overall.recommendation == se.WIN

    def test_single_dimension(self, sample_df):
        res = se.analyze_segments(sample_df, ["device"])
        assert set(res["segment"]) == {"mobile", "desktop"}
        by_seg = res.set_index("segment")
        assert by_seg.loc["mobile", "recommendation"] == se.WIN
        assert by_seg.loc["desktop", "recommendation"] == se.INCONCLUSIVE

    def test_cross_segments(self, sample_df):
        res = se.analyze_segments(sample_df, ["device", "traffic_source"])
        assert len(res) == 6
        by_seg = res.set_index("segment")
        assert by_seg.loc["paid mobile", "recommendation"] == se.WIN
        others = by_seg.drop(index="paid mobile")
        assert (others["recommendation"] == se.INCONCLUSIVE).all()

    def test_segment_counts_sum_to_overall(self, sample_df):
        res = se.analyze_segments(sample_df, ["device", "traffic_source"])
        overall = se.analyze_overall(sample_df)
        assert res["control_visitors"].sum() == overall.control_visitors
        assert res["variant_conversions"].sum() == overall.variant_conversions

    def test_missing_arm_is_marked_insufficient(self):
        df = se.prepare_dataframe(
            pd.DataFrame(
                {
                    "device": ["mobile", "mobile", "desktop"],
                    "traffic_source": ["paid", "paid", "paid"],
                    "group": ["control", "variant", "control"],
                    "visitors": [100, 100, 50],
                    "conversions": [10, 12, 5],
                }
            )
        )
        res = se.analyze_segments(df, ["device"]).set_index("segment")
        assert res.loc["desktop", "recommendation"] == "Insufficient data"
        assert res.loc["mobile", "recommendation"] == se.INCONCLUSIVE


def _seg(segment, recommendation, incremental, rel=0.1):
    return {
        "segment": segment,
        "recommendation": recommendation,
        "incremental_conversions": incremental,
        "relative_lift": rel,
    }


class TestSummarizeSegments:
    def test_sample_data_summary_mentions_paid_mobile(self, sample_df):
        res = se.analyze_segments(sample_df, ["device", "traffic_source"])
        summary = se.summarize_segments(res.to_dict("records"), se.analyze_overall(sample_df))
        assert "Effect is concentrated in paid mobile traffic" in summary
        assert summary.startswith("Overall, the variant wins")

    def test_multiple_winners(self):
        summary = se.summarize_segments(
            [_seg("paid mobile", se.WIN, 50), _seg("organic desktop", se.WIN, 45), _seg("email mobile", se.INCONCLUSIVE, 5)]
        )
        assert "paid mobile and organic desktop" in summary
        assert "concentrated" not in summary

    def test_single_winner_below_threshold_is_not_concentrated(self):
        summary = se.summarize_segments(
            [_seg("paid mobile", se.WIN, 30), _seg("organic desktop", se.INCONCLUSIVE, 40)]
        )
        assert "concentrated" not in summary
        assert "Significant positive effect in paid mobile traffic" in summary

    def test_no_significant_segments(self):
        summary = se.summarize_segments([_seg("a", se.INCONCLUSIVE, 2), _seg("b", se.INCONCLUSIVE, -1)])
        assert "No individual segment" in summary
        assert "false positives" not in summary

    def test_losers_are_reported(self):
        summary = se.summarize_segments(
            [_seg("paid mobile", se.WIN, 80), _seg("email desktop", se.LOSS, -20)]
        )
        assert "underperforms in email desktop" in summary

    def test_no_valid_segments(self):
        summary = se.summarize_segments([{"segment": "x", "recommendation": "Insufficient data"}])
        assert "Not enough data" in summary

    def test_overall_inconclusive_sentence(self):
        overall = se.analyze_test(1000, 100, 1000, 105)
        summary = se.summarize_segments([_seg("a", se.INCONCLUSIVE, 1)], overall)
        assert summary.startswith("Overall, the result is inconclusive")
