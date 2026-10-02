"""Pure statistical functions for A/B test analysis.

Nothing in this module touches the UI. Every function is deterministic and
side-effect free, which keeps the math easy to unit test. Only the standard
library is used for the core statistics; pandas is needed only by the
segment helpers that operate on uploaded CSV data.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import asdict, dataclass
from statistics import NormalDist
from typing import Iterable, Sequence

DEFAULT_ALPHA = 0.05

WIN = "Win"
LOSS = "Loss"
INCONCLUSIVE = "Inconclusive"

_STD_NORMAL = NormalDist()


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate_inputs(
    control_visitors: int,
    control_conversions: int,
    variant_visitors: int,
    variant_conversions: int,
) -> None:
    """Raise ``ValueError`` if the four counts cannot describe a valid test."""
    counts = {
        "control visitors": control_visitors,
        "control conversions": control_conversions,
        "variant visitors": variant_visitors,
        "variant conversions": variant_conversions,
    }
    for name, value in counts.items():
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise ValueError(f"{name.capitalize()} must be a number.")
        if not isinstance(value, numbers.Integral) and (
            math.isnan(value) or not float(value).is_integer()
        ):
            raise ValueError(f"{name.capitalize()} must be a whole number.")
        if value < 0:
            raise ValueError(f"{name.capitalize()} cannot be negative.")

    if control_visitors == 0:
        raise ValueError("Control visitors must be greater than zero.")
    if variant_visitors == 0:
        raise ValueError("Variant visitors must be greater than zero.")
    if control_conversions > control_visitors:
        raise ValueError("Control conversions cannot exceed control visitors.")
    if variant_conversions > variant_visitors:
        raise ValueError("Variant conversions cannot exceed variant visitors.")


def _validate_alpha(alpha: float) -> None:
    if not 0 < alpha < 1:
        raise ValueError("Alpha must be between 0 and 1.")


# --------------------------------------------------------------------------- #
# Core statistics
# --------------------------------------------------------------------------- #
def conversion_rate(conversions: int, visitors: int) -> float:
    """Conversions divided by visitors."""
    if visitors <= 0:
        raise ValueError("Visitors must be greater than zero.")
    if conversions < 0 or conversions > visitors:
        raise ValueError("Conversions must be between 0 and visitors.")
    return conversions / visitors


def absolute_lift(control_cvr: float, variant_cvr: float) -> float:
    """Variant CVR minus control CVR (in proportion units, not points)."""
    return variant_cvr - control_cvr


def relative_lift(control_cvr: float, variant_cvr: float) -> float | None:
    """(variant - control) / control. ``None`` when control CVR is zero."""
    if control_cvr == 0:
        return None
    return (variant_cvr - control_cvr) / control_cvr


def two_proportion_z_test(
    control_visitors: int,
    control_conversions: int,
    variant_visitors: int,
    variant_conversions: int,
) -> tuple[float, float]:
    """Two-sided two-proportion z-test using the pooled standard error.

    Returns ``(z_statistic, p_value)``. When both arms have identical 0% or
    100% conversion the variance is zero, so there is no evidence of a
    difference: ``(0.0, 1.0)`` is returned.
    """
    validate_inputs(control_visitors, control_conversions, variant_visitors, variant_conversions)
    p_c = control_conversions / control_visitors
    p_v = variant_conversions / variant_visitors
    pooled = (control_conversions + variant_conversions) / (control_visitors + variant_visitors)
    se = math.sqrt(pooled * (1 - pooled) * (1 / control_visitors + 1 / variant_visitors))
    if se == 0:
        return 0.0, 1.0
    z = (p_v - p_c) / se
    p_value = math.erfc(abs(z) / math.sqrt(2))  # == 2 * (1 - Phi(|z|))
    return z, min(1.0, p_value)


def difference_confidence_interval(
    control_visitors: int,
    control_conversions: int,
    variant_visitors: int,
    variant_conversions: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Wald confidence interval for (variant CVR - control CVR).

    Uses the unpooled standard error, which is the standard choice for an
    interval estimate (the pooled SE assumes the null hypothesis is true).
    """
    validate_inputs(control_visitors, control_conversions, variant_visitors, variant_conversions)
    if not 0 < confidence < 1:
        raise ValueError("Confidence must be between 0 and 1.")
    p_c = control_conversions / control_visitors
    p_v = variant_conversions / variant_visitors
    se = math.sqrt(p_c * (1 - p_c) / control_visitors + p_v * (1 - p_v) / variant_visitors)
    z_crit = _STD_NORMAL.inv_cdf(1 - (1 - confidence) / 2)
    diff = p_v - p_c
    return diff - z_crit * se, diff + z_crit * se


def recommend(p_value: float, abs_lift: float, alpha: float = DEFAULT_ALPHA) -> str:
    """Win / Loss / Inconclusive decision rule at significance level ``alpha``."""
    _validate_alpha(alpha)
    if p_value < alpha and abs_lift > 0:
        return WIN
    if p_value < alpha and abs_lift < 0:
        return LOSS
    return INCONCLUSIVE


@dataclass(frozen=True)
class TestResult:
    control_visitors: int
    control_conversions: int
    variant_visitors: int
    variant_conversions: int
    control_cvr: float
    variant_cvr: float
    absolute_lift: float
    relative_lift: float | None
    z_statistic: float
    p_value: float
    ci_low: float
    ci_high: float
    alpha: float
    recommendation: str

    @property
    def is_significant(self) -> bool:
        return self.p_value < self.alpha

    def to_dict(self) -> dict:
        return asdict(self)


def analyze_test(
    control_visitors: int,
    control_conversions: int,
    variant_visitors: int,
    variant_conversions: int,
    alpha: float = DEFAULT_ALPHA,
) -> TestResult:
    """Run the full MVP analysis on aggregate counts."""
    validate_inputs(control_visitors, control_conversions, variant_visitors, variant_conversions)
    _validate_alpha(alpha)
    c_v, c_c = int(control_visitors), int(control_conversions)
    v_v, v_c = int(variant_visitors), int(variant_conversions)

    control_cvr = conversion_rate(c_c, c_v)
    variant_cvr = conversion_rate(v_c, v_v)
    abs_lift = absolute_lift(control_cvr, variant_cvr)
    z, p_value = two_proportion_z_test(c_v, c_c, v_v, v_c)
    ci_low, ci_high = difference_confidence_interval(c_v, c_c, v_v, v_c, confidence=1 - alpha)

    return TestResult(
        control_visitors=c_v,
        control_conversions=c_c,
        variant_visitors=v_v,
        variant_conversions=v_c,
        control_cvr=control_cvr,
        variant_cvr=variant_cvr,
        absolute_lift=abs_lift,
        relative_lift=relative_lift(control_cvr, variant_cvr),
        z_statistic=z,
        p_value=p_value,
        ci_low=ci_low,
        ci_high=ci_high,
        alpha=alpha,
        recommendation=recommend(p_value, abs_lift, alpha),
    )


# --------------------------------------------------------------------------- #
# Segment analysis (Phase 2)
# --------------------------------------------------------------------------- #
CONTROL_LABELS = {"control", "a", "baseline"}
VARIANT_LABELS = {"variant", "b", "treatment", "test"}
DEFAULT_SEGMENT_COLUMNS = ("device", "traffic_source")


def prepare_dataframe(df, segment_columns: Sequence[str] = DEFAULT_SEGMENT_COLUMNS):
    """Normalise an uploaded CSV into aggregated ``visitors`` / ``conversions``.

    Accepted shapes:
      * aggregated: ``group, visitors, conversions`` (+ segment columns)
      * user-level: ``group, converted`` with one row per visitor

    Returns a new DataFrame with a canonical ``group`` column holding
    ``"control"`` / ``"variant"``. Raises ``ValueError`` on bad input.
    """
    import pandas as pd

    data = df.copy()
    data.columns = [str(c).strip().lower() for c in data.columns]

    if "group" not in data.columns:
        raise ValueError("CSV must include a 'group' column (control / variant).")
    missing = [c for c in segment_columns if c not in data.columns]
    if missing:
        raise ValueError(f"CSV is missing segment column(s): {', '.join(missing)}.")

    if {"visitors", "conversions"}.issubset(data.columns):
        pass
    elif "converted" in data.columns:
        data["visitors"] = 1
        data["conversions"] = data["converted"]
    else:
        raise ValueError(
            "CSV must have either 'visitors' and 'conversions' columns (aggregated) "
            "or a 'converted' column (one row per visitor)."
        )

    for col in ("visitors", "conversions"):
        data[col] = pd.to_numeric(data[col], errors="coerce")
        if data[col].isna().any():
            raise ValueError(f"Column '{col}' contains non-numeric or empty values.")
        if (data[col] < 0).any():
            raise ValueError(f"Column '{col}' cannot contain negative values.")
        if not (data[col] % 1 == 0).all():
            raise ValueError(f"Column '{col}' must contain whole numbers.")
        data[col] = data[col].astype(int)

    if (data["conversions"] > data["visitors"]).any():
        raise ValueError("Conversions cannot exceed visitors in any row.")

    group = data["group"].astype(str).str.strip().str.lower()
    canonical = group.map(
        lambda g: "control" if g in CONTROL_LABELS else ("variant" if g in VARIANT_LABELS else None)
    )
    unknown = sorted(set(group[canonical.isna()]))
    if unknown:
        raise ValueError(
            f"Unrecognised group label(s): {', '.join(unknown)}. Use 'control' or 'variant'."
        )
    data["group"] = canonical

    for col in segment_columns:
        data[col] = data[col].astype(str).str.strip().str.lower()

    if not {"control", "variant"}.issubset(set(data["group"])):
        raise ValueError("CSV must contain both control and variant rows.")

    return data[[*segment_columns, "group", "visitors", "conversions"]]


def _segment_label(keys: Iterable[str], columns: Sequence[str]) -> str:
    """Human-readable label, e.g. ('mobile', 'paid') -> 'paid mobile'."""
    values = dict(zip(columns, keys))
    # Put traffic source before device so labels read naturally ("paid mobile").
    ordered = sorted(columns, key=lambda c: 0 if c == "traffic_source" else 1)
    return " ".join(values[c] for c in ordered)


def analyze_segments(df, segment_columns: Sequence[str], alpha: float = DEFAULT_ALPHA):
    """Run :func:`analyze_test` for every combination of ``segment_columns``.

    ``df`` must already be normalised by :func:`prepare_dataframe`. Segments
    with zero visitors in either arm are kept but marked as having
    insufficient data rather than raising.
    """
    import pandas as pd

    _validate_alpha(alpha)
    cols = list(segment_columns)
    grouped = df.groupby([*cols, "group"], as_index=False)[["visitors", "conversions"]].sum()
    pivot = grouped.pivot_table(
        index=cols, columns="group", values=["visitors", "conversions"], fill_value=0, aggfunc="sum"
    )

    rows = []
    for keys, row in pivot.iterrows():
        keys = keys if isinstance(keys, tuple) else (keys,)
        c_v = int(row.get(("visitors", "control"), 0))
        c_c = int(row.get(("conversions", "control"), 0))
        v_v = int(row.get(("visitors", "variant"), 0))
        v_c = int(row.get(("conversions", "variant"), 0))
        record = {col: key for col, key in zip(cols, keys)}
        record["segment"] = _segment_label(keys, cols)
        if c_v == 0 or v_v == 0:
            record.update(
                control_visitors=c_v, control_conversions=c_c,
                variant_visitors=v_v, variant_conversions=v_c,
                recommendation="Insufficient data",
            )
        else:
            res = analyze_test(c_v, c_c, v_v, v_c, alpha=alpha)
            record.update(res.to_dict())
            # Expected extra conversions attributable to the variant in this segment.
            record["incremental_conversions"] = res.absolute_lift * v_v
        rows.append(record)

    return pd.DataFrame(rows)


def _join_labels(labels: list[str]) -> str:
    if len(labels) == 1:
        return labels[0]
    return ", ".join(labels[:-1]) + " and " + labels[-1]


def summarize_segments(
    segment_results: Sequence[dict],
    overall: TestResult | None = None,
    concentration_threshold: float = 0.5,
) -> str:
    """Plain-English summary of where the effect lives.

    ``segment_results`` is a list of dicts as produced by
    :func:`analyze_segments` (``DataFrame.to_dict("records")``). A segment is
    called out if its result is significant. A single winning segment that
    accounts for at least ``concentration_threshold`` of all positive
    incremental conversions is described as the place the effect is
    "concentrated".
    """
    valid = [r for r in segment_results if r.get("recommendation") in (WIN, LOSS, INCONCLUSIVE)]
    if not valid:
        return "Not enough data in any segment to compare control and variant."

    sentences: list[str] = []
    if overall is not None:
        if overall.recommendation == WIN:
            sentences.append(
                f"Overall, the variant wins ({overall.relative_lift:+.1%} relative lift, "
                f"p = {overall.p_value:.3g})."
                if overall.relative_lift is not None
                else f"Overall, the variant wins (p = {overall.p_value:.3g})."
            )
        elif overall.recommendation == LOSS:
            sentences.append(f"Overall, the variant loses (p = {overall.p_value:.3g}).")
        else:
            sentences.append(
                f"Overall, the result is inconclusive (p = {overall.p_value:.3g})."
            )

    winners = sorted(
        (r for r in valid if r["recommendation"] == WIN),
        key=lambda r: r["incremental_conversions"],
        reverse=True,
    )
    losers = sorted(
        (r for r in valid if r["recommendation"] == LOSS),
        key=lambda r: r["incremental_conversions"],
    )
    total_positive = sum(max(r["incremental_conversions"], 0.0) for r in valid)

    if winners:
        top = winners[0]
        share = top["incremental_conversions"] / total_positive if total_positive > 0 else 0.0
        if len(winners) == 1 and share >= concentration_threshold:
            sentences.append(
                f"Effect is concentrated in {top['segment']} traffic "
                f"({top['relative_lift']:+.1%} lift there, ~{share:.0%} of incremental conversions)."
                if top["relative_lift"] is not None
                else f"Effect is concentrated in {top['segment']} traffic "
                f"(~{share:.0%} of incremental conversions)."
            )
        else:
            labels = [w["segment"] for w in winners]
            sentences.append(f"Significant positive effect in {_join_labels(labels)} traffic.")
    else:
        sentences.append("No individual segment shows a statistically significant lift.")

    if losers:
        labels = [loser["segment"] for loser in losers]
        sentences.append(f"The variant significantly underperforms in {_join_labels(labels)} traffic.")

    if winners or losers:
        sentences.append(
            "Treat segment findings as hypotheses: testing many segments raises the "
            "chance of false positives, so confirm with a follow-up test."
        )
    return " ".join(sentences)


def analyze_overall(df, alpha: float = DEFAULT_ALPHA) -> TestResult:
    """Pool every row of a normalised DataFrame into a single control-vs-variant test."""
    totals = df.groupby("group")[["visitors", "conversions"]].sum()
    return analyze_test(
        int(totals.loc["control", "visitors"]),
        int(totals.loc["control", "conversions"]),
        int(totals.loc["variant", "visitors"]),
        int(totals.loc["variant", "conversions"]),
        alpha=alpha,
    )
