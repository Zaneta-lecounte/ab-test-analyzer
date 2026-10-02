"""A/B Test Results Analyzer — Streamlit UI.

All statistics live in ``stats_engine.py``; this file only handles input,
layout and formatting.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

import stats_engine as se

SAMPLE_CSV = Path(__file__).parent / "sample_data.csv"

RECOMMENDATION_STYLE = {
    se.WIN: ("success", "✅ Win — ship the variant."),
    se.LOSS: ("error", "❌ Loss — keep control."),
    se.INCONCLUSIVE: ("warning", "⚖️ Inconclusive — not enough evidence of a difference."),
}


def pct(x: float | None, digits: int = 2, signed: bool = False) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    return f"{x:+.{digits}%}" if signed else f"{x:.{digits}%}"


def pp(x: float) -> str:
    """Format a proportion difference as percentage points."""
    return f"{x * 100:+.2f} pp"


def render_result(result: se.TestResult) -> None:
    confidence = 1 - result.alpha
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Control CVR", pct(result.control_cvr))
    c2.metric("Variant CVR", pct(result.variant_cvr))
    c3.metric("Absolute lift", pp(result.absolute_lift))
    c4.metric("Relative lift", pct(result.relative_lift, 1, signed=True))

    c5, c6, c7 = st.columns(3)
    c5.metric("p-value (two-sided)", f"{result.p_value:.4f}")
    c6.metric("z-statistic", f"{result.z_statistic:.3f}")
    c7.metric(f"{confidence:.0%} CI for difference", f"[{pp(result.ci_low)}, {pp(result.ci_high)}]")

    kind, message = RECOMMENDATION_STYLE[result.recommendation]
    getattr(st, kind)(f"**Recommendation: {message}** (α = {result.alpha:g})")

    if result.recommendation == se.INCONCLUSIVE:
        st.caption(
            "Inconclusive does not mean 'no effect'. The confidence interval shows the range of "
            "effects still consistent with the data; consider a larger sample if that range "
            "includes a lift you would care about."
        )


def manual_tab(alpha: float) -> None:
    st.subheader("Enter aggregate results")
    left, right = st.columns(2)
    with left:
        st.markdown("**Control (A)**")
        c_visitors = st.number_input("Control visitors", min_value=0, value=10000, step=100)
        c_conversions = st.number_input("Control conversions", min_value=0, value=500, step=10)
    with right:
        st.markdown("**Variant (B)**")
        v_visitors = st.number_input("Variant visitors", min_value=0, value=10000, step=100)
        v_conversions = st.number_input("Variant conversions", min_value=0, value=580, step=10)

    try:
        result = se.analyze_test(
            int(c_visitors), int(c_conversions), int(v_visitors), int(v_conversions), alpha=alpha
        )
    except ValueError as exc:
        st.error(f"Invalid input: {exc}")
        return
    st.divider()
    render_result(result)


def format_segment_table(df: pd.DataFrame, segment_cols: list[str]) -> pd.DataFrame:
    out = pd.DataFrame({"Segment": df["segment"]})
    out["Control visitors"] = df["control_visitors"]
    out["Variant visitors"] = df["variant_visitors"]
    for src, dst in [("control_cvr", "Control CVR"), ("variant_cvr", "Variant CVR")]:
        out[dst] = df[src].map(pct) if src in df else "n/a"
    out["Abs. lift"] = df["absolute_lift"].map(lambda x: "n/a" if pd.isna(x) else pp(x)) if "absolute_lift" in df else "n/a"
    out["Rel. lift"] = df["relative_lift"].map(lambda x: pct(x, 1, signed=True)) if "relative_lift" in df else "n/a"
    out["p-value"] = df["p_value"].map(lambda x: "n/a" if pd.isna(x) else f"{x:.4f}") if "p_value" in df else "n/a"
    if "ci_low" in df:
        out["CI"] = [
            "n/a" if pd.isna(lo) else f"[{pp(lo)}, {pp(hi)}]" for lo, hi in zip(df["ci_low"], df["ci_high"])
        ]
    out["Result"] = df["recommendation"]
    return out


def highlight_result(val: str) -> str:
    colors = {se.WIN: "#1a7f37", se.LOSS: "#cf222e", se.INCONCLUSIVE: "#9a6700"}
    return f"color: {colors[val]}; font-weight: 600" if val in colors else ""


def csv_tab(alpha: float) -> None:
    st.subheader("Upload segmented results")
    st.markdown(
        "CSV columns: `device`, `traffic_source`, `group` (control / variant), and either "
        "`visitors` + `conversions` (aggregated) **or** `converted` (0/1, one row per visitor)."
    )
    uploaded = st.file_uploader("Upload CSV", type="csv")
    use_sample = st.checkbox("Use bundled synthetic sample data", value=uploaded is None)

    if uploaded is not None and not use_sample:
        source = uploaded
    elif use_sample:
        source = SAMPLE_CSV
        with open(SAMPLE_CSV, "rb") as f:
            st.download_button("Download sample CSV", f, file_name="sample_data.csv", mime="text/csv")
    else:
        st.info("Upload a CSV or tick the sample data box to get started.")
        return

    try:
        raw = pd.read_csv(source)
        data = se.prepare_dataframe(raw, se.DEFAULT_SEGMENT_COLUMNS)
        overall = se.analyze_overall(data, alpha=alpha)
    except (ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        st.error(f"Could not analyse this file: {exc}")
        return

    with st.expander("Preview data"):
        st.dataframe(raw.head(50), width="stretch")

    st.markdown("### Overall result")
    render_result(overall)

    cross = se.analyze_segments(data, ["device", "traffic_source"], alpha=alpha)
    st.markdown("### Summary")
    st.info(se.summarize_segments(cross.to_dict("records"), overall))

    st.markdown("### Results by segment")
    views = {
        "Device × traffic source": ["device", "traffic_source"],
        "Device": ["device"],
        "Traffic source": ["traffic_source"],
    }
    for tab, (label, cols) in zip(st.tabs(list(views)), views.items()):
        with tab:
            res = cross if cols == views["Device × traffic source"] else se.analyze_segments(data, cols, alpha=alpha)
            table = format_segment_table(res, cols)
            st.dataframe(
                table.style.map(highlight_result, subset=["Result"]),
                width="stretch",
                hide_index=True,
            )
    st.caption(
        "Each segment is tested independently at the chosen α. With many segments some "
        "'wins' can appear by chance, so treat segment findings as hypotheses for a follow-up test."
    )


def main() -> None:
    st.set_page_config(page_title="A/B Test Results Analyzer", page_icon="🧪", layout="wide")
    st.title("🧪 A/B Test Results Analyzer")
    st.caption("Two-proportion z-test · confidence interval for the difference · segment breakdown")

    with st.sidebar:
        st.header("Settings")
        alpha = st.select_slider("Significance level (α)", options=[0.01, 0.05, 0.10], value=se.DEFAULT_ALPHA)
        st.markdown(
            f"**Decision rule** at α = {alpha:g}:\n"
            f"- **Win**: p < {alpha:g} and variant CVR > control\n"
            f"- **Loss**: p < {alpha:g} and variant CVR < control\n"
            "- **Inconclusive**: otherwise"
        )

    manual, segmented = st.tabs(["Quick analysis", "CSV + segments"])
    with manual:
        manual_tab(alpha)
    with segmented:
        csv_tab(alpha)


if __name__ == "__main__":
    main()
