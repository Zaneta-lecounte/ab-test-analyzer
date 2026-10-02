# 🧪 A/B Test Results Analyzer

A Streamlit app that turns raw A/B test counts into a clear ship / don't-ship decision. It also breaks results down by segment to show **where** an effect comes from.

---

## Problem

Growth teams run many experiments, but reading the results is often ad hoc. Someone pastes numbers into an online calculator, eyeballs a "lift", and calls a winner. That causes three recurring problems:

1. **False wins.** A +10% relative lift on a small sample is often noise. Without a significance test and a confidence interval, teams ship changes that do nothing.
2. **No shared decision rule.** "Is p = 0.07 a win?" gets re-argued in every review.
3. **Blended averages hide the story.** A test can look flat overall while working well for one audience, or "win" overall because of a single segment. Without a segment breakdown the team learns the wrong lesson and builds the wrong follow-up.

## Solution

A lightweight analyzer that:

- Takes control and variant visitors and conversions, then reports conversion rates, absolute and relative lift, a **two-proportion z-test p-value**, and a **95% confidence interval** for the difference.
- Applies a fixed, transparent decision rule: **Win / Loss / Inconclusive** at α = 0.05.
- Accepts a **CSV with segment columns** (`device`, `traffic_source`), runs the same test per segment, and writes a **plain-English summary** such as *"Effect is concentrated in paid mobile traffic."*

## What I Built

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI with two tabs: **Quick analysis** (manual inputs) and **CSV + segments**. Handles only layout and formatting. |
| `stats_engine.py` | Pure, side-effect-free math. The core stats use only the Python standard library (`math`, `statistics.NormalDist`). |
| `tests/test_stats_engine.py` | 66 pytest cases covering validation, hand-computed z-test and CI values, a SciPy cross-check, decision-rule edge cases, CSV parsing and segment summaries. |
| `sample_data.csv` | **Synthetic** data (no real users) with a planted effect in paid mobile traffic. |

**Statistics**

- **CVR** = conversions ÷ visitors
- **Absolute lift** = CVR<sub>B</sub> − CVR<sub>A</sub> (shown in percentage points); **relative lift** = absolute lift ÷ CVR<sub>A</sub>
- **Two-proportion z-test** (two-sided) with the *pooled* standard error, since the null hypothesis assumes equal rates
- **Confidence interval for the difference** with the *unpooled* (Wald) standard error, since an interval estimate shouldn't assume the null
- **Decision rule:** Win if p < α and lift > 0; Loss if p < α and lift < 0; otherwise Inconclusive

**Input validation:** visitors must be greater than 0, conversions must be between 0 and visitors, and counts must be whole numbers. Uploaded CSVs are checked for required columns, numeric values, recognised group labels (`control/variant`, `A/B`, `treatment`…), and the presence of both arms.

**Segment analysis:** results are computed for device, traffic source, and device × traffic source. Each segment's *incremental conversions* (lift × variant visitors) is used to judge concentration. The effect counts as "concentrated" in a segment when it is the only significant winner and accounts for at least 50% of the positive incremental conversions. Segments with a missing arm are flagged as *Insufficient data* instead of crashing the analysis.

**CSV format** (aggregated):

```csv
device,traffic_source,group,visitors,conversions
mobile,paid,control,8200,287
mobile,paid,variant,8150,391
```

User-level rows also work: replace `visitors,conversions` with a single `converted` column (0/1).

## Growth Methodology

The tool sits in the last two steps of the experiment loop, and its output feeds the next round:

1. **Signal**: a metric moves or a funnel leaks. *Example:* mobile checkout conversion is about 30% below desktop.
2. **Diagnosis**: find the cause using funnel analysis, session recordings and segment cuts. *Example:* the mobile payment form needs a lot of typing, and paid-social visitors (new, low-intent) drop off most.
3. **Hypothesis**: state a falsifiable, measurable prediction. *Example:* "Adding express wallet pay will raise mobile checkout CVR by ≥ 10% relative, because it removes form friction."
4. **Experiment**: randomise visitors into control and variant, fix the sample size and α **before** launch, and run it to completion (no peeking).
5. **Decision**: use this analyzer. Apply the pre-agreed rule (Win / Loss / Inconclusive), read the CI to see the plausible range of impact, and use the segment summary to shape the next hypothesis. *Example:* "Effect is concentrated in paid mobile traffic" suggests rolling out to mobile first and testing a stronger variant for organic users.

## How to Run

```bash
git clone https://github.com/zaneta-lecounte/ab-test-analyzer.git
cd ab-test-analyzer

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py              # opens http://localhost:8501
pytest                            # run the test suite
```

In the **CSV + segments** tab, tick *Use bundled synthetic sample data* to see the segment summary right away.

## Tech

- **Python 3.10+**
- **Streamlit**: UI
- **pandas**: CSV parsing and segment aggregation
- **pytest**: tests (SciPy is used only for an optional cross-check if it's installed)
- Core statistics use only the standard library, with no SciPy or statsmodels dependency

## Future Improvements

- **Multiple-comparison correction** (Bonferroni or Benjamini–Hochberg) for segment results
- **Power and sample-size calculator** to plan tests before launch (minimum detectable effect, test duration)
- **Sequential testing / always-valid p-values** so teams can monitor tests without inflating false positives
- **Bayesian view**: probability that the variant beats control and expected loss
- **Sample ratio mismatch (SRM) check** to catch broken randomisation
- **Continuous metrics** (revenue per visitor, AOV) using Welch's t-test or a bootstrap
- **Configurable segment columns** and multi-variant (A/B/n) tests
- **Charts**: CI forest plot per segment and cumulative CVR over time
- **Exportable decision memo** (Markdown/PDF) for experiment reviews
