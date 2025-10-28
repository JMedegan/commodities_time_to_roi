# app_trendy.py — Aesthetic & trendy Streamlit dashboard for ROI models

import os
import pickle
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =========================
# ---- CONFIG: PATHS -----
# =========================
REG_RESULTS_PATH = os.getenv(
    "REG_RESULTS_PATH", "data/08_reporting/time_to_roi_results.pkl"
)
REG_TEST_PATH = os.getenv(
    "REG_TEST_PATH", "data/07_model_output/test_set_with_time_to_roi_preds.csv"
)
CLF_RESULTS_PATH = os.getenv(
    "CLF_RESULTS_PATH", "data/08_reporting/roi_within_horizon_results.pkl"
)
CLF_TEST_PATH = os.getenv(
    "CLF_TEST_PATH", "data/07_model_output/test_set_with_roi_within_horizon_preds.csv"
)

# =========================
# ---- THEME & STYLES ----
# =========================
st.set_page_config(page_title="ROI Models", layout="wide", page_icon="💹")
mode_vibe = st.sidebar.selectbox("Theme", ["Minimal Light", "Charcoal Dark"], index=0)

BASE_BG = "#ffffff" if mode_vibe == "Minimal Light" else "#0E1117"
CARD_BG = "#f7f8fb" if mode_vibe == "Minimal Light" else "#151a22"
TEXT = "#1f2937" if mode_vibe == "Minimal Light" else "#e5e7eb"
SUBTLE = "#6b7280" if mode_vibe == "Minimal Light" else "#9ca3af"
ACCENT = "#2563eb" if mode_vibe == "Minimal Light" else "#60a5fa"

st.markdown(
    f"""
<style>
/* Page background + text */
[data-testid="stAppViewContainer"] {{
  background: {BASE_BG};
  color: {TEXT};
}}
h1, h2, h3, h4 {{ color: {TEXT}; }}
/* Section subtitles */
.block-container {{ padding-top: 2rem; }}
/* Card look */
.card {{
  background: {CARD_BG};
  border: 1px solid rgba(125, 125, 125, 0.08);
  padding: 14px 16px;
  border-radius: 14px;
  box-shadow: 0 2px 6px rgba(0,0,0,0.06);
}}
.card-title {{
  font-size: 0.9rem;
  color: {SUBTLE};
  margin-bottom: 6px;
}}
.card-value {{
  font-size: 1.6rem;
  font-weight: 600;
  color: {TEXT};
  line-height: 1.1;
}}
.card-subtle {{
  font-size: 0.8rem;
  color: {SUBTLE};
  margin-top: 4px;
}}
/* Selectbox label tweak */
.stSelectbox label {{
  font-weight: 600;
  color: {SUBTLE};
}}
/* Plotly container radius illusion */
.js-plotly-plot .plotly, .js-plotly-plot .main-svg {{
  border-radius: 10px !important;
}}
</style>
""",
    unsafe_allow_html=True,
)


# =========================
# Helpers
# =========================
@st.cache_data(show_spinner=False)
def _load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner=False)
def _load_csv(path: str) -> pd.DataFrame:
    sample = pd.read_csv(path, nrows=1)
    parse_dates = ["Date"] if "Date" in sample.columns else None
    return pd.read_csv(path, parse_dates=parse_dates)


def _ensure_files_exist(paths: list[str]) -> tuple[bool, list[str]]:
    missing = [p for p in paths if not os.path.exists(p)]
    return (len(missing) == 0, missing)


def _guess_reg_targets(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns if c.startswith("time_to_") and f"{c}_pred" in df.columns
    ]


def _guess_clf_targets(df: pd.DataFrame) -> list[str]:
    return [
        c
        for c in df.columns
        if c.startswith("roi_") and f"{c}_proba_pred" in df.columns
    ]


def _has_pi(df: pd.DataFrame, tgt: str) -> tuple[bool, str, str]:
    low = f"{tgt}_pred_low_5"
    high = f"{tgt}_pred_high_95"
    return (low in df.columns and high in df.columns, low, high)


def _rmse_gauge(rmse_months: float) -> str:
    if rmse_months <= 5:
        return "🟢 High"
    if rmse_months <= 10:
        return "🟡 Moderate"
    return "🔴 Low"


def _auc_gauge(auc: float) -> str:
    if np.isnan(auc):
        return "⚪ N/A"
    if auc >= 0.8:
        return "🟢 Strong"
    if auc >= 0.65:
        return "🟡 Moderate"
    return "🔴 Weak"


def _friendly_reg_name(col: str) -> str:
    m = re.search(r"time_to_(\d+)pct", col)
    if m:
        pct = int(m.group(1))
        return f"Time to {pct}% ROI"
    return col


def _friendly_clf_name(col: str) -> str:
    m = re.search(r"roi_(\d+)pct_within_(\d+)", col)
    if m:
        pct = int(m.group(1))
        horizon = int(m.group(2))
        return f"{pct}% ROI within {horizon} months"
    return col


def _line_fig(x, y_true, y_pred, title, y_band=None, template="plotly_white"):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=x, y=y_true, name="Observed", mode="lines", line=dict(width=2))
    )
    fig.add_trace(
        go.Scatter(x=x, y=y_pred, name="Predicted", mode="lines", line=dict(width=2))
    )
    if y_band is not None:
        y_low, y_high = y_band
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([x, x[::-1]]),
                y=np.concatenate([y_high, y_low[::-1]]),
                fill="toself",
                name="5–95% PI",
                line=dict(width=0),
                opacity=0.15,
                hoverinfo="skip",
            )
        )
        # Hover lines for band
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y_high,
                name="95% bound",
                mode="lines",
                line=dict(width=0.5, color="rgba(0,0,0,0)"),
                hovertemplate="Upper PI (95%): %{y:.2f}<extra></extra>",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y_low,
                name="5% bound",
                mode="lines",
                line=dict(width=0.5, color="rgba(0,0,0,0)"),
                hovertemplate="Lower PI (5%): %{y:.2f}<extra></extra>",
                showlegend=False,
            )
        )
    # y padding
    vals = [y_true, y_pred] + ([y_low, y_high] if y_band else [])
    ymin = np.nanmin([v.min() for v in vals])
    ymax = np.nanmax([v.max() for v in vals])
    pad = (ymax - ymin) * 0.08 if ymax > ymin else 1.0
    fig.update_yaxes(
        range=[ymin - pad, ymax + pad],
        zeroline=False,
        showgrid=True,
        gridcolor="rgba(0,0,0,0.06)",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_layout(
        title=title,
        height=430,
        template=template,
        legend=dict(orientation="h"),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def _scatter_fig(y_true, y_pred, title, template="plotly_white"):
    fig = px.scatter(x=y_true, y=y_pred, labels={"x": "Observed", "y": "Predicted"})
    lims = [min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))]
    fig.add_trace(
        go.Scatter(x=lims, y=lims, mode="lines", name="y = x", line=dict(dash="dash"))
    )
    fig.update_layout(
        title=title, height=400, template=template, margin=dict(l=10, r=10, t=60, b=10)
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.06)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.06)")
    return fig


def _prob_ts_fig(
    df_dates: pd.Series, proba: np.ndarray, title, template="plotly_white"
):
    month = pd.to_datetime(df_dates).dt.to_period("M").dt.to_timestamp()
    dfp = (
        pd.DataFrame({"Month": month, "proba": proba})
        .groupby("Month", as_index=False)["proba"]
        .mean()
    )
    x, y = dfp["Month"], dfp["proba"].to_numpy(float)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Monthly mean"))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    pad = max(0.02, (ymax - ymin) * 0.08)
    fig.update_yaxes(
        range=[max(0.0, ymin - pad), min(1.0, ymax + pad)],
        title="Probability",
        tickformat=".0%",
    )
    fig.update_xaxes(title="Month")
    fig.update_layout(
        title=title,
        height=430,
        template=template,
        legend=dict(orientation="h"),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


PLOTLY_TEMPLATE = "plotly_white" if mode_vibe == "Minimal Light" else "plotly_dark"

# =========================
# Header
# =========================
st.markdown(
    """
<div style="text-align:center; margin-top:-1rem; margin-bottom:0.5rem;">
  <h1 style="font-weight:600; letter-spacing:-0.5px; margin-bottom:0.3rem;">
    🪙 Gold ROI Outlook
  </h1>
  <p style="color:#666; font-size:1.05rem; margin:0;">
    Time-to-ROI & Probability of ROI within a horizon — based on historical patterns
  </p>
</div>
""",
    unsafe_allow_html=True,
)
# Explanatory block (Beginner / Technical)
mode = st.radio("Explain mode", ["Beginner", "Technical"], horizontal=True)

if mode == "Beginner":
    st.markdown("""
---
### 📌 What this dashboard shows

We estimate **how long** it might take to reach a chosen **Return on Investment (ROI)** in gold based on historical price patterns, and the **probability** of hitting that ROI within a defined timeframe.

**Two complementary perspectives:**

1) **⏱ Time-to-ROI (Regression)** — *How many months until +X%?*  
   We show:
   - the predicted waiting time (in months)
   - and, when available, a shaded **uncertainty interval** (5–95%).

2) **🎯 ROI Within a Horizon (Classification)** — *What’s the chance of reaching +X% ROI within Y months?*  
   We show:
   - a probability between **0% and 100%**, which reflects how likely that outcome is.

---

### 📐 RMSE (Root Mean Squared Error), in plain words

RMSE tells you how **accurate the time predictions** are — measured directly in **months**.

- RMSE = 3 → Predictions are typically off by **around 3 months**.  
- **Lower RMSE = better accuracy.**  
Because RMSE penalizes large mistakes more, it responds strongly when the market shifts abruptly.

> You can think of RMSE as the **uncertainty in the waiting time** estimate.

---

### 🎯 AUC (Area Under the ROC Curve), in plain words

AUC tells you how well the **probability model separates cases where ROI was reached vs. not reached**.

- AUC = **1.0** → Perfect distinction (ideal, rarely achievable).
- AUC = **0.5** → No skill at all (like flipping a coin).
- AUC between **0.65–0.80** → **Useful but not perfect**.
- AUC above **0.80** → **Strong predictive separation**.

> You can think of AUC as **how confident the model is when deciding whether ROI is likely**.

---

""")

else:
    st.markdown("""
---
### Model overview (technical)

- **Regression (time-to-ROI):** Predicts the expected number of months to reach a given ROI level.  
  Optional 5–95% prediction intervals indicate uncertainty.
- **Classification (ROI within horizon):** Estimates the probability of achieving a given ROI within a specified time period.
- **Evaluation:** Models are trained on past data and tested on the most recent period (chronological split).

---

### RMSE (regression metric)
Reported in **months**. Indicates the typical error in predicted time-to-ROI.  
Lower RMSE → more accurate timing estimates.

---

### AUC (classification metric)
Measures how well the model separates cases where ROI was achieved vs. not achieved within the horizon.  
Higher AUC → stronger discrimination.

---
""")

# =========================
# Load artifacts
# =========================
ok, missing = _ensure_files_exist(
    [REG_RESULTS_PATH, REG_TEST_PATH, CLF_RESULTS_PATH, CLF_TEST_PATH]
)
if not ok:
    st.error("Missing required files:\n\n" + "\n".join(f"- {m}" for m in missing))
    st.stop()

reg_results = _load_pickle(REG_RESULTS_PATH)
reg_test = _load_csv(REG_TEST_PATH)
clf_results = _load_pickle(CLF_RESULTS_PATH)
clf_test = _load_csv(CLF_TEST_PATH)

# =========================
# Target selectors
# =========================
reg_targets = _guess_reg_targets(reg_test)
clf_targets = _guess_clf_targets(clf_test)
reg_display = {_friendly_reg_name(t): t for t in reg_targets}
clf_display = {_friendly_clf_name(t): t for t in clf_targets}

sel1, sel2 = st.columns(2)
with sel1:
    reg_label = st.selectbox(
        "⏱️ Time-to-ROI (regression)",
        list(reg_display.keys()),
        index=0 if reg_targets else None,
    )
    reg_target = reg_display.get(reg_label)
with sel2:
    clf_label = st.selectbox(
        "🎯 ROI within horizon (probability)",
        list(clf_display.keys()),
        index=0 if clf_targets else None,
    )
    clf_target = clf_display.get(clf_label)

st.divider()

# =========================
# Panels
# =========================
left, right = st.columns(2)

# ----- Left: Regression -----
with left:
    st.subheader("⏱ Time-to-ROI")

    if not reg_targets:
        st.warning("No regression targets with predictions found in the test file.")
    else:
        y_true = reg_test[reg_target].to_numpy(float)
        y_pred = reg_test[f"{reg_target}_pred"].to_numpy(float)
        stored_metrics = (reg_results.get(reg_target, {}) or {}).get("metrics", {})
        rmse = stored_metrics.get("RMSE")
        if rmse is None or (isinstance(rmse, float) and np.isnan(rmse)):
            rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

        # KPI row in cards
        colA, colB, colC = st.columns([1, 2, 1])
        with colA:
            st.markdown(
                f"<div class='card'><div class='card-title'>RMSE (months)</div><div class='card-value'>{rmse:.0f}</div></div>",
                unsafe_allow_html=True,
            )
        with colB:
            st.markdown(
                f"<div class='card'><div class='card-title'>Model confidence</div><div class='card-value'>{_rmse_gauge(rmse)}</div></div>",
                unsafe_allow_html=True,
            )
        with colC:
            latest_date = (
                reg_test["Date"].iloc[-1].strftime("%Y-%m")
                if "Date" in reg_test.columns
                else "Latest"
            )
            latest_pred = float(reg_test[f"{reg_target}_pred"].iloc[-1])
            has_band, low_col, high_col = _has_pi(reg_test, reg_target)
            if has_band:
                latest_low = float(reg_test[low_col].iloc[-1])
                latest_high = float(reg_test[high_col].iloc[-1])
                extra = f"<div class='card-subtle'>5–95%: {latest_low:.0f}–{latest_high:.0f} mo</div>"
            else:
                extra = ""
            st.markdown(
                f"<div class='card'><div class='card-title'>From {latest_date}</div><div class='card-value'>{latest_pred:.0f} months</div>{extra}</div>",
                unsafe_allow_html=True,
            )

        # Chart
        x = reg_test["Date"] if "Date" in reg_test.columns else np.arange(len(reg_test))
        has_band, low_col, high_col = _has_pi(reg_test, reg_target)
        band = None
        if has_band:
            y_low = reg_test[low_col].to_numpy(float)
            y_high = reg_test[high_col].to_numpy(float)
            band = (y_low, y_high)

        fig = _line_fig(
            x,
            y_true,
            y_pred,
            title=f"{reg_label} — observed vs predicted",
            y_band=band,
            template=PLOTLY_TEMPLATE,
        )
        st.plotly_chart(fig, use_container_width=True)

# ----- Right: Classification -----
with right:
    st.subheader("🎯 ROI within horizon")

    if not clf_targets:
        st.warning("No classification targets with probability predictions found.")
    else:
        proba_col = f"{clf_target}_proba_pred"
        y_true_clf = clf_test[clf_target].astype(int).to_numpy()
        proba = clf_test[proba_col].to_numpy(float)
        stored_metrics = (clf_results.get(clf_target, {}) or {}).get("metrics", {})
        auc = stored_metrics.get("AUC")
        if auc is None:
            # only compute if balanced classes; otherwise leave as None/NaN
            auc = np.nan

        # KPI row in cards
        colA, colB, colC = st.columns([1, 2, 1])
        with colA:
            val = "N/A" if auc is None or np.isnan(auc) else f"{auc:.3f}"
            st.markdown(
                f"<div class='card'><div class='card-title'>AUC</div><div class='card-value'>{val}</div></div>",
                unsafe_allow_html=True,
            )
        with colB:
            st.markdown(
                f"<div class='card'><div class='card-title'>Model separation</div><div class='card-value'>{_auc_gauge(auc if auc is not None else np.nan)}</div></div>",
                unsafe_allow_html=True,
            )
        with colC:
            latest_date = (
                clf_test["Date"].iloc[-1].strftime("%Y-%m")
                if "Date" in clf_test.columns
                else "Latest"
            )
            latest_proba = float(proba[-1]) * 100.0
            st.markdown(
                f"<div class='card'><div class='card-title'>From {latest_date}</div><div class='card-value'>{latest_proba:.1f}%</div><div class='card-subtle'>P(ROI within horizon)</div></div>",
                unsafe_allow_html=True,
            )

        # Probabilities over time (monthly mean)
        if "Date" in clf_test.columns:
            pfig = _prob_ts_fig(
                clf_test["Date"],
                proba,
                title=f"{clf_label} — predicted probability by month",
                template=PLOTLY_TEMPLATE,
            )
            st.plotly_chart(pfig, use_container_width=True)

# Footer
st.divider()
st.markdown(
    f"<div style='color:{SUBTLE};font-size:0.85rem;'>Built with Streamlit · Plotly</div>",
    unsafe_allow_html=True,
)
