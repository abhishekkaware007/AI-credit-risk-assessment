import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CreditIQ — Loan Risk Dashboard",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.stApp {
    background-color: #07090f;
    color: #e2e8f0;
}
#MainMenu, footer { visibility: hidden; }
.block-container {
    padding: 3.5rem 2.5rem 2rem 2.5rem;
    max-width: 1400px;
}
[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #161d27;
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stNumberInput label {
    color: #475569 !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}
[data-testid="stSlider"] > div > div > div {
    background: #1e3a5f !important;
}
[data-testid="stSlider"] > div > div > div > div {
    background: #3b82f6 !important;
}
[data-testid="metric-container"] {
    background: #0d1117;
    border: 1px solid #161d27;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
}
[data-testid="metric-container"] label {
    color: #475569 !important;
    font-size: 0.7rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.09em !important;
    font-weight: 600 !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #f1f5f9 !important;
    font-size: 1.5rem !important;
    font-weight: 600 !important;
    font-family: 'DM Mono', monospace !important;
}
[data-testid="stMetricDelta"] {
    font-size: 0.72rem !important;
}
.section-label {
    font-size: 0.82rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #e2e8f0;
    margin: 1.2rem 0 0.6rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e2a3a;
}
.panel {
    background: #0d1117;
    border: 1px solid #161d27;
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    height: 100%;
}
.panel-title {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #e2e8f0;
    margin-bottom: 0.8rem;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid #1e2a3a;
}
.risk-minimal  { background:rgba(22,163,74,0.12);  border:1px solid rgba(22,163,74,0.4);  color:#4ade80; }
.risk-low      { background:rgba(101,163,13,0.12); border:1px solid rgba(101,163,13,0.4); color:#a3e635; }
.risk-moderate { background:rgba(217,119,6,0.12);  border:1px solid rgba(217,119,6,0.4);  color:#fbbf24; }
.risk-high     { background:rgba(234,88,12,0.12);  border:1px solid rgba(234,88,12,0.4);  color:#fb923c; }
.risk-critical { background:rgba(220,38,38,0.12);  border:1px solid rgba(220,38,38,0.4);  color:#f87171; }
.risk-badge {
    display: inline-block;
    padding: 0.3rem 1rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.risk-title-minimal  { color: #4ade80; }
.risk-title-low      { color: #a3e635; }
.risk-title-moderate { color: #fbbf24; }
.risk-title-high     { color: #fb923c; }
.risk-title-critical { color: #f87171; }
.verdict-card {
    border-radius: 16px;
    padding: 1.6rem 2rem;
    margin: 1rem 0;
}
.verdict-minimal  { background:rgba(22,163,74,0.07);  border:1px solid rgba(22,163,74,0.2);  border-left:5px solid #22c55e; }
.verdict-low      { background:rgba(101,163,13,0.07); border:1px solid rgba(101,163,13,0.2); border-left:5px solid #84cc16; }
.verdict-moderate { background:rgba(217,119,6,0.07);  border:1px solid rgba(217,119,6,0.2);  border-left:5px solid #f59e0b; }
.verdict-high     { background:rgba(234,88,12,0.07);  border:1px solid rgba(234,88,12,0.2);  border-left:5px solid #f97316; }
.verdict-critical { background:rgba(220,38,38,0.07);  border:1px solid rgba(220,38,38,0.2);  border-left:5px solid #ef4444; }
.verdict-heading {
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 0.4rem;
}
.verdict-body {
    font-size: 0.88rem;
    color: #94a3b8;
    line-height: 1.7;
}
.verdict-action {
    margin-top: 0.8rem;
    font-size: 0.82rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #64748b;
}
.factor-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.6rem 0;
    border-bottom: 1px solid #0f1520;
}
.factor-name { font-size:0.82rem; color:#64748b; }
.factor-val  { font-size:0.82rem; font-family:'DM Mono',monospace; color:#e2e8f0; }
.factor-ok   { font-size:0.72rem; background:rgba(34,197,94,0.1);  border:1px solid rgba(34,197,94,0.3);  color:#4ade80;  padding:0.12rem 0.5rem; border-radius:999px; }
.factor-warn { font-size:0.72rem; background:rgba(251,191,36,0.1); border:1px solid rgba(251,191,36,0.3); color:#fbbf24;  padding:0.12rem 0.5rem; border-radius:999px; }
.factor-bad  { font-size:0.72rem; background:rgba(239,68,68,0.1);  border:1px solid rgba(239,68,68,0.3);  color:#f87171;  padding:0.12rem 0.5rem; border-radius:999px; }
.prog-wrap {
    background: #161d27;
    border-radius: 999px;
    height: 5px;
    width: 100%;
    overflow: hidden;
    margin-top: 4px;
}
.prog-fill {
    height: 100%;
    border-radius: 999px;
}
.brand     { font-size:1.3rem; font-weight:700; color:#3b82f6 !important; letter-spacing:-0.02em; }
.brand-tag { font-size:0.68rem; color:#334155 !important; text-transform:uppercase; letter-spacing:0.1em; }

/* ── Indicator row for key financials ── */
.indicator-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.55rem 0.8rem;
    border-radius: 8px;
    margin-bottom: 0.3rem;
    background: #0a0e1a;
    border: 1px solid #161d27;
}
.indicator-label {
    font-size: 0.8rem;
    color: #64748b;
}
.indicator-value {
    font-size: 0.82rem;
    font-family: 'DM Mono', monospace;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        base   = os.path.dirname(os.path.abspath(__file__))
        mpath  = os.path.join(base, '..', 'models')
        model  = joblib.load(os.path.join(mpath, 'random_forest.pkl'))
        scaler = joblib.load(os.path.join(mpath, 'scaler.pkl'))
        return model, scaler, mpath, None
    except Exception as e:
        return None, None, None, str(e)

model, scaler, mpath, err = load_model()
if err:
    st.error(f"Could not load model: {err}")
    st.stop()

# ─────────────────────────────────────────────
# HISTORY FILE PATH
# ─────────────────────────────────────────────
HISTORY_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'assessment_history.csv'
)

# ─────────────────────────────────────────────
# 5 RISK CATEGORIES
# ─────────────────────────────────────────────
def get_risk(prob_pct):
    if prob_pct < 20:
        return {
            'level':   'MINIMAL',
            'css':     'minimal',
            'emoji':   '🟢',
            'color':   '#22c55e',
            'decision':'APPROVE',
            'heading': '✓ Recommend Full Approval',
            'body':    f"This applicant presents minimal default risk at {prob_pct:.1f}%. Financial indicators are strong — stable income, low debt obligations, and a clean payment history. Standard loan terms are appropriate.",
            'action':  '→ Proceed with standard loan processing'
        }
    elif prob_pct < 40:
        return {
            'level':   'LOW',
            'css':     'low',
            'emoji':   '🟡',
            'color':   '#84cc16',
            'decision':'APPROVE',
            'heading': '✓ Recommend Approval with Standard Review',
            'body':    f"This applicant carries low default risk at {prob_pct:.1f}%. Most financial indicators are within acceptable range. A standard documentation review is sufficient before disbursement.",
            'action':  '→ Approve — verify income documents before disbursement'
        }
    elif prob_pct < 60:
        return {
            'level':   'MODERATE',
            'css':     'moderate',
            'emoji':   '🟠',
            'color':   '#f59e0b',
            'decision':'REVIEW',
            'heading': '⚠ Requires Enhanced Review',
            'body':    f"This applicant shows moderate default risk at {prob_pct:.1f}%. Some financial indicators are outside optimal range. Consider requesting additional collateral, a co-applicant, or reducing the loan amount before approval.",
            'action':  '→ Escalate to senior credit officer for manual review'
        }
    elif prob_pct < 80:
        return {
            'level':   'HIGH',
            'css':     'high',
            'emoji':   '🔴',
            'color':   '#f97316',
            'decision':'REJECT',
            'heading': '✗ Recommend Rejection',
            'body':    f"This applicant presents high default risk at {prob_pct:.1f}%. Multiple financial indicators suggest difficulty in maintaining repayments. Approval at this risk level is not recommended under standard bank policy.",
            'action':  '→ Reject application — advise applicant on credit improvement'
        }
    else:
        return {
            'level':   'CRITICAL',
            'css':     'critical',
            'emoji':   '🚨',
            'color':   '#ef4444',
            'decision':'REJECT',
            'heading': '✗ Critical Risk — Immediate Rejection',
            'body':    f"This applicant presents critical default risk at {prob_pct:.1f}%. Severe indicators across multiple financial dimensions. Loan disbursement would represent an unacceptable credit risk. Do not proceed under any standard lending criteria.",
            'action':  '→ Reject immediately — flag for credit counselling referral'
        }

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="brand">💎 CreditIQ</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-tag">Loan Risk Intelligence</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div class="section-label">Personal Information</div>', unsafe_allow_html=True)
    age        = st.slider("Age", 18, 80, 35)
    dependents = st.slider("Number of Dependents", 0, 10, 1)

    st.markdown('<div class="section-label">Financial Information</div>', unsafe_allow_html=True)
    income = st.number_input(
        "Monthly Income ($)",
        min_value=0, max_value=100000,
        value=5000, step=500
    )
    debt_ratio     = st.slider("Debt Ratio", 0.0, 5.0, 0.35, step=0.01)
    revolving_util = st.slider("Credit Card Utilization", 0.0, 1.0, 0.30, step=0.01)

    st.markdown('<div class="section-label">Loan Information</div>', unsafe_allow_html=True)
    open_loans  = st.slider("Open Credit Lines", 0, 30, 5)
    real_estate = st.slider("Real Estate Loans", 0, 10, 1)

    st.markdown('<div class="section-label">Payment History</div>', unsafe_allow_html=True)
    late_30_59 = st.slider("Times 30–59 Days Late", 0, 10, 0)
    late_60_89 = st.slider("Times 60–89 Days Late", 0, 10, 0)
    late_90    = st.slider("Times 90+ Days Late",   0, 10, 0)

    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.68rem;color:#1e293b;text-align:center">'
        'Abhishek Kaware · VIT Pune · 2025'
        '</div>',
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────
# COMPUTE PREDICTION
# ─────────────────────────────────────────────
total_late = late_30_59 + late_60_89 + late_90

input_df = pd.DataFrame([{
    'RevolvingUtilizationOfUnsecuredLines': revolving_util,
    'age':                                   age,
    'NumberOfTime30-59DaysPastDueNotWorse': late_30_59,
    'DebtRatio':                             debt_ratio,
    'MonthlyIncome':                         income,
    'NumberOfOpenCreditLinesAndLoans':       open_loans,
    'NumberOfTimes90DaysLate':               late_90,
    'NumberRealEstateLoansOrLines':          real_estate,
    'NumberOfTime60-89DaysPastDueNotWorse':  late_60_89,
    'NumberOfDependents':                    dependents,
    'TotalLatePayments':                     total_late,
    'IncomePerDependent':                    income / (dependents + 1),
}])

X_input      = input_df.values
prob         = model.predict_proba(X_input)[0][1]
prob_pct     = round(prob * 100, 1)
risk         = get_risk(prob_pct)
monthly_free = max(income - (income * debt_ratio), 0)
income_dep   = income / (dependents + 1)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
h1, h2 = st.columns([5, 1])
with h1:
    st.markdown(
        '<div style="font-size:1.5rem;font-weight:600;color:#f1f5f9;letter-spacing:-0.02em">'
        'Loan Application Risk Assessment'
        '</div>'
        '<div style="font-size:0.82rem;color:#94a3b8;margin-top:0.2rem">'
        'AI-driven credit risk analysis · Adjust applicant details in the sidebar'
        '</div>',
        unsafe_allow_html=True
    )
with h2:
    st.markdown(
        f'<div style="text-align:right;padding-top:0.6rem">'
        f'<span class="risk-badge risk-{risk["css"]}">'
        f'{risk["emoji"]} {risk["level"]}'
        f'</span></div>',
        unsafe_allow_html=True
    )

st.markdown("---")

# ─────────────────────────────────────────────
# SECTION 1 — 5 KPI CARDS
# ─────────────────────────────────────────────
st.markdown('<div class="section-label">Applicant Overview</div>', unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Default Probability", f"{prob_pct}%")
with c2:
    st.metric("Risk Category", risk['level'])
with c3:
    st.metric("Decision", risk['decision'])
with c4:
    st.metric("Free Monthly Cash", f"${monthly_free:,.0f}")
with c5:
    st.metric("Income / Dependent", f"${income_dep:,.0f}")

st.markdown("")

# ─────────────────────────────────────────────
# SECTION 2 — VERDICT
# ─────────────────────────────────────────────
st.markdown('<div class="section-label">Risk Verdict</div>', unsafe_allow_html=True)

st.markdown(f"""
<div class="verdict-card verdict-{risk['css']}">
    <div class="verdict-heading risk-title-{risk['css']}">{risk['heading']}</div>
    <div class="verdict-body">{risk['body']}</div>
    <div class="verdict-action">{risk['action']}</div>
</div>
""", unsafe_allow_html=True)

st.markdown("")

# ─────────────────────────────────────────────
# SECTION 3 — RISK SCALE
# ─────────────────────────────────────────────
st.markdown('<div class="section-label">Risk Scale Position</div>', unsafe_allow_html=True)

levels = [
    ('MINIMAL',  '0–20%',   'minimal',  '#22c55e'),
    ('LOW',      '20–40%',  'low',      '#84cc16'),
    ('MODERATE', '40–60%',  'moderate', '#f59e0b'),
    ('HIGH',     '60–80%',  'high',     '#f97316'),
    ('CRITICAL', '80–100%', 'critical', '#ef4444'),
]

scale_cols = st.columns(5)
for col, (lvl, rng, css, color) in zip(scale_cols, levels):
    is_active = risk['level'] == lvl
    r, g, b   = int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)
    border    = f"border:2px solid {color};" if is_active else "border:1px solid #161d27;"
    bg        = f"background:rgba({r},{g},{b},0.15);" if is_active else "background:#0d1117;"
    txt_size  = "font-size:0.95rem;font-weight:700;" if is_active else "font-size:0.82rem;font-weight:500;opacity:0.45;"
    you_here  = f'<div style="font-size:0.62rem;color:{color};margin-top:0.3rem;font-weight:700">▲ HERE</div>' if is_active else ''

    col.markdown(f"""
    <div style="border-radius:12px;padding:0.9rem;text-align:center;{border}{bg}">
        <div style="color:{color};{txt_size}">{lvl}</div>
        <div style="font-size:0.68rem;color:#64748b;margin-top:0.2rem">{rng}</div>
        {you_here}
    </div>
    """, unsafe_allow_html=True)

st.markdown("")

# ─────────────────────────────────────────────
# SECTION 4 — GAUGE + FACTORS + RADAR
# ─────────────────────────────────────────────
st.markdown('<div class="section-label">Detailed Analysis</div>', unsafe_allow_html=True)

col_g, col_f, col_r = st.columns([1.1, 1, 1])

with col_g:
    st.markdown('<div class="panel"><div class="panel-title">Default Probability Meter</div>', unsafe_allow_html=True)

    r, g, b = int(risk['color'][1:3],16), int(risk['color'][3:5],16), int(risk['color'][5:7],16)

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob_pct,
        number={
            'suffix': '%',
            'font': {'size': 40, 'color': risk['color'], 'family': 'DM Mono'}
        },
        gauge={
            'axis': {
                'range':    [0, 100],
                'tickvals': [0, 20, 40, 60, 80, 100],
                'ticktext': ['0', '20', '40', '60', '80', '100'],
                'tickfont': {'color': '#64748b', 'size': 10},
                'tickwidth': 0
            },
            'bar':         {'color': risk['color'], 'thickness': 0.22},
            'bgcolor':     'rgba(0,0,0,0)',
            'borderwidth': 0,
            'steps': [
                {'range': [0,  20], 'color': 'rgba(34,197,94,0.12)'},
                {'range': [20, 40], 'color': 'rgba(132,204,22,0.12)'},
                {'range': [40, 60], 'color': 'rgba(245,158,11,0.12)'},
                {'range': [60, 80], 'color': 'rgba(249,115,22,0.12)'},
                {'range': [80,100], 'color': 'rgba(239,68,68,0.12)'},
            ],
            'threshold': {
                'line':      {'color': '#475569', 'width': 2},
                'thickness': 0.75,
                'value':     prob_pct
            }
        }
    ))
    fig_gauge.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=250,
        margin=dict(l=20, r=20, t=30, b=0),
        font={'color': '#94a3b8'}
    )
    st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})

    st.markdown(f"""
    <div style="text-align:center;margin-top:-1rem;padding-bottom:0.5rem">
        <span class="risk-badge risk-{risk['css']}">{risk['emoji']} {risk['level']} RISK</span>
    </div>
    </div>
    """, unsafe_allow_html=True)

with col_f:
    st.markdown('<div class="panel"><div class="panel-title">Risk Factor Breakdown</div>', unsafe_allow_html=True)

    def make_factor(label, val_str, status_cls, status_txt, prog_pct, prog_color):
        return f"""
        <div class="factor-row">
            <div style="flex:1">
                <div class="factor-name">{label}</div>
                <div class="prog-wrap">
                    <div class="prog-fill" style="width:{min(prog_pct,100)}%;background:{prog_color}"></div>
                </div>
            </div>
            <div style="margin-left:1rem;text-align:right">
                <div class="factor-val">{val_str}</div>
                <div style="margin-top:3px"><span class="{status_cls}">{status_txt}</span></div>
            </div>
        </div>"""

    fh = ""
    fh += make_factor("Age",            f"{age} yrs",                  "factor-bad"  if age<25             else "factor-ok",                                          "High Risk"  if age<25          else "Good",                              max(0,(30-age)*3)           if age<30      else 5,   "#ef4444" if age<25             else "#22c55e")
    fh += make_factor("Monthly Income", f"${income:,}",                "factor-bad"  if income<2000         else "factor-warn" if income<4000 else "factor-ok",        "Low"        if income<2000     else "Moderate" if income<4000 else "Good", max(0,min(100,(6000-income)/60)),                       "#ef4444" if income<2000         else "#22c55e")
    fh += make_factor("Debt Ratio",     f"{debt_ratio:.2f}",           "factor-bad"  if debt_ratio>1        else "factor-warn" if debt_ratio>0.4 else "factor-ok",     "Dangerous"  if debt_ratio>1    else "Elevated"  if debt_ratio>0.4 else "Normal",  min(100,debt_ratio*20),    "#ef4444" if debt_ratio>1        else "#f59e0b" if debt_ratio>0.4 else "#22c55e")
    fh += make_factor("Credit Usage",   f"{revolving_util*100:.0f}%",  "factor-bad"  if revolving_util>0.8  else "factor-warn" if revolving_util>0.5 else "factor-ok", "Maxed Out"  if revolving_util>0.8 else "High" if revolving_util>0.5 else "Normal",  revolving_util*100,        "#ef4444" if revolving_util>0.8  else "#f59e0b" if revolving_util>0.5 else "#22c55e")
    fh += make_factor("Late Payments",  f"{total_late}x",              "factor-bad"  if total_late>5        else "factor-warn" if total_late>0 else "factor-ok",       "Critical"   if total_late>5    else "Warning"   if total_late>0 else "Clean",       min(100,total_late*15),    "#ef4444" if total_late>0        else "#22c55e")
    fh += make_factor("90+ Days Late",  f"{late_90}x",                 "factor-bad"  if late_90>2           else "factor-warn" if late_90>0 else "factor-ok",          "Severe"     if late_90>2       else "Concerning" if late_90>0 else "None",         min(100,late_90*20),       "#ef4444" if late_90>0           else "#22c55e")

    st.markdown(fh + "</div>", unsafe_allow_html=True)

with col_r:
    st.markdown('<div class="panel"><div class="panel-title">Risk Profile Radar</div>', unsafe_allow_html=True)

    cats = ['Age', 'Income', 'Debt', 'Credit\nUsage', 'Late\nPayments', 'Loan\nBurden']
    vals = [
        max(0, min(100, (30-age)*3))       if age<30      else 5,
        max(0, min(100, (5000-income)/50)) if income<5000  else 5,
        min(100, debt_ratio*20),
        revolving_util*100,
        min(100, total_late*18),
        min(100, open_loans*5),
    ]

    rc = risk['color']
    rr, rg, rb = int(rc[1:3],16), int(rc[3:5],16), int(rc[5:7],16)

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=vals+[vals[0]],
        theta=cats+[cats[0]],
        fill='toself',
        fillcolor=f"rgba({rr},{rg},{rb},0.18)",
        line=dict(color=rc, width=2),
    ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(
                visible=True, range=[0,100],
                tickfont=dict(color='#475569', size=8),
                gridcolor='#1e2a3a', linecolor='#1e2a3a'
            ),
            angularaxis=dict(
                tickfont=dict(color='#94a3b8', size=10),
                gridcolor='#1e2a3a', linecolor='#1e2a3a'
            )
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=290,
        margin=dict(l=30, r=30, t=20, b=10),
        showlegend=False
    )
    st.plotly_chart(fig_radar, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("")

# ─────────────────────────────────────────────
# SECTION 5 — SHAP + KEY INDICATORS (FIXED)
# ─────────────────────────────────────────────
st.markdown('<div class="section-label">Why This Prediction Was Made</div>', unsafe_allow_html=True)

col_shap, col_stats = st.columns([1.6, 1])

with col_shap:

    # ── Visible title ──────────────────────────
    st.markdown("""
    <div style="font-size:0.78rem;font-weight:700;text-transform:uppercase;
                letter-spacing:0.1em;color:#e2e8f0;margin-bottom:0.4rem">
        Feature Impact
    </div>
    <div style="font-size:0.78rem;color:#64748b;margin-bottom:1rem">
        Which features pushed the risk
        <span style="color:#f87171;font-weight:600">higher ↑</span>
        or
        <span style="color:#60a5fa;font-weight:600">lower ↓</span>
        for this specific applicant
    </div>
    """, unsafe_allow_html=True)

    try:
        with st.spinner("Computing impact..."):
            explainer  = shap.TreeExplainer(model)
            shap_vals  = explainer.shap_values(input_df)

            # Larger figure so all features are visible
            fig2, ax = plt.subplots(figsize=(9, 6))
            fig2.patch.set_facecolor('#0d1117')
            ax.set_facecolor('#0d1117')

            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_vals[0][0],
                    base_values=explainer.expected_value[0],
                    data=input_df.iloc[0],
                    feature_names=input_df.columns.tolist()
                ),
                show=False,
                max_display=12
            )

            # Style all text visible
            ax.tick_params(colors='#94a3b8', labelsize=10)
            for text in ax.texts:
                text.set_color('#e2e8f0')
                text.set_fontsize(10)
            for spine in ax.spines.values():
                spine.set_edgecolor('#1e2a3a')

            plt.rcParams['text.color']      = '#94a3b8'
            plt.rcParams['axes.labelcolor'] = '#94a3b8'
            plt.rcParams['xtick.color']     = '#64748b'
            plt.rcParams['ytick.color']     = '#94a3b8'
            plt.tight_layout(pad=2.0)
            st.pyplot(fig2, transparent=True)
            plt.clf()
            plt.close()

    except Exception as e:
        st.info(f"Feature impact unavailable: {e}")

with col_stats:

    # ── Visible title ──────────────────────────
    st.markdown("""
    <div style="font-size:0.78rem;font-weight:700;text-transform:uppercase;
                letter-spacing:0.1em;color:#e2e8f0;margin-bottom:1rem">
        Key Financial Indicators
    </div>
    """, unsafe_allow_html=True)

    indicators = [
        ("Age",               f"{age} yrs",                 "#f1f5f9"),
        ("Monthly Income",    f"${income:,}",               "#f1f5f9"),
        ("Debt Ratio",        f"{debt_ratio:.2f}",          "#f87171" if debt_ratio>0.5     else "#4ade80"),
        ("Credit Usage",      f"{revolving_util*100:.0f}%", "#f87171" if revolving_util>0.6 else "#4ade80"),
        ("Total Late Pmts",   f"{total_late}x",             "#f87171" if total_late>0       else "#4ade80"),
        ("30–59 Days Late",   f"{late_30_59}x",             "#fbbf24" if late_30_59>0       else "#4ade80"),
        ("60–89 Days Late",   f"{late_60_89}x",             "#f87171" if late_60_89>0       else "#4ade80"),
        ("90+ Days Late",     f"{late_90}x",                "#ef4444" if late_90>0          else "#4ade80"),
        ("Open Loans",        f"{open_loans}",              "#f1f5f9"),
        ("Real Estate Loans", f"{real_estate}",             "#f1f5f9"),
        ("Dependents",        f"{dependents}",              "#f1f5f9"),
        ("Free Monthly Cash", f"${monthly_free:,.0f}",      "#4ade80" if monthly_free>2000  else "#f87171"),
    ]

    rows = ""
    for label, val, color in indicators:
        rows += f"""
        <div class="indicator-row">
            <span class="indicator-label">{label}</span>
            <span class="indicator-value" style="color:{color}">{val}</span>
        </div>"""
    st.markdown(rows, unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────
# SECTION 6 — WHAT-IF ANALYSIS
# ─────────────────────────────────────────────
st.markdown('<div class="section-label">What-If Analysis — How to Reduce Risk</div>', unsafe_allow_html=True)

if prob_pct >= 40:
    st.markdown(
        '<div style="font-size:0.82rem;color:#64748b;margin-bottom:1rem">'
        'Showing what specific changes would reduce this applicant\'s default probability. Sorted by biggest impact first.'
        '</div>',
        unsafe_allow_html=True
    )

    suggestions = []

    if late_90 > 0:
        test_df = input_df.copy()
        test_df['NumberOfTimes90DaysLate'] = 0
        test_df['TotalLatePayments']       = late_30_59 + late_60_89 + 0
        new_prob = model.predict_proba(test_df.values)[0][1] * 100
        drop     = prob_pct - new_prob
        new_risk = get_risk(new_prob)
        suggestions.append({
            'Improvement':    'Clear all 90+ day late payments',
            'Current Value':  f"{late_90}x late",
            'Target Value':   '0x late',
            'New Risk %':     f"{new_prob:.1f}%",
            'New Category':   new_risk['level'],
            'Risk Reduction': f"↓ {drop:.1f}%",
            'Impact':         'High' if drop > 15 else 'Medium',
            '_drop':          drop
        })

    if debt_ratio > 0.5:
        test_df = input_df.copy()
        test_df['DebtRatio'] = 0.30
        new_prob = model.predict_proba(test_df.values)[0][1] * 100
        drop     = prob_pct - new_prob
        new_risk = get_risk(new_prob)
        suggestions.append({
            'Improvement':    'Reduce debt ratio to 0.30',
            'Current Value':  f"{debt_ratio:.2f}",
            'Target Value':   '0.30',
            'New Risk %':     f"{new_prob:.1f}%",
            'New Category':   new_risk['level'],
            'Risk Reduction': f"↓ {drop:.1f}%",
            'Impact':         'High' if drop > 15 else 'Medium',
            '_drop':          drop
        })

    if revolving_util > 0.5:
        test_df = input_df.copy()
        test_df['RevolvingUtilizationOfUnsecuredLines'] = 0.20
        new_prob = model.predict_proba(test_df.values)[0][1] * 100
        drop     = prob_pct - new_prob
        new_risk = get_risk(new_prob)
        suggestions.append({
            'Improvement':    'Reduce credit card usage to 20%',
            'Current Value':  f"{revolving_util*100:.0f}%",
            'Target Value':   '20%',
            'New Risk %':     f"{new_prob:.1f}%",
            'New Category':   new_risk['level'],
            'Risk Reduction': f"↓ {drop:.1f}%",
            'Impact':         'High' if drop > 15 else 'Medium',
            '_drop':          drop
        })

    if income < 4000:
        test_df = input_df.copy()
        new_income = income * 1.5
        test_df['MonthlyIncome']      = new_income
        test_df['IncomePerDependent'] = new_income / (dependents + 1)
        new_prob = model.predict_proba(test_df.values)[0][1] * 100
        drop     = prob_pct - new_prob
        new_risk = get_risk(new_prob)
        suggestions.append({
            'Improvement':    'Increase monthly income by 50%',
            'Current Value':  f"${income:,}",
            'Target Value':   f"${new_income:,.0f}",
            'New Risk %':     f"{new_prob:.1f}%",
            'New Category':   new_risk['level'],
            'Risk Reduction': f"↓ {drop:.1f}%",
            'Impact':         'High' if drop > 15 else 'Medium',
            '_drop':          drop
        })

    if late_30_59 > 2:
        test_df = input_df.copy()
        test_df['NumberOfTime30-59DaysPastDueNotWorse'] = 0
        test_df['TotalLatePayments'] = 0 + late_60_89 + late_90
        new_prob = model.predict_proba(test_df.values)[0][1] * 100
        drop     = prob_pct - new_prob
        new_risk = get_risk(new_prob)
        suggestions.append({
            'Improvement':    'Clear 30–59 day late payments',
            'Current Value':  f"{late_30_59}x late",
            'Target Value':   '0x late',
            'New Risk %':     f"{new_prob:.1f}%",
            'New Category':   new_risk['level'],
            'Risk Reduction': f"↓ {drop:.1f}%",
            'Impact':         'High' if drop > 15 else 'Medium',
            '_drop':          drop
        })

    if suggestions:
        suggestions   = sorted(suggestions, key=lambda x: x['_drop'], reverse=True)
        display_sug   = [{k: v for k, v in s.items() if k != '_drop'} for s in suggestions]
        st.dataframe(pd.DataFrame(display_sug), hide_index=True, use_container_width=True)

        best          = suggestions[0]
        best_new_risk = get_risk(float(best['New Risk %'].replace('%','')))
        st.markdown(f"""
        <div style="background:rgba(59,130,246,0.08);border:1px solid rgba(59,130,246,0.2);
                    border-left:4px solid #3b82f6;border-radius:12px;
                    padding:1.2rem 1.6rem;margin-top:0.8rem">
            <div style="font-size:0.68rem;font-weight:700;text-transform:uppercase;
                        letter-spacing:0.12em;color:#3b82f6;margin-bottom:0.5rem">
                Top Recommendation
            </div>
            <div style="font-size:0.95rem;color:#e2e8f0;font-weight:600;margin-bottom:0.4rem">
                {best['Improvement']}
            </div>
            <div style="font-size:0.83rem;color:#64748b;line-height:1.6">
                This single change would reduce the default probability from
                <span style="color:#f87171;font-weight:600">{prob_pct}%</span>
                to
                <span style="color:#4ade80;font-weight:600">{best['New Risk %']}</span>
                — moving the applicant from
                <span style="color:{risk['color']};font-weight:600">{risk['level']}</span>
                to
                <span style="color:{best_new_risk['color']};font-weight:600">{best_new_risk['level']}</span>
                risk category.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No specific improvements identified for this profile.")
else:
    st.markdown(f"""
    <div style="background:rgba(34,197,94,0.07);border:1px solid rgba(34,197,94,0.2);
                border-radius:12px;padding:1.2rem 1.6rem">
        <div style="color:#4ade80;font-size:0.92rem;font-weight:600;margin-bottom:0.3rem">
            ✓ Risk is already low at {prob_pct}%
        </div>
        <div style="color:#64748b;font-size:0.83rem">
            No improvements needed. This applicant meets all standard lending criteria.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────
# SECTION 7 — ASSESSMENT HISTORY
# ─────────────────────────────────────────────
st.markdown('<div class="section-label">Assessment History</div>', unsafe_allow_html=True)

def save_assessment():
    new_row = {
        'Timestamp':       datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'Age':             age,
        'Monthly Income':  f"${income:,}",
        'Debt Ratio':      debt_ratio,
        'Credit Usage':    f"{revolving_util*100:.0f}%",
        'Late Payments':   total_late,
        '90+ Days Late':   late_90,
        'Default %':       prob_pct,
        'Risk Level':      risk['level'],
        'Decision':        risk['decision'],
    }
    if os.path.exists(HISTORY_PATH):
        hist_df = pd.read_csv(HISTORY_PATH)
        hist_df = pd.concat(
            [hist_df, pd.DataFrame([new_row])],
            ignore_index=True
        )
    else:
        hist_df = pd.DataFrame([new_row])
    hist_df.to_csv(HISTORY_PATH, index=False)
    return len(hist_df)

btn_col, toggle_col = st.columns([1, 2])

with btn_col:
    if st.button("💾  Save This Assessment", use_container_width=True):
        count = save_assessment()
        st.success(f"✅ Saved! Total assessments: {count}")

with toggle_col:
    show_history = st.checkbox("📋  View All Past Assessments")

if show_history:
    if os.path.exists(HISTORY_PATH):
        hist_df = pd.read_csv(HISTORY_PATH)

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Total Assessed", len(hist_df))
        s2.metric("Approved",       len(hist_df[hist_df['Decision'] == 'APPROVE']))
        s3.metric("Rejected",       len(hist_df[hist_df['Decision'] == 'REJECT']))
        s4.metric("Under Review",   len(hist_df[hist_df['Decision'] == 'REVIEW']))

        st.markdown("")
        st.dataframe(hist_df, hide_index=True, use_container_width=True)
        st.markdown("")

        dl_col, clear_col = st.columns([1, 1])
        with dl_col:
            st.download_button(
                label="📥  Download History as CSV",
                data=hist_df.to_csv(index=False),
                file_name="assessment_history.csv",
                mime="text/csv",
                use_container_width=True
            )
        with clear_col:
            if st.button("🗑  Clear All History", use_container_width=True):
                os.remove(HISTORY_PATH)
                st.warning("History cleared.")
                st.rerun()
    else:
        st.markdown("""
        <div style="background:#0d1117;border:1px solid #161d27;border-radius:12px;
                    padding:1.2rem 1.6rem;color:#64748b;font-size:0.83rem">
            No assessments saved yet. Click
            <strong style="color:#94a3b8">Save This Assessment</strong>
            above to start building history.
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────
# SECTION 8 — BATCH PREDICTION
# ─────────────────────────────────────────────
st.markdown('<div class="section-label">Batch Prediction — Assess Multiple Applicants at Once</div>', unsafe_allow_html=True)

st.markdown(
    '<div style="font-size:0.82rem;color:#64748b;margin-bottom:1rem">'
    'Upload a CSV file containing multiple applicants. '
    'The system will predict default risk for all of them instantly '
    'and generate a downloadable results report.'
    '</div>',
    unsafe_allow_html=True
)

with st.expander("📋  Required CSV columns — click to expand"):
    st.markdown("Your CSV must contain these columns (names must match exactly):")
    st.code("""RevolvingUtilizationOfUnsecuredLines
age
NumberOfTime30-59DaysPastDueNotWorse
DebtRatio
MonthlyIncome
NumberOfOpenCreditLinesAndLoans
NumberOfTimes90DaysLate
NumberRealEstateLoansOrLines
NumberOfTime60-89DaysPastDueNotWorse
NumberOfDependents""")
    st.caption("TotalLatePayments and IncomePerDependent will be calculated automatically.")

    sample_data = pd.DataFrame([
        [0.10, 52, 0, 0.15, 8000, 6, 0, 1, 0, 2],
        [0.35, 38, 1, 0.30, 5500, 5, 0, 1, 0, 1],
        [0.62, 29, 2, 0.55, 3200, 8, 0, 0, 1, 2],
        [0.82, 26, 3, 0.90, 2000, 10, 2, 0, 2, 3],
        [0.96, 23, 5, 1.80, 1200, 14, 5, 0, 4, 4],
    ], columns=[
        'RevolvingUtilizationOfUnsecuredLines','age',
        'NumberOfTime30-59DaysPastDueNotWorse','DebtRatio',
        'MonthlyIncome','NumberOfOpenCreditLinesAndLoans',
        'NumberOfTimes90DaysLate','NumberRealEstateLoansOrLines',
        'NumberOfTime60-89DaysPastDueNotWorse','NumberOfDependents'
    ])
    st.download_button(
        label="📥  Download Sample CSV Template",
        data=sample_data.to_csv(index=False),
        file_name="sample_applicants.csv",
        mime="text/csv"
    )

uploaded_file = st.file_uploader(
    "Upload your CSV file here",
    type=['csv'],
    label_visibility="collapsed"
)

if uploaded_file is not None:
    try:
        batch_df = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded {len(batch_df)} applicants from file")

        if 'MonthlyIncome' in batch_df.columns:
            batch_df['MonthlyIncome'].fillna(batch_df['MonthlyIncome'].median(), inplace=True)
        if 'NumberOfDependents' in batch_df.columns:
            batch_df['NumberOfDependents'].fillna(0, inplace=True)

        if 'TotalLatePayments' not in batch_df.columns:
            batch_df['TotalLatePayments'] = (
                batch_df['NumberOfTime30-59DaysPastDueNotWorse'] +
                batch_df['NumberOfTime60-89DaysPastDueNotWorse'] +
                batch_df['NumberOfTimes90DaysLate']
            )
        if 'IncomePerDependent' not in batch_df.columns:
            batch_df['IncomePerDependent'] = (
                batch_df['MonthlyIncome'] / (batch_df['NumberOfDependents'] + 1)
            )

        feature_cols = [
            'RevolvingUtilizationOfUnsecuredLines', 'age',
            'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio',
            'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans',
            'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines',
            'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents',
            'TotalLatePayments', 'IncomePerDependent'
        ]

        X_batch = batch_df[feature_cols].values
        probs   = model.predict_proba(X_batch)[:, 1] * 100

        def batch_risk(p):
            if p < 20:   return 'MINIMAL 🟢'
            elif p < 40: return 'LOW 🟡'
            elif p < 60: return 'MODERATE 🟠'
            elif p < 80: return 'HIGH 🔴'
            else:        return 'CRITICAL 🚨'

        def batch_decision(p):
            if p < 40:   return '✅ APPROVE'
            elif p < 60: return '⚠ REVIEW'
            else:        return '❌ REJECT'

        results_df = pd.DataFrame({
            'Applicant #':         range(1, len(batch_df)+1),
            'Age':                 batch_df['age'].values,
            'Monthly Income':      batch_df['MonthlyIncome'].apply(lambda x: f"${x:,.0f}"),
            'Default Probability': [f"{p:.1f}%" for p in probs],
            'Risk Category':       [batch_risk(p)     for p in probs],
            'Decision':            [batch_decision(p) for p in probs],
        })

        st.markdown("")

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Total",    len(probs))
        m2.metric("Minimal",  sum(p < 20       for p in probs))
        m3.metric("Low",      sum(20 <= p < 40 for p in probs))
        m4.metric("Moderate", sum(40 <= p < 60 for p in probs))
        m5.metric("High",     sum(60 <= p < 80 for p in probs))
        m6.metric("Critical", sum(p >= 80      for p in probs))

        st.markdown("")
        st.dataframe(results_df, hide_index=True, use_container_width=True)
        st.markdown("")

        risk_counts = {
            'MINIMAL':  sum(p < 20       for p in probs),
            'LOW':      sum(20 <= p < 40 for p in probs),
            'MODERATE': sum(40 <= p < 60 for p in probs),
            'HIGH':     sum(60 <= p < 80 for p in probs),
            'CRITICAL': sum(p >= 80      for p in probs),
        }

        fig_dist = go.Figure(go.Bar(
            x=list(risk_counts.keys()),
            y=list(risk_counts.values()),
            marker_color=['#22c55e','#84cc16','#f59e0b','#f97316','#ef4444'],
            text=list(risk_counts.values()),
            textposition='outside',
            textfont=dict(color='#94a3b8', size=12)
        ))
        fig_dist.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=300,
            margin=dict(l=10, r=10, t=40, b=10),
            xaxis=dict(
                tickfont=dict(color='#94a3b8', size=12),
                gridcolor='rgba(0,0,0,0)'
            ),
            yaxis=dict(
                tickfont=dict(color='#475569', size=11),
                gridcolor='#1e2a3a'
            ),
            title=dict(
                text='Risk Distribution of Uploaded Applicants',
                font=dict(color='#94a3b8', size=13),
                x=0.5
            )
        )
        st.plotly_chart(fig_dist, use_container_width=True, config={'displayModeBar': False})

        st.download_button(
            label="📥  Download All Results as CSV",
            data=results_df.to_csv(index=False),
            file_name="batch_risk_results.csv",
            mime="text/csv",
            use_container_width=True
        )

    except KeyError as e:
        st.error(f"Missing column in your CSV: {e}")
        st.info("Make sure all required columns are present and spelled correctly.")
    except Exception as e:
        st.error(f"Error processing file: {e}")

st.markdown("---")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;color:#1e293b;font-size:0.75rem;padding:0.5rem'>
    CreditIQ · Abhishek Kaware · VIT Pune · 2025
</div>
""", unsafe_allow_html=True)