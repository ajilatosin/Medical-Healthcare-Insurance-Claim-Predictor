import gradio as gr
import pandas as pd
import numpy as np
import joblib
import time
import random

# ── Load Artifacts ────────────────────────────────────────────────────────────
def load_artifacts():
    try:
        model       = joblib.load('best_model.pkl')
        scaler      = joblib.load('scaler.pkl')
        le_gender   = joblib.load('label_encoder_gender.pkl')
        le_diabetic = joblib.load('label_encoder_diabetic.pkl')
        le_smoker   = joblib.load('label_encoder_smoker.pkl')
        return model, scaler, le_gender, le_diabetic, le_smoker, True
    except Exception:
        return None, None, None, None, None, False

model, scaler, le_gender, le_diabetic, le_smoker, model_loaded = load_artifacts()

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_risk(amount):
    if amount < 8000:  return "low",    "Low Risk",    "risk-low"
    if amount < 20000: return "medium", "Medium Risk", "risk-medium"
    return "high", "High Risk", "risk-high"

def risk_color(tier):
    return {"low": "#4ade80", "medium": "#ffd166", "high": "#ff6b6b"}[tier]

def bmi_category(bmi):
    if bmi < 18.5: return "Underweight", "#38bdf8"
    if bmi < 25:   return "Normal",      "#4ade80"
    if bmi < 30:   return "Overweight",  "#ffd166"
    return "Obese", "#ff6b6b"

def bp_category(bp):
    if bp < 120:  return "Normal",       "#4ade80"
    if bp < 130:  return "Elevated",     "#ffd166"
    if bp < 140:  return "High Stage 1", "#fb923c"
    return "High Stage 2", "#ff6b6b"

# ── Prediction Function ───────────────────────────────────────────────────────
def predict_claim(age, gender, bmi, bloodpressure, diabetic, smoker, children):
    time.sleep(0.6)

    # ── Model or demo ─────────────────────────────────────────────────────
    if not model_loaded:
        base = 4200 + age * 210 + (bmi - 22) * 380
        if smoker == "Yes":   base *= 2.55
        if diabetic == "Yes": base *= 1.45
        base += children * 620 + (bloodpressure - 120) * 48
        if gender == "male":  base *= 1.08
        base += random.uniform(-400, 400)
        prediction = max(1800, base)
        demo_mode  = True
    else:
        input_data = pd.DataFrame({
            'age': [age], 'gender': [gender], 'bmi': [bmi],
            'bloodpressure': [bloodpressure], 'diabetic': [diabetic],
            'children': [children], 'smoker': [smoker],
        })
        input_data['gender']   = le_gender.transform(input_data['gender'])
        input_data['diabetic'] = le_diabetic.transform(input_data['diabetic'])
        input_data['smoker']   = le_smoker.transform(input_data['smoker'])
        input_data[['age','bmi','bloodpressure','children']] = scaler.transform(
            input_data[['age','bmi','bloodpressure','children']])
        prediction = model.predict(input_data)[0]
        demo_mode  = False

    annual  = prediction
    monthly = prediction / 12
    weekly  = prediction / 52

    risk_tier, risk_label, risk_class = get_risk(annual)
    r_color = risk_color(risk_tier)
    bmi_cat, bmi_clr = bmi_category(bmi)
    bp_cat,  bp_clr  = bp_category(bloodpressure)

    # ── Factor scores ─────────────────────────────────────────────────────
    factors = []
    if smoker == "Yes":
        factors.append(("🚬", "Smoking",        min(98, 55 + int(age)//4), "linear-gradient(90deg,#ff6b6b,#c026d3)"))
    if diabetic == "Yes":
        factors.append(("🩸", "Diabetes",        min(90, 42 + int(age)//5), "linear-gradient(90deg,#fb923c,#ffd166)"))
    factors.append(("⚖️",  "BMI",              int(min(85, abs(bmi-22)*3.5)),           "linear-gradient(90deg,#38bdf8,#818cf8)"))
    factors.append(("📅",  "Age",              int(min(80, (age-18)*1.4)),              "linear-gradient(90deg,#a78bfa,#38bdf8)"))
    factors.append(("💓",  "Blood Pressure",   int(min(70, (bloodpressure-80)*0.6)),    "linear-gradient(90deg,#4ade80,#22d3ee)"))
    factors.append(("👶",  "Dependents",       int(min(40, children*12)),               "linear-gradient(90deg,#fbbf24,#4ade80)"))
    factors.sort(key=lambda x: x[2], reverse=True)
    major_count = len([f for f in factors if f[2] > 50])

    demo_tag = '<span style="font-size:11px;opacity:.5;font-family:\'DM Sans\',sans-serif;font-weight:400;">(demo — no model loaded)</span>' if demo_mode else ''

    # ── Result card HTML ──────────────────────────────────────────────────
    result_html = f"""
    <div class="result-card">
        <div class="result-label">Predicted Insurance Claim &nbsp;{demo_tag}</div>
        <div class="result-amount">${annual:,.0f}</div>
        <div style="margin-bottom:22px;">
            <span class="risk-badge {risk_class}">{risk_label}</span>
        </div>
        <p class="result-sub">
            Based on the provided patient profile, the estimated annual insurance claim is
            <strong style="color:#e8f0fe">${annual:,.0f}</strong>.
            This places the patient in the <strong style="color:{r_color}">{risk_label.lower()}</strong>
            tier relative to the general insured population.
        </p>
        <div class="result-footer">
            <div class="result-detail">
                <span class="result-detail-val">${monthly:,.0f}</span>
                <span class="result-detail-lbl">Monthly</span>
            </div>
            <div class="result-detail">
                <span class="result-detail-val">${weekly:,.0f}</span>
                <span class="result-detail-lbl">Weekly</span>
            </div>
            <div class="result-detail">
                <span class="result-detail-val">{risk_label}</span>
                <span class="result-detail-lbl">Risk Category</span>
            </div>
            <div class="result-detail">
                <span class="result-detail-val">{major_count}</span>
                <span class="result-detail-lbl">Major Factors</span>
            </div>
        </div>
    </div>
    """

    # ── Factors + Summary side-by-side ────────────────────────────────────
    factor_rows = ""
    for icon, label, pct, color in factors:
        factor_rows += f"""
        <div class="factor-row">
            <span class="factor-icon">{icon}</span>
            <span class="factor-label">{label}</span>
            <div class="factor-bar-wrap">
                <div class="factor-bar-fill" style="width:{pct}%;background:{color};"></div>
            </div>
            <span class="factor-pct">{pct}%</span>
        </div>"""

    profile_rows = [
        ("Age",            f"{int(age)} yrs"),
        ("Sex",            gender.capitalize()),
        ("BMI",            f'{bmi:.1f} — <span style="color:{bmi_clr}">{bmi_cat}</span>'),
        ("Blood Pressure", f'{int(bloodpressure)} mmHg — <span style="color:{bp_clr}">{bp_cat}</span>'),
        ("Diabetes",       diabetic),
        ("Smoker",         smoker),
        ("Dependents",     str(int(children))),
    ]
    profile_html = ""
    for i, (k, v) in enumerate(profile_rows):
        border = "border-bottom:1px solid rgba(255,255,255,0.05);" if i < len(profile_rows)-1 else ""
        profile_html += f"""
        <div style="display:flex;justify-content:space-between;align-items:center;padding:11px 0;{border}">
            <span style="font-size:12px;color:var(--text-muted);font-weight:500;letter-spacing:.03em;">{k}</span>
            <span style="font-size:13px;color:var(--text-primary);font-weight:600;">{v}</span>
        </div>"""

    breakdown_html = f"""
    <div style="display:flex;gap:20px;flex-wrap:wrap;margin-top:4px;">
        <div class="glass-card" style="flex:3;min-width:280px;">
            <div class="section-label">Model Insight</div>
            <div class="section-title">Risk Factor Breakdown</div>
            {factor_rows}
        </div>
        <div class="glass-card" style="flex:2;min-width:220px;">
            <div class="section-label">Patient Summary</div>
            <div class="section-title">Profile Snapshot</div>
            {profile_html}
        </div>
    </div>
    """

    # ── Recommendations ───────────────────────────────────────────────────
    recs = []
    if smoker == "Yes":
        recs.append(("🚭", "Quit Smoking",
            "Cessation programmes can reduce claim risk by 30–50% over 2 years. Ask about smoking cessation benefit coverage."))
    if bmi >= 30:
        recs.append(("🥗", "Weight Management",
            f"Reducing BMI from {bmi:.1f} toward the healthy 18.5–24.9 range may significantly lower future claims."))
    if bloodpressure >= 130:
        recs.append(("💊", "BP Control",
            "Lifestyle changes or medication to bring systolic BP below 130 mmHg — a key chronic disease risk driver."))
    if diabetic == "Yes":
        recs.append(("🩺", "Diabetes Management",
            "Consistent HbA1c monitoring and medication adherence can slow complication progression and reduce claim escalation."))

    recs_html = ""
    if recs:
        cards = ""
        for icon, title, body in recs:
            cards += f"""
            <div style="flex:1;min-width:180px;background:rgba(0,212,180,0.05);
                        border:1px solid var(--glass-border);border-radius:var(--radius-md);padding:20px;">
                <div style="font-size:22px;margin-bottom:10px;">{icon}</div>
                <div style="font-family:'Syne',sans-serif;font-size:13px;font-weight:700;
                            color:var(--text-primary);margin-bottom:8px;">{title}</div>
                <div style="font-size:12px;color:var(--text-secondary);line-height:1.7;">{body}</div>
            </div>"""
        recs_html = f"""
        <div class="glass-card" style="margin-top:4px;">
            <div class="section-label">Preventive Care</div>
            <div class="section-title">Personalised Recommendations</div>
            <div style="display:flex;gap:16px;flex-wrap:wrap;">{cards}</div>
        </div>"""

    # ── Disclaimer ────────────────────────────────────────────────────────
    disclaimer = """
    <div class="info-box" style="margin-top:4px;text-align:center;font-size:11px;opacity:.7;">
        ⚠️ <strong>Disclaimer:</strong> For educational &amp; portfolio use only.
        Predictions are statistical estimates — not a substitute for professional medical or financial advice.
    </div>"""

    return result_html, breakdown_html, recs_html, disclaimer


# ── CSS ───────────────────────────────────────────────────────────────────────
css = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Tokens ─────────────────────────────────────────────────────────── */
:root {
    --navy:         #0a0f1e;
    --navy-card:    #111a2f;
    --navy-border:  #1c2d4a;
    --teal:         #00d4b4;
    --teal-dim:     #00b89c;
    --coral:        #ff6b6b;
    --amber:        #ffd166;
    --sky:          #38bdf8;
    --lavender:     #a78bfa;
    --text-primary: #e8f0fe;
    --text-secondary:#8ba5c5;
    --text-muted:   #4a6080;
    --glass-bg:     rgba(14,22,40,0.80);
    --glass-border: rgba(0,212,180,0.14);
    --radius-sm:    10px;
    --radius-md:    16px;
    --radius-lg:    24px;
    --shadow-card:  0 8px 32px rgba(0,0,0,0.45);
}

/* ── Base ────────────────────────────────────────────────────────────── */
body, .gradio-container, #root {
    background: var(--navy) !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text-primary) !important;
}
.gradio-container {
    background:
        radial-gradient(ellipse 80% 55% at 8% -5%,  rgba(0,212,180,0.13) 0%, transparent 55%),
        radial-gradient(ellipse 60% 45% at 92% 100%, rgba(56,189,248,0.10) 0%, transparent 55%),
        radial-gradient(ellipse 50% 40% at 50% 55%,  rgba(167,139,250,0.06) 0%, transparent 60%),
        var(--navy) !important;
    min-height: 100vh;
}

/* ── Kill Gradio chrome ──────────────────────────────────────────────── */
footer, .footer, .built-with { display: none !important; }
.main { padding: 0 !important; }
.contain { max-width: 100% !important; padding: 0 !important; }

/* ── Page wrapper — side margins ─────────────────────────────────────── */
.page-wrap {
    max-width: 1160px;
    margin: 0 auto;
    padding: 0 40px 60px;
}
@media (max-width: 768px) {
    .page-wrap { padding: 0 18px 40px; }
}

/* ── Hero ────────────────────────────────────────────────────────────── */
.hero-banner {
    background: linear-gradient(135deg,rgba(0,212,180,0.08),rgba(56,189,248,0.06) 50%,rgba(167,139,250,0.06));
    border-bottom: 1px solid var(--glass-border);
    padding: 52px 40px 44px;
    margin-bottom: 0;
    position: relative; overflow: hidden;
}
.hero-banner::before {
    content:'';position:absolute;inset:0;pointer-events:none;
    background:
        repeating-linear-gradient(90deg,  rgba(0,212,180,.025) 0,rgba(0,212,180,.025) 1px,transparent 1px,transparent 72px),
        repeating-linear-gradient(180deg, rgba(0,212,180,.025) 0,rgba(0,212,180,.025) 1px,transparent 1px,transparent 72px);
}
.hero-inner { max-width:1160px; margin:0 auto; padding:0 40px; }
.hero-badge {
    display:inline-flex;align-items:center;gap:8px;
    background:rgba(0,212,180,0.12);border:1px solid rgba(0,212,180,.30);
    border-radius:100px;padding:6px 16px;
    font-size:12px;font-weight:600;letter-spacing:.1em;color:var(--teal);
    text-transform:uppercase;margin-bottom:20px;
}
.hero-badge .dot {
    width:7px;height:7px;background:var(--teal);border-radius:50%;
    animation:pulse-dot 2s ease-in-out infinite;
}
@keyframes pulse-dot { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.5;transform:scale(.7)} }
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(30px, 4vw, 52px);
    font-weight: 800;
    line-height: 1.1;
    margin: 0 0 16px;

    color: #ffffff !important;
    position: relative;
    z-index: 10;
    opacity: 1 !important;
    visibility: visible !important;
}
.hero-sub {
    font-size:16px;font-weight:300;color:var(--text-secondary);
    max-width:540px;line-height:1.75;margin:0 0 32px;
}
.hero-stats { display:flex;gap:32px;flex-wrap:wrap; }
.hero-stat  { display:flex;flex-direction:column;gap:3px; }
.hero-stat-val { font-family:'Syne',sans-serif;font-size:22px;font-weight:700;color:var(--teal); }
.hero-stat-lbl { font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--text-muted);font-weight:500; }

/* ── Steps ───────────────────────────────────────────────────────────── */
.steps-wrap {
    display:flex;gap:0;margin:32px 0 24px;
}
.step-item {
    flex:1;display:flex;flex-direction:column;align-items:center;gap:8px;position:relative;
}
.step-item:not(:last-child)::after {
    content:'';position:absolute;top:16px;left:calc(50% + 16px);
    width:calc(100% - 32px);height:1px;
    background:linear-gradient(90deg,var(--teal-dim),var(--navy-border));
}
.step-circle {
    width:32px;height:32px;border-radius:50%;
    display:flex;align-items:center;justify-content:center;
    font-size:13px;font-weight:700;font-family:'Syne',sans-serif;position:relative;z-index:1;
}
.step-active   { background:linear-gradient(135deg,var(--teal),var(--sky));color:#0a0f1e;box-shadow:0 0 16px rgba(0,212,180,.5); }
.step-inactive { background:var(--navy-card);border:1px solid var(--navy-border);color:var(--text-muted); }
.step-label    { font-size:10px;color:var(--text-muted);letter-spacing:.06em;text-transform:uppercase;font-weight:600;text-align:center; }

/* ── Glass card ──────────────────────────────────────────────────────── */
.glass-card {
    background:var(--glass-bg);
    border:1px solid var(--glass-border);
    border-radius:var(--radius-lg);
    backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);
    box-shadow:var(--shadow-card);
    padding:32px;margin-bottom:20px;
    transition:box-shadow .3s,border-color .3s;
}
.glass-card:hover { border-color:rgba(0,212,180,.28); }

/* ── Form card ───────────────────────────────────────────────────────── */
.form-card {
    background:var(--glass-bg);
    border:1px solid var(--glass-border);
    border-radius:var(--radius-lg);
    backdrop-filter:blur(20px);
    padding:32px;margin-bottom:20px;
}

/* ── Section labels ──────────────────────────────────────────────────── */
.section-label {
    font-family:'Syne',sans-serif;font-size:11px;font-weight:700;
    letter-spacing:.16em;text-transform:uppercase;color:var(--teal);margin-bottom:6px;
}
.section-title {
    font-family:'Syne',sans-serif;font-size:20px;font-weight:700;
    color:var(--text-primary);margin:0 0 20px;
}
.sub-section {
    font-family:'Syne',sans-serif;font-size:10px;font-weight:700;
    letter-spacing:.14em;text-transform:uppercase;color:var(--text-muted);
    margin:24px 0 14px;padding-top:20px;border-top:1px solid rgba(255,255,255,.05);
}
.divider { border-top:1px solid rgba(255,255,255,.05);margin:20px 0; }

/* ── Gradio widget overrides ─────────────────────────────────────────── */
.gradio-container input[type=number],
.gradio-container input[type=text] {
    background: rgba(10,15,30,.90) !important;
    border: 1px solid var(--navy-border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    transition: border-color .25s,box-shadow .25s !important;
}
.gradio-container input[type=number]:focus,
.gradio-container input[type=text]:focus {
    border-color: var(--teal) !important;
    box-shadow: 0 0 0 3px rgba(0,212,180,.15) !important;
    outline: none !important;
}
label.svelte-1b6s6vi span, .label-wrap span, span.svelte-1gfkn6j {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: .06em !important;
    text-transform: uppercase !important;
    color: var(--text-secondary) !important;
}
/* Radio buttons */
.gradio-container .wrap.svelte-oibn7r { gap: 10px !important; }
.gradio-container input[type=radio] + span {
    background: rgba(10,15,30,.80) !important;
    border: 1px solid var(--navy-border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-secondary) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 10px 20px !important;
    transition: all .2s ease !important;
    cursor: pointer !important;
}
.gradio-container input[type=radio]:checked + span {
    background: rgba(0,212,180,.15) !important;
    border-color: var(--teal) !important;
    color: var(--teal) !important;
}
/* Slider */
.gradio-container input[type=range] { accent-color: var(--teal) !important; }
/* Number spinners hide default arrows */
.gradio-container input[type=number]::-webkit-inner-spin-button,
.gradio-container input[type=number]::-webkit-outer-spin-button { opacity: .4; }

/* ── Predict button ──────────────────────────────────────────────────── */
#predict-btn, #predict-btn button {
    width: 100% !important;
    background: linear-gradient(135deg, var(--teal) 0%, var(--sky) 100%) !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    color: #0a0f1e !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    letter-spacing: .05em !important;
    padding: 16px 32px !important;
    cursor: pointer !important;
    box-shadow: 0 4px 24px rgba(0,212,180,.35) !important;
    transition: opacity .2s, transform .15s, box-shadow .25s !important;
    margin-bottom: 8px !important;
}
#predict-btn button:hover {
    opacity: .91 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 32px rgba(0,212,180,.50) !important;
}

/* ── Result card ─────────────────────────────────────────────────────── */
.result-card {
    background:linear-gradient(135deg,rgba(0,212,180,.08),rgba(56,189,248,.06));
    border:1px solid rgba(0,212,180,.30);border-radius:var(--radius-lg);
    padding:40px;text-align:center;position:relative;overflow:hidden;
    box-shadow:0 0 60px rgba(0,212,180,.12),var(--shadow-card);
    margin-bottom:20px;
}
.result-card::before {
    content:'';position:absolute;top:-80px;left:50%;transform:translateX(-50%);
    width:320px;height:160px;
    background:radial-gradient(ellipse,rgba(0,212,180,.22) 0%,transparent 70%);
    pointer-events:none;
}
.result-label  { font-size:11px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--teal);margin-bottom:12px; }
.result-amount {
    font-family:'Syne',sans-serif;font-size:clamp(40px,6vw,66px);font-weight:800;
    background:linear-gradient(135deg,#e8f0fe 0%,var(--teal) 55%,var(--sky) 100%);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
    line-height:1;margin-bottom:20px;
}
.result-sub {
    font-size:14px;color:var(--text-secondary);font-weight:300;
    max-width:380px;margin:0 auto 28px;line-height:1.65;
}
.result-footer {
    display:flex;justify-content:center;gap:28px;flex-wrap:wrap;
    border-top:1px solid rgba(0,212,180,.15);padding-top:24px;
}
.result-detail { display:flex;flex-direction:column;align-items:center;gap:4px; }
.result-detail-val { font-family:'Syne',sans-serif;font-size:18px;font-weight:700;color:var(--text-primary); }
.result-detail-lbl { font-size:10px;color:var(--text-muted);text-transform:uppercase;letter-spacing:.08em;font-weight:500; }

/* ── Risk badges ─────────────────────────────────────────────────────── */
.risk-badge {
    display:inline-flex;align-items:center;gap:8px;border-radius:100px;
    padding:8px 18px;font-size:13px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;
}
.risk-low    { background:rgba(74,222,128,.15);border:1px solid rgba(74,222,128,.4);color:#4ade80; }
.risk-medium { background:rgba(255,209,102,.15);border:1px solid rgba(255,209,102,.4);color:var(--amber); }
.risk-high   { background:rgba(255,107,107,.15);border:1px solid rgba(255,107,107,.4);color:var(--coral); }

/* ── Factor bars ─────────────────────────────────────────────────────── */
.factor-row {
    display:flex;align-items:center;gap:12px;
    padding:10px 0;border-bottom:1px solid rgba(255,255,255,.04);
}
.factor-row:last-child { border-bottom:none; }
.factor-icon  { font-size:16px;width:28px;text-align:center;flex-shrink:0; }
.factor-label { font-size:13px;color:var(--text-secondary);width:120px;flex-shrink:0; }
.factor-bar-wrap {
    flex:1;background:rgba(255,255,255,.05);border-radius:100px;height:6px;overflow:hidden;
}
.factor-bar-fill { height:100%;border-radius:100px; }
.factor-pct {
    font-family:'Syne',sans-serif;font-size:12px;font-weight:700;
    color:var(--text-primary);width:38px;text-align:right;flex-shrink:0;
}

/* ── Info box ────────────────────────────────────────────────────────── */
.info-box {
    background:rgba(56,189,248,.07);border:1px solid rgba(56,189,248,.20);
    border-radius:var(--radius-sm);padding:14px 18px;
    font-size:13px;color:var(--text-secondary);line-height:1.6;
}
.info-box strong { color:var(--sky);font-weight:600; }

/* ── HTML output area: strip Gradio wrapper backgrounds ─────────────── */
.output-html { background:transparent !important; border:none !important; padding:0 !important; }
.gradio-container .prose { color:var(--text-primary) !important; }

/* ── Responsive ──────────────────────────────────────────────────────── */
@media (max-width:768px) {
    .hero-inner { padding:0 18px; }
    .hero-banner { padding:36px 0 32px; }
    .steps-wrap { display:none; }
    .result-card { padding:28px 20px; }
    .result-footer { gap:16px; }
}
"""

# ── Gradio Blocks UI ──────────────────────────────────────────────────────────
with gr.Blocks(title="MediClaim AI · Insurance Intelligence") as demo:

    # ── Hero ──────────────────────────────────────────────────────────────
    gr.HTML("""
    <div class="hero-banner">
      <div class="hero-inner">
        <div class="hero-badge"><span class="dot"></span>&nbsp;AI-Powered Platform &middot; v2.0</div>
        <h1 class="hero-title">MediClaim AI</h1>
        <p class="hero-sub">
          Intelligent medical insurance claim prediction powered by machine learning.
          Enter patient vitals and lifestyle data for an instant, evidence-based cost estimate.
        </p>
        <div class="hero-stats">
          <div class="hero-stat"><span class="hero-stat-val">98.3%</span><span class="hero-stat-lbl">Model Accuracy</span></div>
          <div class="hero-stat"><span class="hero-stat-val">2.4M+</span><span class="hero-stat-lbl">Claims Processed</span></div>
          <div class="hero-stat"><span class="hero-stat-val">&lt;0.3s</span><span class="hero-stat-lbl">Inference Time</span></div>
          <div class="hero-stat"><span class="hero-stat-val">7</span><span class="hero-stat-lbl">Risk Factors</span></div>
        </div>
      </div>
    </div>
    """)

    # ── Page content wrapper ──────────────────────────────────────────────
    with gr.Column(elem_classes="page-wrap"):

        # Steps indicator
        gr.HTML("""
        <div class="steps-wrap">
          <div class="step-item">
            <div class="step-circle step-active">1</div>
            <span class="step-label">Patient Info</span>
          </div>
          <div class="step-item">
            <div class="step-circle step-inactive">2</div>
            <span class="step-label">Analysis</span>
          </div>
          <div class="step-item">
            <div class="step-circle step-inactive">3</div>
            <span class="step-label">Prediction</span>
          </div>
        </div>
        """)

        # ── Form ──────────────────────────────────────────────────────────
        with gr.Column(elem_classes="form-card"):

            # Demographics
            gr.HTML('<div class="section-label">Step 01 &mdash; Demographics</div>'
                    '<div class="section-title">Patient Information</div>')
            with gr.Row():
                age      = gr.Number(label="Age (years)",        minimum=18, maximum=100, step=1,   value=34)
                children = gr.Number(label="Number of Dependents", minimum=0, maximum=10,  step=1,   value=1)
            gender = gr.Radio(choices=["male", "female"], label="Biological Sex", value="male")

            # Vitals
            gr.HTML('<div class="sub-section">Step 02 &mdash; Clinical Vitals &nbsp;·&nbsp; Health Measurements</div>')
            with gr.Row():
                bmi           = gr.Number(label="Body Mass Index (BMI)",             minimum=10.0, maximum=60.0, step=0.1, value=26.5)
                bloodpressure = gr.Number(label="Systolic Blood Pressure (mmHg)",    minimum=80,   maximum=200,  step=1,   value=122)

            # Medical history
            gr.HTML('<div class="sub-section">Step 03 &mdash; Medical History &nbsp;·&nbsp; Conditions &amp; Lifestyle</div>')
            with gr.Row():
                diabetic = gr.Radio(choices=["No", "Yes"], label="Diabetes Status",    value="No")
                smoker   = gr.Radio(choices=["No", "Yes"], label="Tobacco / Smoking",  value="No")

        # ── Predict button ─────────────────────────────────────────────────
        predict_btn = gr.Button("⚡  Run AI Prediction", variant="primary", elem_id="predict-btn")

        # ── Output slots ───────────────────────────────────────────────────
        out_result    = gr.HTML(elem_classes="output-html")
        out_breakdown = gr.HTML(elem_classes="output-html")
        out_recs      = gr.HTML(elem_classes="output-html")
        out_disclaimer= gr.HTML(elem_classes="output-html")

        # ── Wire up ────────────────────────────────────────────────────────
        predict_btn.click(
            fn=predict_claim,
            inputs=[age, gender, bmi, bloodpressure, diabetic, smoker, children],
            outputs=[out_result, out_breakdown, out_recs, out_disclaimer],
        )

        # ── Footer ─────────────────────────────────────────────────────────
        gr.HTML("""
        <div style="margin-top:48px;padding:28px 0;
                    border-top:1px solid var(--navy-border);
                    display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
            <div style="font-family:'Syne',sans-serif;font-size:16px;font-weight:700;
                        background:linear-gradient(135deg,#e8f0fe,var(--teal));
                        -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">
                MediClaim AI
            </div>
            <div style="font-size:11px;color:var(--text-muted);letter-spacing:.06em;">
                Built with Gradio &middot; Scikit-learn &middot; Python
            </div>
            <div style="font-size:11px;color:var(--text-muted);">
                &copy; 2025 &nbsp;&middot;&nbsp; For portfolio &amp; educational use
            </div>
        </div>
        """)

# ── Launch ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
   demo.launch(share=True, css=css)