import streamlit as st
import joblib
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent

# -------------------- Page + theme --------------------
st.set_page_config(
    page_title="Triage — Ticket Classification Console",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg: #0B0E14;
  --panel: #121722;
  --raised: #171D2B;
  --border: #232B3A;
  --text: #E7EBF0;
  --muted: #8B96A5;
  --blue: #5B8CFF;
  --green: #34D399;
  --yellow: #FBBF24;
  --coral: #FB8A65;
  --red: #F0475A;
}
.stApp { background: var(--bg); color: var(--text); }
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { max-width: 1440px; padding-top: 1.6rem; padding-bottom: 3rem; }
h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif !important; color: var(--text) !important; }
p, label, span, div { color: inherit; }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
section[data-testid="stSidebar"] { background: var(--panel); }
div[data-testid="stTextArea"] textarea {
  background: #0E131D !important; color: var(--text) !important;
  border: 1px solid var(--border) !important; border-radius: 12px !important;
  font-family: 'Inter', sans-serif !important; line-height: 1.6;
}
div[data-testid="stTextArea"] textarea:focus {
  border-color: var(--blue) !important;
  box-shadow: 0 0 0 1px var(--blue) !important;
}
.stButton > button {
  border-radius: 10px; border: 1px solid var(--border);
  background: var(--raised); color: var(--text);
  min-height: 42px; font-weight: 600; transition: all .15s ease;
}
.stButton > button:hover { border-color: var(--blue); color: white; }
div.stButton > button[kind="primary"] {
  background: var(--blue); color: white; border: 1px solid var(--blue);
}
div.stButton > button[kind="primary"]:hover { background: #4778EC; }
div[data-testid="stAlert"] { border-radius: 10px; }
hr { border-color: var(--border) !important; }
.topbar {
  display:flex; align-items:center; justify-content:space-between; gap:16px;
  padding: 0 0 22px 0; border-bottom:1px solid var(--border); margin-bottom:26px;
}
.brand { display:flex; align-items:center; gap:12px; }
.brand-icon {
  width:42px; height:42px; display:flex; align-items:center; justify-content:center;
  border-radius:12px; background:#1C2A47; color:var(--blue); font-size:22px;
  border:1px solid #2B416B;
}
.brand-title { font-family:'Space Grotesk',sans-serif; font-size:22px; font-weight:700; letter-spacing:-.5px; }
.brand-sub { color:var(--muted); font-size:12px; margin-top:3px; }
.status {
  color:var(--green); background:#102820; border:1px solid #1D4939;
  border-radius:999px; padding:7px 11px; font-size:11px; font-weight:700;
  letter-spacing:.5px; white-space:nowrap;
}
.panel {
  background:var(--panel); border:1px solid var(--border); border-radius:16px;
  padding:22px; min-height:100%;
}
.panel-kicker { color:var(--muted); text-transform:uppercase; font-size:10px; letter-spacing:1.5px; font-weight:700; }
.panel-title { font-family:'Space Grotesk',sans-serif; font-size:20px; font-weight:600; margin-top:6px; margin-bottom:5px; }
.panel-desc { color:var(--muted); font-size:12px; margin-bottom:20px; }
.sample-label { color:var(--muted); font-size:11px; margin:10px 0 8px; }
.result-card {
  background:var(--raised); border:1px solid var(--border); border-radius:12px;
  padding:15px 16px; margin-bottom:12px;
}
.result-label { color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:1px; font-weight:700; }
.result-value { font-family:'Space Grotesk',sans-serif; font-size:20px; font-weight:600; margin-top:7px; overflow-wrap:anywhere; }
.result-dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:7px; }
.small-muted { color:var(--muted); font-size:11px; }
.history-item {
  border-bottom:1px solid var(--border); padding:12px 0;
}
.history-text { color:var(--text); font-size:12px; line-height:1.5; }
.history-meta { color:var(--muted); font-family:'JetBrains Mono',monospace; font-size:10px; margin-top:6px; }
.empty-state {
  text-align:center; padding:38px 15px; border:1px dashed #2A3447; border-radius:12px;
  color:var(--muted); background:#0E131D;
}
.empty-icon { font-size:26px; margin-bottom:8px; }
@media (max-width: 700px) {
  .block-container { padding-left: 1rem; padding-right: 1rem; }
  .panel { padding:16px; }
  .brand-title { font-size:19px; }
}
</style>
""", unsafe_allow_html=True)

# -------------------- Model loading (same paths) --------------------
@st.cache_resource
def load_models():
    department = joblib.load(BASE_DIR / "artifacts" / "model_department.pkl")
    priority = joblib.load(BASE_DIR / "artifacts" / "model_priority.pkl")
    topic = joblib.load(BASE_DIR / "artifacts_data3" / "model_topic.pkl")
    return department, priority, topic

def predict_with_confidence(model, text):
    """Predict with the existing model; show confidence only if supported."""
    prediction = model.predict([text])[0]
    confidence = None
    try:
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([text])
            if probabilities is not None and len(probabilities) and len(probabilities[0]):
                confidence = float(max(probabilities[0]))
        elif hasattr(model, "decision_function"):
            # Decision scores are not probabilities, so deliberately don't label them confidence.
            confidence = None
    except Exception:
        confidence = None
    return str(prediction), confidence

def priority_color(value):
    value = str(value).lower()
    if "critical" in value or "urgent" in value:
        return "#F0475A"
    if "high" in value:
        return "#FB8A65"
    if "medium" in value or "normal" in value:
        return "#FBBF24"
    if "low" in value:
        return "#34D399"
    return "#5B8CFF"

# -------------------- Header --------------------
st.markdown("""
<div class="topbar">
  <div class="brand">
    <div class="brand-icon">✳</div>
    <div>
      <div class="brand-title">Triage</div>
      <div class="brand-sub">Ticket classification console</div>
    </div>
  </div>
  <div class="status">● &nbsp; CLASSIFIER READY</div>
</div>
""", unsafe_allow_html=True)

# Session state
if "history" not in st.session_state:
    st.session_state.history = []
if "ticket_input" not in st.session_state:
    st.session_state.ticket_input = ""
if "latest" not in st.session_state:
    st.session_state.latest = None

samples = {
    "Printer issue": "The office printer keeps jamming every time I try to print documents. Please help resolve this.",
    "Production outage": "The production checkout server is down and customers cannot complete purchases. This is urgent.",
    "Monitor request": "I would like to request an additional monitor for my desk. No rush, whenever available.",
    "Password locked": "I am locked out of my account after several failed password attempts and need access restored.",
}

left, right = st.columns([1.04, 0.96], gap="large")

with left:
    st.markdown("""
    <div class="panel">
      <div class="panel-kicker">01 / Intake</div>
      <div class="panel-title">Ticket details</div>
      <div class="panel-desc">Paste a support request below or start from a sample ticket.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sample-label">TRY A SAMPLE TICKET</div>', unsafe_allow_html=True)
    sample_cols = st.columns(2)
    sample_names = list(samples.keys())
    for idx, col in enumerate(sample_cols):
        with col:
            if st.button(sample_names[idx], use_container_width=True, key=f"sample_{idx}"):
                st.session_state.ticket_input = samples[sample_names[idx]]
                st.rerun()
    sample_cols2 = st.columns(2)
    for idx, col in enumerate(sample_cols2, start=2):
        with col:
            if st.button(sample_names[idx], use_container_width=True, key=f"sample_{idx}"):
                st.session_state.ticket_input = samples[sample_names[idx]]
                st.rerun()

    ticket = st.text_area(
        "Ticket description",
        key="ticket_input",
        height=210,
        placeholder="Describe the issue, request, or incident in detail...",
        label_visibility="collapsed",
    )
    st.markdown(
        f'<div class="small-muted" style="text-align:right;margin-top:-8px;margin-bottom:12px;">{len(ticket)} characters</div>',
        unsafe_allow_html=True,
    )
    classify = st.button("✦  Classify ticket", type="primary", use_container_width=True)

    st.markdown("""
    <div class="small-muted" style="margin-top:12px;">
      Predictions are generated by your saved machine-learning models.
    </div>
    """, unsafe_allow_html=True)

    if classify:
        if not ticket.strip():
            st.warning("Please enter a ticket description.")
        else:
            try:
                department_model, priority_model, topic_model = load_models()
                dept, dept_conf = predict_with_confidence(department_model, ticket.strip())
                pri, pri_conf = predict_with_confidence(priority_model, ticket.strip())
                top, top_conf = predict_with_confidence(topic_model, ticket.strip())
                result = {
                    "ticket": ticket.strip(),
                    "department": dept, "department_confidence": dept_conf,
                    "priority": pri, "priority_confidence": pri_conf,
                    "topic": top, "topic_confidence": top_conf,
                    "time": datetime.now().strftime("%H:%M:%S"),
                }
                st.session_state.latest = result
                st.session_state.history.insert(0, result)
                st.session_state.history = st.session_state.history[:8]
            except Exception as e:
                st.error(f"Prediction failed: {e}")

with right:
    st.markdown("""
    <div class="panel">
      <div class="panel-kicker">02 / Readout</div>
      <div class="panel-title">Classification results</div>
      <div class="panel-desc">Model output for the submitted support ticket.</div>
    </div>
    """, unsafe_allow_html=True)

    latest = st.session_state.latest
    if latest is None:
        st.markdown("""
        <div class="empty-state">
          <div class="empty-icon">⌁</div>
          <div style="font-family:'Space Grotesk',sans-serif;font-size:16px;color:#E7EBF0;margin-bottom:6px;">Awaiting ticket</div>
          <div style="font-size:12px;">Your classification results will appear here.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="result-card"><div class="result-label">Department</div>'
            f'<div class="result-value"><span class="result-dot" style="background:#5B8CFF"></span>{latest["department"]}</div></div>',
            unsafe_allow_html=True,
        )
        pri_color = priority_color(latest["priority"])
        st.markdown(
            f'<div class="result-card"><div class="result-label">Priority</div>'
            f'<div class="result-value"><span class="result-dot" style="background:{pri_color}"></span>{latest["priority"]}</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="result-card"><div class="result-label">Topic</div>'
            f'<div class="result-value"><span class="result-dot" style="background:#34D399"></span>{latest["topic"]}</div></div>',
            unsafe_allow_html=True,
        )
        confidences = [
            ("Department", latest["department_confidence"]),
            ("Priority", latest["priority_confidence"]),
            ("Topic", latest["topic_confidence"]),
        ]
        available = [(name, conf) for name, conf in confidences if conf is not None]
        if available:
            st.markdown("#### Model confidence")
            for name, conf in available:
                st.markdown(f'<div class="small-muted" style="margin-top:10px;margin-bottom:5px;">{name} <span style="float:right;color:#E7EBF0;">{conf:.0%}</span></div>', unsafe_allow_html=True)
                st.progress(max(0.0, min(1.0, conf)))
            st.caption("Confidence is shown only when the model provides prediction probabilities; it is not a guarantee of correctness.")
        else:
            st.caption("Your models returned predictions successfully. Probability-based confidence is not exposed by one or more models, so no confidence score is displayed.")

# -------------------- Session queue --------------------
st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
st.markdown("""
<div class="panel-kicker">03 / Session queue</div>
<div class="panel-title" style="font-size:20px;margin-bottom:5px;">Recent classifications</div>
<div class="panel-desc">Your latest predictions in this session. This history is cleared when the session resets.</div>
""", unsafe_allow_html=True)

if not st.session_state.history:
    st.markdown("""
    <div class="empty-state">
      <div style="font-size:13px;">No classifications yet. Submit a ticket to populate the queue.</div>
    </div>
    """, unsafe_allow_html=True)
else:
    for i, item in enumerate(st.session_state.history):
        preview = item["ticket"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        if len(preview) > 145:
            preview = preview[:142] + "..."
        st.markdown(
            f'<div class="history-item"><div class="history-text">{preview}</div>'
            f'<div class="history-meta">{item["time"]} &nbsp;·&nbsp; DEPT: {item["department"]} '
            f'&nbsp;·&nbsp; PRIORITY: {item["priority"]} &nbsp;·&nbsp; TOPIC: {item["topic"]}</div></div>',
            unsafe_allow_html=True,
        )

st.markdown("""
<div style="border-top:1px solid #232B3A;margin-top:28px;padding-top:14px;color:#687486;font-size:11px;display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;">
  <span>TRIAGE · ML-POWERED TICKET CLASSIFICATION</span>
  <span>Department · Priority · Topic</span>
</div>
""", unsafe_allow_html=True)
