"""
Healthcare Chatbot — Streamlit GUI
====================================
Run with:  streamlit run streamlit_app.py

This file is the COMPLETE standalone GUI.
It imports all logic from app.py (Flask is NOT needed to run this).
Make sure app.py is in the same folder.

Install:
    pip install streamlit flask flask-session pandas numpy scikit-learn
                wikipedia spacy fuzzywuzzy python-Levenshtein
    python -m spacy download en_core_web_sm
"""

import streamlit as st
import sys
import os

# ── Import backend logic from app.py ─────────────────────────────────────────
# We import the pure functions directly — no Flask server needed
sys.path.insert(0, os.path.dirname(__file__))

from app import (
    extract_symptoms,
    predict_disease,
    pick_best_question,
    _rerank_by_demographics,
    wiki_answer,
    extract_wiki_query,
    is_wiki_question,
    get_disease_description,
    get_precautions,
    _local_description_fallback,
    CONFIDENCE_THRESHOLD,
    MAX_CONFIRM_QUESTIONS,
    disease_symptom_map,
    symptoms_dict,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HealthBot AI",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=DM+Serif+Display:ital@0;1&display=swap');

/* ── Root variables ── */
:root {
    --teal:        #0d9488;
    --teal-light:  #ccfbf1;
    --teal-dark:   #0f766e;
    --navy:        #0f172a;
    --slate:       #334155;
    --muted:       #94a3b8;
    --white:       #ffffff;
    --offwhite:    #f8fafc;
    --card:        #ffffff;
    --border:      #e2e8f0;
    --user-bg:     #0d9488;
    --bot-bg:      #f1f5f9;
    --bot-text:    #1e293b;
    --shadow:      0 4px 24px rgba(13,148,136,0.10);
    --radius:      16px;
}

/* ── Reset & base ── */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background: var(--offwhite) !important;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 0 !important;
    max-width: 780px !important;
    margin: 0 auto !important;
}

/* ── Top header bar ── */
.hb-header {
    background: linear-gradient(135deg, #0f766e 0%, #0d9488 50%, #14b8a6 100%);
    padding: 22px 32px 18px;
    display: flex;
    align-items: center;
    gap: 14px;
    position: sticky;
    top: 0;
    z-index: 100;
    box-shadow: 0 2px 20px rgba(13,148,136,0.25);
}

.hb-header-icon {
    width: 46px; height: 46px;
    background: rgba(255,255,255,0.2);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 24px;
    backdrop-filter: blur(10px);
}

.hb-header-text h1 {
    font-family: 'DM Serif Display', serif !important;
    font-size: 22px !important;
    color: white !important;
    margin: 0 !important;
    
    padding: 0 !important;
    letter-spacing: -0.3px;
}

.hb-header-text p {
    font-size: 12px !important;
    color: rgba(255,255,255,0.75) !important;
    margin: 2px 0 0 !important;
    font-weight: 400;
}

.hb-status {
    margin-left: auto;
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    color: rgba(255,255,255,0.85);
    font-weight: 500;
}

.hb-status-dot {
    width: 8px; height: 8px;
    background: #4ade80;
    border-radius: 50%;
    animation: pulse-green 2s infinite;
}

@keyframes pulse-green {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.6; transform: scale(1.3); }
}

/* ── Chat container ── */
.hb-chat-area {
    padding: 24px 20px 12px;
    min-height: 60vh;
    display: flex;
    flex-direction: column;
    gap: 16px;
}

/* ── Message bubbles ── */
.hb-msg-row {
    display: flex;
    align-items: flex-end;
    gap: 10px;
    animation: slideUp 0.3s ease;
}

@keyframes slideUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}

.hb-msg-row.user { flex-direction: row-reverse; }

.hb-avatar {
    width: 34px; height: 34px;
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px;
    flex-shrink: 0;
}

.hb-avatar.bot {
    background: linear-gradient(135deg, #0d9488, #14b8a6);
    box-shadow: 0 4px 12px rgba(13,148,136,0.3);
}

.hb-avatar.user {
    background: linear-gradient(135deg, #334155, #475569);
}

.hb-bubble {
    max-width: 78%;
    padding: 13px 16px;
    border-radius: 18px;
    font-size: 14.5px;
    line-height: 1.65;
    white-space: pre-wrap;
    word-wrap: break-word;
    position: relative;
}

.hb-bubble.bot {
    background: var(--white);
    color: var(--bot-text);
    border-bottom-left-radius: 4px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    border: 1px solid var(--border);
}

.hb-bubble.user {
    background: linear-gradient(135deg, #0d9488, #0f766e);
    color: white;
    border-bottom-right-radius: 4px;
    box-shadow: 0 4px 14px rgba(13,148,136,0.35);
}

.hb-time {
    font-size: 10.5px;
    color: var(--muted);
    margin-top: 4px;
    text-align: right;
}

.hb-time.bot { text-align: left; padding-left: 44px; }

/* ── Typing indicator ── */
.hb-typing {
    display: flex;
    align-items: flex-end;
    gap: 10px;
    padding: 0 20px;
}

.hb-typing-bubble {
    background: white;
    border: 1px solid var(--border);
    border-radius: 18px;
    border-bottom-left-radius: 4px;
    padding: 14px 18px;
    display: flex;
    gap: 5px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
}

.hb-dot {
    width: 7px; height: 7px;
    background: var(--teal);
    border-radius: 50%;
    animation: bounce 1.2s infinite;
}
.hb-dot:nth-child(2) { animation-delay: 0.2s; }
.hb-dot:nth-child(3) { animation-delay: 0.4s; }

@keyframes bounce {
    0%, 100% { transform: translateY(0); opacity: 0.5; }
    50%       { transform: translateY(-6px); opacity: 1; }
}

/* ── Input area ── */
.hb-input-area {
    background: white;
    border-top: 1px solid var(--border);
    padding: 14px 16px;
    position: sticky;
    bottom: 0;
    box-shadow: 0 -4px 20px rgba(0,0,0,0.06);
}

/* Override Streamlit input */
.stTextInput > div > div > input {
    border: 2px solid var(--border) !important;
    border-radius: 14px !important;
    padding: 13px 18px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 14px !important;
    background: var(--offwhite) !important;
    color: var(--navy) !important;
    transition: all 0.2s ease !important;
    box-shadow: none !important;
}

.stTextInput > div > div > input:focus {
    border-color: var(--teal) !important;
    background: white !important;
    box-shadow: 0 0 0 3px rgba(13,148,136,0.12) !important;
}

/* Send button */
.stButton > button {
    background: linear-gradient(135deg, #0d9488, #0f766e) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 13px 22px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 14px rgba(13,148,136,0.35) !important;
    width: 100% !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(13,148,136,0.45) !important;
}

.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── Sidebar progress ── */
.hb-progress-card {
    background: white;
    border-radius: var(--radius);
    padding: 20px;
    border: 1px solid var(--border);
    margin-bottom: 14px;
    box-shadow: var(--shadow);
}

.hb-step {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 0;
    font-size: 13px;
    color: var(--muted);
    font-weight: 500;
}

.hb-step.done { color: var(--teal-dark); }
.hb-step.active { color: var(--navy); font-weight: 600; }

.hb-step-icon {
    width: 28px; height: 28px;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 13px;
    background: var(--border);
    flex-shrink: 0;
}

.hb-step.done .hb-step-icon  { background: var(--teal-light); }
.hb-step.active .hb-step-icon { background: var(--teal); color: white; }

/* ── Quick tip pills ── */
.hb-tips {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    padding: 0 20px 16px;
}

.hb-tip {
    background: white;
    border: 1.5px solid var(--border);
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 12.5px;
    color: var(--slate);
    cursor: pointer;
    transition: all 0.15s;
    font-weight: 500;
}

.hb-tip:hover {
    border-color: var(--teal);
    color: var(--teal-dark);
    background: var(--teal-light);
}

/* ── Report card styling ── */
.hb-report {
    background: linear-gradient(135deg, #f0fdf4, #ecfdf5);
    border: 1.5px solid #6ee7b7;
    border-radius: 16px;
    padding: 16px 20px;
    font-size: 14px;
}

/* ── Divider ── */
.hb-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 8px 20px;
}

/* Streamlit sidebar */
section[data-testid="stSidebar"] {
    background: var(--offwhite) !important;
    border-right: 1px solid var(--border) !important;
}

/* Remove streamlit form padding */
[data-testid="stForm"] {
    border: none !important;
    padding: 0 !important;
}

/* Hide streamlit label */
.stTextInput label { display: none !important; }

</style>
""", unsafe_allow_html=True)


# ── Session state init ────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "step":                "welcome",
        "messages":            [],          # list of {role, text, time}
        "name":                "",
        "age":                 None,
        "gender":              "",
        "symptoms":            [],
        "asked_symptoms":      [],
        "questions_asked":     0,
        "current_confirm_sym": None,
        "pred_disease":        "",
        "pred_conf":           0,
        "severity":            0,
        "days":                0,
        "initialized":         False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ── Time helper ───────────────────────────────────────────────────────────────
from datetime import datetime
def now_str():
    return datetime.now().strftime("%I:%M %p")


# ── Add message helper ────────────────────────────────────────────────────────
def add_msg(role: str, text: str):
    st.session_state.messages.append({
        "role": role,
        "text": text,
        "time": now_str()
    })


# ── Core chatbot logic (mirrors Flask chat() route) ───────────────────────────
import re

def process_message(msg: str) -> str:
    msg = msg.strip()
    step = st.session_state.step

    # Welcome — fire on first interaction
    if step == "welcome":
        st.session_state.step = "name"
        return (
            "👋 Hi there! I'm your personal Health Assistant 😊\n\n"
            "I'll chat with you about how you're feeling, ask a few simple "
            "questions, and help figure out what condition you might have — "
            "based on your symptoms.\n\n"
            "I won't guess until I'm confident, so the more honestly you "
            "answer, the better I can help you! 💙\n\n"
            "Let's start — what's your name?"
        )

    if not msg:
        return "⚠️ Please type something."

    # ── Wikipedia intercept ───────────────────────────────────────────────────
    if is_wiki_question(msg):
        query = extract_wiki_query(msg)
        if query:
            info = wiki_answer(query)
            return f"📖 {info}\n\n_(Type anything to continue.)_"

    # ── Name ──────────────────────────────────────────────────────────────────
    if step == "name":
        name_clean = re.sub(r"[^a-zA-Z\s]", "", msg).strip()
        if not name_clean:
            return "⚠️ Please enter a valid name (letters only)."
        st.session_state.name = name_clean.title()
        st.session_state.step = "age"
        return f"Nice to meet you, {st.session_state.name}! 😊\n\nHow old are you?"

    # ── Age ───────────────────────────────────────────────────────────────────
    elif step == "age":
        nums = re.findall(r'\b(\d+)\b', msg)
        if not nums or not (1 <= int(nums[0]) <= 120):
            return "⚠️ Please enter a valid age between 1 and 120."
        st.session_state.age = int(nums[0])
        st.session_state.step = "gender"
        return "What is your gender?\n\n👉 Male / Female / Other"

    # ── Gender ────────────────────────────────────────────────────────────────
    elif step == "gender":
        g = msg.strip().lower()
        if g in ("m", "male", "man", "boy"):
            st.session_state.gender = "male"
        elif g in ("f", "female", "woman", "girl"):
            st.session_state.gender = "female"
        elif g in ("other", "o", "prefer not to say", "none"):
            st.session_state.gender = "other"
        else:
            return (
                "⚠️ Please enter a valid option:\n"
                "  • Male\n  • Female\n  • Other"
            )
        st.session_state.step = "symptoms"
        return (
            f"Got it! 😊\n\n"
            f"Now tell me how you're feeling. Describe your symptoms "
            f"in simple words — like:\n"
            f"'I have fever, headache and stomach pain'"
        )

    # ── Symptoms ──────────────────────────────────────────────────────────────
    elif step == "symptoms":
        symptoms = extract_symptoms(msg)
        if not symptoms:
            return (
                "❌ I couldn't detect any recognizable symptoms.\n\n"
                "Try describing them more clearly, e.g.:\n"
                "'I have fever, cough and fatigue'\n\n"
                "Or ask me 'What is fever?' to learn about a symptom."
            )

        st.session_state.symptoms        = symptoms
        st.session_state.asked_symptoms  = list(symptoms)
        st.session_state.questions_asked = 0
        st.session_state.current_confirm_sym = None
        st.session_state.step = "confirm"

        disease, conf, top3 = predict_disease(symptoms)
        sym_text = ", ".join(s.replace("_", " ") for s in symptoms)

        if conf >= CONFIDENCE_THRESHOLD:
            return _start_confirm(symptoms, disease, conf)

        next_sym = pick_best_question(symptoms, set(symptoms))
        if not next_sym:
            return _start_confirm(symptoms, top3[0][0], top3[0][1])

        st.session_state.current_confirm_sym = next_sym
        st.session_state.asked_symptoms.append(next_sym)
        sym_display = next_sym.replace("_", " ")
        return (
            f"🩺 I detected: {sym_text}\n\n"
            f"Let me ask a few more questions to be accurate.\n\n"
            f"Do you also have {sym_display}? (yes / no)"
        )

    # ── Confirm loop ──────────────────────────────────────────────────────────
    elif step == "confirm":
        return _handle_confirm(msg)

    # ── Days ──────────────────────────────────────────────────────────────────
    elif step == "days":
        nums = re.findall(r'\b(\d+)\b', msg)
        if not nums or not (1 <= int(nums[0]) <= 365):
            return "⚠️ Please enter the number of days (e.g. 2, 5, 10)."
        st.session_state.days = int(nums[0])
        st.session_state.step = "severity"
        return "On a scale of 1–10, how severe are your symptoms?\n(1 = very mild, 10 = unbearable)"

    # ── Severity ──────────────────────────────────────────────────────────────
    elif step == "severity":
        nums = re.findall(r'\b(\d+)\b', msg)
        if not nums or not (1 <= int(nums[0]) <= 10):
            return "⚠️ Please enter a number between 1 and 10."
        st.session_state.severity = int(nums[0])
        st.session_state.step = "sym_confirm"
        symptoms = st.session_state.symptoms
        sym_text = "\n".join(f"   ✔️ {s.replace('_', ' ').title()}" for s in symptoms)
        return (
            f"Before I generate your report, let me confirm the symptoms I recorded:\n\n"
            f"{sym_text}\n\n"
            f"Is this correct? (yes / no)"
        )

    # ── Symptom confirmation ──────────────────────────────────────────────────
    elif step == "sym_confirm":
        ans = msg.strip().lower()
        if ans in ("yes", "y", "yeah", "yep", "correct", "right"):
            st.session_state.step = "final"
            return _generate_report()
        elif ans in ("no", "n", "nope", "wrong", "incorrect"):
            st.session_state.symptoms        = []
            st.session_state.asked_symptoms  = []
            st.session_state.questions_asked = 0
            st.session_state.current_confirm_sym = None
            st.session_state.step = "symptoms"
            return (
                "No problem! Let's redo it. 😊\n\n"
                "Please describe your symptoms again clearly:"
            )
        else:
            symptoms = st.session_state.symptoms
            sym_text = "\n".join(f"   ✔️ {s.replace('_', ' ').title()}" for s in symptoms)
            return (
                f"Please reply yes or no.\n\n"
                f"Symptoms recorded:\n{sym_text}\n\n"
                f"Is this correct? (yes / no)"
            )

    # ── Final ─────────────────────────────────────────────────────────────────
    elif step == "final":
        return (
            "Your report is above ☝️\n\n"
            "You can ask me 'What is [disease]?' or 'What causes [disease]?' "
            "to learn more.\n\n"
            "Type 'restart' to start a new session."
        )

    return "⚠️ Something went wrong. Type 'restart' to begin again."


def _start_confirm(symptoms, disease, conf):
    st.session_state.pred_disease = disease
    st.session_state.pred_conf    = conf
    st.session_state.step         = "days"
    sym_text = ", ".join(s.replace("_", " ") for s in symptoms)
    return (
        f"I think I have a good idea! 🩺\n\n"
        f"Based on: {sym_text}\n"
        f"I'm leaning towards: {disease} ({conf}% confident)\n\n"
        f"A couple more quick questions.\n"
        f"How many days have you had these symptoms?"
    )


def _handle_confirm(msg: str) -> str:
    answer          = msg.lower().strip()
    current_sym     = st.session_state.current_confirm_sym
    symptoms        = st.session_state.symptoms
    asked           = set(st.session_state.asked_symptoms)
    questions_asked = st.session_state.questions_asked

    YES = {"yes","y","yeah","yep","yup","sure","correct","true","yea"}
    NO  = {"no","n","nope","nah","not","never","false"}

    if current_sym:
        if answer in YES:
            if current_sym not in symptoms:
                symptoms.append(current_sym)
            st.session_state.symptoms = symptoms
        elif answer in NO:
            pass
        else:
            sym_display = current_sym.replace("_", " ")
            return f"⚠️ Please reply yes or no — do you have {sym_display}?"

        asked.add(current_sym)
        st.session_state.asked_symptoms  = list(asked)
        st.session_state.questions_asked = questions_asked + 1

    disease, conf, top3 = predict_disease(symptoms)
    top3 = _rerank_by_demographics(top3, st.session_state.age or 30, st.session_state.gender)
    disease, conf = top3[0]

    if conf >= CONFIDENCE_THRESHOLD:
        st.session_state.pred_disease = disease
        st.session_state.pred_conf    = conf
        st.session_state.step         = "days"
        return (
            f"I'm confident now! 😊\n\n"
            f"It looks like you may have: {disease} ({conf}% confidence)\n\n"
            f"💡 You can ask me 'What is {disease}?' anytime.\n\n"
            f"How many days have you been feeling this way?"
        )

    if st.session_state.questions_asked >= MAX_CONFIRM_QUESTIONS:
        st.session_state.pred_disease = top3[0][0]
        st.session_state.pred_conf    = top3[0][1]
        st.session_state.step         = "days"
        top3_text = "\n".join(f"   {i+1}. {d} ({c}%)" for i,(d,c) in enumerate(top3))
        return (
            f"Here are the most likely conditions based on your symptoms:\n\n"
            f"{top3_text}\n\n"
            f"How many days have you been feeling this way?"
        )

    next_sym = pick_best_question(symptoms, asked)
    if not next_sym:
        st.session_state.pred_disease = top3[0][0]
        st.session_state.pred_conf    = top3[0][1]
        st.session_state.step         = "days"
        return (
            f"Based on everything, it looks like: {top3[0][0]} ({top3[0][1]}%)\n\n"
            f"How many days have you been feeling this way?"
        )

    st.session_state.current_confirm_sym = next_sym
    asked.add(next_sym)
    st.session_state.asked_symptoms = list(asked)
    return f"Do you also have {next_sym.replace('_', ' ')}? (yes / no)"


def _generate_report() -> str:
    symptoms  = st.session_state.symptoms
    name      = st.session_state.name or "User"
    age       = st.session_state.age or "N/A"
    severity  = st.session_state.severity or 0
    days      = st.session_state.days or "N/A"

    disease, conf, top3 = predict_disease(symptoms)
    top3 = _rerank_by_demographics(top3, age if isinstance(age, int) else 30, st.session_state.gender)
    disease, conf = top3[0]

    urgency = (
        "🔴 High severity — please see a doctor immediately."
        if severity >= 7 else
        "🟡 Moderate — monitor your condition and rest well."
        if severity >= 4 else
        "🟢 Mild — rest, stay hydrated, and observe."
    )

    sym_text = ", ".join(s.replace("_", " ") for s in symptoms)

    if conf >= CONFIDENCE_THRESHOLD:
        description = get_disease_description(disease)
        desc_block  = f"\n📖 About {disease}:\n   {description}\n" if description else ""
        prec        = get_precautions(disease)
        prec_text   = "\n".join(f"   ✅ {p}" for p in prec)
        diag_block  = f"📌 Diagnosed Condition: {disease}\n🔎 Confidence: {conf}%"
        prec_block  = f"\n🛡️ Precautions:\n{prec_text}\n"
    else:
        t3          = "\n".join(f"   {i+1}. {d} ({c}%)" for i,(d,c) in enumerate(top3))
        diag_block  = f"📊 Possible Conditions:\n{t3}"
        desc_block  = ""
        prec        = get_precautions(top3[0][0])
        prec_text   = "\n".join(f"   ✅ {p}" for p in prec)
        prec_block  = f"\n🛡️ General Precautions:\n{prec_text}\n"

    top_disease = disease if conf >= CONFIDENCE_THRESHOLD else top3[0][0]

    return (
        f"╔══════════════════════════════════╗\n"
        f"   🌸  HEALTH REPORT — {name.upper()}\n"
        f"╚══════════════════════════════════╝\n\n"
        f"👤 Age: {age}  |  Severity: {severity}/10  |  Duration: {days} day(s)\n\n"
        f"🩺 Symptoms confirmed:\n   {sym_text}\n\n"
        f"{diag_block}\n"
        f"{desc_block}"
        f"{prec_block}\n"
        f"{urgency}\n\n"
        f"💡 Ask me:\n"
        f"   • 'What is {top_disease}?'\n"
        f"   • 'What causes {top_disease}?'\n"
        f"   • 'How to treat {top_disease}?'\n\n"
        f"⚠️  This is NOT a medical diagnosis.\n"
        f"   Please consult a qualified doctor."
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 8px 0 16px;'>
        <div style='font-family:"DM Serif Display",serif; font-size:20px; color:#0f172a; margin-bottom:4px;'>
            🩺 HealthBot AI
        </div>
        <div style='font-size:12px; color:#94a3b8;'>Smart symptom checker</div>
    </div>
    """, unsafe_allow_html=True)

    # Progress tracker
    steps = [
        ("welcome", "👋", "Welcome"),
        ("name",    "✏️", "Your Name"),
        ("age",     "🎂", "Age"),
        ("gender",  "👤", "Gender"),
        ("symptoms","🤒", "Symptoms"),
        ("confirm", "🔍", "Diagnosis"),
        ("days",    "📅", "Duration"),
        ("severity","📊", "Severity"),
        ("sym_confirm", "✔️", "Confirm"),
        ("final",   "📋", "Report"),
    ]

    step_keys = [s[0] for s in steps]
    current_idx = step_keys.index(st.session_state.step) if st.session_state.step in step_keys else 0

    st.markdown("<div class='hb-progress-card'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:12px;font-weight:600;color:#94a3b8;letter-spacing:0.08em;margin-bottom:8px;'>PROGRESS</div>", unsafe_allow_html=True)

    for i, (key, icon, label) in enumerate(steps):
        if i < current_idx:
            cls = "done"; tick = "✓"
        elif i == current_idx:
            cls = "active"; tick = icon
        else:
            cls = ""; tick = icon

        st.markdown(f"""
        <div class='hb-step {cls}'>
            <div class='hb-step-icon'>{tick}</div>
            {label}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Restart button
    st.markdown("<div style='margin-top:8px;'>", unsafe_allow_html=True)
    if st.button("🔄 Start New Session", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div style='margin-top:20px; padding:14px; background:white; border-radius:12px;
                border:1px solid #e2e8f0; font-size:12.5px; color:#64748b; line-height:1.7;'>
        <div style='font-weight:600; color:#0f172a; margin-bottom:6px;'>💡 Tips</div>
        • Describe symptoms naturally<br>
        • Answer yes/no clearly<br>
        • Ask 'What is [disease]?' anytime<br>
        • More symptoms = better accuracy
    </div>
    """, unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hb-header'>
    <div class='hb-header-icon'>🩺</div>
    <div class='hb-header-text'>
        <h1>HealthBot AI</h1>
        <p>Intelligent symptom checker & disease predictor</p>
    </div>
    <div class='hb-status'>
        <div class='hb-status-dot'></div>
        Online
    </div>
</div>
""", unsafe_allow_html=True)


# ── Auto-fire welcome on first load ──────────────────────────────────────────
if not st.session_state.initialized:
    welcome = process_message("")
    add_msg("bot", welcome)
    st.session_state.initialized = True


# ── Render chat messages ──────────────────────────────────────────────────────
st.markdown("<div class='hb-chat-area'>", unsafe_allow_html=True)

for msg in st.session_state.messages:
    role = msg["role"]
    text = msg["text"]
    time = msg["time"]

    if role == "bot":
        # Check if it's a report
        is_report = "HEALTH REPORT" in text
        bubble_class = "hb-report" if is_report else "bot"
        st.markdown(f"""
        <div class='hb-msg-row'>
            <div class='hb-avatar bot'>🩺</div>
            <div>
                <div class='hb-bubble {bubble_class}'>{text}</div>
                <div class='hb-time bot'>{time}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class='hb-msg-row user'>
            <div class='hb-avatar user'>👤</div>
            <div>
                <div class='hb-bubble user'>{text}</div>
                <div class='hb-time' style='text-align:right;padding-right:44px;'>{time}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ── Quick suggestion pills (shown only at symptoms step) ─────────────────────
if st.session_state.step == "symptoms":
    st.markdown("<div class='hb-tips'>", unsafe_allow_html=True)
    tips = ["fever & headache", "stomach pain & nausea", "cough & cold", "fatigue & body ache"]
    cols = st.columns(len(tips))
    for i, tip in enumerate(tips):
        with cols[i]:
            if st.button(f"💊 {tip}", key=f"tip_{i}", use_container_width=True):
                add_msg("user", tip)
                reply = process_message(tip)
                add_msg("bot", reply)
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


# ── Input area ────────────────────────────────────────────────────────────────
st.markdown("<div class='hb-input-area'>", unsafe_allow_html=True)

with st.form(key="chat_form", clear_on_submit=True):
    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_input(
            label="message",
            placeholder="Type your message here...",
            label_visibility="collapsed",
        )
    with col2:
        submitted = st.form_submit_button("Send ➤")

st.markdown("</div>", unsafe_allow_html=True)


# ── Handle submission ─────────────────────────────────────────────────────────
if submitted and user_input:
    user_text = user_input.strip()

    # Handle restart command
    if user_text.lower() in ("restart", "start over", "reset", "new"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

    add_msg("user", user_text)
    reply = process_message(user_text)
    add_msg("bot", reply)
    st.rerun()