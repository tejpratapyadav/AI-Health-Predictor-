import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from fpdf import FPDF
import streamlit.components.v1 as components

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="AI Health Predictor", layout="centered")

# ---------- GLOBAL LIGHT CSS ----------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #e0f7fa, #fce4ec);
    color: #333;
    font-family: 'Poppins', sans-serif;
}
h1, h2, h3, h4 {
    color: #2c3e50;
}
div[data-testid="stForm"] {
    background-color: rgba(255,255,255,0.9);
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 4px 25px rgba(0,0,0,0.1);
    backdrop-filter: blur(10px);
}
button[kind="secondary"] {
    display:none !important;
}
.result-card {
    background: linear-gradient(135deg, #d1f4f2, #e1f5fe);
    border-radius: 12px;
    padding: 20px;
    margin-top: 25px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.1);
}
.stDownloadButton > button {
    background: linear-gradient(90deg, #00bcd4, #2196f3);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 10px 20px;
    box-shadow: 0 4px 15px rgba(33,150,243,0.4);
}
.stDownloadButton > button:hover {
    background: linear-gradient(90deg, #26c6da, #00acc1);
    color: #fff;
}
.predict-btn {
    width: 200px;
    height: 50px;
    border: none;
    border-radius: 25px;
    color: white;
    font-weight: 600;
    font-size: 16px;
    background: linear-gradient(45deg, #00bcd4, #2196f3);
    cursor: pointer;
    box-shadow: 0 4px 15px rgba(33,150,243,0.4);
    animation: pulse 2s infinite;
}
@keyframes pulse {
  0% {box-shadow: 0 0 0 0 rgba(33,150,243,0.4);}
  70% {box-shadow: 0 0 0 20px rgba(33,150,243,0);}
  100% {box-shadow: 0 0 0 0 rgba(33,150,243,0);}
}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
header_html = """
<div style="display:flex;align-items:center;gap:15px;margin-bottom:10px;">
  <div style="width:80px;height:80px;border-radius:20px;display:flex;align-items:center;justify-content:center;
              background:linear-gradient(135deg,#42a5f5,#26c6da);box-shadow:0 5px 20px rgba(66,165,245,0.3);">
    <svg width="48" height="48" viewBox="0 0 24 24" fill="none">
      <path d="M12 21C12 21 5 14 5 9.5C5 7 7 5 9 5C10.5 5 12 6 12 6C12 6 13.5 5 15 5C17 5 19 7 19 9.5C19 14 12 21 12 21Z"
            fill="white" opacity="0.95">
        <animate attributeName="opacity" values="1;0.6;1" dur="1.5s" repeatCount="indefinite"/>
      </path>
    </svg>
  </div>
  <div>
    <h1 style="margin:0;">AI Health Predictor</h1>
    <p style="margin:0;color:#555;">Instant Symptom → Diagnosis & Treatment</p>
  </div>
</div>
<hr>
"""
components.html(header_html, height=110)

# ---------- SYNTHETIC LARGE DATASET ----------
@st.cache_data
def load_big_dataset():
    np.random.seed(42)
    diseases = ["Common Cold", "COVID-19", "Diabetes", "Asthma", "Pneumonia", "Migraine",
                "Malaria", "Allergic Rhinitis", "Sinusitis", "Hypertension", "Bronchitis"]
    treatments = {
        "Common Cold": "Rest and hydration; Paracetamol if needed",
        "COVID-19": "Isolation; Antivirals; Medical monitoring",
        "Diabetes": "Insulin therapy; Diet control; Exercise",
        "Asthma": "Inhaler; Avoid triggers; Regular checkup",
        "Pneumonia": "Antibiotics; Hospitalization if severe",
        "Migraine": "Pain relief; Rest; Dark quiet room",
        "Malaria": "Antimalarial drugs; Hydration",
        "Allergic Rhinitis": "Antihistamines; Avoid allergens",
        "Sinusitis": "Nasal spray; Antibiotics if bacterial",
        "Hypertension": "Lifestyle change; Medication",
        "Bronchitis": "Cough syrup; Rest; Steam inhalation"
    }
    rows = []
    for _ in range(120):
        age = np.random.randint(18, 70)
        gender = np.random.choice(["M", "F"])
        disease = np.random.choice(diseases)
        row = {
            "Fever": np.random.choice(["Yes", "No"]),
            "Cough": np.random.choice(["Yes", "No"]),
            "Fatigue": np.random.choice(["Yes", "No"]),
            "Breathlessness": np.random.choice(["Yes", "No"]),
            "Headache": np.random.choice(["Yes", "No"]),
            "Age": age,
            "Gender": gender,
            "Diagnosis": disease,
            "Treatment": treatments[disease]
        }
        rows.append(row)
    return pd.DataFrame(rows)

# ---------- MODEL TRAINING ----------
@st.cache_data
def train_model():
    df = load_big_dataset().copy()
    for c in ["Fever","Cough","Fatigue","Breathlessness","Headache"]:
        df[c] = df[c].str.lower().map({"yes":1,"no":0})
    df["Gender"] = df["Gender"].str.lower().map({"m":0,"male":0,"f":1,"female":1})
    le = LabelEncoder()
    df["Label"] = le.fit_transform(df["Diagnosis"])
    X = df[["Fever","Cough","Fatigue","Breathlessness","Headache","Age","Gender"]]
    y = df["Label"]
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X, y)
    return model, le, df

model, le, df = train_model()

# ---------- INPUT FORM ----------
with st.form("symptom_form"):
    st.subheader("🩺 Enter Symptoms")
    c1, c2, c3 = st.columns(3)
    with c1:
        fever = st.selectbox("Fever", ["No","Yes"])
        cough = st.selectbox("Cough", ["No","Yes"])
    with c2:
        fatigue = st.selectbox("Fatigue", ["No","Yes"])
        breath = st.selectbox("Breathlessness", ["No","Yes"])
    with c3:
        headache = st.selectbox("Headache", ["No","Yes"])
        age = st.number_input("Age", 0, 120, 25)
        gender = st.selectbox("Gender", ["Male","Female"])
    submitted = st.form_submit_button("✨ Predict Disease")

# ---------- PREDICTION ----------
if submitted:
    input_df = pd.DataFrame([{
        "Fever": 1 if fever=="Yes" else 0,
        "Cough": 1 if cough=="Yes" else 0,
        "Fatigue": 1 if fatigue=="Yes" else 0,
        "Breathlessness": 1 if breath=="Yes" else 0,
        "Headache": 1 if headache=="Yes" else 0,
        "Age": int(age),
        "Gender": 0 if gender=="Male" else 1
    }])
    pred = model.predict(input_df)[0]
    disease = le.inverse_transform([pred])[0]
    treat = df.loc[df["Diagnosis"]==disease, "Treatment"].iloc[0]
    conf = model.predict_proba(input_df).max()*100

    st.markdown(f"""
    <div class="result-card">
        <h3>Prediction Result 🎉</h3>
        <p><b>Disease:</b> {disease}</p>
        <p><b>Treatment:</b> {treat}</p>
        <p><b>Confidence:</b> {conf:.2f}%</p>
    </div>
    """, unsafe_allow_html=True)

    # ---------- DOWNLOAD SECTION ----------
    result = pd.DataFrame([{
        "Fever": fever, "Cough": cough, "Fatigue": fatigue, "Breathlessness": breath,
        "Headache": headache, "Age": age, "Gender": gender,
        "Predicted Disease": disease, "Suggested Treatment": treat, "Confidence": f"{conf:.2f}%"
    }])
    csv = result.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download CSV", csv, "prediction_result.csv", "text/csv")

    def make_pdf(record):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "AI Health Prediction Report", ln=True, align="C")
        pdf.ln(8)
        pdf.set_font("Arial", size=12)
        for k, v in record.items():
            pdf.multi_cell(0, 8, f"{k}: {v}")
        return pdf.output(dest="S").encode("latin-1")

    pdf = make_pdf(result.iloc[0].to_dict())
    st.download_button("📄 Download PDF Report", pdf, "prediction_report.pdf", "application/pdf")


