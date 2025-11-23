import os
import pandas as pd
import streamlit as st
from datetime import datetime
from fpdf import FPDF

# ------------------ FOLDERS ------------------
UPLOAD_FOLDER = "uploads"
REPORT_FOLDER = "reports"
SIGNATURE_FOLDER = "signatures"
SIGNATURE_PATH = os.path.join(SIGNATURE_FOLDER, "digital_sign.png")
HISTORY_FILE = "history.csv"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)
os.makedirs(SIGNATURE_FOLDER, exist_ok=True)

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Brain Tumor Detection", layout="wide")

st.title("🧠 Brain Tumor Detection System")

# ------------------ USER SELECTION ------------------
user_type = st.radio("Select User Type", ["Doctor Visit", "Patient Visit"])

doctor_name = ""
doctor_phone = ""

if user_type == "Doctor Visit":
    doctor_name = st.text_input("Enter Doctor Name")
    doctor_phone = st.text_input("Enter Doctor Phone")

else:
    doctor_name = "Brain Tumor Detection AI Model"
    doctor_phone = "9885698725"

# ------------------ PATIENT DETAILS ------------------
st.subheader("Patient Information")

col1, col2 = st.columns(2)

with col1:
    patient_name = st.text_input("Patient Name")
    patient_age = st.text_input("Patient Age")

with col2:
    patient_address = st.text_area("Patient Address")

# ------------------ UPLOAD MRI ------------------
st.subheader("Upload MRI Image")

uploaded_file = st.file_uploader("Upload Brain MRI", type=["jpg", "png", "jpeg"])

# ------------------ TUMOR INFO ------------------
tumor_info = {
    "glioma": {
        "risk": "High Risk",
        "desc": "Aggressive tumor originating from glial cells.",
        "symptoms": "Headache, seizures, vision problems"
    },
    "meningioma": {
        "risk": "Medium Risk",
        "desc": "Slow growing tumor from brain membranes.",
        "symptoms": "Vision issues, headache"
    },
    "pituitary": {
        "risk": "Low Risk",
        "desc": "Tumor in pituitary gland affecting hormones.",
        "symptoms": "Hormonal imbalance"
    },
    "notumor": {
        "risk": "No Risk",
        "desc": "No abnormality detected.",
        "symptoms": "None"
    }
}

# ------------------ FAKE MODEL (Replace with your real model) ------------------
import random

def predict_tumor(image_path):
    labels = list(tumor_info.keys())
    result = random.choice(labels)
    confidence = random.uniform(55, 98)
    return result, confidence

# ------------------ GENERATE PDF ------------------
def generate_pdf(result, confidence, image_path):

    safe_name = patient_name.strip().replace(" ", "_") if patient_name else "Patient"
    file_name = f"{REPORT_FOLDER}/Medical_Report_{safe_name}.pdf"

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # ---- HEADER ----
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Brain Tumor Detection Medical Report", ln=True, align="C")

    pdf.set_font("Arial", size=12)
    pdf.cell(0, 8, f"Doctor: {doctor_name}", ln=True)
    pdf.cell(0, 8, f"Contact: {doctor_phone}", ln=True)

    pdf.ln(3)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)

    # ---- PATIENT DETAILS ----
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 10, "Patient Details", ln=True)

    pdf.set_font("Arial", size=12)
    pdf.cell(0, 8, f"Name: {patient_name}", ln=True)
    pdf.cell(0, 8, f"Age: {patient_age}", ln=True)
    pdf.multi_cell(0, 8, f"Address: {patient_address}")
    pdf.cell(0, 8, f"Scan Date: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}", ln=True)

    pdf.ln(5)

    # ---- RESULT ----
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 10, "MRI Analysis Result", ln=True)

    pdf.set_font("Arial", size=12)
    pdf.cell(0, 8, f"Tumor Type: {result.upper()}", ln=True)
    pdf.cell(0, 8, f"Confidence: {confidence:.2f}%", ln=True)
    pdf.cell(0, 8, f"Risk Level: {tumor_info[result]['risk']}", ln=True)

    pdf.multi_cell(0, 8, f"Description: {tumor_info[result]['desc']}")
    pdf.multi_cell(0, 8, f"Symptoms: {tumor_info[result]['symptoms']}")

    # ---------- SIGNATURE ----------
    pdf.ln(15)

    # Doctor Visit → Manual
    if user_type == "Doctor Visit":
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Doctor Signature: ____________________", ln=True, align="R")
        pdf.set_font("Arial", "I", 10)
        pdf.cell(0, 8, f"({doctor_name})", ln=True, align="R")

    # Patient Visit → Digital
    else:
        if os.path.exists(SIGNATURE_PATH):
            y = pdf.get_y()
            pdf.image(SIGNATURE_PATH, x=140, y=y, w=50)
            pdf.ln(25)
            pdf.set_font("Arial", "I", 10)
            pdf.cell(0, 8, "(Digital Signature)", ln=True, align="R")
        else:
            pdf.cell(0, 10, "Doctor Signature: [Not Available]", ln=True, align="R")

    # ---- DISCLAIMER ----
    pdf.ln(3)
    pdf.set_font("Arial", "I", 9)
    pdf.multi_cell(0, 8, "Disclaimer: This is an AI-generated medical assistance report.")

    # ---- MRI IMAGE ----
    if image_path and os.path.exists(image_path):
        pdf.add_page()
        pdf.image(image_path, w=170)

    pdf.output(file_name)

    return file_name

# ------------------ ANALYZE BUTTON ------------------
if st.button("Analyze Image") and uploaded_file is not None:

    image_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)

    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(image_path, caption="Uploaded MRI Image", use_container_width=True)

    # PREDICT
    result, confidence = predict_tumor(image_path)

    st.success(f"Prediction: {result.upper()}")
    st.info(f"Confidence: {confidence:.2f}%")

    # SAVE HISTORY
    record = pd.DataFrame([{
        "Name": patient_name,
        "Age": patient_age,
        "Prediction": result,
        "Date": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
        "Address": patient_address,
        "Confidence": confidence
    }])

    if os.path.exists(HISTORY_FILE) and os.path.getsize(HISTORY_FILE) > 0:
        record.to_csv(HISTORY_FILE, mode='a', header=False, index=False)
    else:
        record.to_csv(HISTORY_FILE, index=False)

    # GENERATE PDF
    pdf_file = generate_pdf(result, confidence, image_path)

    with open(pdf_file, "rb") as f:
        st.download_button("📄 Download Medical Report", f, file_name=os.path.basename(pdf_file))

# ------------------ DOCTOR DASHBOARD ------------------
st.subheader("Doctor Dashboard (History)")

if os.path.exists(HISTORY_FILE) and os.path.getsize(HISTORY_FILE) > 0:
    df = pd.read_csv(HISTORY_FILE)
    st.dataframe(df, use_container_width=True)
else:
    st.info("No patient history available yet.")
