import streamlit as st
import hashlib
import numpy as np
import tensorflow as tf
import wikipediaapi
import re
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="DiagAI", layout="centered")

# ---------------- AUTH ----------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def check_login(username, password):
    usernames = st.secrets["auth"]["usernames"]
    password_hashes = st.secrets["auth"]["password_hashes"]

    user_dict = dict(zip(usernames, password_hashes))
    return username in user_dict and user_dict[username] == hash_password(password)

# ---------------- SESSION STATE ----------------
defaults = {
    "logged_in": False,
    "username": "",
    "prediction_en": None,
    "classification_en": None,
    "prediction_sw": None,
    "classification_sw": None
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ---------------- LOGIN PAGE ----------------
if not st.session_state.logged_in:
    st.title("DiagAI Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if check_login(username, password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("Login successful")
            st.rerun()
        else:
            st.error("Invalid username or password")

    st.stop()

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("malar_model.keras")

model = load_model()

# ---------------- SYMPTOMS ----------------
symptoms_en = [
    "Fever", "Vomiting", "Cough", "Diarrhoea", "Headache", "Body Pain",
    "Abdominal Pain", "Loss of Appetite", "Body Weakness", "Blood in Urine",
    "Dizziness", "Epigastric Pain", "Eye Pain", "Fungal Infection", "Generalized Rash",
    "Joint Pain", "Numbness", "Pain Urinating", "Palpitations", "Vaginal Discharge",
    "Runny Nose", "Scabies", "Chest Pain", "Ear Pain", "Back Pain", "Treated for Malaria Recently"
]

symptoms_sw = [
    "Homa", "Kutapika", "Kikohozi", "Kuhara", "Kichwa Kuuma", "Maumivu ya Mwili",
    "Maumivu ya Tumbo", "Kupoteza Hamu ya Kula", "Udhaifu wa Mwili", "Damu Katika Mkojo",
    "Kizunguzungu", "Maumivu ya Epigastriki", "Maumivu ya Macho", "Maambukizi ya Kuvu",
    "Upele wa Mwili", "Maumivu ya Viungo", "Kufa Ganzi", "Maumivu Wakati wa Mkojo",
    "Mapigo ya Moyo Kasi", "Uchafu wa Uke", "Mafua", "Kaskasi", "Maumivu ya Kifua",
    "Maumivu ya Sikio", "Maumivu ya Mgongo", "Umetibiwa Malaria Karibuni"
]

# ---------------- UI TEXT ----------------
translations = {
    "title": {
        "en": "DiagAI/1.0 for Rapid Malaria Diagnosis",
        "sw": "DiagAI/1.0 kwa Uchunguzi wa Haraka wa Malaria"
    },
    "sidebar_content": {
        "en": """
DiagAI is a web application designed for rapid disease diagnosis based on symptoms, signs, and patient history input.

This first version of the application utilizes a neural network model that predicts the likelihood of malaria based on the selected symptoms, signs, or patient history.

**How to Use:**
1. Select the symptoms and signs you are experiencing or history from the dropdown menu.
2. Click **Get Malaria Results**.
3. Review the prediction.
4. Optionally save the response to the database.

*This application is a rapid screening tool and not a substitute for professional medical advice.*
""",
        "sw": """
DiagAI ni programu ya mtandao iliyoundwa kwa uchunguzi wa haraka wa magonjwa kulingana na dalili, ishara, na historia ya mgonjwa.

Toleo hili la kwanza linatumia mtandao wa neva kutabiri uwezekano wa malaria kwa kuzingatia historia, dalili na ishara zilizoainishwa na mgonjwa au mtabibu wake.

**Maelekezo:**
1. Chagua dalili, ishara au historia kuhusiana na ugonjwa wako kutoka kwenye menyu.
2. Bonyeza **Matokeo ya Malaria**.
3. Angalia matokeo.
4. Unaweza kuhifadhi taarifa kwenye kanzidata.

*Programu hii ni chombo cha uchunguzi wa haraka na si mbadala wa ushauri wa kitaalamu wa matibabu.*
"""
    }
}

# ---------------- HELPERS ----------------
def get_wikipedia_summary(disease, num_sentences=5, lang="en"):
    wiki_wiki = wikipediaapi.Wikipedia(language=lang, user_agent="DiagAI/1.0")
    page = wiki_wiki.page(disease)
    if page.exists():
        sentences = re.split(r'(?<=[.!?])\s*', page.summary)
        return " ".join(sentences[:num_sentences])
    return f"No information found for {disease}."

def send_email(subject, body, receiver_email):
    sender_email = st.secrets["email"]["sender_email"]
    sender_password = st.secrets["email"]["sender_password"]

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
        return True
    except Exception as e:
        st.error(f"Error sending email: {str(e)}")
        return False

def submit_to_database(username, language, selected_symptoms, other_symptoms, prediction, classification):
    url = "http://127.0.0.1:8000/submit"   # replace later with deployed API URL

    payload = {
        "username": username,
        "language": language,
        "selected_symptoms": selected_symptoms,
        "other_symptoms": other_symptoms,
        "prediction": float(prediction),
        "classification": classification
    }

    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            return True
        else:
            st.error(f"Submission failed: {response.text}")
            return False
    except Exception as e:
        st.error(f"Error connecting to database server: {str(e)}")
        return False

# ---------------- SIDEBAR ----------------
st.sidebar.write(f"Logged in as: **{st.session_state.username}**")
st.sidebar.header("About This App / Kuhusu Programu Hii")
st.sidebar.write(translations["sidebar_content"]["en"])

if st.sidebar.button("Logout"):
    for key in defaults:
        st.session_state[key] = defaults[key]
    st.rerun()

# ---------------- TABS ----------------
tab_en, tab_sw = st.tabs(["English", "Kiswahili"])

# =========================================================
# ENGLISH TAB
# =========================================================
with tab_en:
    st.title(translations["title"]["en"])

    selected_symptoms_en = st.multiselect(
        "Select all history, symptoms or signs you have:",
        symptoms_en + ["Others"],
        placeholder="Choose options:"
    )

    other_symptoms_en = ""
    if "Others" in selected_symptoms_en:
        other_symptoms_en = st.text_area("Please list any other symptoms or signs you have:")

        if st.button("📧 Submit Symptoms", key="email_en"):
            if other_symptoms_en.strip():
                subject = "Additional Symptoms Submitted via App"
                body = f"The user has submitted additional symptoms:\n\n{other_symptoms_en}"
                if send_email(subject, body, "diagai2024@gmail.com"):
                    st.success("Your symptoms have been sent successfully.")
            else:
                st.warning("Please describe additional symptoms before sending.")

    if st.button("🐜 Get Malaria Results", key="predict_en"):
        features = [1 if symptom in selected_symptoms_en else 0 for symptom in symptoms_en]
        prediction = model.predict(np.array(features).reshape(1, -1), verbose=0)[0][0]

        st.session_state.prediction_en = float(prediction)

        if prediction > 0.43:
            st.session_state.classification_en = "Probably positive for malaria"
        else:
            st.session_state.classification_en = "Probably negative for malaria"

    # Show saved result if available
    if st.session_state.prediction_en is not None:
        if st.session_state.classification_en == "Probably positive for malaria":
            st.success(st.session_state.classification_en)
            st.write(f"**Malaria Summary:** {get_wikipedia_summary('malaria', lang='en')}")
        else:
            st.info(st.session_state.classification_en)

        st.write(f"**Prediction Score:** {st.session_state.prediction_en:.3f}")

        if st.button("💾 Save Response", key="save_en"):
            saved = submit_to_database(
                username=st.session_state.username,
                language="English",
                selected_symptoms=selected_symptoms_en,
                other_symptoms=other_symptoms_en,
                prediction=st.session_state.prediction_en,
                classification=st.session_state.classification_en
            )

            if saved:
                st.success("Response saved to database.")

# =========================================================
# SWAHILI TAB
# =========================================================
with tab_sw:
    st.title(translations["title"]["sw"])

    selected_symptoms_sw = st.multiselect(
        "Chagua historia, dalili au ishara zote ulizonazo:",
        symptoms_sw + ["Dalili Nyingine"],
        placeholder="Chagua zinazokuhusu:"
    )

    other_symptoms_sw = ""
    if "Dalili Nyingine" in selected_symptoms_sw:
        other_symptoms_sw = st.text_area("Andika dalili nyingine unazopata")

        if st.button("📧 Tuma Dalili", key="email_sw"):
            if other_symptoms_sw.strip():
                subject = "Dalili za Ziada Zimetumwa Kupitia Programu"
                body = f"Mtumiaji ametuma dalili zifuatazo:\n\n{other_symptoms_sw}"
                if send_email(subject, body, "diagai2024@gmail.com"):
                    st.success("Dalili zako zimetumwa kikamilifu.")
            else:
                st.warning("Tafadhali andika dalili nyingine kabla ya kutuma ujumbe.")

    selected_symptoms_mapped = [
        symptoms_en[symptoms_sw.index(symptom)]
        for symptom in selected_symptoms_sw
        if symptom != "Dalili Nyingine"
    ]

    if st.button("🐜 Matokeo ya Malaria", key="predict_sw"):
        features = [1 if symptom in selected_symptoms_mapped else 0 for symptom in symptoms_en]
        prediction = model.predict(np.array(features).reshape(1, -1), verbose=0)[0][0]

        st.session_state.prediction_sw = float(prediction)

        if prediction > 0.24:
            st.session_state.classification_sw = "Inawezekana una malaria"
        else:
            st.session_state.classification_sw = "Inawezekana huna malaria"

    # Show saved result if available
    if st.session_state.prediction_sw is not None:
        if st.session_state.classification_sw == "Inawezekana una malaria":
            st.success(st.session_state.classification_sw)
            st.write(f"**Muhtasari wa Malaria:** {get_wikipedia_summary('malaria', lang='sw')}")
        else:
            st.info(st.session_state.classification_sw)

        st.write(f"**Prediction Score:** {st.session_state.prediction_sw:.3f}")

        if st.button("💾 Hifadhi Taarifa", key="save_sw"):
            saved = submit_to_database(
                username=st.session_state.username,
                language="Kiswahili",
                selected_symptoms=selected_symptoms_sw,
                other_symptoms=other_symptoms_sw,
                prediction=st.session_state.prediction_sw,
                classification=st.session_state.classification_sw
            )

            if saved:
                st.success("Taarifa zimehifadhiwa kwenye kanzidata.")