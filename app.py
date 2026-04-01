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
st.set_page_config(page_title="DiagAI", page_icon="🩺", layout="centered")

# ---------------- HELPERS ----------------
def normalize_username(username):
    return username.lower().strip()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def check_login(username, password):
    usernames = [normalize_username(u) for u in st.secrets["auth"]["usernames"]]
    password_hashes = st.secrets["auth"]["password_hashes"]

    user_dict = dict(zip(usernames, password_hashes))
    username = normalize_username(username)

    if username in user_dict:
        return user_dict[username] == hash_password(password)
    return False

def is_valid_patient_id(patient_id):
    return bool(re.fullmatch(r"[A-Za-z0-9]+", patient_id.strip()))

# ---------------- SESSION STATE ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

# ---------------- LOGIN ----------------
if not st.session_state.logged_in:
    st.title("DiagAI Login")
    st.caption("Please log in to continue.")

    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_submitted = st.form_submit_button("Login")

    if login_submitted:
        if not username.strip() or not password.strip():
            st.warning("Please enter both username and password.")
        elif check_login(username, password):
            st.session_state.logged_in = True
            st.session_state.username = normalize_username(username)
            st.success("Login successful")
            st.rerun()
        else:
            st.error("Invalid username or password")

    st.stop()

# ---------------- MODEL ----------------
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

# ---------------- TRANSLATIONS ----------------
translations = {
    "title": {
        "en": "DiagAI/1.0 for Rapid Malaria Diagnosis",
        "sw": "DiagAI/1.0 kwa Uchunguzi wa Haraka wa Malaria"
    },

    "sidebar_header": {
        "en": "About This App",
        "sw": "Kuhusu Programu Hii"
    },

    "sidebar_content": {
        "en": """
DiagAI is a web application designed for rapid disease diagnosis based on symptoms, signs, and patient history input.

This first version of the application utilizes a neural network model that predicts the likelihood of malaria based on the selected symptoms, signs, or patient history.

**How to Use:**
1. Enter Patient ID and select location.
2. Select the symptoms, signs, or history from the dropdown menu.
3. Click the button to check the malaria result.
4. Optionally save the response to the database.

*Please remember that this application is a rapid diagnostic tool and not a substitute for professional medical advice.*
        """,

        "sw": """
DiagAI ni programu ya mtandao iliyoundwa kwa uchunguzi wa haraka wa magonjwa kulingana na dalili, ishara, na historia ya mgonjwa.

Toleo hili la kwanza linatumia mtandao wa neva kutabiri uwezekano wa malaria kwa kuzingatia historia, dalili na ishara zilizoainishwa na mgonjwa au mtabibu wake.

**Maelekezo:**
1. Weka namba ya mgonjwa na uchague mahali.
2. Chagua dalili, ishara au historia kutoka kwenye menyu.
3. Bonyeza kitufe ili kuangalia matokeo ya malaria.
4. Unaweza kuhifadhi taarifa kwenye kanzidata.

*Tafadhali kumbuka kuwa hii programu imeandaliwa kwa ajili ya uchunguzi wa haraka wa malaria na si mbadala wa ushauri wa kitaalamu wa matibabu.*
        """
    },

    "symptoms_prompt": {
        "en": "Select all history, symptoms or signs you have:",
        "sw": "Chagua historia, dalili au ishara zote ulizonazo:"
    },

    "symptoms_placeholder": {
        "en": "Choose options:",
        "sw": "Chagua zinazokuhusu:"
    },

    "button_results": {
        "en": "🐜 Malaria Results",
        "sw": "🐜 Matokeo ya Malaria"
    },

    "positive_result": {
        "en": "Probably positive for malaria",
        "sw": "Inawezekana una malaria"
    },

    "negative_result": {
        "en": "Probably negative for malaria",
        "sw": "Inawezekana huna malaria"
    },

    "send_email_button": {
        "en": "📧 Submit Symptoms",
        "sw": "📧 Tuma Dalili"
    },

    "send_email_warning": {
        "en": "Please describe additional symptoms before sending.",
        "sw": "Tafadhali eleza dalili zaidi kabla ya kutuma."
    },

    "patient_id_label": {
        "en": "Patient ID",
        "sw": "Namba ya Mgonjwa"
    },

    "patient_id_help": {
        "en": "Enter an alphanumeric patient ID (required).",
        "sw": "Weka namba ya mgonjwa yenye herufi na namba (inahitajika)."
    },

    "patient_id_error": {
        "en": "Patient ID is required and must be alphanumeric.",
        "sw": "Namba ya mgonjwa inahitajika na lazima iwe na herufi na namba pekee."
    },

    "location_label": {
        "en": "Location",
        "sw": "Mahali"
    },

    "location_options": {
        "en": ["Rural", "Peri-Urban", "Urban"],
        "sw": ["Vijijini", "Pembezoni mwa Mji", "Mjini"]
    },

    "save_success": {
        "en": "Response saved to database.",
        "sw": "Taarifa zimehifadhiwa kwenye kanzidata."
    },
    "predictive_score_label": {
    "en": "Malaria predictive score",
    "sw": "Alama ya utabiri wa malaria"
    },

    "save_button": {
        "en": "💾 Save Response",
        "sw": "💾 Hifadhi Taarifa"
    }
}

# Swahili display -> English standardized save
location_map_sw_to_en = {
    "Vijijini": "Rural",
    "Pembezoni mwa Mji": "Peri-Urban",
    "Mjini": "Urban"
}

# ---------------- WIKIPEDIA ----------------
def get_wikipedia_summary(disease, num_sentences=5, lang="en"):
    wiki_wiki = wikipediaapi.Wikipedia(language=lang, user_agent="DiagAI/1.0")
    page = wiki_wiki.page(disease)
    if page.exists():
        sentences = re.split(r'(?<=[.!?])\s*', page.summary)
        return " ".join(sentences[:num_sentences])
    else:
        return f"No information found for {disease}."

# ---------------- EMAIL ----------------
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

# ---------------- API SUBMISSION ----------------
def submit_to_database(username, patient_id, location, language, selected_symptoms, other_symptoms, prediction, classification):
    url = "http://127.0.0.1:8000/submit"  # replace later with deployed API URL

    payload = {
        "username": username,
        "patient_id": patient_id,
        "location": location,
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
            st.warning(f"Submission failed: {response.text}")
            return False

    except requests.exceptions.ConnectionError:
        st.info("Database saving is not yet active in the cloud version of this app.")
        return False

    except requests.exceptions.Timeout:
        st.warning("Database server did not respond in time.")
        return False

    except Exception as e:
        st.warning(f"Unexpected error while saving: {str(e)}")
        return False

# ---------------- SIDEBAR ----------------
st.sidebar.write(f"**Logged in as:** {st.session_state.username}")

sidebar_language = st.sidebar.radio(
    "🌐 Language / Lugha",
    ["en", "sw"],
    format_func=lambda x: "English" if x == "en" else "Kiswahili",
    index=0
)

if sidebar_language == "en":
    st.sidebar.header(translations["sidebar_header"]["en"])
    st.sidebar.write(translations["sidebar_content"]["en"])
else:
    st.sidebar.header(translations["sidebar_header"]["sw"])
    st.sidebar.write(translations["sidebar_content"]["sw"])

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.rerun()

# ---------------- MAIN TABS ----------------
tab_en, tab_sw, tab_lab = st.tabs(["English", "Kiswahili", "Lab Confirmation"])

# ================= ENGLISH TAB =================
with tab_en:
    st.title(translations["title"]["en"])

    patient_id_en = st.text_input(
        translations["patient_id_label"]["en"],
        help=translations["patient_id_help"]["en"],
        key="patient_id_en"
    )

    location_en = st.selectbox(
        translations["location_label"]["en"],
        translations["location_options"]["en"],
        key="location_en"
    )

    selected_symptoms_en = st.multiselect(
        translations["symptoms_prompt"]["en"],
        symptoms_en + ["Others"],
        placeholder=translations["symptoms_placeholder"]["en"]
    )

    other_symptoms_en = ""

    if "Others" in selected_symptoms_en:
        other_symptoms_en = st.text_area(
            "Please list any other symptoms or signs you have:",
            key="other_symptoms_en"
        )

        if st.button(translations["send_email_button"]["en"], key="send_email_en"):
            if other_symptoms_en.strip():
                subject = "Additional Symptoms Submitted via App"
                body = f"The user has submitted additional symptoms:\n\n{other_symptoms_en}"
                receiver_email = "diagai2024@gmail.com"

                if send_email(subject, body, receiver_email):
                    st.success("Your symptoms have been sent successfully! Thank you.")
            else:
                st.warning(translations["send_email_warning"]["en"])

    if st.button(translations["button_results"]["en"], key="predict_en"):
        if not is_valid_patient_id(patient_id_en):
            st.error(translations["patient_id_error"]["en"])
        else:
            features = [1 if symptom in selected_symptoms_en else 0 for symptom in symptoms_en]
            prediction = model.predict(np.array(features).reshape(1, -1), verbose=0)[0][0]
            st.write(f"**{translations['predictive_score_label']['en']}:** {prediction * 100:.1f}%")
            st.session_state.prediction_en = float(prediction)
            st.session_state.selected_symptoms_saved_en = selected_symptoms_en
            st.session_state.other_symptoms_saved_en = other_symptoms_en
            st.session_state.patient_id_saved_en = patient_id_en.strip()
            st.session_state.location_saved_en = location_en

            if prediction > 0.43:
                st.session_state.classification_en = "Probably positive for malaria"
                st.success(translations["positive_result"]["en"])
                st.write(f"**Malaria Summary:** {get_wikipedia_summary('malaria', lang='en')}")
            else:
                st.session_state.classification_en = "Probably negative for malaria"
                st.info(translations["negative_result"]["en"])

    if "prediction_en" in st.session_state:
        if st.button(translations["save_button"]["en"], key="save_en"):
            if not is_valid_patient_id(st.session_state.patient_id_saved_en):
                st.error(translations["patient_id_error"]["en"])
            else:
                saved = submit_to_database(
                    username=st.session_state.username,
                    patient_id=st.session_state.patient_id_saved_en,
                    location=st.session_state.location_saved_en,
                    language="English",
                    selected_symptoms=st.session_state.selected_symptoms_saved_en,
                    other_symptoms=st.session_state.other_symptoms_saved_en,
                    prediction=st.session_state.prediction_en,
                    classification=st.session_state.classification_en
                )

                if saved:
                    st.success(translations["save_success"]["en"])
# ================= SWAHILI TAB =================
with tab_sw:
    st.title(translations["title"]["sw"])

    patient_id_sw = st.text_input(
        translations["patient_id_label"]["sw"],
        help=translations["patient_id_help"]["sw"],
        key="patient_id_sw"
    )

    location_sw = st.selectbox(
        translations["location_label"]["sw"],
        translations["location_options"]["sw"],
        key="location_sw"
    )

    selected_symptoms_sw = st.multiselect(
        translations["symptoms_prompt"]["sw"],
        symptoms_sw + ["Dalili Nyingine"],
        placeholder=translations["symptoms_placeholder"]["sw"]
    )

    other_symptoms_sw = ""

    if "Dalili Nyingine" in selected_symptoms_sw:
        other_symptoms_sw = st.text_area(
            "Andika dalili nyingine unazopata",
            key="other_symptoms_sw"
        )

        if st.button(translations["send_email_button"]["sw"], key="send_email_sw"):
            if other_symptoms_sw.strip():
                subject = "Dalili za Ziada Zimetumwa Kupitia Programu"
                body = f"Mtumiaji ametuma dalili zifuatazo:\n\n{other_symptoms_sw}"
                receiver_email = "diagai2024@gmail.com"

                if send_email(subject, body, receiver_email):
                    st.success("Dalili zako zimetumwa kikamilifu! Asante.")
            else:
                st.warning(translations["send_email_warning"]["sw"])

    selected_symptoms_mapped = [
        symptoms_en[symptoms_sw.index(symptom)]
        for symptom in selected_symptoms_sw
        if symptom != "Dalili Nyingine"
    ]

    if st.button(translations["button_results"]["sw"], key="predict_sw"):
        if not is_valid_patient_id(patient_id_sw):
            st.error(translations["patient_id_error"]["sw"])
        else:
            features = [1 if symptom in selected_symptoms_mapped else 0 for symptom in symptoms_en]
            prediction = model.predict(np.array(features).reshape(1, -1), verbose=0)[0][0]
            st.write(f"**{translations['predictive_score_label']['sw']}:** {prediction * 100:.1f}%")
            st.session_state.prediction_sw = float(prediction)
            st.session_state.selected_symptoms_saved_sw = selected_symptoms_sw
            st.session_state.other_symptoms_saved_sw = other_symptoms_sw
            st.session_state.patient_id_saved_sw = patient_id_sw.strip()
            st.session_state.location_saved_sw = location_map_sw_to_en[location_sw]

            if prediction > 0.43:
                st.session_state.classification_sw = "Inawezekana una malaria"
                st.success(translations["positive_result"]["sw"])
                st.write(f"**Muhtasari wa Malaria:** {get_wikipedia_summary('malaria', lang='sw')}")
            else:
                st.session_state.classification_sw = "Inawezekana huna malaria"
                st.info(translations["negative_result"]["sw"])

    if "prediction_sw" in st.session_state:
        if st.button(translations["save_button"]["sw"], key="save_sw"):
            if not is_valid_patient_id(st.session_state.patient_id_saved_sw):
                st.error(translations["patient_id_error"]["sw"])
            else:
                saved = submit_to_database(
                    username=st.session_state.username,
                    patient_id=st.session_state.patient_id_saved_sw,
                    location=st.session_state.location_saved_sw,
                    language="Kiswahili",
                    selected_symptoms=st.session_state.selected_symptoms_saved_sw,
                    other_symptoms=st.session_state.other_symptoms_saved_sw,
                    prediction=st.session_state.prediction_sw,
                    classification=st.session_state.classification_sw
                )

                if saved:
                    st.success(translations["save_success"]["sw"])

def search_patient_records(patient_id):
    url = f"http://127.0.0.1:8000/search_by_patient_id/{patient_id}"

    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json().get("results", [])
        else:
            st.error(f"Search failed: {response.text}")
            return []
    except Exception as e:
        st.error(f"Error searching patient records: {str(e)}")
        return []

def update_lab_result(record_id, lab_result, lab_test_type, confirmed_by):
    url = "http://127.0.0.1:8000/update_lab_result"

    payload = {
        "record_id": record_id,
        "lab_result": lab_result,
        "lab_test_type": lab_test_type,
        "confirmed_by": confirmed_by
    }

    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            return True
        else:
            st.error(f"Update failed: {response.text}")
            return False
    except Exception as e:
        st.error(f"Error updating lab result: {str(e)}")
        return False

# ================= LAB CONFIRMATION TAB =================
with tab_lab:
    st.title("Lab Confirmation")

    st.write("Search for a patient record and update the laboratory confirmation result.")

    patient_search_id = st.text_input(
        "Enter Patient ID to search",
        key="lab_patient_search"
    )

    if st.button("Search Records", key="search_records_btn"):
        if not patient_search_id.strip():
            st.warning("Please enter a Patient ID.")
        elif not is_valid_patient_id(patient_search_id):
            st.warning("Patient ID must be alphanumeric.")
        else:
            records = search_patient_records(patient_search_id.strip())
            st.session_state.lab_search_results = records

    if "lab_search_results" in st.session_state and st.session_state.lab_search_results:
        records = st.session_state.lab_search_results

        record_options = {
            f"Record ID {r['id']} | {r['timestamp']} | Score: {r['prediction']:.3f} | {r['classification']}": r
            for r in records
        }

        selected_label = st.selectbox(
            "Select a record to update",
            list(record_options.keys()),
            key="record_select_lab"
        )

        selected_record = record_options[selected_label]

        st.markdown("### Selected Record")
        st.write(f"**Record ID:** {selected_record['id']}")
        st.write(f"**Patient ID:** {selected_record['patient_id']}")
        st.write(f"**Timestamp:** {selected_record['timestamp']}")
        st.write(f"**Prediction Score:** {selected_record['prediction']:.3f}")
        st.write(f"**Classification:** {selected_record['classification']}")
        st.write(f"**Symptoms:** {', '.join(selected_record['selected_symptoms']) if selected_record['selected_symptoms'] else 'None'}")
        st.write(f"**Other Symptoms:** {selected_record['other_symptoms'] if selected_record['other_symptoms'] else 'None'}")
        st.write(f"**Existing Lab Result:** {selected_record['lab_result'] if selected_record['lab_result'] else 'Not yet entered'}")
        st.write(f"**Existing Test Type:** {selected_record['lab_test_type'] if selected_record['lab_test_type'] else 'Not yet entered'}")

        st.markdown("### Enter Lab Confirmation")

        lab_result = st.selectbox(
            "Lab Result",
            ["Positive", "Negative", "Pending"],
            key="lab_result_input"
        )

        lab_test_type = st.selectbox(
            "Test Type",
            ["Microscopy", "RDT", "PCR", "Clinical only"],
            key="lab_test_type_input"
        )

        if st.button("Save Lab Confirmation", key="save_lab_confirmation"):
            saved = update_lab_result(
                record_id=selected_record["id"],
                lab_result=lab_result,
                lab_test_type=lab_test_type,
                confirmed_by=st.session_state.username
            )

            if saved:
                st.success("Lab confirmation updated successfully.")