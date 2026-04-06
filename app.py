import streamlit as st
import hashlib
import numpy as np
import tensorflow as tf
import wikipediaapi
import re
import requests
import pandas as pd
import matplotlib.pyplot as plt
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

API_BASE_URL = "http://127.0.0.1:8000"

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
    roles = st.secrets["auth"]["roles"]

    user_dict = {
        usernames[i]: {
            "password_hash": password_hashes[i],
            "role": roles[i]
        }
        for i in range(len(usernames))
    }

    username = normalize_username(username)

    if username in user_dict:
        if user_dict[username]["password_hash"] == hash_password(password):
            return True, user_dict[username]["role"]

    return False, None

def is_valid_patient_id(patient_id):
    return bool(re.fullmatch(r"[A-Za-z0-9]+", patient_id.strip()))

# ---------------- SESSION STATE ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""
    
if "role" not in st.session_state:
    st.session_state.role = None

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
        else:
            login_success, role = check_login(username, password)

            if login_success:
                st.session_state.logged_in = True
                st.session_state.username = normalize_username(username)
                st.session_state.role = role
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
    },
    
    "lab_sidebar_header": {
    "en": "Lab Confirmation",
    "sw": "Uthibitisho wa Maabara"
    },
    
    "lab_sidebar_content": {
    "en": """
        This section is for laboratory confirmation of previously submitted patient records.

        **How to Use:**
        1. Enter the Patient ID to search saved records.
        2. Select the correct visit/record.
        3. Enter the laboratory result and test type.
        4. Save the lab confirmation.

        *Only authorized laboratory personnel should use this section.*
    """,
    "sw": """
        Sehemu hii ni kwa ajili ya kuthibitisha matokeo ya maabara kwa kumbukumbu za wagonjwa zilizohifadhiwa.

        **Jinsi ya Kutumia:**
        1. Weka Patient ID kutafuta kumbukumbu zilizohifadhiwa.
        2. Chagua rekodi sahihi ya mgonjwa.
        3. Weka matokeo ya maabara na aina ya kipimo.
        4. Hifadhi uthibitisho wa maabara.

        *Sehemu hii inapaswa kutumiwa na wahusika wa maabara walioidhinishwa pekee.*
    """
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
def safe_api_request(method, url, payload=None, connection_message="This feature is not yet active in the cloud version of this app."):
    try:
        if method.upper() == "POST":
            response = requests.post(url, json=payload, timeout=10)
        elif method.upper() == "GET":
            response = requests.get(url, timeout=10)
        else:
            st.warning("Unsupported API request method.")
            return None

        if response.status_code == 200:
            return response
        else:
            st.warning(f"Request failed: {response.text}")
            return None

    except requests.exceptions.ConnectionError:
        st.info(connection_message)
        return None

    except requests.exceptions.Timeout:
        st.warning("Database server did not respond in time.")
        return None

    except Exception as e:
        st.warning(f"Unexpected error: {str(e)}")
        return None

def submit_to_database(username, role, patient_id, location, language, selected_symptoms, other_symptoms, prediction, classification):
    url = f"{API_BASE_URL}/submit"

    payload = {
        "username": username,
        "role": role,
        "patient_id": patient_id,
        "location": location,
        "language": language,
        "selected_symptoms": selected_symptoms,
        "other_symptoms": other_symptoms,
        "prediction": float(prediction),
        "classification": classification
    }

    response = safe_api_request(
        method="POST",
        url=url,
        payload=payload,
        connection_message="Database saving is not yet active in the cloud version of this app."
    )

    if response is None:
        return False

    if response.status_code == 200:
        return True
    elif response.status_code == 403:
        st.warning("Database saving is not authorized for your account.")
        return False
    else:
        st.warning(f"Submission failed: {response.text}")
        return False

def get_export_csv():
    url = f"{API_BASE_URL}/export_csv"

    try:
        response = requests.get(
            url,
            params={"role": st.session_state.role},
            timeout=10
        )

        if response.status_code == 200:
            return response.content
        elif response.status_code == 403:
            st.warning("You are not authorized to export the dataset.")
            return None
        else:
            st.error(f"CSV export failed: {response.text}")
            return None

    except Exception:
        st.info("CSV export is not yet active in the cloud version of this app.")
        return None

def get_admin_records(
    location=None,
    classification=None,
    lab_result=None,
    username=None,
    patient_id=None,
    start_date=None,
    end_date=None,
    limit=500
):
    url = f"{API_BASE_URL}/admin_records"

    params = {
        "role": st.session_state.role,
        "limit": limit
    }

    if location:
        params["location"] = location
    if classification:
        params["classification"] = classification
    if lab_result:
        params["lab_result"] = lab_result
    if username:
        params["username"] = username
    if patient_id:
        params["patient_id"] = patient_id
    if start_date:
        params["start_date"] = str(start_date)
    if end_date:
        params["end_date"] = str(end_date)

    try:
        response = requests.get(url, params=params, timeout=15)

        if response.status_code == 200:
            return response.json()
        elif response.status_code == 403:
            st.warning("You are not authorized to access the admin dashboard.")
            return []
        else:
            st.error(f"Admin dashboard query failed: {response.text}")
            return []
    except Exception:
        st.info("Admin dashboard is not yet active in the cloud version of this app.")
        return []

def records_to_csv(records):
    import csv
    import io

    if not records:
        return None

    output = io.StringIO()
    writer = csv.writer(output)

    columns = list(records[0].keys())
    writer.writerow(columns)

    for r in records:
        row = r.copy()
        if isinstance(row.get("selected_symptoms"), list):
            row["selected_symptoms"] = "; ".join(row["selected_symptoms"])
        writer.writerow([row.get(col, "") for col in columns])

    return output.getvalue().encode("utf-8")


def get_dashboard_summary(filters):
    url = f"{API_BASE_URL}/dashboard_summary"

    params = {"role": st.session_state.role}
    params.update(filters)

    try:
        response = requests.get(url, params=params, timeout=10)

        if response.status_code == 200:
            return response.json()
        elif response.status_code == 403:
            st.warning("You are not authorized to view dashboard summary.")
            return None
        else:
            st.error(f"Dashboard summary failed: {response.text}")
            return None

    except Exception:
        st.info("Dashboard summary is not yet active in the cloud version of this app.")
        return None

def get_dashboard_records(filters, page=1, page_size=25):
    url = f"{API_BASE_URL}/dashboard_records"

    params = {
        "role": st.session_state.role,
        "page": page,
        "page_size": page_size
    }
    params.update(filters)

    try:
        response = requests.get(url, params=params, timeout=10)

        if response.status_code == 200:
            return response.json()
        elif response.status_code == 403:
            st.warning("You are not authorized to view dashboard records.")
            return None
        else:
            st.error(f"Dashboard records failed: {response.text}")
            return None

    except Exception:
        st.info("Dashboard records are not yet active in the cloud version of this app.")
        return None

def get_dashboard_filter_options():
    url = f"{API_BASE_URL}/dashboard_filter_options"

    try:
        response = requests.get(
            url,
            params={"role": st.session_state.role},
            timeout=10
        )

        if response.status_code == 200:
            return response.json()
        elif response.status_code == 403:
            st.warning("You are not authorized to view dashboard filters.")
            return None
        else:
            st.error(f"Dashboard filters failed: {response.text}")
            return None

    except Exception:
        st.info("Dashboard filters are not yet active in the cloud version of this app.")
        return None

def get_filtered_export_csv(filters):
    url = f"{API_BASE_URL}/export_csv"

    params = {"role": st.session_state.role}
    params.update(filters)

    try:
        response = requests.get(url, params=params, timeout=15)

        if response.status_code == 200:
            return response.content
        elif response.status_code == 403:
            st.warning("You are not authorized to export filtered data.")
            return None
        else:
            st.error(f"Filtered CSV export failed: {response.text}")
            return None

    except Exception:
        st.info("Filtered CSV export is not yet active in the cloud version of this app.")
        return None

def get_all_records():
    url = f"{API_BASE_URL}/all_records"

    try:
        response = requests.get(
            url,
            params={"role": st.session_state.role},
            timeout=10
        )

        if response.status_code == 200:
            return response.json()
        elif response.status_code == 403:
            st.warning("You are not authorized to view all records.")
            return []
        else:
            st.error(f"Failed to load records: {response.text}")
            return []

    except Exception:
        st.info("Admin dashboard data is not yet active in the cloud version of this app.")
        return []

def search_patient_records(patient_id):
    url = f"{API_BASE_URL}/search_by_patient_id/{patient_id}"

    try:
        response = requests.get(url, params={"role": st.session_state.role})

        if response.status_code == 200:
            return response.json()
        elif response.status_code == 403:
            st.warning("You are not authorized to search patient records.")
            return []
        else:
            st.error(f"Search failed: {response.text}")
            return []
    except Exception:
        st.info("Database search is not yet active in the cloud version of this app.")
        return []
        

def update_lab_result(record_id, lab_result, lab_test_type, confirmed_by):
    url = f"{API_BASE_URL}/update_lab_result"

    payload = {
        "role": st.session_state.role,
        "record_id": record_id,
        "lab_result": lab_result,
        "lab_test_type": lab_test_type,
        "confirmed_by": confirmed_by
    }

    try:
        response = requests.post(url, json=payload)

        if response.status_code == 200:
            return True
        elif response.status_code == 403:
            st.warning("You are not authorized to update lab results.")
            return False
        else:
            st.error(f"Update failed: {response.text}")
            return False
    except Exception:
        st.info("Lab confirmation saving is not yet active in the cloud version of this app.")
        return False

# ---------------- SIDEBAR ----------------
st.sidebar.write(f"**Logged in as:** {st.session_state.username}")
st.sidebar.write(f"Role: **{st.session_state.role}**")

sidebar_language = st.sidebar.radio(
    "🌐 Language / Lugha",
    ["en", "sw"],
    format_func=lambda x: "English" if x == "en" else "Kiswahili",
    index=0
)

sidebar_lang = sidebar_language

# Role-based sidebar content
if st.session_state.role in ["admin", "clinician"]:
     st.sidebar.header(translations["sidebar_header"][sidebar_lang])
     st.sidebar.write(translations["sidebar_content"][sidebar_lang])

elif st.session_state.role == "lab":
     st.sidebar.header(translations["lab_sidebar_header"][sidebar_lang])
     st.sidebar.write(translations["lab_sidebar_content"][sidebar_lang])

# ================= ADMIN TOOLS =================
if st.session_state.role == "admin":
    st.sidebar.markdown("---")
    st.sidebar.subheader("Admin Tools")

    if st.sidebar.button("Prepare Dataset CSV", key="prepare_csv_btn"):
        csv_data = get_export_csv()
        if csv_data is not None:
            st.session_state.export_csv_data = csv_data

    if "export_csv_data" in st.session_state and st.session_state.export_csv_data is not None:
        st.sidebar.download_button(
            label="Download Dataset CSV",
            data=st.session_state.export_csv_data,
            file_name="responses_export.csv",
            mime="text/csv",
            key="download_csv_btn"
        )
        
# ================= LOG OUT =================
if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.role = None
    st.rerun()

# ---------------- MAIN TABS ----------------
allowed_diag = st.session_state.role in ["admin", "clinician"]
allowed_lab = st.session_state.role in ["admin", "lab"]
allowed_admin = st.session_state.role == "admin"

if allowed_diag and allowed_lab and allowed_admin:
    tab_en, tab_sw, tab_lab, tab_admin = st.tabs(
        ["English", "Kiswahili", "Lab Confirmation", "Admin Dashboard"]
    )
elif allowed_diag and allowed_admin:
    tab_en, tab_sw, tab_admin = st.tabs(
        ["English", "Kiswahili", "Admin Dashboard"]
    )
    tab_lab = None
elif allowed_diag:
    tab_en, tab_sw = st.tabs(["English", "Kiswahili"])
    tab_lab = None
    tab_admin = None
elif allowed_lab and allowed_admin:
    tab_lab, tab_admin = st.tabs(["Lab Confirmation", "Admin Dashboard"])
    tab_en = None
    tab_sw = None
elif allowed_lab:
    tab_lab, = st.tabs(["Lab Confirmation"])
    tab_en = None
    tab_sw = None
    tab_admin = None
else:
    st.error("You do not have access to any part of this app.")
    st.stop()

# ================= ENGLISH TAB =================
if tab_en is not None:    
    with tab_en:
        if st.session_state.role not in ["admin", "clinician"]:
            st.warning("You are not authorized to access the diagnosis section.")
            st.stop()
        
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
                        role=st.session_state.role,
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
if tab_sw is not None:
    with tab_sw:
        if st.session_state.role not in ["admin", "clinician"]:
            st.warning("Huna ruhusa ya kutumia sehemu ya uchunguzi.")
            st.stop()
        
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
                        role=st.session_state.role,
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

# ================= LAB CONFIRMATION TAB =================
if tab_lab is not None:
    with tab_lab:
        st.title("Lab Confirmation")

        # Extra protection even if tab is somehow reached
        if st.session_state.role not in ["admin", "lab"]:
            st.warning("You are not authorized to access the Lab Confirmation section.")
            st.stop()

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

# ================= ADMIN DASHBOARD =================
if st.session_state.role == "admin":
    st.markdown("---")
    st.subheader("📊 Admin Dashboard")

    with st.expander("Open Admin Dashboard", expanded=False):
        st.write("Filter, review, and analyze saved diagnosis records.")

        # -------- FILTERS --------
        colf1, colf2, colf3 = st.columns(3)

        with colf1:
            filter_location = st.selectbox(
                "Filter by Location",
                ["All", "Rural", "Peri-Urban", "Urban"],
                key="admin_filter_location_v5"
            )

            filter_lab = st.selectbox(
                "Filter by Lab Result",
                ["All", "Positive", "Negative", "Pending", "Blank"],
                key="admin_filter_lab_v5"
            )

        with colf2:
            filter_classification = st.selectbox(
                "Filter by Classification",
                ["All", "Probably positive for malaria", "Probably negative for malaria"],
                key="admin_filter_classification_v5"
            )

            filter_username = st.text_input(
                "Filter by Username",
                key="admin_filter_username_v5"
            )

        with colf3:
            filter_patient_id = st.text_input(
                "Filter by Patient ID",
                key="admin_filter_patient_id_v5"
            )

            filter_limit = st.selectbox(
                "Max records",
                [100, 250, 500, 1000],
                index=2,
                key="admin_filter_limit_v5"
            )

        col_date1, col_date2 = st.columns(2)
        with col_date1:
            filter_start_date = st.date_input(
                "Start Date",
                key="admin_filter_start_date_v5"
            )
        with col_date2:
            filter_end_date = st.date_input(
                "End Date",
                key="admin_filter_end_date_v5"
            )

        if st.button("🔍 Load Dashboard Data", key="admin_dashboard_load_v5"):
            records = get_admin_records(
                location=None if filter_location == "All" else filter_location,
                classification=None if filter_classification == "All" else filter_classification,
                lab_result=None if filter_lab == "All" else filter_lab,
                username=filter_username.strip() or None,
                patient_id=filter_patient_id.strip() or None,
                start_date=filter_start_date,
                end_date=filter_end_date,
                limit=filter_limit
            )
            st.session_state.admin_dashboard_records = records

        # -------- DISPLAY --------
        if "admin_dashboard_records" in st.session_state:
            records = st.session_state.admin_dashboard_records

            if not records:
                st.info("No records found for the selected filters.")
            else:
                df = pd.DataFrame(records)

                # Make symptoms readable
                if "selected_symptoms" in df.columns:
                    df["selected_symptoms_display"] = df["selected_symptoms"].apply(
                        lambda x: ", ".join(x) if isinstance(x, list) else ""
                    )

                # Safer datetime parsing
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

                if "confirmation_timestamp" in df.columns:
                    df["confirmation_timestamp"] = pd.to_datetime(df["confirmation_timestamp"], errors="coerce")

                # -------- KPI SUMMARY --------
                total_records = len(df)
                positive_pred = (df["classification"] == "Probably positive for malaria").sum() if "classification" in df.columns else 0
                negative_pred = (df["classification"] == "Probably negative for malaria").sum() if "classification" in df.columns else 0
                lab_positive = (df["lab_result"] == "Positive").sum() if "lab_result" in df.columns else 0
                lab_negative = (df["lab_result"] == "Negative").sum() if "lab_result" in df.columns else 0
                lab_pending = ((df["lab_result"] == "Pending") | (df["lab_result"].isna()) | (df["lab_result"] == "")).sum() if "lab_result" in df.columns else 0

                st.markdown("### Summary")
                k1, k2, k3, k4, k5, k6 = st.columns(6)
                k1.metric("Total", total_records)
                k2.metric("Pred +", int(positive_pred))
                k3.metric("Pred -", int(negative_pred))
                k4.metric("Lab +", int(lab_positive))
                k5.metric("Lab -", int(lab_negative))
                k6.metric("Pending", int(lab_pending))

                # -------- ANALYTICS --------
                st.markdown("### 📈 Analytics")

                # Prepare helper columns
                if "classification" in df.columns:
                    df["prediction_label"] = df["classification"].replace({
                        "Probably positive for malaria": "Predicted Positive",
                        "Probably negative for malaria": "Predicted Negative",
                        "Inawezekana una malaria": "Predicted Positive",
                        "Inawezekana huna malaria": "Predicted Negative"
                    })

                if "timestamp" in df.columns:
                    df["date_only"] = df["timestamp"].dt.date

                # ---- Row 1 Charts ----
                c1, c2 = st.columns(2)

                with c1:
                    st.markdown("**Prediction Counts**")
                    if "prediction_label" in df.columns:
                        pred_counts = df["prediction_label"].value_counts()
                        fig, ax = plt.subplots()
                        pred_counts.plot(kind="bar", ax=ax)
                        ax.set_ylabel("Count")
                        ax.set_xlabel("")
                        plt.xticks(rotation=20)
                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        st.info("Prediction data not available.")

                with c2:
                    st.markdown("**Lab Result Counts**")
                    if "lab_result" in df.columns:
                        lab_counts = df["lab_result"].fillna("Blank").replace("", "Blank").value_counts()
                        fig, ax = plt.subplots()
                        lab_counts.plot(kind="bar", ax=ax)
                        ax.set_ylabel("Count")
                        ax.set_xlabel("")
                        plt.xticks(rotation=20)
                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        st.info("Lab result data not available.")

                # ---- Row 2 Charts ----
                c3, c4 = st.columns(2)

                with c3:
                    st.markdown("**Records by Location**")
                    if "location" in df.columns:
                        loc_counts = df["location"].fillna("Unknown").value_counts()
                        fig, ax = plt.subplots()
                        loc_counts.plot(kind="bar", ax=ax)
                        ax.set_ylabel("Count")
                        ax.set_xlabel("")
                        plt.xticks(rotation=20)
                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        st.info("Location data not available.")

                with c4:
                    st.markdown("**Prediction by Location**")
                    if "location" in df.columns and "prediction_label" in df.columns:
                        pred_loc = pd.crosstab(df["location"], df["prediction_label"])
                        fig, ax = plt.subplots()
                        pred_loc.plot(kind="bar", ax=ax)
                        ax.set_ylabel("Count")
                        ax.set_xlabel("")
                        plt.xticks(rotation=20)
                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        st.info("Prediction/location data not available.")

                # ---- Row 3 Charts ----
                c5, c6 = st.columns(2)

                with c5:
                    st.markdown("**Lab Result by Location**")
                    if "location" in df.columns and "lab_result" in df.columns:
                        lab_loc = pd.crosstab(
                            df["location"],
                            df["lab_result"].fillna("Blank").replace("", "Blank")
                        )
                        fig, ax = plt.subplots()
                        lab_loc.plot(kind="bar", ax=ax)
                        ax.set_ylabel("Count")
                        ax.set_xlabel("")
                        plt.xticks(rotation=20)
                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        st.info("Lab/location data not available.")

                with c6:
                    st.markdown("**Records Over Time**")
                    if "date_only" in df.columns:
                        trend = df.groupby("date_only").size().sort_index()
                        fig, ax = plt.subplots()
                        trend.plot(kind="line", marker="o", ax=ax)
                        ax.set_ylabel("Count")
                        ax.set_xlabel("Date")
                        plt.xticks(rotation=30)
                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        st.info("Timestamp data not available.")

                # ---- Agreement Table ----
                st.markdown("### 🧪 Prediction vs Lab Agreement")
                if "prediction_label" in df.columns and "lab_result" in df.columns:
                    agreement = pd.crosstab(
                        df["prediction_label"],
                        df["lab_result"].fillna("Blank").replace("", "Blank")
                    )
                    st.dataframe(agreement, use_container_width=True)
                else:
                    st.info("Agreement table not available.")

                # -------- TABLE VIEW --------
                st.markdown("### Table View")

                display_columns = [
                    "id", "timestamp", "username", "patient_id", "location",
                    "language", "prediction", "classification",
                    "lab_result", "lab_test_type", "confirmed_by",
                    "confirmation_timestamp", "selected_symptoms_display", "other_symptoms"
                ]

                display_columns = [c for c in display_columns if c in df.columns]

                sort_column = st.selectbox(
                    "Sort table by",
                    display_columns,
                    index=1 if "timestamp" in display_columns else 0,
                    key="admin_sort_column_v5"
                )

                sort_ascending = st.checkbox("Sort ascending", value=False, key="admin_sort_ascending_v5")

                df_display = df[display_columns].sort_values(
                    by=sort_column,
                    ascending=sort_ascending,
                    na_position="last"
                )

                st.dataframe(df_display, use_container_width=True, hide_index=True)

                # -------- FILTERED CSV DOWNLOAD --------
                filtered_csv = records_to_csv(records)
                if filtered_csv is not None:
                    st.download_button(
                        label="⬇️ Download Filtered Results as CSV",
                        data=filtered_csv,
                        file_name="filtered_admin_dashboard_export.csv",
                        mime="text/csv",
                        key="admin_filtered_csv_download_v5"
                    )

                # -------- RECORD DETAIL PREVIEW --------
                st.markdown("### Record Detail Preview")

                record_map = {
                    f"ID {r['id']} | {r['patient_id']} | {r['timestamp']}": r
                    for r in records
                }

                selected_record_label = st.selectbox(
                    "Select one record to preview",
                    list(record_map.keys()),
                    key="admin_record_preview_v5"
                )

                selected_record = record_map[selected_record_label]

                st.write(f"**Record ID:** {selected_record.get('id', '')}")
                st.write(f"**Timestamp:** {selected_record.get('timestamp', '')}")
                st.write(f"**Username:** {selected_record.get('username', '')}")
                st.write(f"**Patient ID:** {selected_record.get('patient_id', '')}")
                st.write(f"**Location:** {selected_record.get('location', '')}")
                st.write(f"**Language:** {selected_record.get('language', '')}")
                st.write(f"**Prediction Score:** {selected_record.get('prediction', '')}")
                st.write(f"**Classification:** {selected_record.get('classification', '')}")
                st.write(f"**Symptoms:** {', '.join(selected_record.get('selected_symptoms', [])) if selected_record.get('selected_symptoms') else 'None'}")
                st.write(f"**Other Symptoms:** {selected_record.get('other_symptoms', '') or 'None'}")
                st.write(f"**Lab Result:** {selected_record.get('lab_result', '') or 'Not entered'}")
                st.write(f"**Lab Test Type:** {selected_record.get('lab_test_type', '') or 'Not entered'}")
                st.write(f"**Confirmed By:** {selected_record.get('confirmed_by', '') or 'Not entered'}")
                st.write(f"**Confirmation Timestamp:** {selected_record.get('confirmation_timestamp', '') or 'Not entered'}")