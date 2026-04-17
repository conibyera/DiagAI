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
        "en": "DiagAI/1.0 Triage Tool for Malaria Diagnosis",
        "sw": "DiagAI/1.0 Chombo cha Upangaji wa Uchunguzi wa Malaria"
    },

    "sidebar_header": {
        "en": "About This App",
        "sw": "Kuhusu Programu Hii"
    },

    "sidebar_content": {
        "en": """
DiagAI is a web application designed for rapid disease diagnosis and triage based on symptoms, signs, and patient history input.

This first version of the application utilizes a neural network model that predicts the likelihood of malaria for the purpose of triage based on the selected symptoms, signs, or patient history.

**How to Use:**
1. Enter Patient ID and select location.
2. Select the symptoms, signs, or history from the dropdown menu.
3. Click the button to check the malaria result.
4. Optionally save the response to the database.

*Please remember that this application is a rapid diagnostic tool for triage and not a substitute for professional medical advice.*
        """,

        "sw": """
DiagAI ni programu ya mtandao iliyoundwa kwa uchunguzi wa haraka wa magonjwa na upangaji wa uchunguzi zaidi kulingana na dalili, ishara, na historia ya mgonjwa.

Toleo hili la kwanza linatumia mtandao wa neva kutabiri uwezekano wa malaria na upangaji wa uchunguzi zaidi kwa kuzingatia historia, dalili na ishara zilizoainishwa na mgonjwa au mtabibu wake.

**Maelekezo:**
1. Weka namba ya mgonjwa na uchague mahali.
2. Chagua dalili, ishara au historia kutoka kwenye menyu.
3. Bonyeza kitufe ili kuangalia matokeo ya malaria.
4. Unaweza kuhifadhi taarifa kwenye kanzidata.

*Tafadhali kumbuka kuwa hii programu imeandaliwa kwa ajili ya uchunguzi wa haraka wa malaria kwa ajili ya uchunguzi zaidi na si mbadala wa ushauri wa kitaalamu wa matibabu.*
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
    
    "facility_name_label": {
    "en": "Facility Name",
    "sw": "Jina la Kituo cha Afya"
    },

    "facility_name_help": {
        "en": "Enter the name of the health facility.",
        "sw": "Weka jina la kituo cha afya."
    },

    "facility_name_error": {
        "en": "Facility name is required.",
        "sw": "Jina la kituo cha afya linahitajika."
    },

    "date_of_birth_label": {
        "en": "Date of Birth",
        "sw": "Tarehe ya Kuzaliwa"
    },

    "date_of_birth_help": {
        "en": "Select the patient's date of birth.",
        "sw": "Chagua tarehe ya kuzaliwa ya mgonjwa."
    },

    "sex_label": {
        "en": "Sex",
        "sw": "Jinsia"
    },

    "sex_options": {
        "en": ["Male", "Female"],
        "sw": ["Mwanaume", "Mwanamke"]
    },

    "sex_error": {
        "en": "Sex is required.",
        "sw": "Jinsia inahitajika."
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

sex_map_sw_to_en = {
    "Mwanaume": "Male",
    "Mwanamke": "Female"
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

def submit_to_database(
    username,
    role,
    patient_id,
    facility_name,
    date_of_birth,
    sex,
    location,
    language,
    selected_symptoms,
    other_symptoms,
    prediction,
    classification
):
    url = f"{API_BASE_URL}/submit"

    payload = {
        "username": username,
        "role": role,
        "patient_id": patient_id,
        "facility_name": facility_name,
        "date_of_birth": str(date_of_birth),
        "sex": sex,
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

def get_filtered_records(
    location="",
    classification="",
    lab_result="",
    sex="",
    facility_name="",
    username="",
    keyword="",
    start_date="",
    end_date="",
    sort_by="timestamp",
    sort_order="desc",
    limit=500
):
    url = f"{API_BASE_URL}/filtered_records"

    params = {
        "role": st.session_state.role,
        "location": location,
        "classification": classification,
        "lab_result": lab_result,
        "sex": sex,
        "facility_name": facility_name,
        "username": username,
        "keyword": keyword,
        "start_date": start_date,
        "end_date": end_date,
        "sort_by": sort_by,
        "sort_order": sort_order,
        "limit": limit
    }

    try:
        response = requests.get(url, params=params, timeout=15)

        if response.status_code == 200:
            return response.json()
        elif response.status_code == 403:
            st.warning("You are not authorized to view dashboard records.")
            return []
        else:
            st.warning(f"Failed to load filtered records: {response.text}")
            return []

    except Exception:
        st.info("Admin dashboard is not yet active in the cloud version of this app.")
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

        facility_name_en = st.text_input(
            translations["facility_name_label"]["en"],
            help=translations["facility_name_help"]["en"],
            key="facility_name_en"
        )

        date_of_birth_en = st.date_input(
            translations["date_of_birth_label"]["en"],
            help=translations["date_of_birth_help"]["en"],
            key="date_of_birth_en"
        )

        sex_en = st.selectbox(
            translations["sex_label"]["en"],
            translations["sex_options"]["en"],
            key="sex_en"
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
            elif not facility_name_en.strip():
                st.error(translations["facility_name_error"]["en"])
            else:
                features = [1 if symptom in selected_symptoms_en else 0 for symptom in symptoms_en]
                prediction = model.predict(np.array(features).reshape(1, -1), verbose=0)[0][0]
                st.write(f"**{translations['predictive_score_label']['en']}:** {prediction * 100:.1f}%")
                st.session_state.prediction_en = float(prediction)
                st.session_state.selected_symptoms_saved_en = selected_symptoms_en
                st.session_state.other_symptoms_saved_en = other_symptoms_en
                st.session_state.patient_id_saved_en = patient_id_en.strip()
                st.session_state.facility_name_saved_en = facility_name_en.strip()
                st.session_state.date_of_birth_saved_en = str(date_of_birth_en)
                st.session_state.sex_saved_en = sex_en
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
                        facility_name=st.session_state.facility_name_saved_en,
                        date_of_birth=st.session_state.date_of_birth_saved_en,
                        sex=st.session_state.sex_saved_en,
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

        facility_name_sw = st.text_input(
            translations["facility_name_label"]["sw"],
            help=translations["facility_name_help"]["sw"],
            key="facility_name_sw"
        )

        date_of_birth_sw = st.date_input(
            translations["date_of_birth_label"]["sw"],
            help=translations["date_of_birth_help"]["sw"],
            key="date_of_birth_sw"
        )

        sex_sw = st.selectbox(
            translations["sex_label"]["sw"],
            translations["sex_options"]["sw"],
            key="sex_sw"
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
            elif not facility_name_sw.strip():
                st.error(translations["facility_name_error"]["sw"])
            else:
                features = [1 if symptom in selected_symptoms_mapped else 0 for symptom in symptoms_en]
                prediction = model.predict(np.array(features).reshape(1, -1), verbose=0)[0][0]
                st.write(f"**{translations['predictive_score_label']['sw']}:** {prediction * 100:.1f}%")
                st.session_state.prediction_sw = float(prediction)
                st.session_state.selected_symptoms_saved_sw = selected_symptoms_sw
                st.session_state.other_symptoms_saved_sw = other_symptoms_sw
                st.session_state.patient_id_saved_sw = patient_id_sw.strip()
                st.session_state.facility_name_saved_sw = facility_name_sw.strip()
                st.session_state.date_of_birth_saved_sw = str(date_of_birth_sw)
                st.session_state.sex_saved_sw = sex_map_sw_to_en[sex_sw]
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
                        facility_name=st.session_state.facility_name_saved_sw,
                        date_of_birth=st.session_state.date_of_birth_saved_sw,
                        sex=st.session_state.sex_saved_sw,
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
            st.write(f"**Facility Name:** {selected_record.get('facility_name', '') or 'Not entered'}")
            st.write(f"**Date of Birth:** {selected_record.get('date_of_birth', '') or 'Not entered'}")
            st.write(f"**Sex:** {selected_record.get('sex', '') or 'Not entered'}")
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

    st.caption("Search, filter, review, and export submitted diagnosis and lab records.")

    # ---------- Load Data ----------
    admin_records = get_all_records()

    if not admin_records:
        st.info("No records found or dashboard is not connected to the database server.")
    else:
        import pandas as pd

        df = pd.DataFrame(admin_records)

        # ---------- Safe cleanup ----------
        if "selected_symptoms" in df.columns:
            df["selected_symptoms_display"] = df["selected_symptoms"].apply(
                lambda x: ", ".join(x) if isinstance(x, list) else ""
            )
        else:
            df["selected_symptoms_display"] = ""

        # Ensure expected columns exist
        expected_cols = [
            "id", "timestamp", "username", "patient_id", "facility_name",
            "date_of_birth", "sex", "location", "language", "prediction",
            "classification", "lab_result", "lab_test_type", "confirmed_by",
            "confirmation_timestamp", "selected_symptoms_display", "other_symptoms"
        ]

        for col in expected_cols:
            if col not in df.columns:
                df[col] = None

        # ---------- Datetime handling ----------
        if "timestamp" in df.columns:
            df["timestamp_dt"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df["date_only"] = df["timestamp_dt"].dt.date
        else:
            df["timestamp_dt"] = pd.NaT
            df["date_only"] = None

        if "confirmation_timestamp" in df.columns:
            df["confirmation_timestamp_dt"] = pd.to_datetime(df["confirmation_timestamp"], errors="coerce")
        else:
            df["confirmation_timestamp_dt"] = pd.NaT

        # ---------- Numeric cleanup ----------
        if "prediction" in df.columns:
            df["prediction_percent"] = pd.to_numeric(df["prediction"], errors="coerce") * 100
        else:
            df["prediction_percent"] = None

        # ---------- Top metrics ----------
        total_records = len(df)
        positive_cases = (df["classification"] == "Probably positive for malaria").sum() + \
                         (df["classification"] == "Inawezekana una malaria").sum()

        negative_cases = (df["classification"] == "Probably negative for malaria").sum() + \
                         (df["classification"] == "Inawezekana huna malaria").sum()

        confirmed_positive = (df["lab_result"] == "Positive").sum()
        confirmed_negative = (df["lab_result"] == "Negative").sum()
        pending_lab = df["lab_result"].isna().sum() + (df["lab_result"] == "").sum() + (df["lab_result"] == "Pending").sum()

        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Total", total_records)
        col2.metric("Pred +", int(positive_cases))
        col3.metric("Pred -", int(negative_cases))
        col4.metric("Lab +", int(confirmed_positive))
        col5.metric("Lab -", int(confirmed_negative))
        col6.metric("Pending", int(pending_lab))

        st.markdown("### 🔎 Search & Filters")

        # ---------- Search ----------
        quick_search = st.text_input(
            "Quick Search",
            placeholder="Search patient ID, username, facility, symptoms, or notes...",
            key="admin_quick_search_v9"
        )

        # ---------- Filters ----------
        filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)

        with filter_col1:
            location_options = sorted([x for x in df["location"].dropna().unique().tolist() if x != ""])
            location_filter = st.multiselect("Location", location_options, key="admin_location_filter_v9")

        with filter_col2:
            classification_options = sorted([x for x in df["classification"].dropna().unique().tolist() if x != ""])
            classification_filter = st.multiselect("Classification", classification_options, key="admin_classification_filter_v9")

        with filter_col3:
            lab_result_options = sorted([x for x in df["lab_result"].dropna().unique().tolist() if x != ""])
            lab_result_filter = st.multiselect("Lab Result", lab_result_options, key="admin_lab_result_filter_v9")

        with filter_col4:
            sex_options = sorted([x for x in df["sex"].dropna().unique().tolist() if x != ""])
            sex_filter = st.multiselect("Sex", sex_options, key="admin_sex_filter_v9")

        # ---------- Date Filter ----------
        st.markdown("#### 📅 Date Filter")
        date_col1, date_col2 = st.columns(2)

        min_date = df["date_only"].dropna().min() if df["date_only"].notna().any() else None
        max_date = df["date_only"].dropna().max() if df["date_only"].notna().any() else None

        with date_col1:
            start_date = st.date_input(
                "Start Date",
                value=min_date if min_date else None,
                key="admin_start_date_v9"
            )

        with date_col2:
            end_date = st.date_input(
                "End Date",
                value=max_date if max_date else None,
                key="admin_end_date_v9"
            )

        # ---------- Apply Filters ----------
        filtered_df = df.copy()

        if quick_search.strip():
            q = quick_search.strip().lower()

            search_cols = [
                "patient_id", "username", "facility_name", "selected_symptoms_display",
                "other_symptoms", "classification", "lab_result", "confirmed_by"
            ]

            mask = pd.Series(False, index=filtered_df.index)
            for col in search_cols:
                if col in filtered_df.columns:
                    mask = mask | filtered_df[col].fillna("").astype(str).str.lower().str.contains(q, na=False)

            filtered_df = filtered_df[mask]

        if location_filter:
            filtered_df = filtered_df[filtered_df["location"].isin(location_filter)]

        if classification_filter:
            filtered_df = filtered_df[filtered_df["classification"].isin(classification_filter)]

        if lab_result_filter:
            filtered_df = filtered_df[filtered_df["lab_result"].isin(lab_result_filter)]

        if sex_filter:
            filtered_df = filtered_df[filtered_df["sex"].isin(sex_filter)]

        if min_date is not None and max_date is not None:
            filtered_df = filtered_df[
                (filtered_df["date_only"] >= start_date) &
                (filtered_df["date_only"] <= end_date)
            ]

        st.caption(f"Showing **{len(filtered_df)}** of **{len(df)}** total records.")

        # ---------- Sort ----------
        st.markdown("### ↕️ Sort & Display")

        display_columns = [
            "id",
            "timestamp",
            "username",
            "patient_id",
            "facility_name",
            "date_of_birth",
            "sex",
            "location",
            "language",
            "prediction_percent",
            "classification",
            "lab_result",
            "lab_test_type",
            "confirmed_by",
            "confirmation_timestamp",
            "selected_symptoms_display",
            "other_symptoms"
        ]

        display_columns = [c for c in display_columns if c in filtered_df.columns]

        sort_col1, sort_col2 = st.columns([2, 1])

        with sort_col1:
            sort_column = st.selectbox(
                "Sort table by",
                display_columns,
                index=1 if "timestamp" in display_columns else 0,
                key="admin_sort_column_v9"
            )

        with sort_col2:
            sort_ascending = st.checkbox("Ascending", value=False, key="admin_sort_ascending_v9")

        filtered_df = filtered_df.sort_values(
            by=sort_column,
            ascending=sort_ascending,
            na_position="last"
        )

        # ---------- Friendly display names ----------
        rename_map = {
            "id": "Record ID",
            "timestamp": "Submitted At",
            "username": "Submitted By",
            "patient_id": "Patient ID",
            "facility_name": "Facility",
            "date_of_birth": "Date of Birth",
            "sex": "Sex",
            "location": "Location",
            "language": "Language",
            "prediction_percent": "Prediction Score (%)",
            "classification": "Classification",
            "lab_result": "Lab Result",
            "lab_test_type": "Lab Test Type",
            "confirmed_by": "Confirmed By",
            "confirmation_timestamp": "Confirmed At",
            "selected_symptoms_display": "Symptoms",
            "other_symptoms": "Other Symptoms"
        }

        df_display = filtered_df[display_columns].rename(columns=rename_map)

        # Round prediction display
        if "Prediction Score (%)" in df_display.columns:
            df_display["Prediction Score (%)"] = pd.to_numeric(
                df_display["Prediction Score (%)"], errors="coerce"
            ).round(1)

        # ---------- Table View ----------
        st.markdown("### 📋 Records Table")
        st.dataframe(df_display, use_container_width=True, hide_index=True)

        # ---------- Download Filtered CSV ----------
        st.markdown("### ⬇️ Export Filtered Results")
        filtered_csv = filtered_df[display_columns].copy()
        filtered_csv = filtered_csv.rename(columns=rename_map).to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Filtered Table as CSV",
            data=filtered_csv,
            file_name="filtered_admin_dashboard_records.csv",
            mime="text/csv",
            key="download_filtered_dashboard_csv_v9"
        )

        # ---------- Record Detail Preview ----------
        st.markdown("### 🧾 Record Detail Preview")

        if not filtered_df.empty:
            preview_options = {
                f"Record {row['id']} | Patient {row['patient_id']} | {row['timestamp']}": row["id"]
                for _, row in filtered_df.iterrows()
            }

            selected_preview_label = st.selectbox(
                "Select a record to preview",
                list(preview_options.keys()),
                key="admin_record_preview_select_v9"
            )

            selected_record_id = preview_options[selected_preview_label]
            selected_row = filtered_df[filtered_df["id"] == selected_record_id].iloc[0]

            preview_col1, preview_col2 = st.columns(2)

            with preview_col1:
                st.markdown("#### Patient / Visit Info")
                st.write(f"**Record ID:** {selected_row['id']}")
                st.write(f"**Patient ID:** {selected_row['patient_id']}")
                st.write(f"**Facility:** {selected_row['facility_name'] or '—'}")
                st.write(f"**Date of Birth:** {selected_row['date_of_birth'] or '—'}")
                st.write(f"**Sex:** {selected_row['sex'] or '—'}")
                st.write(f"**Location:** {selected_row['location'] or '—'}")
                st.write(f"**Language:** {selected_row['language'] or '—'}")
                st.write(f"**Submitted By:** {selected_row['username'] or '—'}")
                st.write(f"**Submitted At:** {selected_row['timestamp'] or '—'}")

            with preview_col2:
                st.markdown("#### Diagnosis / Lab Info")
                pred_pct = selected_row["prediction_percent"]
                pred_pct_text = f"{pred_pct:.1f}%" if pd.notna(pred_pct) else "—"
                st.write(f"**Prediction Score:** {pred_pct_text}")
                st.write(f"**Classification:** {selected_row['classification'] or '—'}")
                st.write(f"**Lab Result:** {selected_row['lab_result'] or '—'}")
                st.write(f"**Lab Test Type:** {selected_row['lab_test_type'] or '—'}")
                st.write(f"**Confirmed By:** {selected_row['confirmed_by'] or '—'}")
                st.write(f"**Confirmed At:** {selected_row['confirmation_timestamp'] or '—'}")

            st.markdown("#### Symptoms")
            st.write(selected_row["selected_symptoms_display"] if selected_row["selected_symptoms_display"] else "—")

            st.markdown("#### Other Symptoms / Notes")
            st.write(selected_row["other_symptoms"] if selected_row["other_symptoms"] else "—")