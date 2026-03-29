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

# ---------------- AUTH ----------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def check_login(username, password):
    usernames = st.secrets["auth"]["usernames"]
    password_hashes = st.secrets["auth"]["password_hashes"]

    user_dict = dict(zip(usernames, password_hashes))

    if username in user_dict:
        return user_dict[username] == hash_password(password)
    return False

# Session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

# Login page
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

# Load the saved model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("malar_model.keras")

model = load_model()

#model = tf.keras.models.load_model('malar_model.keras')

# Define symptoms in English and Swahili
symptoms_en = [
    "Fever", "Vomiting", "Cough", "Diarrhoea", "Headache", "Body Pain",
    "Abdominal Pain", "Loss of Appetite", "Body Weakness", "Blood in Urine",
    "Dizziness", "Epigastric Pain", "Eye Pain", "Fungal Infection", "Generalized Rash",
    "Joint Pain", "Numbness", "Pain Urinating", "Palpitations", "Vaginal Discharge",
    "Runny Nose", "Scabies", "Chest Pain","Ear Pain","Back Pain", "Treated for Malaria Recently"
]

symptoms_sw = [
    "Homa", "Kutapika", "Kikohozi", "Kuhara", "Kichwa Kuuma", "Maumivu ya Mwili",
    "Maumivu ya Tumbo", "Kupoteza Hamu ya Kula", "Udhaifu wa Mwili", "Damu Katika Mkojo",
    "Kizunguzungu", "Maumivu ya Epigastriki", "Maumivu ya Macho", "Maambukizi ya Kuvu",
    "Upele wa Mwili", "Maumivu ya Viungo", "Kufa Ganzi", "Maumivu Wakati wa Mkojo", 
    "Mapigo ya Moyo Kasi", "Uchafu wa Uke", "Mafua", "Kaskasi", "Maumivu ya Kifua", 
    "Maumivu ya Sikio", "Maumivu ya Mgongo","Umetibiwa Malaria Karibuni"
]

# UI Translations
translations = {
    "title": {"en": "DiagAI/1.0 for Rapid Malaria Diagnosis", "sw": "DiagAI/1.0 kwa Uchunguzi wa Haraka wa Malaria"},
    "sidebar_header": {"en": "About This App", "sw": "Kuhusu Programu Hii"},
    "sidebar_content": {
        "en": """
            DiagAI is a web application designed for rapid disease diagnosis based on symptoms, signs, and patient history input.
            
            This first version of the application utilizes a neural network model that predicts the likelihood of malaria based on the selected symptoms, signs, or patient history.
            
            **How to Use:**
            1. Select the symptoms and signs you are experiencing or history from the dropdown menu.
            2. Click on the buttons to check the disease status for malaria.
            3. The app will indicate whether you are probably positive or negative for malaria.
    
            *Please remember that this application is a rapid diagnostic tool and not a substitute for professional medical advice.*
        """,
        "sw": """
            DiagAI ni programu ya mtandao iliyoundwa kwa uchunguzi wa haraka wa magonjwa kulingana na dalili, ishara, na historia ya mgonjwa.
            
            Toleo hili la kwanza linatumia mtandao wa neva kutabiri uwezekano wa malaria kwa kuzingatia historia, dalili na ishara zilizoainishwa na mgonjwa au mtabibu wake.
            
            **Maelekezo:**
            1. Chagua dalili, ishara au historia kuhusiana na ugonjwa wako kutoka kwenye menyu.
            2. Bonyeza kitufe ili kuangalia kama una uwezekano wa malaria.
            3. Programu itakuonyesha kama una uwezekano wa kuwa na malaria au la.
    
            *Tafadhali kumbuka kuwa hii programu imeandaliwa kwa ajili ya uchunguzi wa haraka wa malaria na si mbadala wa ushauri wa kitaalamu wa matibabu.*
        """
    },
    "symptoms_prompt": {"en": "Select all history, symptoms or signs you have:", "sw": "Chagua historia, dalili au ishara zote ulizonazo:"},
    "symptoms_placeholder": {"en": "Choose options:", "sw": "Chagua zinazokuhusu:"},
    "button_results": {"en": "🐜Malaria Results", "sw": "🐜Matokeo ya Malaria"},
    "positive_result": {"en": "Probably positive for malaria", "sw": "Inawezekana una malaria"},
    "negative_result": {"en": "Probably negative for malaria", "sw": "Inawezekana huna malaria"},
    "send_email_button": {"en": "📧 Submit Symptoms", "sw": "📧 Tuma Dalili"},
    "send_email_warning": {"en": "Please describe additional symptoms before sending.", "sw": "Tafadhali eleza dalili zaidi kabla ya kutuma."}
}

# Function to fetch Wikipedia summary
def get_wikipedia_summary(disease, num_sentences=5, lang="en"):
    wiki_wiki = wikipediaapi.Wikipedia(language=lang, user_agent='DiagAI/1.0')
    page = wiki_wiki.page(disease)
    if page.exists():
        sentences = re.split(r'(?<=[.!?])\s*', page.summary)
        return ' '.join(sentences[:num_sentences])
    else:
        return f"No information found for {disease}."

# Function to send email
def send_email(subject, body, receiver_email):
    sender_email = st.secrets["email"]["sender_email"]
    sender_password = st.secrets["email"]["sender_password"]

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
        return True
    except Exception as e:
        st.error(f"Error sending email: {str(e)}")
        return False

# App Layout with Tabs
tab_en, tab_sw = st.tabs(["English", "Kiswahili"])

#Logged in
st.sidebar.write(f"Logged in as: **{st.session_state.username}**")

def submit_to_database(username, language, selected_symptoms, other_symptoms, prediction, classification):
    url = "http://127.0.0.1:8000/submit"

    payload = {
        "username": username,
        "language": language,
        "selected_symptoms": selected_symptoms,
        "other_symptoms": other_symptoms,
        "prediction": float(prediction),
        "classification": classification
    }

    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return True
        else:
            st.error(f"Submission failed: {response.text}")
            return False
    except Exception as e:
        st.error(f"Error connecting to database server: {str(e)}")
        return False

# English Tab
with tab_en:
    st.title(translations["title"]["en"])
    st.sidebar.header(translations["sidebar_header"]["en"])
    st.sidebar.write(translations["sidebar_content"]["en"])

    selected_symptoms = st.multiselect(translations["symptoms_prompt"]["en"], symptoms_en + ["Others"],placeholder = translations["symptoms_placeholder"]["en"])
    
    if "Others" in selected_symptoms:
        other_symptoms = st.text_area("Please list any other symptoms or signs you have:")
        if other_symptoms:
            if st.button(translations["send_email_button"]["en"]):
                if other_symptoms:
                    subject = "Additional Symptoms Submitted via App"
                    body = f"The user has submitted additional symptoms:\n\n{other_symptoms}"
                    receiver_email = "diagai2024@gmail.com"
                    if send_email(subject, body, receiver_email):
                        st.success("Your symptoms have been sent successfully! Thank you.")
                else:
                    st.warning(translations["send_email_warning"]["en"])
        else:
            st.warning("Please describe additional symptoms before sending.")

    if st.button(translations["button_results"]["en"]):
        features = [1 if symptom in selected_symptoms else 0 for symptom in symptoms_en]
        prediction = model.predict(np.array(features).reshape(1, -1))[0][0]

        if prediction > 0.43:
            classification = "Probably positive for malaria"
            st.success(translations["positive_result"]["en"])
            st.write(f"**Malaria Summary:** {get_wikipedia_summary('malaria', lang='en')}")
        else:
            classification = "Probably negative for malaria"
            st.info(translations["negative_result"]["en"])

        other_text = other_symptoms if "Others" in selected_symptoms else ""

        saved = submit_to_database(
            username=st.session_state.username,
            language="English",
            selected_symptoms=selected_symptoms,
            other_symptoms=other_text,
            prediction=prediction,
            classification=classification
        )

        if saved:
            st.success("Response saved to database.")

# Swahili Tab
with tab_sw:
    st.title(translations["title"]["sw"])
    st.sidebar.header(translations["sidebar_header"]["sw"])
    st.sidebar.write(translations["sidebar_content"]["sw"])
    
    selected_symptoms_sw = st.multiselect(translations["symptoms_prompt"]["sw"], symptoms_sw + ["Dalili Nyingine"],placeholder = translations["symptoms_placeholder"]["sw"])

    if "Dalili Nyingine" in selected_symptoms_sw:
        other_symptoms = st.text_area("Andika dalili nyingine unazopata")
        if other_symptoms:
            if st.button(translations["send_email_button"]["sw"]):
                if other_symptoms:
                    subject = "Dalili za Ziada Zimetumwa Kupitia Programu"
                    body = f"Mtumiaji ametuma dalili zifuatazo:\n\n{other_symptoms}"
                    receiver_email = "diagai2024@gmail.com"
                    if send_email(subject, body, receiver_email):
                        st.success("Dalili zako zimetumwa kikamilifu! Asante.")
                else:
                    st.warning(translations["send_email_warning"]["sw"])
        else:
            st.warning("Tafadhali andika dalili nyingine kabla ya kutuma ujumbe.")

    selected_symptoms = [symptoms_en[symptoms_sw.index(symptom)] for symptom in selected_symptoms_sw if symptom != "Dalili Nyingine"]

    if st.button(translations["button_results"]["sw"]):
        features = [1 if symptom in selected_symptoms else 0 for symptom in symptoms_en]
        prediction = model.predict(np.array(features).reshape(1, -1))[0][0]

        if prediction > 0.43:
            classification = "Inawezekana una malaria"
            st.success(translations["positive_result"]["sw"])
            st.write(f"**Muhtasari wa Malaria:** {get_wikipedia_summary('malaria', lang='sw')}")
        else:
            classification = "Inawezekana huna malaria"
            st.info(translations["negative_result"]["sw"])

        other_text = other_symptoms if "Dalili Nyingine" in selected_symptoms_sw else ""

        saved = submit_to_database(
            username=st.session_state.username,
            language="Kiswahili",
            selected_symptoms=selected_symptoms_sw,
            other_symptoms=other_text,
            prediction=prediction,
            classification=classification
        )

        if saved:
            st.success("Taarifa zimehifadhiwa kwenye kanzidata.")

#Logout
if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.rerun()            