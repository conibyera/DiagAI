import streamlit as st
import numpy as np
import tensorflow as tf
import wikipediaapi
import re
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Load the saved model
model = tf.keras.models.load_model('G:/My Drive/Rapid-AI/malar_model.keras')

# Define the list of symptoms (features)
symptoms = ["Fever", "Vomiting", "Convulsions", "Cough","Yellow Eyes", "Diarrhoea", "Headache", "Body Pain","Abdnominal Pain","Loss of Appetite","Body Weakness"]

# Function to fetch Wikipedia summary with improved sentence splitting
def get_wikipedia_summary(disease, num_sentences=5):
    # Define Wikipedia API with user-agent header
    wiki_wiki = wikipediaapi.Wikipedia(
        language='en', 
        user_agent='DiagAI/1.0'
    )

    # Fetch the page related to the disease
    page = wiki_wiki.page(disease)

    if page.exists():
        # Split summary into sentences
        sentences = re.split(r'(?<=[.!?])\s*', page.summary)
        sentences = [sentence.strip() for sentence in sentences if sentence]  # Clean up

        # Return the desired number of sentences
        return ' '.join(sentences[:num_sentences])
    else:
        return f"No information found for {disease}."


# Streamlit app interface
st.title("DiagAI/1.0 for Rapid Malaria Diagnosis")

# Sidebar for app explanation
st.sidebar.header("About This App")
st.sidebar.write("""
    DiagAI is a web application designed for rapid disease diagnosis based on symptoms and signs input.
    
    This first version of the application utilizes a neural network model that predicts the likelihood of malaria based on the selected symptoms or signs. 
    You can select symptoms or signs you are experiencing, and the model will provide you with a probable diagnosis for malaria.
        
    **How to Use:**
    1. Select the symptoms and signs you are experiencing from the dropdown menu.
    2. Click on the buttons to check the disease status for malaria.
    3. The app will indicate whether you are probably positive or negative for malaria.
    
    *Please remember that this application is a rapid diagnostic tool and not a substitute for professional medical advice.*
""")

# Fetch email and password from Streamlit secrets
sender_email = st.secrets["email"]["sender_email"]
sender_password = st.secrets["email"]["sender_password"]

# Function to send email
def send_email(recipient, subject, body):
    try:
        # Set up the SMTP server
        server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
        server.login(sender_email, sender_password)

        # Create the email
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        # Send the email
        server.sendmail(sender_email, recipient, msg.as_string())
        server.quit()

        return True
    except Exception as e:
        st.error(f"Error sending email: {e}")
        return False

# Input section: Multiselect dropdown for symptoms
selected_symptoms = st.multiselect(
    "Select the symptoms and signs you have:",
    symptoms + ["Others"]
)

# Handle "Others" logic
if "Others" in selected_symptoms:
    other_symptoms = st.text_area(
        "Please list any other symptoms you are experiencing:",
        placeholder="Type additional symptoms here..."
    )

    # Activate the "Send Email" button only if text is entered
    if other_symptoms:
        if st.button("📧 Submit"):
            subject = "Additional Symptoms Submitted via App"
            body = f"The user has submitted the following additional symptoms:\n\n{other_symptoms}"
            receiver_email = "diagai2024@gmail.com"  # Replace with your email address

            # Send the email
            if send_email(subject, body, receiver_email):
                st.success("Your symptoms have been sent successfully! Thank you.")
            else:
                st.error("There was an issue sending your symptoms. Please try again.")
    else:
        st.warning("Please describe additional symptoms before sending.")

# Exclude "Others" from the prediction feature vector
features_for_prediction = [symptom for symptom in selected_symptoms if symptom != "Others"]

# Convert selected symptoms into binary feature vector
input_features = [1 if symptom in features_for_prediction else 0 for symptom in symptoms]


# Display the feature vector (for transparency)
# st.write("Feature vector:", input_features)

if st.button("🐜Malaria Results"):
     # Malaria Prediction Logic
     input_array = np.array(input_features).reshape(1, -1)
     prediction = model.predict(input_array)[0][0]

     if prediction > 0.33:
         st.success("Probably positive for malaria")
         summary = get_wikipedia_summary("malaria")
         st.write(f"**Malaria Summary:** {summary}")
     else:
         st.info("Probably negative for malaria")   