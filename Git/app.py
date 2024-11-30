import streamlit as st
import numpy as np
import tensorflow as tf
import wikipediaapi
import re


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



# Input section: Multiselect dropdown for symptoms
selected_symptoms = st.multiselect(
    "Select the symptoms and signs you have:",
    symptoms
)

# Convert selected symptoms into binary feature vector
input_features = [1 if symptom in selected_symptoms else 0 for symptom in symptoms]

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