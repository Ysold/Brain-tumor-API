import streamlit as st
import requests
import json
import io
from fonctions import read_uploaded_file

# Configuration de la page
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="auto",
)

# Titre de la page
st.title("Brain Tumor Detection")

# Route /model
st.header("Model Information")
if st.button('Get Model Info'):
    response = requests.get('http://127.0.0.1:8000/model')
    st.write(response.text)

# Route /predict
st.header("Predict")
uploaded_file = st.file_uploader("Choose a Brain tumor image", type=['png', 'jpg'], accept_multiple_files=False)
if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=False, width=200)
    if st.button('Predict Image'):
        file_bytes = read_uploaded_file(uploaded_file)
        response = requests.post('http://127.0.0.1:8000/predict', files={"file": file_bytes})
        response_json = json.loads(response.text)

        st.markdown(f"**Prediction:** {response_json['prediction']}")
        st.markdown(f"**Probability:** {response_json['probability']}")
else:
    st.warning("Please upload an image.")

# Route /training
st.header("Training")
uploaded_file = st.file_uploader("Choose a CSV file for training", type=['csv'], accept_multiple_files=False)
if uploaded_file is not None:
    if st.button('Train Model'):
        file_bytes = read_uploaded_file(uploaded_file)
        if file_bytes is not None:

            file_obj = io.BytesIO(file_bytes)
            
            files = {"file": (uploaded_file.name, file_obj)}
            with st.spinner('Training model...'):
                response = requests.post('http://127.0.0.1:8000/training', files=files)
            response_json = json.loads(response.text)

            st.markdown(f"**Message:** {response_json['message']}")
            st.markdown(f"**Précision:** {response_json['accuracy']}")
            st.markdown(f"**Nom du fichier téléchargé:** {response_json['downloaded_filename']}")
            
else:
    st.warning("Please upload a CSV file.")
