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
        st.markdown(f"**Précision:** {response_json['probability']}")
else:
    st.warning("Please upload an image.")

# Route /training
st.header("Training with Random Forest Classifier")
uploaded_file = st.file_uploader("Choose a CSV file for training with Random Forest Classifier", type=['csv'], accept_multiple_files=False)
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
            st.markdown(f"**Model à télécharger:** [{response_json['model_filename']}](http://127.0.0.1:8000/download_model)")
            
else:
    st.warning("Please upload a CSV file.")

# Route /training_tensorflow
st.header("Training with TensorFlow")
uploaded_file = st.file_uploader("Choose a CSV file for training with tensorflow", type=['csv'], accept_multiple_files=False)
if uploaded_file is not None:
    if st.button('Train Model'):
        file_bytes = read_uploaded_file(uploaded_file)
        if file_bytes is not None:

            file_obj = io.BytesIO(file_bytes)
            
            files = {"file": (uploaded_file.name, file_obj)}
            with st.spinner('Training model...'):
                response = requests.post('http://127.0.0.1:8000/train_tensorflow', files=files)
            response_json = json.loads(response.text)

            st.markdown(f"**Message:** {response_json['message']}")
            st.markdown(f"**Précision:** {response_json['accuracy']}")
            st.markdown(f"**Model à télécharger:** [{response_json['model_filename']}](http://127.0.0.1:8000/download_model)")
            
else:
    st.warning("Please upload a CSV file.")