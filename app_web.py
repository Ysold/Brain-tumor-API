import streamlit as st
import requests
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

uploaded_file = st.file_uploader("Choose a Brain tumor image", type=['png', 'jpg'], accept_multiple_files=False)
if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    # Ajouter un bouton pour appeler l'API predict
    if st.button('Predict'):
        # Lire l'image téléchargée
        image_bytes = read_uploaded_file(uploaded_file)
        if image_bytes is not None:
            # Envoi de la requête à l'API
            response = requests.post('http://127.0.0.1:8000/predict', files={"file": image_bytes})
            
            # Affichage de la réponse
            st.write(response.text)
else:
    st.warning("Please upload an image.")