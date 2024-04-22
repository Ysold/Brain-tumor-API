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

# Bloc Brain Tumor Detection
st.markdown("<h1 style='text-align: center;'>Brain Tumor Detection</h1>", unsafe_allow_html=True)

left_co, cent_co,last_co = st.columns(3)
with cent_co:
    st.image('./images/tumor.jpg')
multi = '''
The Brain-tumor-API is a high-tech tool designed to help doctors detect brain tumors with a remarkable accuracy rate of 0.99. It uses machine learning to improve its ability to recognize tumors from brain scans, enhancing its accuracy over time. This tool assists doctors in making accurate diagnoses, providing reliable health assessments to patients. It's a valuable aid in the early detection and treatment of brain tumors.
'''
centered_multi = f'<div style="text-align: center;padding: 20px;margin: 20px;max-width: 820px;margin-left: auto;margin-right: auto;">{multi}</div>'
st.markdown(centered_multi, unsafe_allow_html=True)

# Separateur
st.divider()   

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

# Separateur
st.divider()    
    
# Bloc 'Our Team'
col1, col2 = st.columns([1, 1], gap='small')

# Colonne 1
with col1:
    st.image('./images/our-team.jpg')

# Colonne 2
with col2:
    st.write("""
        ## Our Team
        Led by a shared passion for healing and innovation, our doctors work tirelessly to unravel the complexities of tumors and provide personalized treatment plans tailored to each patient's unique needs. Through rigorous research, cutting-edge technologies, and compassionate care, we strive to not only treat tumors but also to improve outcomes and enhance quality of life. The Brain Tumor Detection tool, that has been developed by our engineers, help us in our everyday job making easier the detection of patients tumor. 
    """)
    
# Separateur
st.divider()       

# Bloc 'Charts'
st.markdown("<h2 style='text-align: center;'>Our Charts</h2>", unsafe_allow_html=True)
col1, col2 = st.columns([1, 1], gap='small')

# Colonne 1
with col1:
    st.image('./images/accuracy_chart.png')
    multi = '''More epochs lead to a better Accuracy, reaching a rate of 0.99 after 8 Epochs'''
    centered_multi = f'<div style="text-align: center;max-width: 430px;margin-right: auto;text-wrap:balance">{multi}</div>'
    st.markdown(centered_multi, unsafe_allow_html=True)

# Colonne 2
with col2:
    st.image('./images/loss_chart.png')
    multi = ''' More epochs lead to a decrease of the Loss, reaching a rate of 0.01 after 8 Epochs'''
    centered_multi = f'<div style="text-align: center;max-width: 430px;margin-right: auto;text-wrap:balance">{multi}</div>'
    st.markdown(centered_multi, unsafe_allow_html=True)


multi = '''According to the results we have experienced, here is the Acurracy and Loss charts that we can expect.'''
centered_multi = f'<div style="text-align: center;padding: 20px;margin: 20px;max-width: 820px;margin-left: auto;margin-right: auto;">{multi}</div>'
st.markdown(centered_multi, unsafe_allow_html=True)