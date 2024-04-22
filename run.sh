# Installation des dépendances
pip install -r requirements.txt

# Démarrage de l'API
uvicorn api:app --reload

# Démarrage de l'interface Streamlit
streamlit run app_web.py