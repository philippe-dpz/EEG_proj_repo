import streamlit as st

puce_icon = ":material/arrow_circle_right:"
projet_page = st.Page("projet/projet.py", title="Présentation", icon=puce_icon)
dataset_page = st.Page(
    "projet/dataset.py", title="Les données utilisées", icon=puce_icon
)
dataset_analyse_page = st.Page(
    "projet/dataset_analyse.py",
    title="Analyse des données",
    icon=puce_icon,
)

dataset_pretraitement_page = st.Page(
    "projet/dataset_pretraitement.py",
    title="Prétraitement des données",
    icon=puce_icon,
)

methode1_page = st.Page("methodes/methode_1.py", title="Méthode 1", icon=puce_icon)
methode2_page = st.Page("methodes/methode_2.py", title="Méthode 2", icon=puce_icon)
methode3_page = st.Page("methodes/methode_3.py", title="Méthode 3", icon=puce_icon)
discussion_page = st.Page("resultats/discussion.py", title="discussion", icon=puce_icon)
exploitation_page = st.Page(
    "resultats/exploitation.py", title="exploitation", icon=puce_icon
)

st.set_page_config(
    page_title="EEG",
    page_icon=":material/neurology:",
    layout="wide",
    initial_sidebar_state="auto",
)

pg = st.navigation(
    {
        "Le projet": [
            projet_page,
            dataset_page,
            dataset_analyse_page,
            dataset_pretraitement_page,
        ],
        "Méthodes et modélisations": [
            methode1_page,
            methode2_page,
            methode3_page,
        ],
        "Résultats": [discussion_page, exploitation_page],
    }
)
pg.run()
