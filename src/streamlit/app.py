import streamlit as st
import base64

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
discussion_page = st.Page("resultats/discussion.py", title="Discussion", icon=puce_icon)
exploitation_page = st.Page(
    "resultats/exploitation.py", title="Exploitation", icon=puce_icon
)

st.set_page_config(
    page_title="EEG-IM",
    page_icon=":material/neurology:",
    layout="wide",
    initial_sidebar_state="auto",
)

def get_base64_of_bin_file(png_file: str) -> str:
    with open(png_file, "rb") as f:
        return base64.b64encode(f.read()).decode()


@st.cache_resource
def build_markup_for_logo(png_file: str) -> str:
    binary_string = get_base64_of_bin_file(png_file)
    return f"""
            <style>
                [data-testid="stSidebarContent"] {{
                    padding: 10px;
                }}
                [data-testid="stSidebarHeader"] {{
                    background-image: url("data:image/png;base64,{binary_string}");
                    background-repeat: no-repeat;
                    background-size: contain;
                    background-position: top center;
                }}
            </style>
            """


st.markdown(
    build_markup_for_logo("assets/brain.jpg"),
    unsafe_allow_html=True,
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
