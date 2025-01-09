import streamlit as st
from utils.utils import insert_png, insert_image_title

st.markdown("# Prétraitement")
st.markdown(
    "Suite aux observations précédentes, nous avons appliqué un filtre passe-bande 1Hz-30hz"
)
st.markdown("## Comparaison des signaux avant/après filtrage")
insert_png("./assets/signals_filtres.png")
insert_image_title("Données EEG avant (a) et après (b) filtrage")
st.markdown(
    "La courbe représentant le signal filtré est plus lisse tout en gardant ses caractéristiques principales."
)
