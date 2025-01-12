import streamlit as st
from utils.utils import insert_png, insert_image_title

st.markdown("# Prétraitement")
st.markdown(
    "Suite aux observations précédentes, nous avons appliqué un filtre passe-bande 1Hz-30hz"
)
st.markdown("## Comparaison des signaux avant/après filtrage")
left_co, right_co = st.columns(2)
with left_co:
    insert_png("app/static/assets/images/signalB1.png")
    insert_image_title("Données EEG avant filtrage")
with right_co:
    insert_png("app/static/assets/images/signalF1.png")
    insert_image_title("Données EEG après filtrage")
st.columns(1)
st.markdown(
    "La courbe représentant le signal filtré est plus lisse tout en gardant ses caractéristiques principales."
)
