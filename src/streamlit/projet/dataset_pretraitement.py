import streamlit as st
from PIL import Image

st.markdown("# Pré-traitement")
st.markdown("Suite aux observations précédentes, nous avons appliqué un filtre passe-bande 1Hz-30hz")
st.markdown("## Comparaison des signaux avant/après filtrage")
img_signals = Image.open('./assets/signals_filtres.png')
st.image(img_signals)
st.html("<u>Données EEG avant (a) et après (b) filtrage</u>")
st.markdown("La courbe représentant le signal filtré est plus lisse tout en gardant ses caractéristiques principales.")
