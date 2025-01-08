import streamlit as st
from PIL import Image

st.markdown("# Analyse des données")
st.markdown("## Visualisation des signaux bruts")
img_signals = Image.open('./assets/signals.png')
st.image(img_signals)
st.html("<u>EEG brut sur 35 secondes (a) et sur 1 seconde (b). <br>Dans le signal (a) on observe l’activité cérébrale de deux epochs : un de l’IM gauche, un de l’IM droite.</u>")
st.markdown("## Puissance du signal")
img_psd = Image.open('./assets/psd.png')
st.image(img_psd)
st.html("<u>Puissance du signal brut sur une session.</u>")
st.markdown("## Comparaison par classe")
img_signals_comparaison = Image.open('./assets/signals_comparaison.png')
st.image(img_signals_comparaison)
st.html("<u>Comparaison des signaux moyens par type de tentative.</u>")

st.markdown("On remarque à travers ces différentes représentations que les signaux présentent du bruit, et que l'on peut réduire la plage des fréquences.")

