import streamlit as st
from utils.utils import insert_png, insert_image_title

st.markdown("# Analyse des données")
st.markdown("## Visualisation des signaux bruts")
insert_png("./assets/signals.png")
insert_image_title(
    "EEG brut sur 35 secondes (a) et sur 1 seconde (b). <br>Dans le signal (a) on observe l’activité cérébrale de deux epochs : un de l’IM gauche, un de l’IM droite."
)
st.markdown("## Puissance du signal")
insert_png("./assets/psd.png")
insert_image_title("Puissance du signal brut sur une session.")
st.markdown("## Comparaison par classe")
insert_png("./assets/signals_comparaison.png")
insert_image_title("Comparaison des signaux moyens par type de tentative")

st.markdown(
    "On remarque à travers ces différentes représentations que les signaux présentent du bruit, et que l'on peut réduire la plage des fréquences."
)
