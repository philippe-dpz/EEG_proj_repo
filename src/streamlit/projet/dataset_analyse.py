import streamlit as st
from utils.utils import insert_png, insert_image_title

st.markdown("# Analyse des données")
st.markdown("## Visualisation des signaux bruts")

left_co, right_co = st.columns(2)
with left_co:
    insert_png("app/static/assets/images/signalB35.png")
    insert_image_title(
        "EEG brut sur 25 secondes. <br>Dans le signal (a) on observe l’activité cérébrale de deux epochs : un de l’IM gauche, un de l’IM droite."
    )
with right_co:
    st.markdown('')
    insert_png("app/static/assets/images/signalB1.png")
    insert_image_title("EEG brut sur 1 seconde.")

st.markdown("## Puissance du signal")
insert_png("app/static/assets/images/psdB.png")
insert_image_title("Puissance du signal brut sur une session.")
st.markdown("## Comparaison par classe")
insert_png("app/static/assets/images/signalsComparaison.png")
insert_image_title("Comparaison des signaux moyens par type de tentative")

st.markdown(
    "On remarque à travers ces différentes représentations que les signaux présentent du bruit, et que l'on peut réduire la plage des fréquences."
)
