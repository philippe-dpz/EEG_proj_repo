import streamlit as st
from PIL import Image

st.markdown("# Prédiction des Mouvements Imaginaires de la Main ")
st.markdown("Dans le cadre de notre formation en data science nous avons choisi d'étudier l'analyse d'enregistrements d'électroencéphalogrammes (EEG) et plus précisément l'imagerie motrice (IM).")
st.markdown("L'IM correspond au schéma enregistré par l'électroencéphalogramme lorsque le sujet imagine l'exécution d'un mouvement sans réaliser celui-ci.")
st.markdown("Le but de ce projet est de créer et d’entraîner un programme permettant de prédire si l’IM d’une personne correspond à un mouvement de la main droite ou de la main gauche.")
st.markdown("Il s'agit donc de classifier selon 2 classes, main gauche (classe 1) et de la main droite (classe 2), les enregistrements des différentes tentatives.")
            
st.markdown("## Protocole d'acquisition des données")
st.markdown("L'activité électrique du cerveau est enregistrée grâce à des capteurs placés sur le cuir chevelu.")
st.markdown("Nous avons utilisé la base de données ‘BCI Competition IV Dataset b’")
st.markdown("Pour recueillir les données 9 sujets ont participé à 5 sessions d'enregistrements EEG comprenant différentes conditions expérimentales.")
st.markdown("Les deux premières contiennent des données d'entraînement sans retour d'information (screening), et les trois dernières ont été enregistrées avec retour d'information (neurofeedback(nf)).")
st.markdown("Chaque session comporte 120 tentatives par classe.")
st.markdown("Les enregistrements EEG ont été effectués à l'aide de trois dérivations bipolaires (C3, Cz et C4) avec une fréquence d'échantillonnage de 250 Hz. Les signaux ont été filtrés dans une bande de fréquence comprise entre 0,5 Hz et 100 Hz, incluant un filtre coupe-bande à 50 Hz. La position des électrodes variait légèrement d'un sujet à l'autre.")
st.markdown("En plus des canaux EEG, l’électro-oculogramme (EOG) a été enregistré avec trois électrodes monopolaires (le mastoïde gauche servant de référence), en utilisant les mêmes réglages d’amplificateur.")
img = Image.open('./assets/electrodes.png')
st.image(img)
st.markdown("Configuration des électrodes de l’électroencéphalogramme (a), et de l’électrooculogramme (b).")
st.markdown("Le détail du déroulement des sessions d'enregistrement se trouve  """"<a href="https://www.bbci.de/competition/iv/desc_2b.pdf">ici</a>""""", unsafe_allow_html=True)