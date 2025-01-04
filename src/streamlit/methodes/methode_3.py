import streamlit as st
from PIL import Image

st.markdown("# Méthode 3")

st.markdown("# Prétraitement")
st.markdown(
    """<div style="text-align: justify;">Dans cette partie nous nous intéresserons principalement aux enregistrements des sessions3. C’est-à-dire les enregistrements NF pour lesquels nous disposons du résultat de classification. Nous constatons que le changement de protocole expérimental et la répétition des sessions induit un changement influençant la structure des données enregistrées. Nous ne nous intéresserons qu’aux données EEG.""",
    unsafe_allow_html=True,
)
st.markdown(
    """<div style="text-align: justify;">Le signal brut est décomposé en epochs correspondant aux différentes tentatives.""",
    unsafe_allow_html=True,
)

st.markdown("## Extraction de caractéristiques")
st.markdown(
    """<div style="text-align: justify;">Nous appliquons une décomposition en ondelettes de Daubechies, sur la différence des échantillons entre C4 et C3.Cela permet d’analyser les fréquences caractéristiques tout en conservant des informations temporelles et réduire la dimensionnalité des données, facilitant ainsi leur interprétation et leur traitement par les algorithmes de classification""",
    unsafe_allow_html=True,
)

st.markdown("## Modélisation")
st.markdown(
    """<div style="text-align: justify;">Le jeu de d'entraînement est composé de la session 3 de 7 candidats tirés au hasard parmi les 9 participants. Le jeu de test / validation est, quant à lui, composé des “runs” 1 et 2 des 2 participants qui n’ont pas été retenus dans le jeu d'entraînement.""",
    unsafe_allow_html=True,
)
st.markdown(
    """<div style="text-align: justify;">Une variété de modèles d’apprentissage automatique a était utilisée pour classifier les IM, parmi lesquels le SVM (Support Vector Machine), la régression logistique, l’analyse discriminante linéaire, et des méthodes avancées comme le Gradient Boosting ou les k-plus-proches voisins (k-NN). Ces approches, combinées à des stratégies de segmentation, ont permis d’améliorer la précision et la fiabilité des prédictions.""",
    unsafe_allow_html=True,
)

st.markdown("## Analyse des résultats")
img_resultat1 = Image.open("./assets/resultat1M3.png")
st.image(img_resultat1)

img_resultat2 = Image.open("./assets/resultat2M3.png")
st.image(img_resultat2)

img_resultat3 = Image.open("./assets/resultat3M3.png")
st.image(img_resultat3)

st.html("<u>Accuracy des différents modèles</u>")
st.markdown(
    """<div style="text-align: justify;">
Pour les résultats présentés ci-dessus, en abscisse sont représentés le temps de départ et de fin de la segmentation après le top départ de l'événement (“EventStart”), en nombre d'échantillons (le sampling rate étant de 250 Hz). Et, en ordonné le ratio obtenu.""",
    unsafe_allow_html=True,
)
