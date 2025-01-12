import streamlit as st
from utils.utils import insert_svg, insert_image_title

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
insert_svg("./static/assets/images/resultat1M3.svg")
insert_svg("./static/assets/images/resultat2M3.svg")
insert_svg("./static/assets/images/resultat3M3.svg")
insert_image_title("Accuracy des différents modèles")
st.markdown(
    """<div style="text-align: justify;">
Pour les résultats présentés ci-dessus, en abscisse sont représentés le temps de départ et de fin de la segmentation après le top départ de l'événement (“EventStart”), en nombre d'échantillons (le sampling rate étant de 250 Hz). Et, en ordonné le ratio obtenu.""",
    unsafe_allow_html=True,
)
st.markdown(
    """<div style="text-align: justify;">
Les meilleurs résultats sont obtenus pour le model de classification DAL.<br><br>
Nous obtenons un score de prédiction de 71% pour une la fenêtre 75-225 (à 300 milliseconde âpr
le début de l’évènement pour se terminer 900 milliseconde après de début de l’évènement, soit une
fenêtre d’environ 600 milliseconde)""",
    unsafe_allow_html=True,
)
