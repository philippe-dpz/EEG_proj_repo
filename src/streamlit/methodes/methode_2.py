import streamlit as st
from utils.utils import insert_png

st.markdown("# Méthode 2")

st.markdown(
    """<div style="text-align: justify;">Dans cette partie nous nous intéresserons principalement aux enregistrements des sessions3. C’est-à-dire les enregistrements NF pour lesquels nous disposons du résultat de classification. Nous constatons que le changement de protocole expérimental et la répétition des sessions induit un changement influençant la structure des données enregistrées. Nous ne nous intéresserons qu’aux données EEG. Nous choisissons ce type de sessions pour lesquelles nous disposons de la classe afin de nous ramener à un cas de classification supervisée.""",
    unsafe_allow_html=True,
)
st.markdown("## Prétraitement")
st.markdown(
    """<div style="text-align: justify;">Nous appliquons une baseline de -0.2s. La valeur moyenne des signaux pendant les 0.2s précédent le début de l’époque est déduite. Cette valeur a été retenue suite à différentes tentatives.
Afin d’atténuer l’effet du bruit, une pratique courante consiste à changer la référence du signal. Il s'agit de prendre un des signaux et de le considérer comme une base pour les autres. Souvent le signal choisi correspond à une électrode placée dans la zone médiane du crâne. Dans notre cas, le signal Cz nous semble un bon candidat. Nous effectuons donc ce changement de référence grâce à la fonction de  <a href="https://mne.tools/stable/index.html">MNE</a> correspondante.
""",
    unsafe_allow_html=True,
)
st.markdown(
    """<div style="text-align: justify;">
L’exploration des différentes valeurs de signal, pour les différents candidats montre des disparités d’intensités assez importantes. Pour tenter de les corriger, nous avons appliqué une mise à l’échelle de type MinMax.""",
    unsafe_allow_html=True,
)
st.markdown("## Extraction de caractéristiques")
st.markdown(
    """<div style="text-align: justify;">
Nous nous plaçons dans un espace Fréquence/Temps grâce au calcul TFR (revient à calculer le PSD en gardant la notion de temporalité). L’analyse des données nous incite à retenir les fréquences de la gamme alpha. On s’intéressera à la différence entre C3 et C4.""",
    unsafe_allow_html=True,
)
st.markdown("## Modélisation")
st.markdown(
    """<div style="text-align: justify;">Le jeu d'entraînement est constitué par les enregistrements des candidats 1 à 7, le jeu de test par ceux des candidats 8 et 9, ce choix est fait de façon arbitraire.Nous entraînons différents modèles, LogisticRegression, AdaBoostClassifier, SVM sur le jeu de données. A la vue des premiers résultats SVM ne semble pas adapté. Afin d'améliorer les performances, nous modifions les paramètres max_iter pour LogistiqueRegression et faisons un GridSearch sur AdaBoostClassifier. Nous effectuons également des entraînements sur une fenêtre temporelle glissante sur chaque époque. Ensuite nous essayons de réduire les fréquences utilisées.<br><br>""",
    unsafe_allow_html=True,
)
insert_png("./assets/PSDM2.png")
st.html(
    "<u>PSD entre 1 et 3 secondes sur la plage de fréquence 10.5 - 12.5Hz. Une différence nette entre le PSD de l’IM main gauche et droite est observable notamment pour les canaux C3 et C4.</u>"
)

st.markdown("## Analyse des résultats")
st.markdown(
    """<div style="text-align: justify;">Les meilleurs résultats sont obtenus avec LogistiqueRegression sur la plage de fréquences 10.5 - 12.5Hz et sur l’intervalle 1s - 3s""",
    unsafe_allow_html=True,
)
