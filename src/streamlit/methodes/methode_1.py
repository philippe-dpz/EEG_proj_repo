import streamlit as st
from PIL import Image

st.markdown("# Méthode 1")
st.markdown(
    """<div style="text-align: justify;">Cette méthode explore l'utilisation des techniques bien établies dans la recherche neuroscientifique.<br><br>L'objectif est de tirer parti des informations spatio-temporelles et fréquentielles contenues dans les signaux cérébraux afin de reconnaître les schémas neuronaux associés à cette activité mentale.<br>Nous nous appuyons sur des approches courantes, telles que l'analyse des spectres de densité de puissance, l'identification des rythmes sensorimoteurs (notamment µ et β), et des outils de filtrage spatial comme les filtres Common Spatial Patterns (CSP).</div>""",
    unsafe_allow_html=True,
)
st.markdown("## Pré-traitement")
st.markdown(
    "- Filtre passe-bande entre 8 et 30 Hz, afin de nous concentrer sur les rythmes alpha (mu) et bêta"
)
st.markdown(
    "- Application d'une régression basée sur les signaux EOG pour réduire les artefacts oculaires"
)
st.markdown(
    "- Correction de la ligne de base effectuée en prenant une période de 0,2 seconde avant chaque epoch"
)
st.markdown("## Extraction de caractéristiques")
st.markdown(
    "Nous avons utilisé l'approche de Blankertz et al. (2007) qui utilise l'application de filtres CSP."
)
st.markdown(
    "Nous avons ensuite appliqué la technique du logarithmic variance (log-var) pour transformer les signaux CSP en valeurs plus distinctes et comparables"
)
st.markdown(
    """<div style="text-align: justify;">Nous avons appliqué les filtres CSP sur les epochs EEG, projetant les données dans un espace où les différences de variance entre les classes sont maximisées, facilitant l'extraction des composantes discriminantes. Après cela, nous avons calculé la variance de chaque composante filtrée et appliqué le logarithme de cette variance pour normaliser les valeurs et atténuer l'impact des grandes variations, afin d'obtenir des caractéristiques adaptées à la classification des intentions motrices.""",
    unsafe_allow_html=True,
)
img_psd = Image.open("./assets/EEGCSP.png")
st.image(img_psd)
st.html(
    "<u>Signal EEG filtré CSP pour C3, Cz, et C4 avec en bleu event_0 (IM main gauche), et en rouge event_1 (IM main droite)</u>"
)

st.markdown(
    """<div style="text-align: justify;">Nous avons également appliqué ce processus aux Power Spectral Densities (PSD). Pour chaque epoch EEG, nous avons d'abord calculé la PSD, puis utilisé les filtres CSP pour extraire les composantes discriminantes, avant de calculer le logarithme de la variance de ces composantes. Cela permet d'extraire des caractéristiques basées sur la puissance spectrale plutôt que sur l'activité brute du signal.""",
    unsafe_allow_html=True,
)
img_sds = Image.open("./assets/SDS.png")
st.image(img_sds)
st.html(
    "<u>Signal de densité spectrale de puissance filtrré CSP de IM gauche (gauche) et im droite (droite) on voit que les PSD des canaux vert et bleu s’échangent en fonction de l’event</u>"
)
st.markdown(
    "Enfin, nous avons appliqué la pipeline sur l'amplitude par le temps (amp by time), qui analyse l'évolution de l'intensité du signal EEG au fil du temps"
)
img_amp = Image.open("./assets/amp.png")
st.image(img_amp, width=600)
st.html(
    "<u>amp by time des 3 canaux en fonction de l’IM (ligne pleine, IM gauche et pointillée IM droite)</u>"
)
st.markdown(
    """<div style="text-align: justify;">Pour les trois facteurs (CSP, PSD et amplitude par le temps), nous avons extrait trois composantes distinctes, permettant de capturer des caractéristiques discriminantes pour la classification des intentions motrices.""",
    unsafe_allow_html=True,
)
st.html(
    "<u>amp by time des 3 canaux en fonction de l’IM (ligne pleine, IM gauche et pointillée IM droite)</u>"
)
img_carac = Image.open("./assets/caractM1.png")
st.image(img_carac, width=600)

st.html(
    "<u>log var des CSP du signal brut, des PSD et de l’amp by time, moyenné par IM gauche (bleu) et droite (orange)</u>"
)
st.markdown("## Modélisation")
st.markdown(
    """<div style="text-align: justify;">
Dans ce processus, nous avons utilisé trois modèles de classification distincts, chacun appliqué à des ensembles de caractéristiques extraites des signaux EEG. Ces modèles sont : un Random Forest, un SVM (Support Vector Machine) et un AdaBoost. 
Nous avons aussi utilisé un réseau de neurones convolutionnels (CNN) pour classifier les signaux EEG en fonction des caractéristiques extraites. Il comprend deux couches de convolution 1D pour extraire les caractéristiques des signaux, suivies d’une couche de Flatten pour aplatir les données et de couches densément connectées pour apprendre des représentations plus complexes. Une couche de Dropout est ajoutée pour éviter le surapprentissage. L'optimiseur Adam et la fonction de perte binary crossentropy sont utilisés pour l’entraînement. Un mécanisme d’early stopping arrête l’entraînement si la performance de validation ne s’améliore pas après un certain nombre d’époques.
Chaque modèle a été testé sur différentes combinaisons de caractéristiques et a été appliqué à trois situations spécifiques:
Pour chaque modèle, la phase d'entraînement a été réalisée sur 80 % des données, tandis que les 20 % restants ont été utilisés pour les tests. Cette répartition a été effectuée de manière aléatoire.
Nous avons appliqué les modèles sur l'ensemble des epochs disponibles, afin de capturer une vue d'ensemble complète des intentions motrices à travers toutes les données EEG. Par la suite, les modèles ont été ajustés en fonction des epochs utilisées lors des premières sessions de screening, puis de NF. Pour chaque condition expérimentale, les modèles ont été optimisés afin d'obtenir la meilleure précision possible.""",
    unsafe_allow_html=True,
)
img_accuracy = Image.open("./assets/accuracyM1.png")
st.image(img_accuracy, width=600)

st.html(
    "<u>Accuracy des différents modèles en fonction du type de modèle et des condition expérimentales</u>"
)

st.markdown("## Analyse des résultats")
st.markdown(
    """<div style="text-align: justify;">
Les résultats obtenus montrent des performances satisfaisantes, avec une précision dépassant 70 % pour les tests du DNN sur l'ensemble des epochs et lors de la phase de neurofeedback. Cependant, une baisse significative des performances a été observée lors des tests sur la phase de screening.""",
    unsafe_allow_html=True,
)
