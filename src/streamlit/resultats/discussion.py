import streamlit as st
from utils.utils import insert_png

st.markdown("# Discussion")
st.markdown(
    """<div style="text-align: justify;">
Nos modèles ont montré des résultats prometteurs, atteignant une précision de 80 % sur les données NF et 73 % sur l'ensemble des données. Ces performances illustrent une intégration réussie des concepts fondamentaux de la recherche en interface cerveau-machine (BCI). En nous concentrant sur les rythmes mu et beta, essentiels pour l’imagerie motrice (IM), nous avons extrait des caractéristiques différenciables, permettant ainsi une classification efficace des états cognitifs.""",
    unsafe_allow_html=True,
)

st.markdown(
    """<div style="text-align: justify;">
Une différence majeure observée réside dans l’écart de précision entre les sessions de screening et celles de NF, respectivement de 63 % contre 80 %. Cette disparité peut être expliquée par l'importance du retour d’information visuel en temps réel, qui joue un rôle crucial dans l’amélioration des performances en BCI. """,
    unsafe_allow_html=True,
)

st.markdown(
    """<div style="text-align: justify;">
Dans notre étude, les sessions de screening correspondent aux deux premières sessions, tandis que pour la session de neurofeedback (NF), les participants sont déjà entraînés, ce qui peut être considéré comme un entrainement préalable. Ce facteur pourrait expliquer la différence notable observée dans nos modèles.""",
    unsafe_allow_html=True,
)

insert_png("app/static/assets/images/spectro.png")
st.html(
    "<u>(a) 50 premiers événements d’imagerie motrice (IM) de main gauche, (b) 50 derniers événements d’imagerie motrice (IM) de main gauche .</u>"
)

st.markdown(
    """<div style="text-align: justify;">
Nos résultats ont également mis en évidence l’importance cruciale de la plage de temps sélectionnée pour extraire les caractéristiques. En effet, nous avons obtenu de meilleurs résultats en utilisant une plage de temps qui prend en compte le temps de réaction au stimulus et qui est plus courte que l’intégralité de l’époque""",
    unsafe_allow_html=True,
)
st.markdown(
    """<div style="text-align: justify;">
Cependant, une question demeure : nos modèles sont-ils exploitables dans un usage courant et sur des données inconnues ? En effet, nos données reposent sur seulement 9 participants, et compte tenu de la sensibilité des signaux EEG, il est légitime de s’interroger sur la généralisabilité de notre paradigme. C'est l'objet dans la partie <a href="./exploitation">Exploitation</a>""",
    unsafe_allow_html=True,
)
