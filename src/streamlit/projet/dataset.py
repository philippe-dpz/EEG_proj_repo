import streamlit as st
import pandas as pd

@st.cache_resource
def load_train_sample():
        return pd.read_csv("./assets/train_B0101T.csv")
        
def load_y_train_sample():
        return pd.read_csv("./assets/y_train_B0101T.csv")

st.markdown("# Données utilisées")
st.markdown("Nous avons utilisé la base de données ‘BCI Competition IV Dataset b’ disponible sur """"<a href="https://www.kaggle.com/competitions/ucsd-neural-data-challenge/data">kaggle</a>""""", unsafe_allow_html=True)
st.markdown("Pour chacun des 9 sujets, nous disposons de deux fichiers par session.")
st.markdown("- Un contenant les données d'enregistrement d'encéphalogramme et d'électrooculogramme. (time Temps depuis le début en ms, C3 Cz C4 EEG, 3 canaux EOG, EventStart Début de tentative)")
df_train = load_train_sample()
st.dataframe(df_train)
st.markdown("- un autre contenant la classe de chacunes des tentatives. 0 Gauche,  1 Droite")
df_y_train = load_y_train_sample()
st.dataframe(df_y_train)

