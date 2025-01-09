import base64
import streamlit as st
from PIL import Image


def get_base64_of_bin_file(path_file: str) -> str:
    with open(path_file, "rb") as f:
        return base64.b64encode(f.read()).decode()


def insert_svg(path: str):
    css_justify = "center"
    css = '<p style="text-align:center; display: flex; justify-content: {};">'.format(
        css_justify
    )
    html = r'{}<img src="data:image/svg+xml;base64,{}"/>'.format(
        css, get_base64_of_bin_file(path)
    )
    st.write(html, unsafe_allow_html=True)


def insert_png(path: str):
    left_co, cent_co, last_co = st.columns([0.1, 0.8, 0.1])
    with cent_co:
        img = Image.open(path)
        st.image(img, width=600)
    st.columns(1)


def insert_image_title(title: str):
    html = r'<div style="text-align: center"><u>{}</u></div>'.format(title)
    st.html(html)
