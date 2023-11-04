import streamlit as st
from helper import *

# Header for the application.
HEADER_HTML = """
<div style="text-align: center;">
<h1>RISE</h1>
<h2>Review Insights and Sentiment Analysis</h2>
</div>
""".strip()


def main() -> None:
    # Header
    st.markdown(HEADER_HTML, unsafe_allow_html=True)
