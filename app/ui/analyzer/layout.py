import streamlit as st


def main():

    topics_tab, search_tab = st.tabs(
        [
            "Explore by Topics",
            "Explore by Keywords / Phrases",
        ]
    )

    return topics_tab, search_tab
