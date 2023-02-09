import streamlit as st


def main():
    # Step 5: Analyze the model
    st.markdown("## Step 5: Analyze the model")

    topics_tab, search_tab = st.tabs(
        [
            "Explore by Topics",
            "Explore by Keywords / Phrases",
        ]
    )

    return topics_tab, search_tab
