import streamlit as st
import pandas as pd

from top2vec import Top2Vec

from app.helper import *


def main(df: pd.DataFrame, model: Top2Vec) -> None:
    st.markdown("### Keywords / Phrases Search")
    mode = st.radio(
        "Search mode",
        ["Keywords", "Phrases"],
        key="mode",
    )

    if mode == "Keywords":
        st.markdown("### Enter keywords separated by commas (e.g. `good, bad, ugly`)")
    else:
        st.markdown("### Enter a phrase (e.g. `good but bad`)")

    keyword_input = st.text_input("Keyword / Phrase", key="keyword_input").strip()
    keywords = None

    try:
        if mode == "Keywords":
            keywords = keyword_input.split(",")
            keywords = [keyword.strip() for keyword in keywords]
        else:
            keywords = keyword_input
    except:
        st.error("Invalid input")

    if not keywords or not all(keywords):
        st.stop()

    # Ask user to select a number of reviews to show
    if mode == "Keywords":
        st.markdown(
            f"Searching for reviews containing the keywords {', '.join([f'`{keyword}`' for keyword in keywords])}"
        )
    else:
        st.markdown(f"Searching for reviews containing phrases similar to `{keywords}`")

    st.markdown("### Select number of reviews to show")
    st.markdown("These are marked in descending order of similarity to the topic.")
    num_docs = st.number_input(
        "Number of reviews",
        min_value=1,
        max_value=1000,
        value=10,
        key="num_keywords_docs_kw",
    )

    search_btn = st.button("Search", key="search_btn")

    if not search_btn:
        st.stop()

    try:
        with st.spinner("Searching..."):
            if mode == "Keywords":
                documents, document_scores, _ = model.search_documents_by_keywords(
                    keywords, num_docs=num_docs
                )

            else:
                documents, document_scores, _ = model.query_documents(
                    keywords, num_docs=num_docs
                )

            document_df = pd.DataFrame({"Review": documents, "Score": document_scores})
            display_dataframe(document_df, key="keyword_search_df")
    except:
        st.error(f"Could not find any documents with keywords: {keywords}")
