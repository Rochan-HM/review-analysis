import streamlit as st
import numpy as np
import pandas as pd

from top2vec import Top2Vec

# Set streamlit configs
st.set_page_config(layout="wide")
st.set_option("deprecation.showPyplotGlobalUse", False)

# Then import the helper functions

from helper import *
from ui.header import main as header
from ui.uploader import main as uploader
from ui.model import main as embedding_model
from ui.trainer import main as trainer
from ui.analyzer import main as analyzer

# Show the header
header()

# Get the dataframe and validate it
df, selected_column = uploader()

# Get the embedding model and the model learning speed
embedding_model, selected_embedding = embedding_model()

# Train the model
model = trainer(df, selected_column, selected_embedding, embedding_model)

# Show the model analysis
analyzer(model)

with topics:
    st.markdown("### Topics")
    num_topics = model.get_num_topics()
    num_docs = model.get_topic_sizes()[0]
    st.write(f"Number of topics: {num_topics}")

    if (
        st.session_state.predicted_topic_labels is None
        and st.session_state.representative_df is None
    ):
        with st.spinner("Auto-generating topics..."):
            # Get the top representative documents for each topic
            representative_documents = []
            for topic in range(num_topics):
                representative_docs, _, _ = model.search_documents_by_topic(
                    topic, num_docs=1
                )
                representative_documents.append(representative_docs[0])

            # Extract cluster labels for the documents
            predicted_topic_labels = extract_labels(model)

            representative_df = pd.DataFrame(
                {
                    "Topic": range(num_topics),
                    "Predicted Topic Labels": predicted_topic_labels,
                    "Top Review": representative_documents,
                }
            )

            # Save the results to the session state
            st.session_state.predicted_topic_labels = predicted_topic_labels
            st.session_state.representative_df = representative_df

    predicted_topic_labels = st.session_state.predicted_topic_labels
    representative_df = st.session_state.representative_df

    # Show it in an expandable table
    st.markdown("### Top reviews for each topic")
    st.markdown(
        "This table shows the top review for each topic. However, it might not be the most representative review for the topic."
    )
    display_dataframe(representative_df.set_index("Topic"), key="representative_df")

    # Show a dropdown to select a topic
    topic_options = [
        f"Topic {i}: {predicted_topic_labels[i]}" for i in range(num_topics)
    ]
    selected_topic = st.selectbox("Select a topic", topic_options)
    topic = int(selected_topic.split(":")[0].split(" ")[1])

    # Show the top 10 words for the selected topic
    st.markdown("### Top 10 words for the selected topic")
    topic_words, word_scores, _ = model.get_topics()
    topic_words = topic_words[topic][:10]
    word_scores = word_scores[topic][:10]
    topics_df = pd.DataFrame({"Word": topic_words, "Score": word_scores})
    display_dataframe(topics_df, key="topic_words_df")

    # Show the word cloud for the selected topic
    st.markdown("### Word cloud for the selected topic")
    fig = model.generate_topic_wordcloud(topic)
    st.pyplot(fig=fig)

    # Ask user to select a number of reviews to show
    st.markdown("### Select number of reviews to show")
    st.markdown("These are marked in descending order of similarity to the topic.")
    num_docs = st.number_input(
        "Number of reviews",
        min_value=1,
        max_value=num_docs[topic],
        value=10,
        key="num_topic_docs",
    )
    documents, document_scores, _ = model.search_documents_by_topic(
        topic, num_docs=num_docs
    )
    document_df = pd.DataFrame({"Review": documents, "Score": document_scores})
    display_dataframe(document_df, key="document_df")

with kw_search:
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
