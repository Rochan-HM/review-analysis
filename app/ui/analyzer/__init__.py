import streamlit as st
import pandas as pd
import plotly.express as px

from top2vec import Top2Vec
from stqdm import stqdm

from helper import *
from .layout import main as layout
from .topics import main as topics
from .search import main as search


def main(df: pd.DataFrame, model: Top2Vec, selected_column: str) -> None:
    # Step 5: Analyze the model
    st.markdown("## Step 5: Analyze the model")

    num_topics = model.get_num_topics()
    st.write(f"Number of topics: {num_topics}")

    if "predicted_topic_labels" not in st.session_state:
        st.session_state.predicted_topic_labels = None

    if "representative_df" not in st.session_state:
        st.session_state.representative_df = None

    if (
        st.session_state.predicted_topic_labels is None
        and st.session_state.representative_df is None
    ):
        with st.spinner("Auto-generating topics..."):
            # Get the top representative documents for each topic
            representative_documents = []
            for topic in stqdm(range(num_topics)):
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
    df = st.session_state.df

    # Check if sentiment analysis is enabled
    if st.session_state.sentiment_analysis is not None:
        # Construct a pie chart to show the sentiment distribution
        st.markdown("### Overall Sentiment Distribution")
        fig = px.pie(
            df,
            names="Predicted Sentiment",
            title="Overall Sentiment Distribution",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Show it in an expandable table
    st.markdown("### Top reviews for each topic")
    st.markdown(
        "This table shows the top review for each topic. However, it might not be the most representative review for the topic."
    )

    display_dataframe(
        representative_df.set_index("Topic"),
        key="representative_df",
    )

    topics_tab, search_tab = layout()

    with topics_tab:
        topics(df, model, predicted_topic_labels, representative_df)

    with search_tab:
        search(df, model)
