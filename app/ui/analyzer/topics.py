import streamlit as st
import pandas as pd
import plotly.express as px

from top2vec import Top2Vec
from typing import List

from helper import *


def main(
    df: pd.DataFrame,
    model: Top2Vec,
    predicted_topic_labels: List[str],
    representative_df: pd.DataFrame,
) -> None:
    num_topics = model.get_num_topics()
    num_docs = model.get_topic_sizes()[0]

    st.markdown("### Topics")

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

    # Show the sentiment distribution for the selected topic
    st.markdown("### Sentiment Distribution for the selected topic")
    sentiment_df = representative_df[representative_df["Topic"] == topic]
    fig = px.pie(
        sentiment_df,
        names="Predicted Sentiment",
        title="Sentiment Distribution for the selected topic",
    )
    st.plotly_chart(fig)

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
