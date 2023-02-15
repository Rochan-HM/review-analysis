import streamlit as st
import pandas as pd
import numpy as np

from typing import Tuple

from top2vec import Top2Vec
from stqdm import stqdm

from helper import *

stqdm.pandas()


def main(
    df: pd.DataFrame,
    selected_column: str,
    selected_embedding: str,
    selected_speed: str,
) -> Tuple[Top2Vec, pd.DataFrame]:
    # Step 4: Ask user to click a button to start training the model
    st.markdown("## Step 4: Train the model")
    st.markdown("This may take a few minutes...")

    checkbox_placeholder = st.empty()
    btn_placeholder = st.empty()
    sentiment_checkbox = checkbox_placeholder.checkbox(
        "Perform sentiment analysis (This may take a long time depending on the size of the dataset.)"
    )
    btn = btn_placeholder.button("Start training the model")

    if "model" not in st.session_state:
        st.session_state.model = None

    if "sentiment_analysis" not in st.session_state:
        st.session_state.sentiment_analysis = None

    if "df" not in st.session_state:
        st.session_state.df = None

    if (
        btn
        and st.session_state.model is None
        and st.session_state.sentiment_analysis is None
    ):
        st.spinner("Training the model...")
        with st.spinner("Training the model..."):
            # Hide the buttons
            btn_placeholder.empty()
            checkbox_placeholder.empty()

            if selected_embedding == "doc2vec":
                model = Top2Vec(
                    documents=df[selected_column].tolist(),
                    speed=selected_speed,
                    workers=get_num_cpu_cores(),
                    embedding_model=selected_embedding,
                )
            else:
                model = Top2Vec(
                    documents=df[selected_column].tolist(),
                    workers=get_num_cpu_cores(),
                    embedding_model=selected_embedding,
                )

        num_topics = model.get_num_topics()

        if sentiment_checkbox:
            with st.spinner("Analyzing sentiment..."):
                sentiment_analysis = get_sentiment_df(df, selected_column)
                # sentiment_analysis = []
                # for chunk in stqdm(np.array_split(df[selected_column], 10)):
                #     chunk_df = pd.DataFrame(chunk, columns=[selected_column])
                #     sentiment_analysis += get_sentiment_df(chunk_df, selected_column)

            col_name = "Predicted Sentiment"
            df = df.assign(**{col_name: sentiment_analysis})

            if len(sentiment_analysis) < num_topics:
                sentiment_analysis = sentiment_analysis + [""] * (
                    num_topics - len(sentiment_analysis)
                )
            else:
                sentiment_analysis = sentiment_analysis[:num_topics]

            st.session_state.sentiment_analysis = sentiment_analysis

        st.session_state.model = model
        st.session_state.df = df
        st.success("Model trained!")

    model = st.session_state.model
    sentiment_analysis = st.session_state.sentiment_analysis

    if model is None:
        st.stop()

    return model, df
