import streamlit as st
import pandas as pd

from top2vec import Top2Vec
from stqdm import stqdm

from app.helper import *

stqdm.pandas()


def main(
    df: pd.DataFrame,
    selected_column: str,
    selected_embedding: str,
    selected_speed: str,
) -> Top2Vec:
    # Step 4: Ask user to click a button to start training the model
    st.markdown("## Step 4: Train the model")
    st.markdown("This may take a few minutes...")

    placeholder = st.empty()
    btn = placeholder.button("Start training the model")

    if "model" not in st.session_state:
        st.session_state.model = None

    if btn and st.session_state.model is None:
        st.spinner("Training the model...")
        with st.spinner("Training the model..."):
            if selected_embedding == "doc2vec":
                # Hide the button
                placeholder.empty()

                model = Top2Vec(
                    documents=df[selected_column].tolist(),
                    speed=selected_speed,
                    workers=get_num_cpu_cores(),
                    embedding_model=selected_embedding,
                )
            else:
                # Hide the button
                placeholder.empty()

                model = Top2Vec(
                    documents=df[selected_column].tolist(),
                    workers=get_num_cpu_cores(),
                    embedding_model=selected_embedding,
                )

        st.session_state.model = model
        st.success("Model trained!")

    model = st.session_state.model

    if model is None:
        st.stop()

    return model
