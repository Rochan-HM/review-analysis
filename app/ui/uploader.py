import streamlit as st
import pandas as pd

from typing import Tuple

from helper import *


def main() -> Tuple[pd.DataFrame, str]:
    # Step 1: Ask user to upload a CSV file
    st.markdown("## Step 1: Upload a CSV file")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    # Preview the uploaded CSV file
    if uploaded_file is None:
        st.session_state.clear()
        st.stop()

    try:
        df = read_data_csv(uploaded_file)
    except Exception as e:
        st.error(e)
        st.stop()

    st.markdown("### Preview of uploaded CSV file")
    df.dropna(inplace=True)
    display_dataframe(df.head(10), key="preview_df")

    # Step 2: Ask user to select a column
    st.markdown("## Step 2: Select the column to analyze")
    column_options = df.columns.tolist()
    selected_column = st.selectbox("Select a column", column_options)

    # Check if the column is a string column
    valid_column = pd.api.types.is_string_dtype(df[selected_column])

    if not valid_column:
        st.error("Please select a string column")
        st.stop()

    return df, selected_column
