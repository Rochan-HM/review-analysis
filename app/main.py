import streamlit as st

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
embedding_model, model_training_speed = embedding_model()

# Train the model
model, df = trainer(df, selected_column, embedding_model, model_training_speed)

# Show the model analysis
analyzer(df, model, selected_column)
