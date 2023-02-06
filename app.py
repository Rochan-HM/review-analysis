import streamlit as st
import numpy as np
import pandas as pd

from top2vec import Top2Vec
from helper import *

st.set_page_config(layout="wide")
st.set_option("deprecation.showPyplotGlobalUse", False)

# Header
st.markdown(HEADER_HTML, unsafe_allow_html=True)

# Step 1: Ask user to upload a CSV file
st.markdown("## Step 1: Upload a CSV file")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# Preview the uploaded CSV file
if uploaded_file is None:
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

# Step 3: Ask user to select an embedding model
st.markdown("## Step 3: Select an embedding model")
with st.expander("What is an embedding model?"):
    st.markdown(
        "An embedding model is a way of representing text as a vector of numbers."
    )
    st.markdown(EMBEDDING_MODEL_MSG)

embedding_options = [
    "all-MiniLM-L6-v2",
    "doc2vec",
    # "universal-sentence-encoder",
    # "universal-sentence-encoder-large",
    # "universal-sentence-encoder-multilingual",
    # "universal-sentence-encoder-multilingual-large",
    "distiluse-base-multilingual-cased",
    "paraphrase-multilingual-MiniLM-L12-v2",
]
selected_embedding = st.selectbox("Select an embedding model", embedding_options)

if "selected_embedding" not in st.session_state:
    st.session_state.selected_embedding = None

if selected_embedding != st.session_state.selected_embedding:
    # Reset the model
    st.session_state.model = None
    st.session_state.predicted_topic_labels = None
    st.session_state.representative_df = None

st.session_state.selected_embedding = selected_embedding


# Ask user to select a learning speed
if selected_embedding == "doc2vec":
    st.markdown("### Select a learning speed")
    with st.expander("What is a learning speed?"):
        st.markdown(SPEED_MSG)

    speed_options = ["fast-learn", "learn", "deep-learn"]
    selected_speed = st.selectbox("Select a learning speed", speed_options)

# Step 4: Ask user to click a button to start training the model
st.markdown("## Step 4: Train the model")
st.markdown("This may take a few minutes...")

placeholder = st.empty()
btn = placeholder.button("Start training the model")

if "model" not in st.session_state:
    st.session_state.model = None

if "predicted_topic_labels" not in st.session_state:
    st.session_state.predicted_topic_labels = None

if "representative_df" not in st.session_state:
    st.session_state.representative_df = None

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

# Step 5: Analyze the model
st.markdown("## Step 5: Analyze the model")

topics, kw_search, query_search = st.tabs(
    ["Explore by Topics", "Explore by Keywords", "Explore by Query"]
)

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
    st.markdown("### Keywords")
    st.markdown("Enter a list of keywords separated by commas.")
    keyword_input = st.text_input("Keyword", key="keyword_input").strip()
    keywords = None

    try:
        keywords = keyword_input.split(",")
        keywords = [keyword.strip() for keyword in keywords]
    except:
        st.error("Invalid input")

    if not keywords or not all(keywords):
        st.stop()

    # Ask user to select a number of reviews to show
    st.markdown("### Select number of reviews to show")
    st.markdown("These are marked in descending order of similarity to the topic.")
    num_docs = st.number_input(
        "Number of reviews",
        min_value=1,
        max_value=1000,
        value=10,
        key="num_keywords_docs_kw",
    )

    search_btn = st.button("Search", key="kw_search_btn")

    if not search_btn:
        st.stop()

    try:
        with st.spinner("Searching..."):
            documents, document_scores, _ = model.search_documents_by_keywords(
                keywords, num_docs=num_docs
            )

            document_df = pd.DataFrame({"Review": documents, "Score": document_scores})
            display_dataframe(document_df, key="keyword_search_df")
    except ValueError:
        st.error(f"Could not find any documents with keywords: {keywords}", key="error")

with query_search:
    st.markdown("### Query")
    st.markdown("Enter a query to search for similar reviews.")
    query_input = st.text_input("Keyword", key="query_input").strip()

    # Ask user to select a number of reviews to show
    st.markdown("### Select number of reviews to show")
    st.markdown("These are marked in descending order of similarity to the topic.")
    num_docs = st.number_input(
        "Number of reviews",
        min_value=1,
        max_value=1000,
        value=10,
        key="num_keywords_docs_query",
    )

    search_btn = st.button("Search", key="query_search_btn")

    if not search_btn:
        st.stop()

    try:
        with st.spinner("Searching..."):
            documents, document_scores, _ = model.query_documents(
                query_input, num_docs=num_docs
            )

            document_df = pd.DataFrame({"Review": documents, "Score": document_scores})
            display_dataframe(document_df, key="query_search_df")
    except ValueError:
        st.error(
            f"Could not find any documents with query: {query_input}", key="query_error"
        )
