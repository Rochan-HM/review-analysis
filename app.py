import helper
import streamlit as st
import numpy as np
import pandas as pd

from top2vec import Top2Vec

st.set_page_config(layout="wide")
st.set_option("deprecation.showPyplotGlobalUse", False)

st.title("Review Analysis")

# Step 1: Ask user to upload a CSV file
st.markdown("## Step 1: Upload a CSV file")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# Preview the uploaded CSV file
if uploaded_file is None:
    st.stop()

try:
    with st.spinner("Loading data..."):
        df = helper.read_data_csv(uploaded_file)
except Exception as e:
    st.error(e)
    st.stop()

st.markdown("### Preview of uploaded CSV file")
df.dropna(inplace=True)
st.write(df.head(10))

# Step 2: Ask user to select a column
st.markdown("## Step 2: Select the column to analyze")
column_options = df.columns.tolist()
selected_column = st.selectbox("Select a column", column_options)

# Step 3: Ask user to select an embedding model
st.markdown("## Step 3: Select an embedding model")
with st.expander("What is an embedding model?"):
    st.markdown(
        "An embedding model is a way of representing text as a vector of numbers."
    )
    st.markdown(helper.EMBEDDING_MODEL)

embedding_options = [
    "doc2vec",
    "universal-sentence-encoder",
    "universal-sentence-encoder-large",
    "universal-sentence-encoder-multilingual",
    "universal-sentence-encoder-multilingual-large",
    "distiluse-base-multilingual-cased",
    "all-MiniLM-L6-v2",
    "paraphrase-multilingual-MiniLM-L12-v2",
]
selected_embedding = st.selectbox("Select an embedding model", embedding_options)

if "selected_embedding" not in st.session_state:
    st.session_state.selected_embedding = None

if selected_embedding != st.session_state.selected_embedding:
    # Reset the model
    st.session_state.model = None

st.session_state.selected_embedding = selected_embedding


# Ask user to select a learning speed
if selected_embedding == "doc2vec":
    st.markdown("### Select a learning speed")
    with st.expander("What is a learning speed?"):
        st.markdown(helper.SPEED)

    speed_options = ["fast-learn", "learn", "deep-learn"]
    selected_speed = st.selectbox("Select a learning speed", speed_options)

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
                workers=helper.get_num_cpu_cores(),
                embedding_model=selected_embedding,
            )
        else:
            # Hide the button
            placeholder.empty()

            model = Top2Vec(
                documents=df[selected_column].tolist(),
                workers=helper.get_num_cpu_cores(),
                embedding_model=selected_embedding,
            )

    st.session_state.model = model
    st.success("Model trained!")

model = st.session_state.model

if model is None:
    st.stop()

# Step 5: Analyze the model
st.markdown("## Step 5: Analyze the model")

topics, search = st.tabs(["Explore by Topics", "Explore by Keywords"])

with topics:
    st.markdown("### Topics")
    num_topics = model.get_num_topics()
    st.write(f"Number of topics: {num_topics}")

    # Show a dropdown to select a topic
    topic_options = [f"Topic {i}" for i in range(num_topics)]
    selected_topic = st.selectbox("Select a topic", topic_options)
    topic = int(selected_topic.split(" ")[1])

    # Show the top 10 words for the selected topic
    st.markdown("### Top 10 words for the selected topic")
    topic_words, word_scores, _ = model.get_topics()
    topic_words = topic_words[topic][:10]
    word_scores = word_scores[topic][:10]
    topics_df = pd.DataFrame({"Word": topic_words, "Score": word_scores})
    st.table(topics_df)

    # Show the word cloud for the selected topic
    st.markdown("### Word cloud for the selected topic")
    fig = model.generate_topic_wordcloud(topic)
    st.pyplot(fig=fig)

    # Ask user to select a number of reviews to show
    st.markdown("### Select number of reviews to show")
    st.markdown("These are marked in descending order of similarity to the topic.")
    num_docs = st.number_input(
        "Number of reviews", min_value=1, max_value=1000, value=10, key="num_topic_docs"
    )
    documents, document_scores, _ = model.search_documents_by_topic(
        topic, num_docs=num_docs
    )
    document_df = pd.DataFrame({"Review": documents, "Score": document_scores})
    st.table(document_df)

with search:
    st.markdown("### Keywords")
    st.markdown("Enter a list of keywords separated by commas.")
    keyword_input = st.text_input("Keyword").strip()
    keywords = None

    try:
        keywords = keyword_input.split(",")
        keywords = [keyword.strip() for keyword in keywords]
    except:
        st.error("Invalid input")

    print(keywords)

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
        key="num_keywords_docs",
    )

    search_btn = st.button("Search")

    if not search_btn:
        st.stop()

    try:
        with st.spinner("Searching..."):
            documents, document_scores, _ = model.search_documents_by_keywords(
                keywords, num_docs=num_docs
            )

            document_df = pd.DataFrame({"Review": documents, "Score": document_scores})
            st.table(document_df)
    except ValueError:
        st.error(f"Could not find any documents with keywords: {keywords}")
