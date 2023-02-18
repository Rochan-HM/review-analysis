import multiprocessing
import spacy
import re
import nltk

import pandas as pd
import numpy as np
import streamlit as st

from io import BytesIO
from stqdm import stqdm
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from scipy.special import softmax
from transformers import (
    AutoModelForSequenceClassification,
    TFAutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
    AutoConfig,
)
from flair.data import Sentence
from flair.models import TextClassifier

from labelling.extract_cluster_labels import api as extract_cluster_labels


stqdm.pandas()

nltk.download("stopwords")
nltk.download("punkt")

sw = set(stopwords.words("english"))
nlp = spacy.load("en_core_web_sm")
sentiment_model = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
classifier = TextClassifier.load("en-sentiment")


@st.cache_data
def get_num_cpu_cores():
    """Get the number of CPU cores."""
    return multiprocessing.cpu_count()


def read_data_csv(path):
    """Read data from a CSV file."""
    try:
        return pd.read_csv(path, engine="python")
    except:
        raise Exception(
            "Could not read CSV file. Please check the encoding of the file."
        )


# Preprocess the data
def _remove_hyperlinks(text):
    """Remove hyperlinks and markup"""
    result = re.sub("<[a][^>]*>(.+?)</[a]>", "Link.", text)
    result = re.sub("&gt;", "", result)
    result = re.sub("&#x27;", "'", result)
    result = re.sub("&quot;", '"', result)
    result = re.sub("&#x2F;", " ", result)
    result = re.sub("<p>", " ", result)
    result = re.sub("</i>", "", result)
    result = re.sub("&#62;", "", result)
    result = re.sub("<i>", " ", result)
    result = re.sub("\n", "", result)
    return result


def _remove_numbers(text):
    """Remove numbers"""
    result = re.sub(r"\d+", "", text)
    return result


def _remove_punctuation(text):
    """Remove punctuation"""
    result = "".join(
        u for u in text if u not in ("?", ".", ";", ":", "!", '"', ",", "-")
    )
    return result


def _remove_whitespace(text):
    """Remove extra whitespace"""
    result = re.sub(" +", " ", text)
    return result


def _apply_preprocessing_text(text):
    """Apply preprocessing to a text"""

    text = _remove_hyperlinks(text)
    text = _remove_numbers(text)
    text = _remove_punctuation(text)
    text = _remove_whitespace(text)

    return text


def extract_labels(model):
    """
    Extract a label for the cluster based on most common verbs, objects, and nouns.
    """
    num_topics = model.get_num_topics()
    num_docs = model.get_topic_sizes()[0]

    input_sentences = []

    for i in stqdm(range(num_topics)):
        topic_docs, _, _ = model.search_documents_by_topic(
            i, num_docs=min(num_docs[i], 10)
        )
        topic_docs = list(set([t.lower().strip() for t in topic_docs]))

        top_sentences_concat = " ".join(topic_docs)
        top_sentences_concat = _apply_preprocessing_text(top_sentences_concat)
        input_sentences.append(top_sentences_concat)

    cluster_labels = extract_cluster_labels(input_sentences)

    return cluster_labels


@st.cache_data
def _export_df_to_excel(df: pd.DataFrame):
    """
    Export a dataframe to an excel file.
    """

    output = BytesIO()
    writer = pd.ExcelWriter(output, engine="xlsxwriter")
    df.to_excel(writer, sheet_name="Sheet1", index=False)
    writer.close()
    processed_data = output.getvalue()
    return processed_data


def display_dataframe(df, key, **kwargs):
    """
    Display a dataframe in a nice format.
    Also provide a download link to the dataframe.
    """

    st.dataframe(df, use_container_width=True, **kwargs)
    st.download_button(
        label="Download Data",
        data=_export_df_to_excel(df),
        file_name=f"{key}.xlsx",
        help="Download the data in the above table as an excel file.",
        key=f"{key}_download_button",
    )


def get_sentiment(text):
    """
    Get the sentiment of a text.
    """
    tokenizer = AutoTokenizer.from_pretrained(sentiment_model)
    config = AutoConfig.from_pretrained(sentiment_model)

    model = AutoModelForSequenceClassification.from_pretrained(sentiment_model)

    inp_text = tokenizer(text, return_tensors="pt")
    output = model(**inp_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    sentiment = np.argmax(scores)
    label = config.id2label[sentiment]

    return label


def get_sentiment_df2(df, text_col):
    """
    Get the sentiment of a dataframe.
    """
    task = pipeline(
        "sentiment-analysis", model=sentiment_model, tokenizer=sentiment_model
    )
    output = task(df[text_col].tolist())

    return [x["label"] for x in output]


def _predict_sentiment(text):
    """
    Get the sentiment of a text.
    """
    sentence = Sentence(text)
    classifier.predict(sentence)
    return sentence.labels[0].value.title()


def get_sentiment_df(df, text_col):
    """
    Get the sentiment of a dataframe.
    """
    return df[text_col].progress_apply(_predict_sentiment)
