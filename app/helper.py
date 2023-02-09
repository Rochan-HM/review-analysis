import multiprocessing
import spacy
import re
import nltk

import pandas as pd
import numpy as np
import streamlit as st

from io import BytesIO
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


nltk.download("stopwords")
nltk.download("punkt")

sw = set(stopwords.words("english"))
nlp = spacy.load("en_core_web_sm")
sentiment_model = f"cardiffnlp/twitter-roberta-base-sentiment-latest"


@st.cache
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
    result = "".join(u for u in text if u not in ("?", ".", ";", ":", "!", '"', ","))
    return result


def _remove_whitespace(text):
    """Remove extra whitespace"""
    result = re.sub(" +", " ", text)
    return result


def _remove_stopwords(text):
    """Remove stopwords"""
    result = [word.lower() for word in text.split() if word.lower() not in sw]
    return " ".join(result)


def _stem_words(text):
    """Stem words"""
    ss = SnowballStemmer("english")
    word_tok = nltk.word_tokenize(text)
    stemmed_words = [ss.stem(word) for word in word_tok]
    return " ".join(stemmed_words)


def _apply_preprocessing(df, column, new_column):
    """Apply preprocessing to a dataframe"""

    df[new_column] = df[column].apply(_remove_hyperlinks)
    df[new_column] = df[column].apply(_remove_numbers)
    df[new_column] = df[column].apply(_remove_punctuation)
    df[new_column] = df[column].apply(_remove_whitespace)
    df[new_column] = df[column].apply(_remove_stopwords)
    df[new_column] = df[column].apply(_stem_words)

    return df


def _compute_IDF(documents):
    """
    Compute the IDF for each word in the corpus.
    """

    res = []

    for doc in documents:
        doc = nlp(doc)
        temp_sentence = ""
        for token in doc:
            if token.pos_ in ["VERB", "NOUN", "ADJ"] or (token.dep_ == "dobj"):
                temp_sentence += token.lemma_.lower() + " "
        res.append(temp_sentence)

    count = Counter()
    for doc in res:
        count.update(set(doc.split()))

    total = np.sum(list(count.values()))
    idf = {word: round(np.log2(total / ct)) for word, ct in count.items()}
    return idf


def _get_suggested_titles(documents):
    """
    Get suggested titles for each cluster.
    """
    cluster_df = pd.DataFrame(documents, columns=["Text"])
    cluster_df = _apply_preprocessing(cluster_df, "Text", "Cleaned")

    idf = _compute_IDF(cluster_df["Cleaned"])
    label = _extract_cluster_label(cluster_df["Cleaned"], idf)

    return label


def _most_common(documents, n_words, idf):
    """
    Get the most common words in the corpus.
    """
    doc_count = Counter(documents)
    for w in list(doc_count):
        if doc_count[w] == 1:
            pass
        else:
            doc_count[w] = doc_count[w] * idf.get(w, 1)

    return doc_count.most_common(n_words)


def _extract_cluster_label(clustered_docs, idf):
    """
    Extract a label for the cluster based on most common verbs, objects, and nouns.
    """
    nlp = spacy.load("en_core_web_sm")

    verbs = []
    objects = []
    nouns = []
    adjectives = []

    for doc in nlp.pipe(clustered_docs):
        for tok in doc:
            if tok.is_stop or not str(tok).strip():
                continue
            if tok.dep_ == "VERB":
                verbs.append(tok.lemma_.lower())
            elif tok.dep_ == "dobj":
                objects.append(tok.lemma_.lower())
            elif tok.pos_ == "NOUN":
                nouns.append(tok.lemma_.lower())
            elif tok.pos_ == "ADJ":
                adjectives.append(tok.lemma_.lower())

    #  Get most common verbs, objects, nouns, and adjectives
    if verbs:
        verb = _most_common(verbs, 1, idf)[0][0]
    else:
        verb = ""

    if objects:
        obj = _most_common(objects, 1, idf)[0][0]
    else:
        obj = ""

    if nouns:
        noun = _most_common(nouns, 1, idf)[0][0]
    else:
        noun = ""

    if len(set(nouns)) > 1:
        noun2 = _most_common(nouns, 2, idf)[1][0]

    if adjectives:
        adj = _most_common(adjectives, 1, idf)[0][0]
    else:
        adj = ""

    # concatenate the most common verb-object-noun1
    label_words = [verb, obj]

    for word in [noun, noun2]:
        if word not in label_words:
            label_words.append(word)

    if "" in label_words:
        label_words.remove("")

    label = "_".join(label_words)

    return label


def extract_labels(model):
    """
    Extract a label for the cluster based on most common verbs, objects, and nouns.
    """

    num_topics = model.get_num_topics()
    num_docs = model.get_topic_sizes()[0]

    labels = []

    for i in range(num_topics):
        topic_docs = model.search_documents_by_topic(i, num_docs=num_docs[i])[0]
        label = _get_suggested_titles(topic_docs)
        labels.append(label)

    return labels


@st.cache
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


def get_sentiment_df(df, text_col):
    """
    Get the sentiment of a dataframe.
    """
    task = pipeline(
        "sentiment-analysis", model=sentiment_model, tokenizer=sentiment_model
    )
    output = task(df[text_col].tolist())

    return [x["label"] for x in output]
